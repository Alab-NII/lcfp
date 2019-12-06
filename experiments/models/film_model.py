import sys
import re
import pdb
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable
import torch.nn.functional as F

from data import STOP_TOKENS
from domain import get_domain
from models import modules


class SelectModel(nn.Module):
    def __init__(self, word_dict, output_length, args, device):
        super(SelectModel, self).__init__()

        domain = get_domain(args.domain)

        self.word_dict = word_dict
        self.args = args
        self.device = device
        self.num_ent = domain.num_ent()
        self.n_obj_attributes = 4
        self.feat_type = args.feat_type
        self.n_gru_direction = 1
        self.n_gru_layers = 1
        self.n_blocks = 4
        self.map_size = 16
        
        if self.feat_type != 'point':
            raise ValueError('feat_type supports only point on this model now.')
        
        if self.args.context_only:
            raise ValueError('conext_only model is not supported.')
        
        if args.use_attention:
            raise ValueError('use_attention is not supported.')
        
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()
        
        self.token_embed = nn.Embedding(len(self.word_dict), args.nembed_word)
        
        self.dialogue_rnn = nn.GRU(
            input_size=args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True,
            bidirectional=(self.n_gru_direction == 2),
            batch_first=True,
            num_layers=self.n_gru_layers,
            dropout=args.dropout
        )
        dialogue_hidden_dim = args.nhid_lang * self.n_gru_direction
        
        self.film_fc = nn.Linear(dialogue_hidden_dim, args.nembed_ctx * self.n_blocks * 2)
        
        self.first_conv = nn.Sequential(
            nn.Conv2d(self.n_obj_attributes, args.nembed_ctx, 1, 1, 0),
            nn.BatchNorm2d(args.nembed_ctx),
            self.relu,
        )
        self.res_blocks = nn.ModuleList([FilmResBlock(args.nembed_ctx, 3) for _ in range(self.n_blocks)])
        self.last_conv = nn.Conv2d(args.nembed_ctx+2, args.nhid_sel, 1, 1, 0)
        
        self.to_prob = nn.Sequential(
            self.dropout,
            nn.Linear(args.nhid_sel, args.nhid_sel, bias=False),
            #nn.BatchNorm1d(args.nhid_sel),
            self.relu,
            self.dropout,
            nn.Linear(args.nhid_sel, args.nhid_sel, bias=False),
            #nn.BatchNorm1d(args.nhid_sel),
            self.relu,
            nn.Linear(args.nhid_sel, 1, bias=False),
        )
        self.to_prob_shuffle_task = nn.Linear(args.nhid_sel, 2, bias=True)
        
        #self.init_weights()
    
    def init_weights(self):
        
        def init_weights(m, init_range=self.args.init_range):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                modules.init_rnn(m, init_range, bidirectional=m.bidirectional)
            else:
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.data.uniform_(-init_range, init_range)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        print('start init_weights')
        self.apply(init_weights)

    def forward_selection(self, sel_in):
        """Forwards selection pass."""
        task_name, words, words_original_len, words_mask, ctx_in = sel_in
        seq_len, batch_size = words.size()
        
        # encode dialogue text
        token_xs = self.token_embed(words)
        token_xs = token_xs.permute(1, 0, 2)
        rnn_x = self.dropout(token_xs)
        rnn_x = nn.utils.rnn.pack_padded_sequence(rnn_x, words_original_len, batch_first=True, enforce_sorted=False)
        _, h_dialogue = self.dialogue_rnn(rnn_x)
        h_dialogue = h_dialogue.view(self.n_gru_layers, self.n_gru_direction, batch_size, self.args.nhid_lang).sum(axis=0)
        h_dialogue = h_dialogue.permute(1, 0, 2).reshape(-1, self.args.nhid_lang * self.n_gru_direction)
        
        # make film beta-gamma
        betagamma = self.film_fc(h_dialogue).view(-1, self.n_blocks, 2, self.args.nembed_ctx)
        
        # encode spatial ctx
        x, cordinates = self.get_ctx_map_2d(ctx_in)
        x = self.first_conv(x)
        for n, block in enumerate(self.res_blocks):
            x = block(x, betagamma[:, n])
        x = self.last_conv(positional_encode(x))
        
        # predict answer
        # shuffle task
        if task_name == 'shuffle':
            x = x.max(axis=2)[0].max(axis=2)[0]
            return self.to_prob_shuffle_task(x)
        
        # selection task
        x = self.relu(x)
        choice_vectors = self.select_feat_by_cordinate(x, cordinates)
        # (batch_size, candidates, hidden_sel)
        choice_vectors = choice_vectors.reshape(-1, self.args.nhid_sel)
        choice_vectors = self.to_prob(choice_vectors).reshape(batch_size, -1)
        return choice_vectors

    def forward_context(self, ctx_in):
        """Run context encoder."""
        # we will use ctx data after gru
        return ctx_in

    def get_ctx_map_2d(self, ctx_in):
        """Map ctx array to 2d array.
        I use this form to enable to embed calculated values into 2d maps.
        If you make 2d maps just from raw attributes, you can calculte them
        at the data preparation phase in advance."""
        ctx_raw = ctx_in[0]
        batch_size = ctx_raw.size(0)
        n_obj = ctx_raw.size(1) // self.n_obj_attributes
        
        y_map = torch.zeros((batch_size, self.n_obj_attributes, self.map_size*self.map_size), device=self.device)
        attributes = ctx_raw.reshape(batch_size, n_obj, self.n_obj_attributes)
        cords = attributes[:,:,:2].cpu().numpy()
        cords = ((self.map_size-1)*(cords + 1)*0.5).astype(np.int32)
        cords = self.map_size*cords[:,:,1] + cords[:,:,0]
        
        cords = torch.from_numpy(cords[:, :, None, None])
        id_range = torch.from_numpy(np.arange(0, self.map_size*self.map_size)[None, None]) # 1, 1, self.map_size*self.map_size
        if self.device is not None:
            cords = cords.to(self.device)
            id_range = id_range.to(self.device)
        
        for i in range(n_obj):
            h = attributes[:, i, :] # batch_size, hidden_size
            c = cords[:, i] # batch_size
            y_map = torch.where(id_range == c, h[:,:,None], y_map)
        y_map = y_map.reshape(batch_size, -1, self.map_size, self.map_size)
        return y_map, cords[:,:,0,0]
    
    def select_feat_by_cordinate(self, x, cords):
        batch_size, feat_dim, ny, nx = x.size()
        offset = torch.arange(batch_size, device=self.device)[:, None]*ny*nx
        choice_vectors = x.permute(0,2,3,1).reshape((-1, feat_dim))[offset+cords]
        return choice_vectors
        

def positional_encode(images):
    """Copied from https://github.com/Lyusungwon/film_pytorch/blob/master/utils.py"""
    
    try:
        device = images.get_device()
    except:
        device = torch.device('cpu')
    n, c, h, w = images.size()
    x_coordinate = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(n, 1, h, w).to(device)
    y_coordinate = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(n, 1, h, w).to(device)
    images = torch.cat([images, x_coordinate, y_coordinate], 1)
    return images


class FilmResBlock(nn.Module):
    """Copied from https://github.com/Lyusungwon/film_pytorch/blob/master/film.py"""
    
    def __init__(self, filter, kernel):
        super(FilmResBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter + 2, filter, 1, 1, 0)
        self.conv2 = nn.Conv2d(filter, filter, kernel, 1, (kernel - 1)//2)
        self.batch_norm = nn.BatchNorm2d(filter)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, betagamma):
        x = positional_encode(x)
        x = self.relu(self.conv1(x))
        residual = x
        beta = betagamma[:, 0].unsqueeze(2).unsqueeze(3).expand_as(x)
        gamma = betagamma[:, 1].unsqueeze(2).unsqueeze(3).expand_as(x)
        x = self.batch_norm(self.conv2(x))
        x = self.relu(x * beta + gamma)
        x = x + residual
        return x

