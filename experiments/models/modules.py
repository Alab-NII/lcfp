from itertools import combinations
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable
import torch.nn.functional as F

import math


def init_rnn(rnn, init_range, weights=None, biases=None, bidirectional=False):
    """Initializes RNN uniformly."""
    weights = weights or ['weight_ih_l0', 'weight_hh_l0']
    biases = biases or ['bias_ih_l0', 'bias_hh_l0']
    # Init weights
    for w in weights:
        rnn._parameters[w].data.uniform_(-init_range, init_range)
    # Init biases
    for b in biases:
        rnn._parameters[b].data.fill_(0)
    if bidirectional:
        reverse_weights = ['weight_ih_l0_reverse', 'weight_hh_l0_reverse']
        reverse_biases = ['bias_ih_l0_reverse', 'bias_hh_l0_reverse']
        for w in reverse_weights:
            rnn._parameters[w].data.uniform_(-init_range, init_range)
        for b in reverse_biases:
            rnn._parameters[b].data.fill_(0)


def init_rnn_cell(rnn, init_range):
    """Initializes RNNCell uniformly."""
    init_rnn(rnn, init_range, ['weight_ih', 'weight_hh'], ['bias_ih', 'bias_hh'])


def init_cont(cont, init_range):
    """Initializes a container uniformly."""
    for m in cont:
        if hasattr(m, 'weight'):
            m.weight.data.uniform_(-init_range, init_range)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0)


class RelationalContextEncoder(nn.Module):
    """A module that encodes dialogues context using an MLP."""
    def __init__(self, num_ent, dim_ent, rel_hidden, hidden_size, dropout, init_range, device):
        super(RelationalContextEncoder, self).__init__()

        self.input_size = num_ent * dim_ent + rel_hidden
        self.num_ent = num_ent
        self.dim_ent = dim_ent
        self.hidden_size = hidden_size
        self.device = device
        self.rel_hidden = rel_hidden

        self.relation = nn.Sequential(
            torch.nn.Linear(2 * self.dim_ent, self.rel_hidden),
            nn.Tanh()
        )

        self.fc1 = nn.Linear(self.input_size, self.hidden_size) 
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        init_cont([self.fc1, self.relation], init_range)

    def forward(self, ctx):
        ctx.to(self.device)
        ents = ctx.view(ctx.size(0), self.num_ent, self.dim_ent)
        rel = torch.cat([self.relation(torch.cat([ents[:,i,:],ents[:,j,:]], 1)) for i, j in combinations(range(7), 2)], 1)
        rel = torch.sum(rel.view(rel.size(0), self.rel_hidden, -1), 2)

        inpt = torch.cat([ctx, rel], 1)
        out = self.dropout(inpt)
        out = self.fc1(inpt)
        out = self.tanh(out)
        return out.unsqueeze(0)


class MlpContextEncoder(nn.Module):
    """A module that encodes dialogues context using an MLP."""
    def __init__(self, input_size, hidden_size, dropout, init_range, device):
        super(MlpContextEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = device

        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        init_cont([self.fc1], init_range)

    def forward(self, ctx):
        ctx.to(self.device)
        out = self.fc1(ctx)
        out = self.dropout(out)
        out = self.tanh(out)
        return out.unsqueeze(0)


class ResidualBlock3x3(nn.Module):
    
    def __init__(self, in_channel, out_channel, stride):
        super(ResidualBlock3x3, self).__init__()
        
        self.ksize = ksize = 3
        
        self.conv1 = nn.Conv2d(in_channel, out_channel, ksize, stride, int((ksize-1)//2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, ksize, 1, int((ksize-1)//2), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        self.activation = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.1)
        
        self.downsampler = None
        if (in_channel != out_channel) or stride != 0:
            self.downsampler = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channel),
            )
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsampler is not None:
            x = self.downsampler(x)
        
        out += x
        out = self.activation(out)
        
        return out


class ResidualBlockNew(nn.Module):

    def __init__(self, in_channel, out_channel, stride):
        super(ResidualBlockNew, self).__init__()

        self.ksize = ksize = 3

        self.conv1 = nn.Conv2d(in_channel, out_channel*2, ksize, 2*stride, int((ksize-1)//2), bias=False)
        self.bn1 = nn.BatchNorm2d(2*out_channel)
        self.conv2 = torch.nn.ConvTranspose2d(out_channel*2, out_channel, ksize+1, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel, ksize, 1, int((ksize-1)//2), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.activation = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.1)

        self.downsampler = None
        if (in_channel != out_channel) or stride != 0:
            self.downsampler = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        
        out = self.dropout(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsampler is not None:
            x = self.downsampler(x)

        out += x
        out = self.activation(out)

        return out


class ContextViewEncoder(nn.Module):
    """A module that encodes dialogues context using an MLP."""
    def __init__(self, n_object, hidden_size, context_view_size, dropout, init_range, device):
        super(ContextViewEncoder, self).__init__()

        self.n_object = n_object
        self.hidden_size = hidden_size
        self.scale_down = int(2**3)
        self.base_dim = base_dim = int(self.hidden_size//self.scale_down)
        #self.base_dim = base_dim = int(self.hidden_size//4)
        self.ctx_annots_dim = base_dim
        self.context_view_size = context_view_size
        self.dropout = dropout
        self.device = device
        
        block = ResidualBlockNew
        self.cnn_1 = nn.Sequential(
            block(1,          base_dim*1, 1),
            block(base_dim*1, base_dim*1, 1),
            block(base_dim*1, base_dim*1, 1),
        )
        self.cnn_2 = nn.Sequential(
            block(base_dim*1, base_dim*2, 2),
            block(base_dim*2, base_dim*4, 2),
            block(base_dim*4, base_dim*8, 2),
        )
        self.pos_emb_y = torch.nn.parameter.Parameter(torch.Tensor(1, self.hidden_size, self.get_output_l(), 1))
        self.pos_emb_x = torch.nn.parameter.Parameter(torch.Tensor(1, self.hidden_size, 1, self.get_output_l()))
        nn.init.kaiming_uniform_(self.pos_emb_y, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pos_emb_x, a=math.sqrt(5))
        
        #self.pos_emb_y_sub = torch.nn.parameter.Parameter(torch.Tensor(self.scale_down, 32))
        #self.pos_emb_x_sub = torch.nn.parameter.Parameter(torch.Tensor(self.scale_down, 32))
        #nn.init.kaiming_uniform_(self.pos_emb_y_sub, a=math.sqrt(5))
        #nn.init.kaiming_uniform_(self.pos_emb_x_sub, a=math.sqrt(5))
        
    def get_output_l(self):
        return self.context_view_size//self.scale_down
       
    def get_output_dim(self):
        return (self.hidden_size)*(self.get_output_l()**2)
    
    def forward(self, ctx_in):
        ctx_raw, ctx_view, ctx_view_annot = ctx_in
        n_batch, n_candidate, _ = ctx_view_annot.shape
        ny, nx = ctx_view.size()[-2:]
        
        # make a low-layer feature map
        y_ctx = 2 * ctx_view[:,None] - 1
        y_ctx = self.cnn_1(y_ctx)
        
        # make candidate vectors
        ids = nx*ctx_view_annot[:, :, 0] + ctx_view_annot[:, :, 1] + np.arange(n_batch)[:, None]*ny*nx
        ids = torch.from_numpy(ids)
        if self.device is not None:
            ids = ids.to(self.device)
        choice_vectors = y_ctx.permute(0,2,3,1).reshape((n_batch*ny*nx, -1))[ids]
        
        # make a high-layer feature map
        y_ctx = self.cnn_2(y_ctx)
        y_ctx += (self.pos_emb_y + self.pos_emb_x)
        
        return y_ctx, choice_vectors


class ContextPointEncoder(nn.Module):
    """A module that encodes dialogues context using an MLP."""
    def __init__(self, n_object, hidden_size, context_view_size, dropout, init_range, device):
        super(ContextPointEncoder, self).__init__()
        
        self.device = device
        self.n_point_feat = 4
        self.hidden_size = hidden_size
        self.ctx_annots_dim = hidden_size
        self.add_relation = True
        
        self.activation = nn.ReLU()
        self.last_activation = nn.ReLU()
        
        self.raw_feat_to_vec = nn.Sequential(
            nn.Linear(self.n_point_feat, self.hidden_size, bias=False),
            nn.BatchNorm1d(self.hidden_size),
            self.last_activation,
        )
        
        if self.add_relation:
            self.pairwise_relation_1 = nn.Sequential(
                nn.Linear(self.n_point_feat, self.hidden_size, bias=False),
                nn.BatchNorm1d(self.hidden_size),
                self.activation,
                nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                nn.BatchNorm1d(self.hidden_size),
                self.activation,
                nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                nn.BatchNorm1d(self.hidden_size),
                self.activation,
            )
            self.pairwise_relation_2 = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                nn.BatchNorm1d(self.hidden_size),
                self.last_activation,
            )
    
    def forward(self, ctx_in):
        ctx_raw = ctx_in[0]
        batch_size = ctx_raw.size(0)
        n_obj = ctx_raw.size(1) // self.n_point_feat
        
        v_sep = self.raw_feat_to_vec(torch.reshape(ctx_raw, (-1, self.n_point_feat)))
        v_sep = v_sep.reshape(batch_size, -1, self.hidden_size)
        y_ctx = v_sep
        
        if self.add_relation:
            ctx_raw_1d = ctx_raw.reshape(batch_size, n_obj, self.n_point_feat)
            #v_rel = torch.cat((ctx_raw_1d[:,:,None,:].repeat(1,1,n_obj,1), ctx_raw_1d[:,None,:,:].repeat(1,n_obj,1,1)), axis=3).reshape(-1, 2*self.n_point_feat)
            v_rel = (ctx_raw_1d[:,:,None,:] - ctx_raw_1d[:,None,:,:]).reshape(-1, self.n_point_feat)
            #v_rel = (v_sep[:,:,None,:] - v_sep[:,None,:,:]).reshape(-1, self.hidden_size)
            v_rel = self.pairwise_relation_1(v_rel).reshape(batch_size, n_obj, n_obj, -1)
            mask = 1 - torch.eye(n_obj, device=self.device)[None,:,:,None]
            v_rel = (v_rel*mask).sum(axis=2) #/ (n_obj*(n_obj-1))
            v_rel = self.pairwise_relation_2(v_rel.reshape(-1, self.hidden_size)).reshape(batch_size, -1, self.hidden_size)
            y_ctx += v_rel
        
        y_candidates = y_ctx
        
        if self.training:
            y_ctx = y_ctx[:, torch.randperm(y_ctx.size(1))]
        
        return y_ctx, y_candidates

