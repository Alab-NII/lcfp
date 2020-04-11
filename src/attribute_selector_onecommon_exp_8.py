# coding: utf-8


import os
import json
import time
import inspect

import numpy as np
import random
import torch
import torchvision
from torchvision.ops import misc as misc_nn_ops
from torch.utils.data import DataLoader

try:
    import custom_dataset
except ModuleNotFoundError as e:
    import sys
    sys.path += ['src']

from custom_dataset import collate_fn, UnifiedTextifier, VisualSelectionDataset


class TrainConfig(object):
    
    name_space = 'attribute_selector_onecommon_exp_8'
    
    # Randomness
    rseed = 8 # None
    cudnn_deterministic = True
    
    # Task definition
    provide_image = False
    coco_image_directories = ['data/img/val2014', 'data/img/train2014']
    dataset_list = {
        'train': [
            {'task_name':'onecommon.selection', 'path':'onecommon_data/converted/train.json', 'provide_image':provide_image},
            #{'task_name':'guesswhat.selection', 'path':'data/guesswhat.train.jsonl', 
            #     'success_only':True, 'coco_image_directories':coco_image_directories},
        ],
        'valid': [
            {'task_name':'onecommon.selection', 'path':'onecommon_data/converted/valid.json', 'provide_image':provide_image},
            #{'task_name':'guesswhat.selection', 'path':'data/guesswhat.valid.jsonl', 
            #     'success_only':True, 'coco_image_directories':coco_image_directories},
        ],
    }
    
    # Model definition
    textifier_dict = 'cache/onecommon_textifier_train_u5.json'
    textifier_use_more_than = 5
    train_textifier_ratio_force_unk = 0
    train_optional_info_ratio_force_zero = 0.01
    model_net_args = {
            'rnn_type': 'gru',
            'n_rnn_directions': 1,
            'n_rnn_layers': 1,
            'dim_token': 256,
            'dim_lang': 512,
            'dim_feature': 256,
            'dim_mlp':1024,
            'dim_object_optional_info': 4,
            'dropout_ratio_lang': 0,
    }
    
    # Optimization
    n_workers = 1
    device_name = 'cuda:0'
    n_epoch = 10
    minibatch_size = 32
    optimizer_name = 'Adam'
    optimizer_args = {'lr':5e-4, 'eps':1e-9, 'weight_decay':0, 'betas':(0.9, 0.999)}
    optimizer_scheduler = None 
    # optimizer_scheduler = {'name': 'StepLR', 'step_size':2, 'gamma':0.5}
    
    # Others
    models_path = 'models'
    cache_path = 'cache'
    weight_name_template = 'weight_ep%d'
    history_name = 'history.txt'
    
    def __init__(self):
        self.model_dir = os.path.join(self.models_path, self.name_space)
        self.weight_path_template = os.path.join(self.model_dir, self.weight_name_template)
        self.history_path = os.path.join(self.model_dir, self.history_name)
    
    @property
    def device(self):
        return torch.device(self.device_name)
        

class SummaryHolder(object):
    
    formatter = {
        float: lambda x: '%.3f'%x,
        None: lambda x: str(x),
    }    

    def __setattr__(self, n, v):
        if not hasattr(self, '_name_list'):
            super(SummaryHolder, self).__setattr__('_name_list', [])
        self._name_list.append(n)
        super(SummaryHolder, self).__setattr__(n, v)
    
    def to_str(self, *name_list, prefix='', no_name=False):
        cells = []
        for n in (name_list or self._name_list):
            v = getattr(self, n)
            v = self.formatter.get(type(v), self.formatter[None])(v)
            cells.append(v if no_name else '%s=%s'%(prefix+n, v))
        return ' '.join(cells)


def run_dataset(config, net, opt, data_loader, i_epoch):
    
    start_time = time.time()
    n_samples = 0
    loss_sum = 0
    n_corrects = 0
    
    def summarize():
        _div = lambda x, y: x / y if y != 0 else 0.
        sh = SummaryHolder()
        sh.s = time.time() - start_time
        sh.n = n_samples
        sh.loss = _div(loss_sum, n_samples)
        sh.acc = _div(n_corrects, n_samples)
        return sh
    
    for i_mb, mb in enumerate(data_loader):
        mb = {k:v.to(config.device) if hasattr(v, 'to') else v for k, v in mb.items()}
        ups = net(mb['image'], mb['tokens'], mb['n_tokens'], mb['object_bboxes'], mb['object_optional_info'])
        loss, is_correct = net.calc_loss(ups, mb['n_objects'], mb['ground_truth_id'])
        
        if net.training:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        n_samples += mb['size']
        loss_sum += (loss.item() * mb['size'])
        n_corrects += is_correct.sum().item()
        
        if (i_mb + 1) % 100 == 0:
            print(i_epoch, summarize().to_str())
    
    return summarize()


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def train(config):
    
    # Reproductivity
    if config.rseed is not None:
        set_random_seed(config.rseed)
    torch.backends.cudnn.deterministic = config.cudnn_deterministic
    
    # Initialization on Environment
    if not os.path.exists(config.cache_path):
        os.makedirs(config.cache_path)
    
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)    
    print('model_dir', config.model_dir)
    
    with open(config.history_path, 'w') as f:
        f.write('')
    
    # Tokenizer
    print('#', 'initializing textifiers', '...')
    textifier = UnifiedTextifier.load_or_make(config.textifier_dict,
        config.dataset_list['train'], config.textifier_use_more_than,
    )
    print('textifier vocab size', textifier.get_len())
    
    # Dataset & Data Loader
    print('#', 'loading datasets', '...')
    datasets = {}
    for key, dataset_list in config.dataset_list.items():
        is_train = key == 'train'
        datasets[key] = dataset = VisualSelectionDataset(
            dataset_list=dataset_list,
            textifier=textifier,
            provide_image=config.provide_image,
            image_shape=(1, 1),
            dim_object_optional_info=config.model_net_args['dim_object_optional_info'],
            ratio_force_unk=config.train_textifier_ratio_force_unk if is_train else 0,
            ratio_force_zero=config.train_optional_info_ratio_force_zero if is_train else 0,
        )
        print('len dataset', key, len(dataset))
    
    # Model
    # Some modifications will be required to use multi GPU
    print('#', 'constructing a model', '...')
    model_net = SelectorNet(textifier.get_len(), **config.model_net_args)
    model_net.to(config.device)
    model_net.device = config.device

    # Optimizer
    opt = getattr(torch.optim, config.optimizer_name)(
        filter(lambda x: x.requires_grad, model_net.parameters()), 
        **config.optimizer_args
    )
    scheduler = None
    if config.optimizer_scheduler is not None:
        s_args = config.optimizer_scheduler.copy()
        name = s_args.pop('name')
        scheduler = getattr(torch.optim.lr_scheduler, name)(opt, **s_args)
    
    # Training loop
    print('#', 'training loop starts')
    print('n_epoch', config.n_epoch)
    for i_epoch in range(config.n_epoch):
        # a setter for worker random number generator's seed
        def worker_init_fn(_id):
            if config.rseed is not None:
                seed = config.rseed + (config.n_workers + 1) * i_epoch + _id
                random.seed(seed)
                np.random.seed(seed)
        
        # training
        data_loader = DataLoader(
            datasets['train'], 
            batch_size=config.minibatch_size, 
            num_workers=config.n_workers, collate_fn=collate_fn, shuffle=True,
            worker_init_fn=worker_init_fn,
        )
        model_net.train()
        train_summary = run_dataset(config, model_net, opt, data_loader, i_epoch)
        if scheduler is not None:
            scheduler.step()
        
        # validation
        data_loader = DataLoader(
            datasets['valid'], 
            batch_size=config.minibatch_size, 
            num_workers=config.n_workers, collate_fn=collate_fn, shuffle=False,
        )
        model_net.eval()
        with torch.no_grad():
            valid_summary = run_dataset(config, model_net, opt, data_loader, i_epoch)
        
        # Save states
        weight_path = config.weight_path_template%(i_epoch)
        checkpoint = {
            'model_net_state_dict': model_net.state_dict(),
        }
        torch.save(checkpoint, weight_path)
        
        # Save history
        with open(config.history_path, 'a') as f:
            f.write(' '.join([
                str(i_epoch),
                train_summary.to_str('loss', 'acc', no_name=True),
                valid_summary.to_str('loss', 'acc', no_name=True),
            ]) + '\n')
        
        print(' '.join([
            'ep=%d'%(i_epoch),
            train_summary.to_str(prefix='t_'),
            valid_summary.to_str(prefix='v_'),
        ]))


class SelectorNet(torch.nn.Module):
    
    def __init__(self, n_tokens,
            rnn_type='gru',
            n_rnn_directions=1,
            n_rnn_layers=1,
            dim_token=256,
            dim_lang=1024,
            dim_feature = 256,
            dim_mlp=1024,
            dim_object_optional_info=4,  # x_center, y_center, color, size
            dropout_ratio_lang=0,
        ):
        super(SelectorNet, self).__init__()
        
        # set arguments as sttributes
        local_dict = locals()
        for a in inspect.getfullargspec(self.__init__).args:
            (a == 'self') or setattr(self, a, local_dict[a])
        del local_dict
        
        self.n_tokens, self.n_sub_tokens = self.n_tokens['main'], self.n_tokens['sub']
        self.device = torch.device('cpu')
        self.dropout_lang = torch.nn.Dropout(self.dropout_ratio_lang)
        
        # Language encoder
        self.embed_token = torch.nn.Embedding(self.n_tokens, self.dim_token, padding_idx=0)
        
        rnn_module = torch.nn.LSTM if self.rnn_type=='lstm' else torch.nn.GRU
        self.rnn_lang = rnn_module(
            input_size=self.dim_token,
            hidden_size=self.dim_lang // self.n_rnn_directions,
            bidirectional=(self.n_rnn_directions == 2),
            num_layers=self.n_rnn_layers,
            dropout=self.dropout_ratio_lang if self.n_rnn_layers > 1 else 0,
            bias=True,
            batch_first=True,
        )
        
        # Candidate Embedding
        self.to_feature = torch.nn.Sequential(
            torch.nn.Linear(4, self.dim_feature),
            torch.nn.ReLU(inplace=True),
        )
        
        self.to_relation = torch.nn.Sequential(
            torch.nn.Linear(4, self.dim_feature),
            torch.nn.ReLU(inplace=True),
        )
        
        dim_pre_mlp = 2*self.dim_feature + self.dim_lang
        self.to_logit = torch.nn.Sequential(
            torch.nn.Linear(dim_pre_mlp, self.dim_mlp),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.dim_mlp, 1),
        )
    
    def forward(self, images, tokens, n_tokens, obj_bboxes, obj_optional_info):
        """returns unnormalized probability maps"""
        
        h_lang = self._forward_lang(tokens, n_tokens)
        
        n_obj = obj_optional_info.size(1)
        obj_feats = self.to_feature(obj_optional_info)
        
        diff = obj_optional_info[:, :, None, :] - obj_optional_info[:, None, :, :]
        rel_feats = self.to_relation(diff).mean(axis=2)
        
        feat_all = torch.cat((obj_feats, rel_feats, h_lang[:,None,:].repeat(1, n_obj, 1)), axis=2)
        output = self.to_logit(feat_all)[:, :, 0]
        return output
    
    def _forward_lang(self, tokens, n_tokens):
        
        embs = self.embed_token(tokens)
        embs = self.dropout_lang(embs)
        packed_embs = torch.nn.utils.rnn.pack_padded_sequence(
                embs, n_tokens, batch_first=True, enforce_sorted=False)
        _, _h = self.rnn_lang(packed_embs)
        h_lang = _h[0] if isinstance(_h, tuple) else _h
        h_lang = h_lang.permute(1, 0, 2).view(h_lang.size(1), -1)
        return h_lang
    
    def calc_loss(self, ups, n_objs, target_obj_ids, with_is_correct=True):
        
        batch_size, n_max = ups.size()
        arange = torch.arange(0, n_max, device=self.device)[None]
        mask = arange < n_objs[:, None]
        label_mask = arange == target_obj_ids[:, None]
        
        ups_filled = ups.masked_fill(~mask, float('-inf'))
        y = ups_filled.logsumexp(axis=1) - (ups * label_mask).sum(axis=1)
        y = y.sum(axis=0) / batch_size
        
        if not with_is_correct:
            return y
        
        pred_ids = ups_filled.detach().argmax(axis=1)
        is_correct = pred_ids == target_obj_ids
        
        return y, is_correct


if __name__ == '__main__':
    config = TrainConfig()
    train(config)
