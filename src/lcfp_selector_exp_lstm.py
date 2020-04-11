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
    
    name_space = 'lcfp_selector_exp_lstm'
    
    # Randomness
    rseed = 0 # None
    cudnn_deterministic = True
    
    # Task definition
    provide_image = True
    coco_image_directories = ['data/img/val2014', 'data/img/train2014']
    dataset_list = {
        'train': [
            #{'task_name':'onecommon.selection', 'path':'onecommon_data/converted/train.json'},
            {'task_name':'guesswhat.selection', 'path':'data/guesswhat.train.jsonl', 
                 'success_only':True, 'coco_image_directories':coco_image_directories},
        ],
        'valid': [
            #{'task_name':'onecommon.selection', 'path':'onecommon_data/converted/valid.json'},
            {'task_name':'guesswhat.selection', 'path':'data/guesswhat.valid.jsonl', 
                 'success_only':True, 'coco_image_directories':coco_image_directories},
        ],
    }
    
    # Model definition
    textifier_dict = 'cache/guesswhat_textifier_train_u5.json'
    textifier_use_more_than = 5
    train_textifier_ratio_force_unk = 0
    train_optional_info_ratio_force_zero = 0.01
    model_net_args = {
        'image_shape': (224, 224),
        'use_bbox': True,
        'resnet_name': 'resnet50',
        'used_pyramids': ('l4', 'l3', 'l2', 'l1', 'l0'),
        'rnn_type': 'lstm',
        'n_rnn_directions': 1,
        'n_rnn_layers': 1,
        'dim_token': 256,
        'dim_lang': 1024,
        'dim_feature': 256,
        'dim_object_optional_info': 256,#256,
        'dim_mlp': 1024,
        'pooling_method': 'mean',
        'dropout_ratio_lang': 0,
        'dropout_ratio_feat': 0,
        'dropout_ratio_mlp': 0,
        'coordinate_noise': 0,
    }
    
    # Optimization
    n_workers = 1
    device_name = 'cuda:0'
    n_epoch = 5
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
            image_shape=config.model_net_args['image_shape'],
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


def positional_encode(images):
    
    try:
        device = images.get_device()
    except:
        device = -1
    if device < 0:
        device = torch.device('cpu')
    
    n, c, h, w = images.size()
    x_coordinate = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(n, 1, h, w).to(device)
    y_coordinate = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(n, 1, h, w).to(device)
    images = torch.cat([images, x_coordinate, y_coordinate], 1)
    return images


class FilmBlock(torch.nn.Module):
    
    def __init__(self, in_channel, out_channel, ksize):
        super(FilmBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel + 2, out_channel, 1, 1, 0)
        self.conv2 = torch.nn.Conv2d(out_channel, out_channel, ksize, 1, (ksize - 1)//2)
        self.batch_norm = torch.nn.BatchNorm2d(out_channel)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x, c):
        x = positional_encode(x)
        x = self.relu(self.conv1(x))
        residual = x
        beta = c[:, 0].unsqueeze(2).unsqueeze(3).expand_as(x)
        gamma = c[:, 1].unsqueeze(2).unsqueeze(3).expand_as(x)
        x = self.batch_norm(self.conv2(x))
        x = self.relu(x * beta + gamma)
        x = x + residual
        return x


class ImageNormalizer(torch.nn.Module):
    
    def __init__(self, mean, std):
        super(ImageNormalizer, self).__init__()
        self.mean = torch.nn.parameter.Parameter(torch.as_tensor(mean)[None, :, None, None], requires_grad=False)
        self.std = torch.nn.parameter.Parameter(torch.as_tensor(std)[None, :, None, None], requires_grad=False)

    def forward(self, x):
        return x.sub_(self.mean).div_(self.std)


class SelectorNet(torch.nn.Module):
    
    # Information about pre-trained resnet
    all_pyramids = ('l4', 'l3', 'l2', 'l1', 'l0')
    defined_dim_pyramids = {
        'resnet50': {'l4':2048, 'l3':1024, 'l2':512, 'l1':256, 'l0':64},
        'resnet34': {'l4':512, 'l3':256, 'l2':128, 'l1':64, 'l0':64},
        'resnet18': {'l4':512, 'l3':256, 'l2':128, 'l1':64, 'l0':64},
    }
    sf_pyramids = {'l4':2, 'l3':2, 'l2':2, 'l1':2, 'l0':1}
    last_scale_factor = 2
    
    # Values from torchvision documentation
    image_normalization_mean = [0.485, 0.456, 0.406]
    image_normalization_std = [0.229, 0.224, 0.225]
    
    def __init__(self, n_tokens,
            image_shape=(224, 224),
            use_bbox=True,
            resnet_name='resnet50',
            used_pyramids=('l4', 'l3', 'l2', 'l1', 'l0'),
            rnn_type='gru',
            n_rnn_directions=1,
            n_rnn_layers=1,
            dim_token=256,
            dim_lang=1024,
            dim_feature=256,
            dim_object_optional_info=512,
            dim_mlp=1024,
            pooling_method='mean',
            dropout_ratio_lang=0,
            dropout_ratio_feat=0,
            dropout_ratio_mlp=0,
            coordinate_noise=0,
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
        self.dropout_feat = torch.nn.Dropout(self.dropout_ratio_feat)
        self.n_used_pyramids = len(self.used_pyramids)
        
        if self.pooling_method == 'mean':
            self.pooling = lambda x: x.mean(axis=(-1, -2))
        else:
            self.pooling = lambda x: x.max(dim=-1)[0].max(dim=-1)[0]
        
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
        
        # Image feature extracter
        self.image_normalizer = ImageNormalizer(
            mean=self.image_normalization_mean,
            std=self.image_normalization_std,
        )
        self.resnet = getattr(torchvision.models, self.resnet_name)(
            pretrained=True,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d
        )
        self.resnet.requires_grad_(False)
        self.dim_pyramids = self.defined_dim_pyramids[self.resnet_name]
        
        # Vision-language interaction
        self.film_fc = torch.nn.Linear(self.dim_lang, self.n_used_pyramids * 2 * self.dim_feature)
        self.film_blocks = torch.nn.ModuleDict({
            k: FilmBlock(d, self.dim_feature, 3) for k, d in self.dim_pyramids.items()
        })
        self.upsamplers = torch.nn.ModuleDict({
            k: torch.nn.Upsample(scale_factor=sf, mode='nearest') if sf != 1 else 
                torch.nn.Identity() for k, sf in self.sf_pyramids.items()
        })
        
        # Optional Information
        self.embed_optional_info = torch.nn.Embedding(
            self.n_sub_tokens, self.dim_object_optional_info, padding_idx=0)
        
        # Probability
        dim_pre_mlp = self.dim_feature + self.dim_object_optional_info
        self.to_logit = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout_ratio_mlp),
            torch.nn.Linear(dim_pre_mlp, self.dim_mlp),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.dim_mlp, 1),
        )
    
    def forward(self, images, tokens, n_tokens, obj_bboxes, obj_optional_info):
        """returns unnormalized probability maps"""
        
        images = self.image_normalizer(images)
        image_feats = self._forward_resnet(images)
        
        h_lang = self._forward_lang(tokens, n_tokens)
        
        feat_all = self._forward_fuse(h_lang, image_feats)
        output = self._forward_candidates(feat_all, obj_bboxes, obj_optional_info)
        return output
    
    def _forward_resnet(self, x):
        
        resnet = self.resnet
        x = resnet.conv1(x)
        x = resnet.bn1(x)
        x = l0 = resnet.relu(x)
        x = resnet.maxpool(x)
        x = l1 = resnet.layer1(x)
        x = l2 = resnet.layer2(x)
        x = l3 = resnet.layer3(x)
        x = l4 = resnet.layer4(x)
        
        return {'l4':l4, 'l3':l3, 'l2':l2, 'l1':l1, 'l0':l0}
    
    def _forward_lang(self, tokens, n_tokens):
        
        embs = self.embed_token(tokens)
        embs = self.dropout_lang(embs)
        packed_embs = torch.nn.utils.rnn.pack_padded_sequence(
                embs, n_tokens, batch_first=True, enforce_sorted=False)
        _, _h = self.rnn_lang(packed_embs)
        h_lang = _h[0] if isinstance(_h, tuple) else _h
        h_lang = h_lang.permute(1, 0, 2).view(h_lang.size(1), -1)
        return h_lang
    
    def _forward_fuse(self, h_lang, image_feats):
        
        betagamma = self.film_fc(h_lang).view(-1, self.n_used_pyramids, 2, self.dim_feature)
        
        batch_size, _, h, w = image_feats[self.all_pyramids[0]].size()
        feat_all = torch.zeros((batch_size, self.dim_feature, h, w), device=self.device)
        
        n_used_film = 0
        for k in self.all_pyramids:
            if k in self.used_pyramids:
                feat_local = self.film_blocks[k](image_feats[k], betagamma[:, n_used_film])
                feat_local = self.dropout_feat(feat_local)
                feat_all += feat_local
                n_used_film += 1
            feat_all = self.upsamplers[k](feat_all)
        
        return feat_all
    
    def _forward_candidates(self, feat_all, obj_bboxes, obj_optional_info):
        
        if self.use_bbox:
            obj_feats = self._forward_map_select_bbox(feat_all, obj_bboxes)
        else:
            obj_centers = 0.5*(obj_bboxes[:,:,:2] + obj_bboxes[:,:,2:])
            obj_feats = self._forward_map_select_center(feat_all, obj_centers)
        
        obj_optional_info = self.embed_optional_info(obj_optional_info)
        
        f = torch.cat((obj_feats, obj_optional_info), axis=2)
        y = self.to_logit(f)[:, :, 0]
        return y
    
    def _forward_map_select_bbox(self, feat_all, obj_bboxes):
        
        if self.training and self.coordinate_noise > 0:
            obj_bboxes += torch.randn_like(obj_bboxes)*(self.coordinate_noise*self.image_shape[0])
        
        batch_size, feat_dim, h, w = feat_all.size()
        n_objs = obj_bboxes.size(1)
        
        # truncating coordinates into integer 
        obj_bboxes /= self.last_scale_factor
        for i, u, f in [(0, w, True), (1, h, True), (2, w, False), (3, h, False)]:
            obj_bboxes[..., i].clamp_(0, u)
            if f:
                obj_bboxes[..., i] = obj_bboxes[..., i].floor()
            else:
                obj_bboxes[..., i] = obj_bboxes[..., i].ceil()
        obj_bboxes = torch.as_tensor(obj_bboxes, dtype=np.int, device=self.device)
        
        # selecting boxes
        obj_feats = []
        z = torch.zeros((feat_dim,), device=self.device, dtype=torch.float32)
        for ib in range(batch_size):
            feats = []
            batch_map = feat_all[ib]
            for io in range(n_objs):
                x_min, y_min, x_max, y_max = obj_bboxes[ib, io]
                bbox = batch_map[:, y_min:y_max, x_min:x_max]
                if bbox.nelement() == 0:
                    feats.append(z)
                else:
                    feats.append(self.pooling(bbox))
            obj_feats.append(torch.stack(feats))
        obj_feats = torch.stack(obj_feats)
        
        return obj_feats
    
    def _forward_map_select_center(self, feat_all, obj_centers):
        
        if self.training and self.coordinate_noise > 0:
            obj_centers += torch.randn_like(obj_centers)*(self.coordinate_noise*self.image_shape[0])
        
        batch_size, _, h, w =  feat_all.size()
        n_obj = obj_centers.size(1)
        
        # truncating coordinates into integer
        obj_centers /= self.last_scale_factor
        obj_centers = obj_centers.round()
        for i, u in [(0, w), (1, h)]:
            obj_centers[..., i] = obj_centers[..., i].clamp(0, u - 1)
        obj_centers = torch.as_tensor(obj_centers, dtype=np.int, device=self.device)
        
        # selecting points
        ib = torch.arange(batch_size, device=self.device).view(-1, 1).repeat(1, n_obj).flatten()
        ix = obj_centers[:, :, 0].flatten()
        iy = obj_centers[:, :, 1].flatten()
        obj_feats = feat_all.permute(0, 2, 3, 1)[(ib, iy, ix)].view(batch_size, n_obj, -1)
        
        return obj_feats
    
    def _forward_map_all(self, feat_all):
        raise NotImplementedError()
    
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
