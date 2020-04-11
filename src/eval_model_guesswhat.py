# coding: utf-8


import os
import json
import time
import inspect

import importlib.machinery as imm

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import custom_dataset
except ModuleNotFoundError as e:
    import sys
    sys.path += ['src']

from custom_dataset import collate_fn, UnifiedTextifier, VisualSelectionDataset


COCO_IMAGE_DIRECTORIES = ['data/img/test2014', 'data/img/val2014', 'data/img/train2014']
TARGET_DATASETS=[
     ('train', {'task_name':'guesswhat.selection', 'path':'data/guesswhat.train.jsonl', 
                 'success_only':True, 'coco_image_directories':COCO_IMAGE_DIRECTORIES}),
     ('valid', {'task_name':'guesswhat.selection', 'path':'data/guesswhat.valid.jsonl', 
                 'success_only':True, 'coco_image_directories':COCO_IMAGE_DIRECTORIES}),
     ('test', {'task_name':'guesswhat.selection', 'path':'data/guesswhat.test.jsonl', 
                 'success_only':True, 'coco_image_directories':COCO_IMAGE_DIRECTORIES}),
]


def eval_model(net_module):
    
    print('#', 'Evaluation')
    
    config = net_module.TrainConfig()
    weight_path = _get_best_weight_path(config)
    print('model_dir', config.model_dir)
    print('best_weight_path', weight_path)
    
    # Tokenizer
    print('#', 'initializing textifiers')
    textifier = UnifiedTextifier.load_or_make(config.textifier_dict,
        config.dataset_list['train'], config.textifier_use_more_than,
    )
    print('textifier vocab size', len(textifier))
    
    # Dataset
    print('#', 'loading dataset definitions')
    datasets = []
    for key, dataset_spec in TARGET_DATASETS:
        dataset = VisualSelectionDataset(
            dataset_list=[dataset_spec],
            textifier=textifier,
            provide_image=config.provide_image,
            image_size=config.model_net_args['image_size'],
            dim_object_optional_info=config.model_net_args['dim_object_optional_info'],
            ratio_force_unk=0, ratio_force_zero=0,
        )
        datasets.append((key, dataset))
        print('len', key, len(dataset))
        
    # Model
    print('#', 'constructing a model')
    model_net = net_module.SelectorNet(len(textifier), **config.model_net_args)
    model_net.to(config.device)
    model_net.device = config.device
    # load
    states = torch.load(weight_path, map_location=torch.device('cpu'))
    model_net.load_state_dict(states['model_net_state_dict'])
   
    # start evaluation
    summaries = []
    for name, dataset in datasets:
        
        print('+', name, 'evaluating...')
        
        data_loader = DataLoader(
            dataset, 
            batch_size=config.minibatch_size, 
            num_workers=config.n_workers, 
            collate_fn=collate_fn, 
            shuffle=False,
        )
        
        model_net.eval()
        with torch.no_grad():
            summary = net_module.run_dataset(config, model_net, None, data_loader, 0)
        
        summaries.append(' '.join([summary.to_str(prefix=name+'_')]))
        print(summaries[-1])

    print('#', 'results')
    print('\n'.join(summaries))


def _get_best_weight_path(config):
    
    history = []
    with open(config.history_path, 'r') as f:
        history = [_.strip().split(' ') for _ in f.readlines()]

    # header: ep, train_loss, train_acc, valid_loss, valid_acc
    min_valid_loss = float('inf')
    min_ep = -1
    for line in history:
        valid_loss = float(line[3])
        ep = int(line[0])
        if min_valid_loss > valid_loss:
            min_valid_loss = valid_loss
            min_ep = ep
    if min_ep < 0:
        raise ValueError('no record in history')
    
    weight_path = config.weight_path_template%(min_ep)
    return weight_path


if __name__ == '__main__':
    
    import sys
    module_path = sys.argv[1]
    net_module = imm.SourceFileLoader('net_module', module_path).load_module()   
    eval_model(net_module)
