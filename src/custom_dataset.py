# coding: utf-8


import os
import json
import inspect

import numpy as np
import torch
import PIL, PIL.Image
from torch.utils.data import Dataset

try:
    import data_reader_onecommon
except ModuleNotFoundError as e:
    import sys
    sys.path += ['src']

# Register data readers
data_reader_router = {}

from data_reader_onecommon import OneCommonDataReader
OneCommonDataReader.register(data_reader_router)

from data_reader_guesswhat import GuessWhatDataReader
GuessWhatDataReader.register(data_reader_router)


class UnifiedTextifier(object):
    
    token_key_main = 'main'
    token_key_sub = 'sub'
    token_keys = (token_key_main, token_key_sub)
    
    token_pad = '<pad>'
    token_unk = '<unk>'
    token_text = '<text>'
    token_eos = '<eos>'
    token_you = '<you>'
    token_them = '<them>'
    token_selection = '<selection>'
    
    control_tokens = {
        token_key_main: [token_pad, token_unk, token_text, token_eos, 
                      token_you, token_them, token_selection],
        token_key_sub: [token_pad, token_unk]
    }
    
    @classmethod
    def from_dict(cls, base_dict):
        return cls(base_dict)
    
    @classmethod
    def load_or_make(cls, dict_path, dataset_list, use_more_than):
        
        if dataset_list is None or os.path.exists(dict_path):
            # load
            with open(dict_path, 'r') as f:
                textifier = cls.from_dict(json.load(f))
            print('restored from', dict_path)
        else:
            # make
            print('creating a new textifier in ', dict_path)
            
            textifier = cls()
            n_tokens = {key:{} for key in cls.token_keys}
            for dataset in dataset_list:
                reader = data_reader_router[dataset['task_name']]
                
                for key in cls.token_keys:
                    for t in reader.control_tokens[key]:
                        textifier.append(t, key=key)
                n_tokens = reader.count_tokens(dataset, textifier, n_tokens)
            
            for key in cls.token_keys:
                for t, _ in filter(lambda _: _[1] >= use_more_than, n_tokens[key].items()):
                    textifier.append(t, key=key)
            
            with open(dict_path, 'w') as f:
                json.dump(textifier.to_dict(), f)
            print('a new textifier created at', dict_path)
        
        return textifier
    
    def to_dict(self):
        return self.token_to_id
    
    def get_id_pad(self, key=token_key_main):
        return self.token_to_id[key][self.token_pad]
    
    def get_id_unk(self, key=token_key_main):
        return self.token_to_id[key][self.token_unk]
    
    def get_id_text(self, key=token_key_main):
        return self.token_to_id[key][self.token_text]
    
    def get_id_eos(self, key=token_key_main):
        return self.token_to_id[key][self.token_eos]
    
    def get_id_you(self, key=token_key_main):
        return self.token_to_id[key][self.token_you]
    
    def get_id_them(self, key=token_key_main):
        return self.token_to_id[key][self.token_them]
    
    def get_id_selection(self, key=token_key_main):
        return self.token_to_id[key][self.token_selection]
    
    def __init__(self, token_to_id=None):
        if token_to_id is not None:
            self.token_to_id = token_to_id
            self.id_to_token = {}
            for key in self.token_keys:
                self.id_to_token[key] = {i:t for t, i in self.token_to_id[key].items()}
        else:
            self.token_to_id = {key:{} for key in self.token_keys}
            self.id_to_token = {key:{} for key in self.token_keys}
            for key in self.token_keys:
                for t in self.control_tokens[key]:
                    self.append(t, key)
    
    def append(self, token, key):
        if token not in self.token_to_id[key]:
            i = len(self.token_to_id[key])
            self.token_to_id[key][token] = i 
            self.id_to_token[key][i] = token
    
    def get_len(self, key=None):
        if key is None:
            return {_:len(self.token_to_id[_]) for _ in self.token_keys}
        return self.token_to_id[key]
    
    def decode(self, ids, key=token_key_main):
        id_to_token = self.id_to_token[key]
        return [id_to_token[i] for i in ids]
    
    def encode(self, tokens, key=token_key_main):
        id_unk = self.get_id_unk(key)
        token_to_id = self.token_to_id[key]
        return [token_to_id.get(t, id_unk) for t in tokens]
    
    def tokenize(self, text):
        return text.split()
    
    def __call__(self, task_name, data, to_ids=True, key=token_key_main):
        text = data_reader_router[task_name].instance_to_text(data)
        text = '%s %s %s'%(task_name, self.token_text, text)
        tokens = self.tokenize(text)
        if to_ids:
            return self.encode(tokens, key)
        return tokens


class VisualSelectionDataset(Dataset):
    
    def __init__(self,
            dataset_list,
            textifier,
            provide_image,
            image_shape,
            dim_object_optional_info,
            ratio_force_unk,
            ratio_force_zero
        ):
        
        # set arguments
        local_dict = locals()
        for a in inspect.getfullargspec(self.__init__).args:
            (a == 'self') or setattr(self, a, local_dict[a])
        del local_dict
        
        self.image_size = (self.image_shape[1], self.image_shape[0])
        self.token_id_unk_main = self.textifier.get_id_unk('main')
        self.token_id_unk_sub = self.textifier.get_id_unk('sub')
        
        self.instances = []
        for dataset_spec in dataset_list:
            reader = data_reader_router[dataset_spec['task_name']]
            self.instances += reader.compile_dataset(dataset_spec, self.textifier)
    
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        inst = self.instances[idx]
        sample = {
            'task_name': inst.task_name,
            'n_tokens': inst.n_tokens,
            'tokens': inst.tokens,
            'n_objects': inst.n_objects,
            'object_bboxes': inst.object_bboxes,
            'object_optional_info': inst.object_optional_info,
            'dim_object_optional_info': self.dim_object_optional_info,
            'ground_truth_id': inst.ground_truth_id,
        }
        
        if not self.provide_image:
            sample['image'] = None
        else:
            image = PIL.Image.open(inst.image_path)
            if image.size != self.image_size:
                sx = self.image_size[0] / image.size[0]
                sy = self.image_size[1] / image.size[1]
                sample['object_bboxes'] = sample['object_bboxes'] * np.asarray([[sx, sy, sx, sy]])
                image = image.resize(self.image_size, PIL.Image.NEAREST)
            image = np.asarray(image, dtype=np.float32) / 255.0
            if len(image.shape) == 2:
                image = np.tile(image[:,:,None], (1,1,3))
            sample['image'] = image
        
        if self.ratio_force_unk > 0:
            tokens = sample['tokens']
            mask = np.random.uniform(size=tokens.shape) >= self.ratio_force_unk
            sample['tokens'] = np.where(mask, tokens, self.token_id_unk_main)
        
        if sample['object_optional_info'] is None:
            sample['object_optional_info'] = np.full(int(inst.n_objects), self.token_id_unk_sub, dtype=np.int)
        else:
            if self.ratio_force_zero > 0:
                tokens = sample['object_optional_info']
                mask = np.random.uniform(size=tokens.shape) >= self.ratio_force_zero
                sample['object_optional_info'] = np.where(mask, tokens, self.token_id_unk_sub)
        
        return sample


def collate_fn(data):
    """collate_fn for DataLoader"""
    
    pad_value = 0
    use_image = (data[0]['image'] is not None)
    dim_object_optional_info = data[0]['dim_object_optional_info']
    
    keys_to_stack = ['ground_truth_id', 'n_objects', 'n_tokens']
    if use_image:
        keys_to_stack.append('image')
    keys_to_pad_stack = [
            ('tokens', 'n_tokens', tuple(), np.int), 
            ('object_bboxes', 'n_objects', (4,), np.float32), 
    ]
    if use_image:
        # object_optional_info is token id
        keys_to_pad_stack.append(('object_optional_info', 'n_objects', tuple(), np.int))
    else:
        # object_optional_info is real values
        keys_to_pad_stack.append(('object_optional_info', 'n_objects', (dim_object_optional_info,), np.float32))
    
    mb = {}
    mb['size'] = batch_size = len(data)
    mb['task_name'] = [_['task_name'] for _ in data]
    mb['image'] = None
    
    for key in keys_to_stack:
        mb[key] = torch.from_numpy(np.stack([_[key] for _ in data]))
    
    for key, n, s, dtype in keys_to_pad_stack:
        n = mb[n]
        max_length = n.max().item()
        shape = (batch_size, max_length) + s
        mat = np.full(shape, pad_value, dtype=dtype)
        for i in range(batch_size):
            if data[i][key] is not None:
                mat[i, :n[i].item()] = data[i][key]
        mb[key] = torch.from_numpy(mat)
    
    if use_image:
        mb['image'] = mb['image'].permute(0, 3, 1, 2)
    
    return mb
