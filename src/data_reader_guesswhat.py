# coding: utf-8


import json
import numpy as np
import torch
import os

from nltk import TweetTokenizer
#from transformers import BertTokenizer, BertModel

try:
    import data_reader_base
except ModuleNotFoundError as e:
    import sys
    sys.path += ['src']
from data_reader_base import DatasetReaderBase, VisualSelectionTaskInstance


class GuessWhatDataReader(DatasetReaderBase):
    
    task_name = 'guesswhat.selection'
    
    token_eos = '<eos>'
    token_you = '<you>'
    token_them = '<them>'
    token_selection = '<selection>'
    
    control_tokens = {
        'main': [task_name, token_eos, token_you, token_them, token_selection],
        'sub': [],
    }
    
    tokenizer = TweetTokenizer(preserve_case=False)
    
    # For Category (Optional Information) 
    #bert_model_name = 'bert-base-uncased'
    #bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    #bert_model = BertModel.from_pretrained(bert_model_name)
    #bert_model.eval()
    
    @classmethod
    def instance_to_text(cls, inst):
        """Returns a space-splitted text given an instance."""
        
        text_segments = []
        for qa in inst['qas']:
            q = ' '.join(cls.tokenizer.tokenize(qa['question']))
            a = qa['answer'].lower()
            assert a in ['yes', 'no', 'n/a']
                         
            # Question
            text_segments.append(cls.token_you)
            text_segments.append(q)
            text_segments.append(cls.token_eos)
            # Answer
            text_segments.append(cls.token_them)
            text_segments.append(a)
            text_segments.append(cls.token_eos)
        
        text_segments.append(cls.token_selection)
        return ' '.join(text_segments)
    
    @classmethod
    def count_tokens(cls, dataset_spec, textifier, n_tokens):
        """Returns a dict whose key and value are token and its frequency."""
        
        with open(dataset_spec['path'], 'r') as f:
            base_annotations = [json.loads(_) for _ in f.readlines()]
        
        for base_ in base_annotations:
            if dataset_spec['success_only'] and base_['status'] != 'success':
                continue
            
            # main text
            target_n_tokens = n_tokens['main']
            tokens = textifier(cls.task_name, base_, to_ids=False)
            for token in tokens:
                target_n_tokens[token] = target_n_tokens.get(token, 0) + 1
            
            # category for objects:
            target_n_tokens = n_tokens['sub']
            for obj in base_['objects']:
                token = obj['category']
                target_n_tokens[token] = target_n_tokens.get(token, 0) + 1
        
        return n_tokens

    @classmethod
    def compile_dataset(cls, dataset_spec, textifier):
        """Returns a list of VisualSelectionTaskInstance."""
        
        def resolve_image_path(file_name):
            for d in dataset_spec['coco_image_directories']:
                p = os.path.join(d, file_name)
                if os.path.exists(p):
                    return p
            return None
        
        def _get_inner_object_id(inst):
            object_id = inst['object_id']
            for i, obj in enumerate(inst['objects']):
                if obj['id'] == object_id:
                    return i
            raise ValueError('no object matched with object_id')
        
        #embedding_cache = {}
        #def get_embedding(text):
        #    if text not in embedding_cache:
        #        with torch.no_grad():
        #            input_ids = torch.tensor([cls.bert_tokenizer.encode(text, add_special_tokens=True)])
        #            hidden = cls.bert_model(input_ids)[0][0]
        #            embedding_cache[text] = hidden[1:].mean(axis=0).numpy().astype(np.float32)
        #    return embedding_cache[text].copy()
        def get_cat_id(category):
            token = category
            return textifier.token_to_id['sub'].get(token, textifier.get_id_unk('sub'))
        
        asarray = lambda x, t: np.asarray(x, dtype=t)
        get_min_max = lambda bbox: (bbox[0], bbox[1], (bbox[0] + bbox[2]), (bbox[1] + bbox[3]))
        
        with open(dataset_spec['path'], 'r') as f:
            dataset = [json.loads(_) for _ in f.readlines()]
        
        instances = []
        for inst in dataset:
            _image = inst['image']
            image_path = resolve_image_path(_image['file_name'])
            
            if dataset_spec['success_only'] and inst['status'] != 'success':
                continue
            
            if image_path is None:
                print('%d image(%s) not found skipped'%(inst['id'], _image['file_name']))
                continue
            
            tokens = asarray(textifier(cls.task_name, inst, to_ids=True), np.int)
            
            object_optional_info = asarray([get_cat_id(_['category']) for _ in inst['objects']], np.int)
            object_bboxes = asarray([get_min_max(o['bbox']) for o in inst['objects']], np.float32)
            
            instances.append(VisualSelectionTaskInstance(
                task_name=cls.task_name,
                instance_id=inst['id'],
                image_path=image_path,
                tokens=tokens,
                n_tokens=asarray(tokens.shape[0], np.int),
                object_bboxes=object_bboxes,
                object_optional_info=object_optional_info,
                n_objects=asarray(object_bboxes.shape[0], np.int),
                ground_truth_id=asarray(_get_inner_object_id(inst), np.int),
            ))
        
        return instances
