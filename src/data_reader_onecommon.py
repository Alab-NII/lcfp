# coding: utf-8


import json
import numpy as np
import os

try:
    import data_reader_base
except ModuleNotFoundError as e:
    import sys
    sys.path += ['src']
from data_reader_base import DatasetReaderBase, VisualSelectionTaskInstance


class OneCommonDataReader(DatasetReaderBase):
    
    task_name = 'onecommon.selection'
    
    token_eos = '<eos>'
    token_selection = '<selection>'
    token_you = '<you>'
    token_them = '<them>'
    control_tokens = {
        'main':[task_name, token_eos, token_selection, token_you, token_them],
        'sub':[],
    }
    
    full_canvas_size = 224
    min_size=0.025
    max_size=0.05
    min_col=0.2
    max_col=0.8
    min_pos = full_canvas_size*max_size*0.5
    max_pos = full_canvas_size*(1 - max_size*0.5)
    
    @classmethod
    def instance_to_text(cls, inst):
        """Returns a space-splitted text given an instance."""
        text = inst['dialogue']
        text = text.replace('YOU:', cls.token_you)
        text = text.replace('THEM:', cls.token_them)
        text = text.lower()
        return text
    
    @classmethod
    def count_tokens(cls, dataset_spec, textifier, n_tokens):
        """Return a dict whose key and value are token and its frequency."""
        with open(dataset_spec['path'], 'r') as f:
            dataset = json.load(f)
        
        target_n_tokens = n_tokens['main']
        for inst in dataset:
            tokens = textifier(cls.task_name, inst, to_ids=False)
            for token in tokens:
                target_n_tokens[token] = target_n_tokens.get(token, 0) + 1
        
        return n_tokens

    @classmethod
    def compile_dataset(cls, dataset_spec, textifier):
        """Returns a list of VisualSelectionTaskInstance."""
        
        provide_image = dataset_spec.get('provide_image', True)
        
        asarray = lambda x, t: np.asarray(x, dtype=t)
        get_min_max = lambda obj: (obj['x_min'], obj['y_min'], obj['x_max'], obj['y_max'])
        
        def get_attributes(obj):
            x_center = 2*(0.5*(obj['x_min'] + obj['x_max']) - cls.min_pos)/(cls.max_pos - cls.min_pos) - 1
            y_center = 2*(0.5*(obj['y_min'] + obj['y_max']) - cls.min_pos)/(cls.max_pos - cls.min_pos) - 1
            size = 2*(obj['size'][0] / (cls.full_canvas_size*(1 - cls.max_size)) - cls.min_size)/(cls.max_size - cls.min_size) - 1
            color = 2*(obj['color']/255 - cls.min_col) / (cls.max_col - cls.min_col) - 1
            return [x_center, y_center, size, color]
        
        with open(dataset_spec['path'], 'r') as f:
            dataset = json.load(f)
        
        instances = []
        for inst in dataset:
            _id = os.path.splitext(os.path.basename(inst['image_path']))[0]
            tokens = asarray(textifier(cls.task_name, inst, to_ids=True), np.int)
            object_bboxes = asarray([get_min_max(o) for o in inst['objects']], np.float32)
            
            if provide_image:
                object_optional_info = None
            else:
                object_optional_info = asarray([get_attributes(o) for o in inst['objects']], np.float32)
            
            instances.append(VisualSelectionTaskInstance(
                task_name=cls.task_name,
                instance_id=_id,
                image_path=inst['image_path'],
                tokens=tokens,
                n_tokens=asarray(tokens.shape[0], np.int),
                object_bboxes=object_bboxes,
                object_optional_info=object_optional_info,
                n_objects=asarray(object_bboxes.shape[0], np.int),
                ground_truth_id=asarray(inst['selected_id'], np.int),
            ))
        
        return instances
