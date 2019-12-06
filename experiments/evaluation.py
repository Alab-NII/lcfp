# coding: utf-8


import numpy as np
import re


def evaluate(file_path, inference_idxs, top_k):
    """file_path: dataset
    inference_idx: results of inference. [[k], [k], ... [k]]"""
    
    re_output = re.compile('<output>\s*([0123456789]+?)\s*</output>')
    
    lines = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    true_labels = []
    for line in lines:
        labels = re_output.findall(line)
        if len(labels) != 1:
            raise ValueError('Wrong label definition: %s'%(line))
        true_labels.append(int(labels[0]))
    
    n_example = len(true_labels)
    n_inference = len(inference_idxs)
    if n_example != n_inference:
        raise ValueError('The number of examples and inference doesn\'t match. %d & %d'%(n_example, n_inference))

    n_correct_k = 0
    
    for truth, infer in zip(true_labels, inference_idxs):
        is_correct = truth in infer[:top_k]
        n_correct_k += is_correct
    
    accuracy = n_correct_k/n_example
    
    return accuracy


if __name__ == '__main__':

    file_path = '../test.txt'
    top_k = 1
    inference_idxs = np.random.randint(0, 7, size=(1350, 1))
    
    accuracy = evaluate(file_path, inference_idxs, top_k)
    print('%.4f@%d'%(accuracy, top_k))
    
