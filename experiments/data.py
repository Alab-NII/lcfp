import os
import random
import sys
import pdb
import copy
import re
from collections import OrderedDict, defaultdict

import torch
import numpy as np

#import make_view
from make_view_imp import make_view_and_label_annot
import json

# special tokens
SPECIAL = [
    '<eos>',
    '<unk>',
    '<selection>',
    '<pad>',
]

# tokens that stops either a sentence or a conversation
STOP_TOKENS = [
    '<eos>',
    '<selection>',
]


def get_tag(tokens, tag):
    """Extracts the value inside the given tag."""
    return tokens[tokens.index('<' + tag + '>') + 1:tokens.index('</' + tag + '>')]


def to_float(tokens):
    return [float(token) for token in tokens.split()]


def read_lines(file_name):
    """Reads all the lines from the file."""
    assert os.path.exists(file_name), 'file does not exists %s' % file_name
    lines = []
    with open(file_name, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines


class Dictionary(object):
    """Maps words into indeces.

    It has forward and backward indexing.
    """

    def __init__(self, init=True):
        self.word2idx = OrderedDict()
        self.idx2word = []
        if init:
            # add special tokens if asked
            for i, k in enumerate(SPECIAL):
                self.word2idx[k] = i
                self.idx2word.append(k)

    def add_word(self, word):
        """Adds a new word, if the word is in the dictionary, just returns its index."""
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def i2w(self, idx):
        """Converts a list of indeces into words."""
        return [self.idx2word[i] for i in idx]

    def w2i(self, words):
        """Converts a list of words into indeces. Uses <unk> for the unknown words."""
        unk = self.word2idx.get('<unk>', None)
        return [self.word2idx.get(w, unk) for w in words]

    def get_idx(self, word):
        """Gets index for the word."""
        unk = self.word2idx.get('<unk>', None)
        return self.word2idx.get(word, unk)

    def get_word(self, idx):
        """Gets word by its index."""
        return self.idx2word[idx]

    def __len__(self):
        return len(self.idx2word)

    def read_tag(file_name, tag, freq_cutoff=-1, init_dict=True):
        """Extracts all the values inside the given tag.

        Applies frequency cuttoff if asked.
        """
        token_freqs = OrderedDict()
        with open(file_name, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                tokens = get_tag(tokens, tag)
                for token in tokens:
                    token_freqs[token] = token_freqs.get(token, 0) + 1
        dictionary = Dictionary(init=init_dict)
        token_freqs = sorted(token_freqs.items(),
                             key=lambda x: x[1], reverse=True)
        for token, freq in token_freqs:
            if freq > freq_cutoff:
                dictionary.add_word(token)
        return dictionary

    def from_file(file_name, freq_cutoff):
        """Constructs a dictionary from the given file."""
        print(file_name)
        assert os.path.exists(file_name)
        word_dict = Dictionary.read_tag(
            file_name, 'dialogue', freq_cutoff=freq_cutoff)
        return word_dict


class WordCorpus(object):
    """An utility that stores the entire dataset.

    It has the train, valid and test datasets and corresponding dictionaries.
    """

    def __init__(self, path, ctx_view_size, freq_cutoff=2, train='train.txt',
                 valid='valid.txt', test='test.txt', verbose=False, word_dict=None):
        self.verbose = verbose
        self.ctx_view_size = ctx_view_size
        self.no_view = ctx_view_size is None
        if word_dict is None:
            self.word_dict = Dictionary.from_file(
                os.path.join(path, train), freq_cutoff=freq_cutoff)
        else:
            self.word_dict = word_dict
        
        self.ref_files = {
            'train': os.path.join(path, train) if train is not None else None,
            'valid': os.path.join(path, valid) if valid is not None else None,
            'test': os.path.join(path, test) if test is not None else None,
        }
        
        # construct all 3 datasets
        self.train = self.tokenize(self.ref_files['train']) if train else []
        self.valid = self.tokenize(self.ref_files['valid']) if valid else []
        self.test = self.tokenize(self.ref_files['test']) if test else []
        
        # find out the output length from the train dataset
        self.output_length = max([len(x) for x in self.train])
       

    def tokenize(self, file_name, test=False):
        """Tokenizes the file and produces a dataset."""
        lines = read_lines(file_name)
        #random.shuffle(lines)

        unk = self.word_dict.get_idx('<unk>')
        dataset, total, unks = [], 0, 0
        for line in lines:
            tokens = line.split()
            input_vals = [float(val) for val in get_tag(tokens, 'input')]
            word_idxs = self.word_dict.w2i(get_tag(tokens, 'dialogue'))
            output_idx = int(get_tag(tokens, 'output')[0])
            if not self.no_view:
                ctx_view, label_annot = make_view_and_label_annot(input_vals, output_idx, self.ctx_view_size)
            else:
                ctx_view = label_annot = None
            dataset.append({
                    'ctx_raw': input_vals,
                    'words': word_idxs,
                    'label': output_idx,
                    'ctx_view': ctx_view,
                    'label_annot': label_annot,
                })
            
            # compute statistics
            total += len(word_idxs)
            unks += np.count_nonzero([idx == unk for idx in word_idxs])

        if self.verbose:
            print('dataset %s, total %d, unks %s, ratio %0.2f%%' % (
                file_name, total, unks, 100. * unks / total))
        return dataset

    def train_dataset(self, bsz, shuffle_instance=True, device=None, annot_noise=0):
        return self._split_into_batches(copy.copy(self.train), bsz, shuffle_instance, annot_noise, device, with_q_data=False)

    def valid_dataset(self, bsz, shuffle_instance=False, device=None):
        return self._split_into_batches(copy.copy(self.valid), bsz, shuffle_instance, 0, device, with_q_data=True)

    def test_dataset(self, bsz, shuffle_instance=False, device=None):
        return self._split_into_batches(copy.copy(self.test), bsz, shuffle_instance, 0, device, with_q_data=True)

    def _split_into_batches(self, dataset, bsz, shuffle_instance, annot_noise, device, with_q_data):
        """Splits given dataset into batches."""
        
        # id for pad token
        pad = self.word_dict.get_idx('<pad>')
        eos = self.word_dict.get_idx('<eos>')
        
        if shuffle_instance:
            random.shuffle(dataset)
        
        batches = []
        stats = {
            'n': 0,
            'nonpadn': 0,
        }
        
        n_short = 0
        for i in range(0, len(dataset), bsz):
            l_ctx_raw = []
            l_words = []
            l_label = []
            l_ctx_view = []
            l_ctx_view_annot = []
            for j in range(i, min(i + bsz, len(dataset))):
                instance = dataset[j]
                
                target_statement = instance['words']
                #target_statement = None
                #n_eos = 0
                ##for k in reversed(range(len(instance['words']))):
                #for k in range(len(instance['words'])):
                #    if instance['words'][k] == eos:
                #        n_eos +=1
                #        if n_eos >= 1:
                #            #target_statement = instance['words'][k+1:]
                #            target_statement = instance['words'][:k+1]
                #            #t = ' '.join(self.word_dict.get_word(t) for t in target_statement)
                #            break
                #if target_statement is None:
                #    target_statement = instance['words']
                #    n_short += 1
                
                l_ctx_raw.append(instance['ctx_raw'])
                l_words.append(target_statement)
                l_label.append(instance['label'])
                if not self.no_view:
                    l_ctx_view_annot.append(instance['label_annot'])
                    ctx_view = instance['ctx_view']
                    if annot_noise > 0:
                        ctx_view = (ctx_view + np.random.normal(0, annot_noise, size=ctx_view.shape)).clip(0.0, 1.0)
                    l_ctx_view.append(ctx_view)
            
            # check
            #for view, text, label in zip(l_ctx_raw, l_words, l_label):
            #    t = ' '.join(self.word_dict.get_word(t) for t in text)
             #    print(t.replace('YOU:', '\nYOU:').replace('THEM:', '\nTHEM:'))
            #    print(view)
            #    print(label)
            #    input()
            
            # original text lengths in the batch
            original_len = [len(text_line) for text_line in l_words]
            max_len = max(original_len)
            words_mask = np.arange(0, max_len)[None, :] < np.asarray(original_len)[:,None]

            # pad all the dialogues to match the longest dialogue
            for j in range(len(l_words)):
                stats['n'] += max_len
                stats['nonpadn'] += len(l_words[j])
                l_words[j] = l_words[j] + [pad] * (max_len - len(l_words[j]))
            
            # construct tensors
            batch_size = len(l_ctx_raw)
            ctx_raw = torch.FloatTensor(l_ctx_raw)
            words = torch.LongTensor(l_words).transpose(0, 1).contiguous()
            words_original_len = torch.LongTensor(original_len)
            words_mask = torch.BoolTensor(words_mask)
            label = torch.LongTensor(l_label)
            if not self.no_view:
                ctx_view = torch.FloatTensor(l_ctx_view)
                #ctx_view_annot = torch.LongTensor(l_ctx_view_annot)
                ctx_view_annot = np.asarray(l_ctx_view_annot)
            
            q_data = {}
            if with_q_data:
                texts = [' '.join(self.word_dict.get_word(t) for t in filter(lambda _: _ != pad, text)) for text in l_words]
                q_data['texts'] = texts
                q_data['ctx_raws'] = l_ctx_raw
            if self.no_view:
                ctx_view = None
                ctx_view_annot = None
            
            task_name = 'selection'
            batches.append((task_name, batch_size, ctx_raw, words, words_original_len, words_mask, label, ctx_view, ctx_view_annot, device, q_data))
        #print(len(dataset), n_short, n_short/len(dataset))
        return batches, stats
    
    def get_train_shuffle_task_data(self, bsz, shuffle_instance=True, device=None):
        return self.make_shuffle_task_data(copy.copy(self.train), bsz, shuffle_instance, device, with_q_data=False)
    
    def get_valid_shuffle_task_data(self, bsz, shuffle_instance=True, device=None):
        return self.make_shuffle_task_data(copy.copy(self.valid), bsz, shuffle_instance, device, with_q_data=False)
    
    def make_shuffle_task_data(self, dataset, bsz, shuffle_instance, device, with_q_data):
        
        pad = self.word_dict.get_idx('<pad>')
        them = self.word_dict.get_idx('THEM:')
        
        batches = []
        
        l_ctx_raw = []
        l_words = []
        l_label = []
        lists = [l_ctx_raw, l_words, l_label]
        
        #debug_dataset = {'data':[]}
        
        def _flush_batch():
            batch_size = len(l_ctx_raw)
            if batch_size <= 0:
                return
            
            task_name = 'shuffle'
            
            #for k in range(0, batch_size):
            #    l_label[k] = 0 # real label
            ## appending swapped negative data
            #for l in lists:
            #    l.extend(l)
            #if batch_size % 2 != 0:
            #    l_ctx_raw.append(l_ctx_raw[0])
            #    l_words.append(l_words[0])
            #    l_label.append(l_label[0])
            #for k in range(0,  batch_size, 2):
            #    tmp = l_ctx_raw[batch_size+k]
            #    l_ctx_raw[batch_size+k] = l_ctx_raw[batch_size+k+1]
            #    l_ctx_raw[batch_size+k+1] = tmp
            #    l_label[batch_size+k] = l_label[batch_size+k+1] = 1 # fake label
            
            for k in range(0, int(batch_size//2), 2):
                tmp = l_ctx_raw[k]
                l_ctx_raw[k] = l_ctx_raw[k+1]
                l_ctx_raw[k+1] = tmp
                l_label[k] = l_label[k+1] = 1 # fake label
            for k in range(int(batch_size//2), batch_size):
                l_label[k] = 0
            
            # debug
            #for k in range(0, batch_size):
            #    t = ' '.join(self.word_dict.get_word(t) for t in filter(lambda x:x != pad, l_words[k]))
            #    debug_dataset['data'].append({'text':t, 'ctx':l_ctx_raw[k], 'label':l_label[k]})
            
            # calc original text lengths in the batch and padding
            original_len = [len(text_line) for text_line in l_words]
            max_len = max(original_len)
            words_mask = np.arange(0, max_len)[None, :] < np.asarray(original_len)[:,None]
            for j in range(len(l_words)):
                l_words[j] = l_words[j] + [pad] * (max_len - len(l_words[j]))
            
            ctx_raw = torch.FloatTensor(l_ctx_raw)
            words = torch.LongTensor(l_words).transpose(0, 1).contiguous()
            words_original_len = torch.LongTensor(original_len)
            words_mask = torch.BoolTensor(words_mask)
            label = torch.LongTensor(l_label)
            
            q_data = {}
            if with_q_data:
                texts = [' '.join(self.word_dict.get_word(t) for t in filter(lambda _: _ != pad, text)) for text in l_words]
                q_data['texts'] = texts
                q_data['ctx_raws'] = l_ctx_raw
            ctx_view = None
            ctx_view_annot = None
            
            batches.append((task_name, batch_size, ctx_raw, words, words_original_len, words_mask, label, ctx_view, ctx_view_annot, device, q_data))
            for l in lists:
                l.clear()
        
        if shuffle_instance:
            random.shuffle(dataset)
        
        #half_batch_size = int(bsz//2)
        for i in range(0, len(dataset)):
            instance = dataset[i]
            
            # to get first 'YOU:' statement which places the head of the dialogue
            # because this often contains a brief description of the view. 
            first_my_statement = None
            for j in range(len(instance['words'])):
                if instance['words'][j] == them:
                    first_my_statement = instance['words'][:j]
                    break
            if first_my_statement is None or len(first_my_statement) < 4:
                continue
            
            l_ctx_raw.append(instance['ctx_raw'])
            l_words.append(first_my_statement)
            l_label.append(instance['label'])
            
            if len(l_ctx_raw) == bsz:
                _flush_batch()
        _flush_batch()
        
        #with open('shuffle_test.json', 'w') as f:
        #    json.dump(debug_dataset, f)
        #    print('shuffle_test.json', 'outputed')
        #    exit()
        
        return batches, None
    
