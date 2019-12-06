import argparse
import random
import pdb
import time
import itertools
import sys
import copy
import re

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from data import STOP_TOKENS

from logger import Logger

class SelectEngine(object):
    """The training engine.

    Performs training and evaluation.
    """
    def __init__(self, model, args, device=None, verbose=False):
        self.model = model
        self.args = args
        self.device = device
        self.verbose = verbose
        self.opt = torch.optim.Adam(model.parameters(), lr=self.args.lr, eps=1e-9, weight_decay=1e-5)
        self.sel_crit = nn.CrossEntropyLoss()
        self.logger = Logger('tensorboard_logs_{}'.format(args.model_file))

    def forward(model, batch, requires_grad=False):
        """A helper function to perform a forward pass on a batch."""
        # extract the batch into contxt, input, target and selection target
        with torch.set_grad_enabled(requires_grad):
            task_name, batch_size, ctx_raw, words, words_original_len, words_mask, label, ctx_view, ctx_view_annot, device, _ = batch
            
            if device is not None:
                ctx_raw = ctx_raw.to(device)
                words = words.to(device)
                words_original_len = words_original_len.to(device)
                words_mask = words_mask.to(device)
                label = label.to(device)
                if ctx_view is not None:
                    ctx_view = ctx_view.to(device)
                    #ctx_view_annot = ctx_view_annot.to(device)
            
            ctx_raw = Variable(ctx_raw)
            words = Variable(words)
            words_original_len = Variable(words_original_len)
            words_mask = Variable(words_mask)
            label = Variable(label)
            
            if ctx_view is not None:
                ctx_view = Variable(ctx_view)
                #ctx_view_annot = Variable(ctx_view_annot)
            
            ctx_in = (ctx_raw, ctx_view, ctx_view_annot)
            ctx_out = model.forward_context(ctx_in)
            
            sel_in = (task_name, words, words_original_len, words_mask, ctx_out)
            sel_out = model.forward_selection(sel_in)
            
            return sel_out, label

    def train_pass(self, N, trainset):
        """Training pass."""
        self.model.train()

        total_loss = 0
        total_correct = 0
        total_sample = 0
        start_time = time.time()

        for batch in trainset:
            sel_out, sel_tgt = SelectEngine.forward(self.model, batch, requires_grad=True)
            loss = self.sel_crit(sel_out, sel_tgt)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.opt.step()
            
            batch_size = batch[1]
            total_sample += batch_size
            total_loss += loss.item()*batch_size
            total_correct += (sel_out.max(dim=1)[1] == sel_tgt).sum().item()
        
        total_sample = total_sample if total_sample != 0 else 1
        total_loss /= total_sample
        total_correct /= total_sample
        time_elapsed = time.time() - start_time
        return total_loss, time_elapsed, total_correct

    def valid_pass(self, N, validset):
        """Validation pass."""
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_sample = 0

        for batch in validset:
            sel_out, sel_tgt = SelectEngine.forward(self.model, batch, requires_grad=False)
            loss = self.sel_crit(sel_out, sel_tgt)
            
            batch_size = batch[1]
            total_sample += batch_size
            total_loss += loss.item()*batch_size
            total_correct += (sel_out.max(dim=1)[1] == sel_tgt).sum().item()
        
        total_sample = (total_sample if total_sample != 0 else 1)
        return  total_loss / total_sample, total_correct / total_sample

    def iter(self, N, epoch, lr, traindata, validdata):
        """Performs on iteration of the training.
        Runs one epoch on the training and validation datasets.
        """
        trainset, _ = traindata
        validset, _ = validdata

        train_loss, train_time, train_accuracy = self.train_pass(N, trainset)
        valid_loss, valid_accuracy = self.valid_pass(N, validset)

        if self.verbose:
            print('| epoch %03d | trainloss %.3f | s/epoch %.2f | trainaccuracy %.3f | lr %0.8f' % (
                epoch, train_loss, train_time, train_accuracy, lr))
            print('| epoch %03d | validloss %.3f | validselectppl %.3f' % (
                epoch, valid_loss, np.exp(valid_loss)))
            print('| epoch %03d | valid_select_accuracy %.3f' % (
                epoch, valid_accuracy))

        # Tensorboard Logging
        # 1. Log scalar values (scalar summary)
        info = {'Train_Loss': train_loss,
                'Train_Accuracy': train_accuracy,
                'Valid_Loss': valid_loss,
                'Valid_Accuracy': valid_accuracy}

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, epoch)
        
        if self.args.log_params:
            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in self.model.named_parameters():
                if value.grad is None:
                    # don't need to log untrainable parameters
                    continue
                tag = tag.replace('.', '/')
                self.logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                self.logger.histo_summary(
                    tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        return train_loss, valid_loss

    def train(self, corpus):
        """Entry point."""
        N = len(corpus.word_dict)
        best_model, best_valid_loss = None, 1e100
        lr = self.args.lr
        
        shuffle_task=False
        #traindata_shuffle_task = [corpus.get_train_shuffle_task_data(self.args.bsz, device=self.device) for _ in range(2)]
        validdata_selection_task = corpus.valid_dataset(self.args.bsz, device=self.device)
        validdata_shuffle_task = corpus.get_valid_shuffle_task_data(self.args.bsz, device=self.device)
        for epoch in range(1, self.args.max_epoch + 1):
            # shuffle_task = True if epoch % 2 == 0 else False
            #shuffle_task = True if epoch < 100 else False
            if shuffle_task:
                #traindata = traindata_shuffle_task[int(np.random.randint(len(traindata_shuffle_task)))]
                #lr = self.args.lr*0.1
                traindata = corpus.get_train_shuffle_task_data(self.args.bsz, device=self.device)
                #traindata = traindata_shuffle_task[0]
                validdata = validdata_shuffle_task
            else:
                traindata = corpus.train_dataset(self.args.bsz, device=self.device, annot_noise=self.args.annot_noise)
                validdata = validdata_selection_task
            
            #if epoch == 5:
            #    lr *= 0.1
            #if epoch == 10:
            #    lr *= 0.1
            
            train_loss, valid_loss = self.iter(N, epoch, lr, traindata, validdata)

            if (not shuffle_task) and valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
            
        return train_loss, best_valid_loss, best_model_state
