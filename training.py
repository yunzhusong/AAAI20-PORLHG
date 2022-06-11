""" module providing basic training utilities"""
import os
import pdb
from os.path import join, exists
from time import time
from datetime import timedelta
from itertools import starmap

from cytoolz import curry, reduce

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tensorboardX


def get_basic_grad_fn(net, clip_grad, max_grad=1e2):
    def f():
        grad_norm = clip_grad_norm_(
            [p for p in net.parameters() if p.requires_grad], clip_grad)
        #grad_norm = grad_norm.item() #yunzhu
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log = {}
        grad_log['grad_norm'] = grad_norm
        return grad_log
    return f

@curry
def compute_loss(net, criterion, fw_args, loss_args, ext_word2id):
    loss = criterion(*((net(*fw_args),) + loss_args), None)
    return loss

@curry
def val_step(loss_step, fw_args, loss_args, ext_word2id):
    loss = loss_step(fw_args, loss_args, None)
    return loss.size(0), loss.sum().item()

@curry
def basic_validate(net, criterion, val_batches):

    print('running validation ... ', end='')
    net.eval()
    start = time()
    with torch.no_grad():
        ext_word2id = None
        #aa, bb,ext_word2id = next(val_batches)
        #validate_fn = val_step(compute_loss(net, criterion, aa, bb, ext_word2id), aa, bb, ext_word2id)
        validate_fn = val_step(compute_loss(net, criterion))
        
        n_data, tot_loss = reduce(
            lambda a, b: (a[0]+b[0], a[1]+b[1]),
            starmap(validate_fn, val_batches),
            (0, 0)
        )
        
        #n_data+= n_data
        #tot_loss += tot_loss
    val_loss = tot_loss / n_data
    print(
        'validation finished in {}'.format(
            timedelta(seconds=int(time()-start)))
    )
    print('validation loss: {:.4f} ... '.format(val_loss))
    return {'loss': val_loss}

@curry
def compute_loss_abs(net, criterion, fw_args, loss_args, ext_word2id):
    loss = criterion(*(net(*fw_args)+ loss_args), ext_word2id)
    return loss

@curry
def val_step_abs(loss_step, fw_args, loss_args, ext_word2id):
    loss, _ = loss_step(fw_args, loss_args, ext_word2id)
    return 1, loss.sum().item()

@curry
def basic_validate_abs(net, criterion, val_batches):
    print('running validation ... ', end='')
    net.eval()
    start = time()
    with torch.no_grad():
        ext_word2id = None
        #aa, bb,ext_word2id = next(val_batches)
        #validate_fn = val_step(compute_loss(net, criterion, aa, bb, ext_word2id), aa, bb, ext_word2id)
        validate_fn = val_step_abs(compute_loss_abs(net, criterion))
        
        n_data, tot_loss = reduce(
            lambda a, b: (a[0]+b[0], a[1]+b[1]),
            starmap(validate_fn, val_batches),
            (0, 0)
        )
        
        #n_data+= n_data
        #tot_loss += tot_loss
    val_loss = tot_loss / n_data
    print(
        'validation finished in {}'.format(
            timedelta(seconds=int(time()-start)))
    )
    print('validation loss: {:.4f} ... '.format(val_loss))
    return {'loss': val_loss}

class BasicPipeline(object):
    def __init__(self, name, net,
                 train_batcher, val_batcher, batch_size,
                 val_fn, criterion, optim, grad_fn=None):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._criterion = criterion
        self._opt = optim
        # grad_fn is calleble without input args that modifyies gradient
        # it should return a dictionary of logging values
        self._grad_fn = grad_fn
        self._val_fn = val_fn

        self._n_epoch = 0  # epoch not very useful?
        self._batch_size = batch_size
        self._batches = self.batches()

    def batches(self):
        while True:
            for fw_args, bw_args, ext_word2id in self._train_batcher(self._batch_size):
                yield fw_args, bw_args, ext_word2id
            self._n_epoch += 1

    def get_loss_args(self, net_out, bw_args):
        if isinstance(net_out, tuple):
            loss_args = net_out + bw_args
        else:
            loss_args = (net_out, ) + bw_args
        return loss_args

    def train_step(self):
        # forward pass of model
        self._net.train()
        fw_args, bw_args, ext_word2id = next(self._batches)
        net_out = self._net(*fw_args)
        # get logs and output for logging, backward
        log_dict = {}
        loss_args = self.get_loss_args(net_out, bw_args)
        # backward and update ( and optional gradient monitoring )
        """
        params = list(self._net.parameters())
        k = 0 
        for i in params:
            l = 1
            for j in i.size():
                l *= j
            print("該層結構:{},参数和:{}".format(str(list(i.size())), str(l)))
            k = k + l
        print("总参数数量和：" + str(k))
        """

        if self.name.split('_')[-1] == 'extractor':
            sent_topics = fw_args[3]
			#loss = self._criterion(*loss_args, ext_word2id).mean() # for extractor
            loss = self._criterion(*loss_args, sent_topics).mean() # for extractor
        else:  
            loss, impov_score = self._criterion(*loss_args, ext_word2id) # for abstractor
            loss = loss.mean()
            log_dict['impov score'] = impov_score
        
        loss.backward()
        log_dict['loss'] = loss.mean().item()

        if self._grad_fn is not None:
            log_dict.update(self._grad_fn())
        self._opt.step()
        self._net.zero_grad()
        return log_dict

    def validate(self): ## Need to pass ext_word2id to validation function)
        #fw_args, bw_args, word2id = next(self._val_batcher(self._batch_size)) # this word2id isnot complete
        return self._val_fn(self._val_batcher(self._batch_size))
        #return self._val_fn(word2id, self._val_batcher(self._batch_size))

    def checkpoint(self, save_path, step, val_metric=None):
        save_dict = {}
        if val_metric is not None:
            name = 'ckpt-{:6f}-{}'.format(val_metric, step)
            save_dict['val_metric'] = val_metric
        else:
            name = 'ckpt-{}'.format(step)

        save_dict['state_dict'] = self._net.state_dict()
        save_dict['optimizer'] = self._opt.state_dict()
        torch.save(save_dict, join(save_path, name))

    def terminate(self):
        self._train_batcher.terminate()
        self._val_batcher.terminate()


class BasicTrainer(object):
    """ Basic trainer with minimal function and early stopping"""
    def __init__(self, pipeline, save_dir, ckpt_freq, patience,
                 scheduler=None, val_mode='loss', trained_step=0):
        assert isinstance(pipeline, BasicPipeline)
        assert val_mode in ['loss', 'score']
        self._pipeline = pipeline
        self._save_dir = save_dir
        self._logger = tensorboardX.SummaryWriter(join(save_dir, 'log'))
        if not exists(join(save_dir, 'ckpt')):  
            os.makedirs(join(save_dir, 'ckpt'))

        self._ckpt_freq = ckpt_freq
        self._patience = patience
        self._sched = scheduler
        self._val_mode = val_mode

        self._step = trained_step
        self._running_loss = None
        # state vars for early stopping
        self._current_p = 0
        self._best_val = None

    def log(self, log_dict):
        loss = log_dict['loss'] if 'loss' in log_dict else log_dict['reward']
        if self._running_loss is not None:
            self._running_loss = 0.99*self._running_loss + 0.01*loss
        else:
            self._running_loss = loss
        print('train step: {}, {}: {:.4f}\r'.format(
            self._step,
            'loss' if 'loss' in log_dict else 'reward',
            self._running_loss), end='')
        for key, value in log_dict.items():
            self._logger.add_scalar(
                '{}_{}'.format(key, self._pipeline.name), value, self._step)

    def validate(self):
        print()
        val_log = self._pipeline.validate()
        for key, value in val_log.items():
            self._logger.add_scalar(
                'val_{}_{}'.format(key, self._pipeline.name),
                value, self._step
            )
        if 'reward' in val_log:
            val_metric = val_log['reward']
        else:
            val_metric = (val_log['loss'] if self._val_mode == 'loss'
                          else val_log['score'])
        return val_metric

    def checkpoint(self):
        val_metric = self.validate()
        self._pipeline.checkpoint(
            join(self._save_dir, 'ckpt'), self._step, val_metric)
        if isinstance(self._sched, ReduceLROnPlateau):
            self._sched.step(val_metric)
        else:
            self._sched.step()
        stop = self.check_stop(val_metric)
        return stop

    def check_stop(self, val_metric):
        if self._best_val is None:
            self._best_val = val_metric
        elif ((val_metric < self._best_val and self._val_mode == 'loss')
              or (val_metric > self._best_val and self._val_mode == 'score')):
            self._current_p = 0
            self._best_val = val_metric
        else:
            self._current_p += 1
        return self._current_p >= self._patience

    def train(self):
        try:
            start = time()
            print('Start training')
            while True:
            #for i in range(20000):
                log_dict = self._pipeline.train_step()
                self._step += 1
                if log_dict != None:
                    self.log(log_dict)

                if self._step % self._ckpt_freq == 0:
                    stop = self.checkpoint()
                    if stop:
                        break
            print('Training finised in ', timedelta(seconds=time()-start))
        finally:
            self._pipeline.terminate()
