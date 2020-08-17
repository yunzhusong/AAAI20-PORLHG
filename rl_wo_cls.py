""" RL training utilities"""
import math
from time import time
from datetime import timedelta

from toolz.sandbox.core import unzip
from cytoolz import concat

import numpy as np
import torch
from torch.nn import functional as F
from torch import autograd
from torch.nn.utils import clip_grad_norm_

from metric import compute_rouge_l, compute_rouge_n
from training import BasicPipeline

import dill
import sys
sys.path.insert(0, '/home/yunzhu/Headline/FASum/FASRL/model/classifier')
from pop_classifier import test_model
from torchtext import data
from torchtext.data import Iterator
from models.LSTM import LSTMClassifier
from torchtext.vocab import Vectors, GloVe

import pdb
batch_size = 32
output_size = 2
hidden_size = 256
embedding_length = 300
conv_hidden = 300
save_path = "/home/yunzhu/Headline/FASum/FASRL/model/classifier/trained_model/lstm_all_2.pt"


def get_gen_score(batch_data, TEXT, vocab_size, word_embeddings):
    if TEXT == None:
        return 0
    LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings, conv_hidden, 0.1)
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict)

    
    test_datafields = [("headline", TEXT), ("comment", LABEL), ("share", None)]
    """
    with open('temp.tsv', 'w') as f:
        f.write("headline\tcomment\tshare\n")
    batch = ""
    for i in range(len(batch_data)):
        sentence = ""
        for j in batch_data[i][0]:
            token =  j+ " "
            sentence += token
        try:
            temp = sentence + '\t1\t1\n'
        except:
            temp = " \t1\t1\n"
        
        batch += temp
    with open('temp.tsv', 'a') as f:
        f.write(batch)
    test_data = data.TabularDataset(path="temp.tsv", format='tsv', skip_header=True, fields=test_datafields)
    """
    examples = [None]*len(batch_data)
    for i in range(len(batch_data)):
        sentence = ""
        if  batch_data[i]: # 若data不為空
            for j in batch_data[i]:
                token = j+ " "
                sentence += token
            temp = [sentence, 1, 1]
        else:
            temp = [" ", 1, 1]
            print("[info] empty sentence for classifer")
        example = data.Example.fromlist(temp, test_datafields)
        examples[i] = example
    test_data = data.Dataset(examples, fields=test_datafields)

    LABEL.build_vocab(test_data)
    test_iter = data.BucketIterator(test_data, batch_size=len(test_data), sort_key=lambda x: len(x.headline), repeat=False, shuffle=True)
    gen_score = test_model(model, test_iter)
    gen_score = torch.softmax(gen_score, dim=1)
    return gen_score


def a2c_validate(agent, abstractor, loader):
    agent.eval()
    start = time()
    print('start running validation...', end='')
    avg_reward = 0
    i = 0
    with torch.no_grad():
        for art_batch, topic_batch, abs_batch in loader:
            ext_sents = []
            ext_inds = []
            for raw_arts, topic in zip(art_batch, topic_batch):
                indices = agent(raw_arts, topic)
                ext_inds += [(len(ext_sents), len(indices)-1)]
                ext_sents += [raw_arts[idx.item()]
                              for idx in indices if idx.item() < len(raw_arts)]
            all_summs = abstractor(ext_sents)
            for (j, n), abs_sents in zip(ext_inds, abs_batch):
                summs = all_summs[j:j+n]
                # python ROUGE-1 (not official evaluation)
                avg_reward += compute_rouge_n(list(concat(summs)),
                                              list(concat(abs_sents)), n=1)
                i += 1
    avg_reward /= (i/100)
    print('finished in {}! avg reward: {:.2f}'.format(
        timedelta(seconds=int(time()-start)), avg_reward))
    return {'reward': avg_reward}


def a2c_train_step(agent, abstractor, loader, opt, grad_fn,
                   gamma=0.99, reward_fn=compute_rouge_l,
                   stop_reward_fn=compute_rouge_n(n=1), stop_coeff=1.0):
    opt.zero_grad()
    indices = []
    probs = []
    baselines = []
    ext_sents = []
    summ_indices = []
    art_batch, topic_batch, abs_batch = next(loader)
    for raw_arts, topic in zip(art_batch, topic_batch):
        (inds, ms), bs = agent(raw_arts, topic)
        baselines.append(bs)
        indices.append(inds)
        probs.append(ms)
        ext_sents += [raw_arts[idx.item()]
                      for idx in inds if idx.item() < len(raw_arts)]
        summ_indices += [[idx.item() for idx in inds if idx.item()<len(raw_arts)]]
        
    with torch.no_grad():
        summaries = abstractor(ext_sents)

    ## collect the generated headline
    summaries_collect = [None]*len(summ_indices)
    cnt = 0
    i = 0
    for summ_inds in summ_indices:
        try:
            if len(summ_inds) != 0:
                summaries_collect[cnt] = summaries[i]
                i = i+len(summ_inds)
            else:
                summaries_collect[cnt] = [" "]
        except:
            pdb.set_trace()
        cnt+=1
    add_cls_loss = False
    if add_cls_loss == True:
        ## collect the gnerated headline
        summaries_collect = [None]*len(summ_indices)
        cnt = 0
        i = 0
        for summ_inds in summ_indices:
            try:
                if len(summ_inds) != 0:
                    summaries_collect[cnt] = summaries[i]
                    i = i+len(summ_inds)
                else:
                    summaries_collect[cnt] = [" "]
            except:
                pdb.set_trace()
            cnt+=1
        with open('/home/yunzhu/Headline/FASum/FASRL/model/classifier/TEXT.Field', 'rb') as f:
            TEXT = dill.load(f)
        vocab_size = len(TEXT.vocab)
        word_embeddings = TEXT.vocab.vectors
    else:
        TEXT = None
        vocab_size = None
        word_embeddings = None
    #prediction = get_gen_score(summaries_collect, TEXT, vocab_size, word_embeddings)
    #cls_score = prediction[:,1].sum().item()

    i = 0
    rewards = []
    avg_reward = 0
    cnt=0
    for inds, abss in zip(indices, abs_batch):
        cls_r = 0
        rs = ([reward_fn(summaries[i+j], abss[j])+cls_r for j in range(min(len(inds)-1, len(abss)))]
              + [0 for _ in range(max(0, len(inds)-1-len(abss)))]
              + [stop_coeff*(stop_reward_fn(
                  list(concat(summaries[i:i+len(inds)-1])),
                  list(concat(abss)))+cls_r)])

        assert len(rs) == len(inds)
        avg_reward += rs[-1]/stop_coeff
        i += len(inds)-1
        # compute discounted rewards
        R = 0
        disc_rs = []
        for r in rs[::-1]:
            R = r + gamma * R
            disc_rs.insert(0, R)
        rewards += disc_rs
        cnt+=1

    indices = list(concat(indices))
    probs = list(concat(probs))
    baselines = list(concat(baselines))
    # standardize rewards
    reward = torch.Tensor(rewards).to(baselines[0].device)
    reward = (reward - reward.mean()) / (
        reward.std() + float(np.finfo(np.float32).eps))
    baseline = torch.cat(baselines).squeeze()
    avg_advantage = 0
    losses = []
    for action, p, r, b in zip(indices, probs, reward, baseline):
        advantage = r - b
        avg_advantage += advantage
        losses.append(-p.log_prob(action)
                      * (advantage/len(indices))) # divide by T*B
    critic_loss = F.mse_loss(baseline, reward)
    # backprop and update
    autograd.backward(
        [critic_loss.unsqueeze(0)] + losses, 
        [torch.ones(1).to('cuda')]*(1+len(losses))
        #[torch.ones(1).to(critic_loss.device)]*(1+len(losses))
    )
   
    grad_log = grad_fn()
    opt.step()
    log_dict = {}
    log_dict.update(grad_log)
    log_dict['reward'] = avg_reward/len(art_batch)
    log_dict['advantage'] = avg_advantage.item()/len(indices)
    log_dict['mse'] = critic_loss.item()
    assert not math.isnan(log_dict['grad_norm'])
    return log_dict


def get_grad_fn(agent, clip_grad, max_grad=1e2):
    """ monitor gradient for each sub-component"""
    params = [p for p in agent.parameters()]
    def f():
        grad_log = {}
        for n, m in agent.named_children():
            tot_grad = 0
            for p in m.parameters():
                if p.grad is not None:
                    tot_grad += p.grad.norm(2) ** 2
            tot_grad = tot_grad ** (1/2)
            grad_log['grad_norm'+n] = tot_grad.item()
        grad_norm = clip_grad_norm_(
            [p for p in params if p.requires_grad], clip_grad)
        #grad_norm = grad_norm.item() # .itmes() attribute is changed in pytorch version 0.4.1
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log['grad_norm'] = grad_norm
        return grad_log
    return f


class A2CPipeline(BasicPipeline):
    def __init__(self, name,
                 net, abstractor,
                 train_batcher, val_batcher,
                 optim, grad_fn,
                 reward_fn, gamma,
                 stop_reward_fn, stop_coeff):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._opt = optim
        self._grad_fn = grad_fn

        self._abstractor = abstractor
        self._gamma = gamma
        self._reward_fn = reward_fn
        self._stop_reward_fn = stop_reward_fn
        self._stop_coeff = stop_coeff

        self._n_epoch = 0  # epoch not very useful?

    def batches(self):
        raise NotImplementedError('A2C does not use batcher')

    def train_step(self):
        # forward pass of model
        self._net.train()
        log_dict = a2c_train_step(
            self._net, self._abstractor,
            self._train_batcher,
            self._opt, self._grad_fn,
            self._gamma, self._reward_fn,
            self._stop_reward_fn, self._stop_coeff
        )
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
        pdb.set_trace()
        """
        return log_dict

    def validate(self):
        return a2c_validate(self._net, self._abstractor, self._val_batcher)

    def checkpoint(self, *args, **kwargs):
        # explicitly use inherited function in case I forgot :)
        return super().checkpoint(*args, **kwargs)

    def terminate(self):
        pass  # No extra processs so do nothing
