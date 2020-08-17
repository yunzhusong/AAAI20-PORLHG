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
#
from metric import compute_rouge_l, compute_rouge_n
from training import BasicPipeline
from model.util import reconstruct_topic_dis_rl

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
save_path = "/home/yunzhu/Headline/FASum/FASRL/model/classifier/trained_model/lstm_all_1.pt"
#save_path = "/home/yunzhu/Headline/FASum/PORLHG_v3/model/classifier/trained_model/lstm_c35s32_without0.pt"


def get_gen_score(batch_data, TEXT, vocab_size, word_embeddings):
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
    test_iter = data.BucketIterator(test_data, batch_size=len(test_data), sort_key=lambda x: len(x.headline), repeat=False, shuffle=False)
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
    ext_sents, ext_sents_topic, ext_sents_probs, summ_indices= [], [], [], []
    #art_batch, topic_batch, abs_batch, topic_label_batch = next(loader)
    art_batch, topic_batch, abs_batch = next(loader)
    for raw_arts, topic in zip(art_batch, topic_batch):
        (inds, ms), bs = agent(raw_arts, topic)
        baselines.append(bs)
        indices.append(inds)
        probs.append(ms)
        num_sents = len(raw_arts)
        ext_sents += [raw_arts[idx.item()]    for idx in inds if idx.item()<num_sents]
        #ext_sents_topic += [topic[idx.item()] for idx in inds if idx.item()<num_sents]
        ext_sents_probs += [ m.probs[0][idx] for idx, m in zip(inds, ms) if idx.item()<num_sents]
        summ_indices += [[idx.item()          for idx in inds if idx.item()<num_sents]]
    if ext_sents == []:
        #print('Reach the end')
        return None
    with torch.no_grad():
        summaries = abstractor(ext_sents)

    # Expand the topic dis of the reference
    #topic_label_expand = []
    #for label, inds  in zip(topic_label_batch, summ_indices):
    #    num_ext = len(inds)
    #    topic_label_expand += [label]*num_ext
    """
    legal_ext_num = len(ext_sents)
    #topic_label_expand = torch.stack(reconstruct_topic_dis(topic_label_expand, legal_ext_num)).squeeze(1).cuda()
    ext_sents_topic    = torch.stack(reconstruct_topic_dis_rl(ext_sents_topic, legal_ext_num)).squeeze(1).cuda()
    ext_sents_probs    = torch.stack(ext_sents_probs).squeeze(1).cuda()
    #rs_topic = 1 - ext_sents_probs * abs(topic_label_expand - ext_sents_topic).sum(dim=1) # by L1 distance
    #rs_topic = 0.01*(ext_sents_probs * abs(topic_label_expand * ext_sents_topic).sum(dim=1)) # by inner product
    rs_topic = 0.05*ext_sents_probs * (ext_sents_topic).sum(dim=1) # by inner product if m2
    rs_topic_avg = rs_topic.sum()/rs_topic.size(0)
    rs_topic_list = rs_topic.cpu().tolist()
 
    """
   
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

    add_cls_loss = True
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
        with open('/home/yunzhu/Headline/FASum/PORLHG_v3/model/classifier/TEXT.Field', 'rb') as f:
        #with open('/home/yunzhu/Headline/FASum/PORLHG_v3/model/classifier/TEXT_abs.Field', 'rb') as f:
            TEXT = dill.load(f)
        vocab_size = len(TEXT.vocab)
        word_embeddings = TEXT.vocab.vectors
        prediction = get_gen_score(summaries_collect, TEXT, vocab_size, word_embeddings)
        cls_score = (prediction[:,1].sum()/prediction.size(0)).item()
    else:
        TEXT = None
        vocab_size = None
        word_embeddings = None
    #prediction = get_gen_score(summaries_collect, TEXT, vocab_size, word_embeddings)
    #cls_score = prediction[:,1].sum().item()

    i, i_topic = 0, 0
    rewards = []
    avg_reward = 0
    cnt=0
    for inds, abss in zip(indices, abs_batch):
        if add_cls_loss == True:
            try:
                #cls_r = prediction[cnt,1].item()*0.2
                cls_r = prediction[cnt,1].item()
            except:
                print('[Info] In rl.py code, cls_r=0')
                cls_r = 0
        else:
            cls_r = 0
        #rs_topic_ = (rs_topic[:len(inds)].sum()/len(inds)).item()
        rs_topic_ = 0
        rs = ([reward_fn(summaries[i+j], abss[j])+ cls_r + rs_topic_ for j in range(min(len(inds)-1, len(abss)))]
              + [0 for _ in range(max(0, len(inds)-1-len(abss)))]
              + [stop_coeff*(stop_reward_fn(
                  list(concat(summaries[i:i+len(inds)-1])),
                  list(concat(abss))))]) 
                  #list(concat(abss)))+cls_r)])  # + cls_r

        #try:
            #rs[:len(inds)-1] = np.add(rs[:len(inds)-1], rs_topic_list[i_topic:i_topic+len(inds)-1])
        #    rs[:-1] = np.add(rs[:-1], rs_topic_list[i:i+len(inds)-1])

        #except:
        #   pdb.set_trace()

        assert len(rs) == len(inds)
        avg_reward += rs[-1]/stop_coeff
        i += len(inds)-1
        i_topic += len(inds)
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
    for action, prob, r, b in zip(indices, probs, reward, baseline):
        advantage = r - b
        avg_advantage += advantage
        losses.append(-prob.log_prob(action)
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
    #log_dict['rs_topic'] = rs_topic_avg.item()
    if add_cls_loss == True:
        log_dict['cls_score'] = cls_score
    assert not math.isnan(log_dict['grad_norm'])
    return log_dict


def get_grad_fn(agent, clip_grad, max_grad=1e2):
    """ monitor gradient for each sub-component"""
    params = [p for p in agent.parameters()]
    def f():
        grad_log = {}
        for n, m in agent.named_children():
            #print("n: ", n)
            #print("m: ", m)
            tot_grad = 0
            for param in m.parameters():
                #print("param: ", param)
                #print("param.grad: ", param.grad)
                #print("    shape: ", param.shape)
                if param.grad is not None:
                    #print("grad_norm: ", param.grad.norm(2).item())
                    tot_grad += param.grad.norm(2) ** 2
            tot_grad = tot_grad ** (1/2)
            #print("tot_grad: ", tot_grad)
            grad_log['grad_norm'+n] = tot_grad
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
