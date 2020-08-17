import math
import pdb
import dill
import sys
sys.path.insert(0, '/home/yunzhu/Headline/FASum/PORLHG_v3/model/classifier')
import torch
from torch.nn import functional as F
import numpy as np
from .additional_function import decode, get_pop_score
from pop_classifier import do_inference
PAD = 0
UNK = 1
START = 2
END = 3


#################### general sequence helper #########################
def len_mask(lens, device):
    """ users are resposible for shaping
    Return: tensor_type [B, T]
    """
    max_len = max(lens)
    batch_size = len(lens)
    mask = torch.ByteTensor(batch_size, max_len).to(device)
    mask.fill_(0)
    for i, l in enumerate(lens):
        mask[i, :l].fill_(1)
    return mask

def sequence_mean(sequence, seq_lens, dim=1):
    if seq_lens:
        assert sequence.size(0) == len(seq_lens)   # batch_size
        sum_ = torch.sum(sequence, dim=dim, keepdim=False)
        mean = torch.stack([s/l for s, l in zip(sum_, seq_lens)], dim=0)
    else:
        mean = torch.mean(sequence, dim=dim, keepdim=False)
    return mean

def sequence_loss(logits, cls_pro, targets, ext_word2id, xent_fn=None, pad_idx=0):
    """ functional interface of SequenceLoss"""
    assert logits.size()[:-1] == targets.size()
    id2word = {i: w for w,i in ext_word2id.items()}
    mask = targets != pad_idx
    target = targets.masked_select(mask)
    logit = logits.masked_select(  ## logits = batch_size*sent_len*dict_size ex. 32*15*30025
        mask.unsqueeze(2).expand_as(logits)
    ).contiguous().view(-1, logits.size(-1))
    ## YUNZHU
    impov_score = 0.
    loss_pop = True
    if loss_pop == True:
        results = torch.max(logits, dim=2, keepdim=True)[1][:,:,0] ## batch_size*sent_len ex.32*15
        tar_dec = decode(targets, id2word, END, UNK)
        gen_dec = decode(results, id2word, END, UNK)
        ## evaluate by svm
        #tar_score = get_pop_score(tar_dec)
        #gen_score = get_pop_score(gen_dec)

        
        ## evaluate by lstm
        with open('/home/yunzhu/Headline/FASum/PORLHG_v3/model/classifier/TEXT.Field', 'rb') as f:
            TEXT = dill.load(f)
        vocab_size = len(TEXT.vocab)
        word_embeddings = TEXT.vocab.vectors
        tar_score = do_inference(tar_dec, TEXT, vocab_size, word_embeddings)
        gen_score = do_inference(gen_dec, TEXT, vocab_size, word_embeddings)
        if tar_score == None or gen_score == None:
            print("[Error] classification error")
        


        impov_score = gen_score - tar_score
        
    if xent_fn:
        loss = xent_fn(logit, target) - int(loss_pop)*impov_score
        #loss = xent_fn(logit, target)
    else:
        loss = F.cross_entropy(logit, target) - int(loss_pop)*impov_score
        #loss = F.cross_entropy(logit, target)
    assert (not math.isnan(loss.mean().item())
            and not math.isinf(loss.mean().item()))
    return loss, impov_score

#def sequence_loss_ext(logits, targets, arts, xent_fn=None, pad_idx=0):
def sequence_loss_ext(logits, targets, batch_topics, xent_fn=None, pad_idx=0):
    """ functional interface of SequenceLoss"""
    assert logits.size()[:-1] == targets.size()
    mask = targets != pad_idx
    target = targets.masked_select(mask)
    logit = logits.masked_select(  ## logits = batch_size*sent_len*dict_size ex. 32*15*30025
        mask.unsqueeze(2).expand_as(logits)
    ).contiguous().view(-1, logits.size(-1))
    ## YUNZHU

    # Add the topic loss ########
    add_topic_loss = False
    if add_topic_loss == True:
        bs, _, max_num_sents = logits.size()
        topic_labels = make_topic_label(bs, max_num_sents, batch_topics) 
        topic_label  = topic_labels.masked_select(mask)
        loss_topic = xent_fn(logit, topic_label)
        if topic_label.size(0) != logit.size(0):
            pdb.set_trace()
    else:
        loss_topic = 0
    
    if xent_fn:
        loss = xent_fn(logit, target) + loss_topic
    else:
        loss = F.cross_entropy(logit, target)
    assert (not math.isnan(loss.mean().item())
            and not math.isinf(loss.mean().item()))
    return loss


#################### LSTM helper #########################

def reorder_sequence(sequence_emb, order, batch_first=False):
    """
    sequence_emb: [T, B, D] if not batch_first
    order: list of sequence length
    """
    batch_dim = 0 if batch_first else 1
    assert len(order) == sequence_emb.size()[batch_dim]

    order = torch.LongTensor(order).to(sequence_emb.device)
    sorted_ = sequence_emb.index_select(index=order, dim=batch_dim)

    return sorted_

def reorder_lstm_states(lstm_states, order):
    """
    lstm_states: (H, C) of tensor [layer, batch, hidden]
    order: list of sequence length
    """
    assert isinstance(lstm_states, tuple)
    assert len(lstm_states) == 2
    assert lstm_states[0].size() == lstm_states[1].size()
    assert len(order) == lstm_states[0].size()[1]

    order = torch.LongTensor(order).to(lstm_states[0].device)
    sorted_states = (lstm_states[0].index_select(index=order, dim=1),
                     lstm_states[1].index_select(index=order, dim=1))

    return sorted_states

#########################################################

def reconstruct_topic_dis(data_topic, topic_num, batch_size=None):
    if batch_size == None:
        num_sent = len(data_topic)
        topic_dis = torch.zeros((num_sent, topic_num), dtype=torch.float).cuda()
        for i in range(num_sent):
            #index = (torch.tensor(data_topic[i][0]),)
            index = (torch.tensor(data_topic[i][0], dtype=torch.long), )
            value = torch.FloatTensor(data_topic[i][1]).cuda()

            topic_dis[i].index_put_(index, value)

        return topic_dis
    else:
        # Deal with batch data
        batch_topic = data_topic
        batch_topic_dis = []
        for b in range(batch_size):
            data_topic = batch_topic[b]
            topic_dis = reconstruct_topic_dis(data_topic, topic_num)
            #topic_dis = reconstruct_topic_dis([data_topic]) !!! for rl
            batch_topic_dis.append(topic_dis)

        return batch_topic_dis

def reconstruct_topic_dis_rl(data_topic, topic_num, batch_size=None):
    if batch_size == None:
        num_sent = len(data_topic)
        topic_dis = torch.zeros((num_sent, topic_num), dtype=torch.float).cuda()
        for i in range(num_sent):
            #index = (torch.tensor(data_topic[i][0]),)
            index = (torch.tensor(data_topic[i][0], dtype=torch.long), )
            value = torch.FloatTensor(data_topic[i][1]).cuda()

            topic_dis[i].index_put_(index, value)

        return topic_dis
    else:
        # Deal with batch data
        batch_topic = data_topic
        batch_topic_dis = []
        for b in range(batch_size):
            data_topic = batch_topic[b]
            topic_dis = reconstruct_topic_dis([data_topic], topic_num)
            batch_topic_dis.append(topic_dis)

        return batch_topic_dis

def make_topic_label(bs, max_num_sents, batch_topics):
    #topic_label = torch.zeros((bs, max_num_sents)).cuda()
    #softmax = torch.nn.Softmax(dim=1)
    topic_labels = torch.zeros(bs, dtype=torch.long).cuda()

    for i in range(bs):
        topic_label = np.argmax(torch.tensor([sum(sent[1]) for sent in batch_topics[i]]))
        topic_labels[i] = topic_label

        #pdb.set_trace()
        #topic_labels[i][:len(topic_label)] = softmax(topic_label)
    return topic_labels.unsqueeze(1)
    #return softmax(topic_labels)
       


