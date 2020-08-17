import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import pickle as pkl

from .rnn import MultiLayerLSTMCells
from .rnn import lstm_encoder
from .util import sequence_mean, len_mask, reconstruct_topic_dis, reconstruct_topic_dis_rl
from .attention import prob_normalize

import pdb
INI = 1e-2

class ConvSentEncoder(nn.Module):
    """
    Convolutional word-level sentence encoder
    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """
    def __init__(self, vocab_size, emb_dim, n_hidden, dropout):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i)
                                     for i in range(3, 6)])
        self._dropout = dropout
        self._grad_handle = None

    def forward(self, input_, weight=None):
        '''
        comment_mtx = pkl.load(open('comment_mtx.pkl', 'rb'))
        comment = torch.FloatTensor(comment_mtx).cuda()
        all_words = []
        for i in range(30004):
            all_words.append(i)
        all_words = torch.LongTensor(all_words).cuda()
        emb_all_words = self._embedding(all_words) ## emb: FloatTensor (30004*128)
        weight = torch.matmul(comment, emb_all_words).unsqueeze(0)
        #weight = weight.unsqueeze(0).unsqueeze(2).expand((1,emb_dim, vocab_size)) ## (1,30004,128)
        '''
        if weight is not None:
            emb_words = self._embedding(input_) ## (30004,128)
            #emb_input = torch.matmul(weight, emb_words).expand((30004,128)) ## (30004,128)
            # revised 0518
            weight_ = (F.softmax(weight.unsqueeze(1), dim=0)+1).expand((30004,128))
            emb_input = weight_ * emb_words
            emb_input = emb_input.unsqueeze(0) ## (1,30004,128)
        else:
            emb_input = self._embedding(input_)  ## (sents_n, sent_n, 128)
        conv_in = F.dropout(emb_input.transpose(1, 2),
                            self._dropout, training=self.training)
        output = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0]
                            for conv in self._convs], dim=1)
        return output

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)
        return embedding


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, n_hidden, n_layer, dropout, bidirectional):
        super().__init__()
        self._init_h = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))  ## 2*256
        self._init_c = nn.Parameter( 
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))  ## 2*256
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        self._lstm = nn.LSTM(input_dim, n_hidden, n_layer,  ## 300, 256, 1
                             dropout=dropout, bidirectional=bidirectional)
    def forward(self, input_, in_lens=None):
        """ [batch_size, max_num_sent, input_dim] Tensor"""
        size = (self._init_h.size(0), input_.size(0), self._init_h.size(1))
        init_states = (self._init_h.unsqueeze(1).expand(*size),
                       self._init_c.unsqueeze(1).expand(*size))
        lstm_out, _ = lstm_encoder(
            input_, self._lstm, in_lens, init_states)
        return lstm_out.transpose(0, 1)

    @property
    def input_size(self):
        return self._lstm.input_size

    @property
    def hidden_size(self):
        return self._lstm.hidden_size

    @property
    def num_layers(self):
        return self._lstm.num_layers

    @property
    def bidirectional(self):
        return self._lstm.bidirectional


class ExtractSumm(nn.Module):
    """ ff-ext """
    def __init__(self, vocab_size, emb_dim,
                 conv_hidden, lstm_hidden, lstm_layer,
                 bidirectional, dropout=0.0):
        super().__init__()
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout)
        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self._sent_linear = nn.Linear(lstm_out_dim, 1)
        self._art_linear = nn.Linear(lstm_out_dim, lstm_out_dim)

    def forward(self, article_sents, sent_nums):
        enc_sent, enc_art = self._encode(article_sents, sent_nums)
        saliency = torch.matmul(enc_sent, enc_art.unsqueeze(2))
        saliency = torch.cat(
            [s[:n] for s, n in zip(saliency, sent_nums)], dim=0)
        content = self._sent_linear(
            torch.cat([s[:n] for s, n in zip(enc_sent, sent_nums)], dim=0)
        )
        logit = (content + saliency).squeeze(1)
        return logit

    def extract(self, article_sents, sent_nums=None, k=4):
        """ extract top-k scored sentences from article (eval only)"""
        enc_sent, enc_art = self._encode(article_sents, sent_nums)
        saliency = torch.matmul(enc_sent, enc_art.unsqueeze(2))
        content = self._sent_linear(enc_sent)
        logit = (content + saliency).squeeze(2)
        if sent_nums is None:  # test-time extract only
            assert len(article_sents) == 1
            n_sent = logit.size(1)
            extracted = logit[0].topk(
                k if k < n_sent else n_sent, sorted=False  # original order
            )[1].tolist()
        else:
            extracted = [l[:n].topk(k if k < n else n)[1].tolist()
                         for n, l in zip(sent_nums, logit)]
        return extracted

    def _encode(self, article_sents, sent_nums):
        if sent_nums is None:  # test-time extract only
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._sent_enc(art_sent)
                         for art_sent in article_sents]
            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.device)],
                           dim=0) if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            )
        lstm_out = self._art_enc(enc_sent, sent_nums)
        enc_art = torch.tanh(
            self._art_linear(sequence_mean(lstm_out, sent_nums, dim=1)))
        return lstm_out, enc_art

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)


class LSTMPointerNet(nn.Module):
    """Pointer network as in Vinyals et al """
    def __init__(self, input_dim, n_hidden, n_layer,
                 dropout, n_hop):
        super().__init__()
        self._init_h = nn.Parameter(torch.Tensor(n_layer, n_hidden)) ## 1*256
        self._init_c = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        init.uniform_(self._init_i, -0.1, 0.1)
        self._lstm = nn.LSTM(
            input_dim, n_hidden, n_layer,
            bidirectional=False, dropout=dropout
        )
        self._lstm_cell = None

        # attention parameters
        print('input_dim', input_dim)
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden)) ## 1024*256, for ext with PTA 
        #self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden)) ## 512*256 
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))  ## 256*256
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))   ## 256
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        init.uniform_(self._attn_v, -INI, INI)

        # hop parameters
        self._hop_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._hop_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._hop_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._hop_wm)
        init.xavier_normal_(self._hop_wq)
        init.uniform_(self._hop_v, -INI, INI)
        self._n_hop = n_hop
    def forward(self, attn_mem, mem_sizes, lstm_in):
        """atten_mem: Tensor of size [batch_size, max_sent_num, input_dim]"""
        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem)
        #lstm_in = torch.cat([init_i, lstm_in], dim=1).transpose(0, 1)
        ## YUNZHU
        #lstm_in[:,0,:] = init_i.squeeze(1) ## lstm_in = (32, 1 ,512)
        lstm_in = lstm_in.transpose(0, 1)  ##(32,1,512) -> (1,32,512)

        query, final_states = self._lstm(lstm_in, lstm_states)
        query = query.transpose(0, 1) ## (32,1,512) -> (1,32,512)
        for _ in range(self._n_hop):
            query = LSTMPointerNet.attention(
                hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
        output = LSTMPointerNet.attention_score(
            attn_feat, query, self._attn_v, self._attn_wq)

        return output  # unormalized extraction logit

    def extract(self, attn_mem, mem_sizes, k):
        """extract k sentences, decode only, batch_size==1"""
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        extracts = []
        for _ in range(k):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = LSTMPointerNet.attention(
                    hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
            score = LSTMPointerNet.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            score = score.squeeze()
            for e in extracts:
                score[e] = -1e6
            ext = score.max(dim=0)[1].item()
            extracts.append(ext)
            lstm_states = (h, c)
            lstm_in = attn_mem[:, ext, :]
        return extracts

    def _prepare(self, attn_mem):
        attn_feat = torch.matmul(attn_mem, self._attn_wm.unsqueeze(0))  ## (32,max_num,512)(1,512,256) -> 32*max_num*256
        hop_feat = torch.matmul(attn_mem, self._hop_wm.unsqueeze(0))
        bs = attn_mem.size(0)
        n_l, d = self._init_h.size() ## 1,256
        size = (n_l, bs, d)  ## 1,32,256
        lstm_states = (self._init_h.unsqueeze(1).expand(*size).contiguous(),
                       self._init_c.unsqueeze(1).expand(*size).contiguous())
        d = self._init_i.size(0) ## 512
        init_i = self._init_i.unsqueeze(0).unsqueeze(1).expand(bs, 1, d)  ## 512 -> 32*1*512
        return attn_feat, hop_feat, lstm_states, init_i

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention.unsqueeze(1) + torch.matmul(
            query, w.unsqueeze(0)
        ).unsqueeze(2)  # [B, Nq, Ns, D]
        score = torch.matmul(
            torch.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        ).squeeze(3)  # [B, Nq, Ns]
        return score

    @staticmethod
    def attention(attention, query, v, w, mem_sizes):
        """ attention context vector""" ## attention : 32*max_num*256
        score = LSTMPointerNet.attention_score(attention, query, v, w)  ## 32*1*max_num
        if mem_sizes is None:
            norm_score = F.softmax(score, dim=-1)
        else:
            mask = len_mask(mem_sizes, score.device).unsqueeze(-2)  ## 32*1*max_num
            norm_score = prob_normalize(score, mask)
        output = torch.matmul(norm_score, attention) ## 32*1*256
        return output


class PtrExtractSumm(nn.Module):
    """ rnn-ext"""
    def __init__(self, emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 n_hop=1, dropout=0.0):
        super().__init__()
        self._sent_enc = ConvSentEncoder(    ## _sent_enc, _art_enc, _extractor will pass to rl
            vocab_size, emb_dim, conv_hidden, dropout)
        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,             
            dropout=dropout, bidirectional=bidirectional
        )
        self._topic_num = 512
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1) + self._topic_num # concatenation !!!
        print('enc_out_dim', enc_out_dim)
        self._extractor = LSTMPointerNet(
            enc_out_dim, lstm_hidden, lstm_layer,
            dropout, n_hop
        )
        self.softmax_2 = torch.nn.Softmax(dim=2)

    def forward(self, article_sents, sent_nums, target, topics):
        enc_out = self._encode(article_sents, sent_nums)
       
        ##pop_info = enc_pop.expand(enc_out.size())*enc_out  ## (32, sents_num, 512)
        bs, nt = target.size() ## 32, 1(usually)
        _, _, d = enc_out.size() ## 512

        topics = reconstruct_topic_dis(topics, self._topic_num, batch_size=bs)
        topics = self._pad_topic(topics, sent_nums)
        topics = self.softmax_2(topics*1e1)

        try:
            enc_out = torch.cat((enc_out, topics), dim=2)
        except:
            pdb.set_trace()

        ptr_in = torch.gather(
            enc_out, dim=1, index=target.unsqueeze(2).expand(bs, nt, d+self._topic_num)  ## (32,nt,512) !!!
        ) ## pick the target sent in pop_info
        """ # Use popularity dictionary
        enc_pop = self._encode_pop('comment_mtx_fre.pkl') ## can open * if want to use comment mtx
        ptr_in = ptr_in * enc_pop ## cn open * if want to use comment mtx"""
        output = self._extractor(enc_out, sent_nums, ptr_in)
        return output

    #def extract(self, article_sents, sent_nums=None, k=4):
    def extract(self, article_sents, topics=None, sent_nums=None, k=1):
        enc_out = self._encode(article_sents, sent_nums)

        topics = reconstruct_topic_dis_rl(topics, self._topic_num)
        topics = self._pad_topic(topics, sent_nums)
        enc_out = torch.cat((enc_out, topics), dim=2)

        output = self._extractor.extract(enc_out, sent_nums, k)
        return output

    def _encode(self, article_sents, sent_nums):
        if sent_nums is None:  # test-time excode only, only 1 articlelstm
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._sent_enc(art_sent)
                         for art_sent in article_sents]
            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z
            ## enc_sent: (32*sent_nums*300)
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.device)], dim=0)
                   if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            )
        lstm_out = self._art_enc(enc_sent, sent_nums) ## 32*sent_nums*512
        return lstm_out

    def _encode_pop(self, comment_mtx_file):
        comment_mtx = pkl.load(open(comment_mtx_file, 'rb'))  ## comment_mtx.pkl
        weight = torch.FloatTensor(comment_mtx).cuda()
        word_list = []
        for i in range(30004):
            word_list.append(i)
        word_list = torch.LongTensor(word_list).cuda()
        enc_sent = self._sent_enc(word_list, weight).unsqueeze(0)
        lstm_out = F.normalize(self._art_enc(enc_sent), dim=2)
        return lstm_out  ## (32,1,512)

    def _pad_topic(self, topics, sent_nums):
        if sent_nums is None:
            outputs = topics.unsqueeze(0)
        else:
            max_n = max(sent_nums)
            def zero(n, device):
                z = torch.zeros([n, self._topic_num], dtype=torch.float).to(device)
                return z

            try:
                outputs = torch.stack(
                [torch.cat([t, zero(max_n-n, t.device)], dim=0) if n!= max_n
                 else t for t,n in zip(topics, sent_nums)], dim=0)
            except:
                pdb.set_trace()
        """
        outputs = torch.zeros(bs, max_sents, 512).cuda()
        for i in range(bs):
            num_sents, _ = topics[i].size()
            try:
                outputs[i,:num_sents,:] = topics[i]
            except:
                pdb.set_trace()  """
        return outputs


    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)
