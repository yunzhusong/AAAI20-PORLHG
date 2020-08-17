import torch
from torch import nn
from torch.nn import init
import pdb


INIT = 1e-2

class LSTMClassifier(nn.Module):
	def __init__(self, vocab_size, emb_dim, n_hidden, bidirectional, n_layer, dropout=0.0):
		super().__init__()
		self._vocab_size = vocab_size
		self._in_projection = nn.Linear(vocab_size, emb_dim, bias=False)
		self._lstm = nn.LSTM(emb_dim, n_hidden, n_layer, 
							 bidirectional=bidirectional, dropout=dropout)
		state_layer = n_layer * (2 if bidirectional else 1)
		self._init_h = nn.Parameter(torch.Tensor(state_layer, n_hidden))
		self._init_c = nn.Parameter(torch.Tensor(state_layer, n_hidden))
		init.uniform_(self._init_h, -INIT, INIT)
		init.uniform_(self._init_c, -INIT, INIT)

		out_dim = n_hidden * (2 if bidirectional else 1)
		self._out_projection = nn.Sequential(nn.Linear(n_hidden, 2, bias=False),
										 nn.Softmax(dim=1))
	def __call__(self, input_, art_lens=None):
		batch_size = len(input_)
		input_ = input_.transpose(0,1)
		size = (self._init_h.size(0), batch_size, self._init_h.size(1))
		init_states = (self._init_h.unsqueeze(1).expand(*size).contiguous(),
					   self._init_c.unsqueeze(1).expand(*size).contiguous())
		assert input_.size(-1) == self._vocab_size
		lstm_in = self._in_projection(input_)
		lstm_out, final_states = self._lstm(lstm_in, init_states)
		final_hidden_state, _ = final_states
		final_out = self._out_projection(final_hidden_state[-1])

		return final_out
