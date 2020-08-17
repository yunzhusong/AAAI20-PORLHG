# _*_ coding: utf-8 _*_
import pdb
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn import functional as F

INI = 1e-2

class LSTMClassifier(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights, conv_hidden, dropout):
		super(LSTMClassifier, self).__init__()
		
		"""
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		"""
		
		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		self._init_h = nn.Parameter(torch.Tensor(self.batch_size, self.hidden_size)).cuda()
		self._init_c = nn.Parameter(torch.Tensor(self.batch_size, self.hidden_size)).cuda()
		init.uniform_(self._init_h, -INI, INI)
		init.uniform_(self._init_c, -INI, INI)
	
		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
		self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
		self.lstm = nn.LSTM(embedding_length, hidden_size)
		self.label = nn.Linear(hidden_size, output_size)
		### Conv sent representation############
		self._convs = nn.ModuleList([nn.Conv1d(embedding_length, conv_hidden, i, padding=0) 
									for i in range(1,4)])
		self._dropout = dropout
		self._grad_handle = None

		
	def forward(self, input_sentence, batch_size=None):
	
		""" 
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
		
		Returns
		-------
		Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
		final_output.shape = (batch_size, output_size)
		
		"""
		
		''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
		input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
		conv_in = F.dropout(input.transpose(1,2), self._dropout, training=self.training)
		#conv_out = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0] for conv in self._convs], dim=1)
		conv_out = torch.cat([F.relu(conv(conv_in)) for conv in self._convs], dim=2)
		input = conv_out.permute(2,0,1)

		#lstm_in = conv_out.unsqueeze(1).permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)

		#size = (1, self.batch_size, self.hidden_size)
		'''
		if batch_size is None:
			h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) # Initial hidden state of the LSTM
			c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) # Initial cell state of the LSTM
		else:
			h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
		'''
		init_states = (self._init_h.unsqueeze(0), self._init_c.unsqueeze(0))
		output, (final_hidden_state, final_cell_state) = self.lstm(input, init_states)
		final_output = self.label(final_hidden_state[-1]) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)

		#return final_output, conv_out
		return final_output
