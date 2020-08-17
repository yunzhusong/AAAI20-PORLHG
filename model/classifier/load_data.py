# _*_ coding: utf-8 _*_
import pdb
import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
import dill
def load_dataset(data_path, bs, test_sen=None, filename=None):

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    
    """
    
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=30)
    #LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    LABEL = data.LabelField()

    if filename != None:
        with open('TEXT_abs.Field', 'rb') as f:
            TEXT = dill.load(f)

    train_datafields = [('headline', TEXT),('comment', LABEL), ('share', None)]
    test_datafields = [('headline', TEXT), ('comment', LABEL), ('share', None)]
    train_data, valid_data = data.TabularDataset.splits(path=data_path,
                                                train='train_h.tsv', validation='val_h.tsv',
                                                format='tsv', 
                                                skip_header=True, 
                                                fields=train_datafields)
    # Choose the target data  ####################################################
    # For the regular test data from CNNDM dataset
    test_data = data.TabularDataset(path=data_path+'/test_h.tsv',
                                        format='tsv',
                                        skip_header=True,
                                        fields=test_datafields)
	# For testing the popularity of generated headlines  (1)#######################
    #gen_data = data.TabularDataset(path=filename,
    #                                    format='tsv',
    #                                    skip_header=True,
    #                                    fields=test_datafields)
    #print("We are testing: {}".format(filename))
    ###############################################################################

    if filename == None:
        TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)


    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32, sort_key=lambda x: len(x.headline), repeat=False, shuffle=True)

    #train_iter, valid_iter, gen_iter = data.BucketIterator.splits((train_data, valid_data, gen_data), batch_size=32, sort_key=lambda x: len(x.headline), repeat=False, shuffle=True)  #(1)

    vocab_size = len(TEXT.vocab)

    #with open('TEXT.Field', 'wb') as f:
    #    dill.dump(TEXT, f)
    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter
    #return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, gen_iter  #(1)
