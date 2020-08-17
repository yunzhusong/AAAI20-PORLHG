from nltk.corpus import stopwords
import os
import pdb
import argparse
import numpy as np
import json
from os.path import join,exists
from collections import defaultdict
import math

from utils import make_vocab
import pickle as pkl

try:
	DATA_DIR = os.environ['DATA']
except KeyError:
	print('please use environment variable to specify data directories')

def read_a_json(data_dir, file):
	with open(join(data_dir, file)) as f:
		data = json.loads(f.read())
	if data['share']!=-1 and data['comment']!=-1:
		shar = data['share']
		comm = data['comment']
		head = data['headline'][0]
		cate = data['category'][0]
	else:
		return None
	return head, shar, comm, cate

def conver2id(unk, word2id, words):
	word2id = defaultdict(lambda: unk, word2id)
	return [word2id[w] for w in words]
	#return [[word2id[w] for w in words.split()] for words in words_list]

def remove_stopwords(inputstr, stop_words):
	out_list = list()
	for word in inputstr.split():
		if word not in stop_words:
			out_list.append(word)
	return out_list

def main(args):

	with open('comment_mtx.pkl', 'rb') as f:
		aa = pkl.load(f)
	with open('comment_mtx_fre.pkl', 'rb') as f:
		bb = pkl.load(f)

	pdb.set_trace()

	with open(join(DATA_DIR, 'vocab_cnt.pkl'), 'rb') as f:
		wc = pkl.load(f)
	word2id = make_vocab(wc, args.vsize)

	comment_mtx = np.zeros(len(word2id))

	#stop_words = set(stopwords.words('english'))
	frequency = np.zeros(len(word2id))
	headlines, shares, comments, categorys = [], [], [], []
	dir_path = os.path.join(DATA_DIR, 'train')
	files = os.listdir( dir_path)
	for file in files:
		if read_a_json(dir_path, file) != None:
			head, shar, comm, cate = read_a_json(dir_path, file)
			#head_list = remove_stopwords(head, stop_words)
			head_list = head.split()
			words_idx = conver2id( 1, word2id, head_list)
			weight = float(comm)/len(words_idx)
			for idx in words_idx:
				comment_mtx[idx] += weight
				frequency[idx] += 1 ## count appearance frequency
			#headlines.append(head)
			#shares.append(shar)
			#comments.append(comm)
			#categorys.append(cate)

	for i in range(len(word2id)):
		temp = comment_mtx[i]/frequency[i]
		if math.isnan(temp):
			temp = 0
		comment_mtx[i] = temp
	with open('comment_mtx_fre.pkl', 'wb') as f:
		pkl.dump(comment_mtx, f)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--vsize', type=int, default=30000)
	args = parser.parse_args()
	
	main(args)



