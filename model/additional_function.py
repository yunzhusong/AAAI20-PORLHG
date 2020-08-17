import os
import pdb
import csv
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


'''
def format_BERT_data(df):
	header = "index\tsentence\n"

	outpath = "/home/yunzhu/Headline/Headline/BERT/glue_data/Headline"

	pdb.set_trace()
	

	cnt = 0
	name = 'comment_high'
	outdir = os.path.join(outpath, name)
	with open(os.path.join(outdir, 'test.csv'), 'w') as f:
		f.writeline(header)
	
'''

def get_pop_score(headlines):

	df_info = get_info(headlines)
	df_feat = get_features(df_info)

	mean = np.array([2.26726999e+01, 1.03902981e+02, 7.82566090e-02, 2.08434508e-01, 1.97875882e+00, 8.94968549e-01, 1.08409289e+00])
	std = np.array([6.11377437, 24.04761313, 0.26857497, 0.40618907, 1.44068698, 0.99228565, 1.12046033])
	X = (df_feat.values - mean)/std	
	model_name = '/home/yunzhu/Headline/Datasets/svm_comment_mean_multi_0.pkl'
	with open(model_name, 'rb') as f:
		svm_comment = pickle.load(f)

	pred_score = svm_comment.predict(X)

	return pred_score

def get_features(df):
	df_feat = pd.DataFrame()
	df_feat['length'] = [ i for i in df['length']]
	df_feat['char_length'] = [ i for i in df['char_length']]
	df_feat['question'] = [int(i) for i in df['question']]
	df_feat['signal_num'] = [ int(i) for i in df['signal_num']]
	df_feat['sentiment_num'] = [ i for i in df['sentiment_num']]
	df_feat['pos_num'] = [ i for i in df['pos_num']]
	df_feat['neg_num'] = [ i for i in df['neg_num']]
	return df_feat

def get_info(headlines):
	df_info = pd.DataFrame()
	lexicon_path = "/home/yunzhu/Headline/Datasets/lexicon/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff"
	set_pos, set_neg, set_both, set_neu = read_lexicon(lexicon_path)
	set_sentiment = set_pos.union(set_neg).union(set_both)
	df_info['length'] = [len(headline.split(" ")) for headline in headlines]
	df_info['char_length'] = [char_num(headline) for headline in headlines]
	df_info['question'] = [has_Qmark(headline) for headline in headlines]
	df_info['signal_num'] = [has_signal_word(headline) for headline in headlines]
	df_info['pos_num'] = [has_word(headline, set_pos) for headline in headlines]
	df_info['neg_num'] = [has_word(headline, set_neg) for headline in headlines]
	df_info['sentiment_num'] = [has_word(headline, set_sentiment) for headline in headlines]
	return df_info

def char_num(inputstr):
	cnt=0
	for i in inputstr.split(' '):
		cnt+=len(i)
	return cnt

def has_Qmark(inputstr):
	return '?' in inputstr

def has_signal_word(inputstr):
	signal_words = ["hence", "this", "therefore", "how", "why", "when", "which", "who", "like that"]
	for i in inputstr.split(" "):
		if i in signal_words:
			return True
	return False

def has_word(inputstr, set):
	cnt=0
	words = inputstr.split(" ")
	for word in words:
		if word in set:
			cnt+=1
	return cnt

def read_lexicon(filepath):
	set_pos, set_neg, set_neu, set_both = set(), set(), set(), set()
	with open(filepath, 'r') as f:
		lines = csv.reader(f, delimiter=' ')
		for line in lines:
			#print(line)
			polarity = line[-1].split("=")[1]
			if polarity == "negative" or polarity == "weakneg" or polarity == "strongneg":
				set_neg.add(line[2].split("=")[1])
			elif polarity == "positive":
				set_pos.add(line[2].split("=")[1])
			elif polarity == "neutral":
				set_neu.add(line[2].split("=")[1])
			elif polarity == "both":
				set_both.add(line[2].split("=")[1])
			else:
				print("[Info] There are other type in the lexicon")
	return set_pos, set_neg, set_both, set_neu


def decode(datas_enc, id2word, END, UNK):

	datas_dec = []
	max_id = len(id2word)-1
	for data in datas_enc:
		temp = ''
		if END in data:
			for word in data:
				id_ = word.item()
				if id_ == END:
					datas_dec.append(temp)
				elif id_ == UNK or id_>max_id:
					temp += '<unk> '
				elif id_==0:
					temp += '<pad>'
				elif id_ == -1:  ## PAD=0
					continue
				else:
					temp = temp + id2word[id_] + ' '
		else:
			for word in data:
				id_ = word.item()
				if id_ == UNK or id_>max_id:
					temp += '<unk> '
				elif id_ == -1 or id_ == 0: 
					continue
				else:
					temp = temp + id2word[id_] + ' '
			datas_dec.append(temp)				

	return datas_dec

def print_result(y ,y_pred, model_name):
	print("Model name: {}".format(model_name))
	print("Misclassified sample: {}".format((y_pred!=y).sum()))
	print("Accuracy: {}".format(accuracy_score(y, y_pred)))
	precision, recall, fscore, support = precision_recall_fscore_support(y, y_pred, average='micro')
	print("precision: {}".format(precision))
	print("recall: {}".format(recall))
	print("fscore: {}".format(fscore))

if __name__=="__main__":
	decode(datas_enc, id2word, END, UNK)
	get_pop_score(headlines)

