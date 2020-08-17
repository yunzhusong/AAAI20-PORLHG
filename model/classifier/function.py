import os
import json
import pdb
import pandas as pd
from collections import Counter
import statistics
import time

def find_median(path, query):
	with open(os.path.join(path, 'train_h.tsv')) as f:
		train_data = f.read().split('\n')[1:-1]
	with open(os.path.join(path, 'val_h.tsv')) as f:
		val_data = f.read().split('\n')[1:-1]
	with open(os.path.join(path, 'test_h.tsv')) as f:
		test_data = f.read().split('\n')[1:-1]

	if query == 'comment':
		index = 1
	else:
		index = 2
	list_ = []

	for i in range(len(train_data)):
		value = int(train_data[i].split('\t')[index])
		list_.append(value)

	for i in range(len(val_data)):
		value = int(val_data[i].split('\t')[index])
		list_.append(value)

	for i in range(len(test_data)):
		value = int(test_data[i].split('\t')[index])
		list_.append(value)

	c_time = time.time()
	mode = Counter(list_).most_common(1)
	#print('cost {}'.format(time.time() - c_time))
	#print('The mode of {} number: {}'.format(query, mode))
	
	c_time = time.time()
	median = statistics.median(list_)
	#print('cost {}'.format(time.time() - c_time))
	print('The median of {} number: {}'.format(query, median))

	return median

def clean_cls_dataset(raw_path, out_path, split):
	print('Romove the data with 0 comment number and 0 share number')

	with open(os.path.join(raw_path, split), encoding='utf-8') as f:
		data = f.read().split('\n')[1:-1] # exclude the first column

	with open(os.path.join(out_path, split), 'w') as f:
		f.write('headline\tcomment\tshare\n')

	f_out = open(os.path.join(out_path, split), 'a')

	cnt_rm = 0 # count the number of removed data

	for i in range(len(data)):
		print('processing {}/{} ({:.2f}%)\r'.format(i, len(data), i*100/len(data)), end='')
		_, comment, share = data[i].split('\t')
		comment, share = int(comment), int(share)
			
		if comment==0 and share==0:
			cnt_rm += 1
			# Remove the data with 0 comment and share
		else:
			f_out.write(data[i]+'\n')
	print('Process done')
	print('{} data are removed.'.format(cnt_rm))


def format_cls_dataset(raw_path, out_path, filename, median_c, median_s):
	print('Formatting the {} dataset for Classifer'.format(filename))
	#print('The median for comment is {}'.format(median_c))
	#print('The median for share is {}'.format(median_s))

	with open(os.path.join(raw_path, filename)) as f:
		data = f.read().split('\n')[1:-1]

	with open(os.path.join(out_path, filename), 'w') as f:
		f.write('headline\tcomment\tshare\n')

	f_out = open(os.path.join(out_path, filename), 'a')


	ckt_comment = 0
	for i in range(len(data)):
		headline, comment, share = data[i].split('\t')
		comment, share = int(comment), int(share)
		print('{}/{} ({:.2f}%) done\r'.format(i, len(data), 100*i/len(data)), end='')

		if comment > median_c and share > median_s:
			line = headline+'\t1\t1\n'
			ckt_comment+=1
		if comment > median_c and share <= median_s:
			line = headline+'\t1\t0\n'
			ckt_comment+=1
		if comment <= median_c and share > median_s:
			line = headline+'\t0\t1\n'
		else:
			line = headline+'\t1\t1\n'
		f_out.write(line)

	print('checking: {}/{} ({:.2f}) should equal to 0.5'.format(ckt_comment, len(data), ckt_comment/len(data)))

def format_cls_abs_dataset(headline_path, label_path, out_path, split):

	with open(os.path.join(label_path, "{}_h.tsv".format(split))) as f:
		labels = f.read().split('\n')[1:-1]
	with open(os.path.join(out_path, "{}_h.tsv".format(split)), 'w') as f:
		f.write('headline\tcomment\tshare\n')

	f_out = open(os.path.join(out_path, "{}_h.tsv".format(split)))
	
	n_data = len(labels)
	for i in range(n_data):
		_, comment, share = labels[i].split('\t')
	
		with open(os.path.join(headline_path, "{}\output\{}.dec".format(split, i))) as f:
			headline = f.read()

		line = "{}\t{}\t{}\n".format(headline, comment, share)
		f_out.write(line)

		print('{}/{} ({:.2f}%) done\r'.format(i, n_data, 100*i/n_data), end='')



ori_path = "/home/yunzhu/Headline/Datasets/dataset_cls/ori"
ori_without0_path = "/home/yunzhu/Headline/Datasets/dataset_cls/ori_without0"
m36s29_without0_path = "/home/yunzhu/Headline/Datasets/dataset_cls/2_class_m36s29_without0"
m35s32_without0_path = "/home/yunzhu/Headline/Datasets/dataset_cls/2_class_m35s32_without0"

#clean_cls_dataset(ori_path, ori_without0_path, 'val_h.tsv')
#clean_cls_dataset(ori_path, ori_without0_path, 'test_h.tsv')
#clean_cls_dataset(ori_path, ori_without0_path, 'train_h.tsv')

median_c = find_median(ori_without0_path, 'comment')
median_s = find_median(ori_without0_path, 'share')

#format_cls_dataset(ori_without0_path, m35s32_without0_path, 'val_h.tsv', median_c, median_s)
#format_cls_dataset(ori_without0_path, m35s32_without0_path, 'test_h.tsv', median_c, median_s)
#format_cls_dataset(ori_without0_path, m35s32_without0_path, 'train_h.tsv', median_c, median_s)

headline_path = "/home/yunzhu/Headline/PORLHG_v3/save_decode_result/exp_0901/abs"
label_path = "/home/yunzhu/Headline/Datasets/dataset_cls/2_class_m24s10"
out_path = "/home/yunzhu/Headline/Datasets/dataset_cls/2_class_m24s10_abs"
format_cls_abs_dataset(headline_path, label_path, out_path, split="test")
format_cls_abs_dataset(headline_path, label_path, out_path, split="val")
format_cls_abs_dataset(headline_path, label_path, out_path, split="train")
