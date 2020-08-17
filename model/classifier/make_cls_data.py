import os
import json
import pdb

def read_file(path, index):
	with open(os.path.join(index+'.ref')) as f:
		return f.read()

def write_file(pathin, fileout):
	path = "/home/yunzhu/Headline/Datasets/CNNDM/finished_files_cleaned/refs/test"
	num = len(os.listdir(pathin))
	with open(fileout, 'w') as f:
		line = "headline\tcomment\tshare\n"
		f.write(line)
	f_tsv = open('infer_cls_ref.tsv', 'a')
	for i in range(num):
		with open(os.path.join(path, str(i)+'.ref')) as f:
			headline = f.read()
		line = headline + "\t0\t0\n"
		f_tsv.write(line)

if __name__ == "__main__":
	write_file()


