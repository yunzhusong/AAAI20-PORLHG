import json
import os
import pdb
import argparse


def making(args, split):
	json_dir = os.path.join(args.path, split)
	n_data = len(os.listdir(json_dir))
	all_headline = ""
	all_extsents = ""
	for i in range(n_data):
		print("{}/{}".format(i, n_data))
		with open(os.path.join(json_dir, "{}.json".format(i))) as f:
			data = json.load(f)
		article = data['article']
		indices = data['extracted_headline']
		headline = data['headline'][0]
		matched_sents = [article[idx] for idx in indices]
		matched_sents = " ".join(matched_sents)
		if i == n_data-1:
			all_headline += headline
			all_extsents += matched_sents
		else:
			all_headline += (headline + "\n")
			all_extsents += (matched_sents + "\n")
	with open(os.path.join(args.path, "ExtSents_{}.txt".format(split)), 'w') as f:
		f.write(all_extsents)
	with open(os.path.join(args.path, "Headline_{}.txt".format(split)), 'w') as f:
		f.write(all_headline)

def main(args):
	splits = ['val', 'train', 'test']
	for split in splits:
		making(args, split)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--path')
	args = parser.parse_args()
	main(args)
