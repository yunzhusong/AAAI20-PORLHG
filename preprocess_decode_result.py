import os
import argparse
import shutil


def main(args):
	if os.path.exists(args.dir_path):
		shutil.rmtree(args.dir_path)
		#print("Remove the previous output")
		print("Start decoding the input article...")
	else:
		print("Start decoding the input article...")

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Remove the previous decode file.')
	parser.add_argument('--dir_path', required=True)
	args = parser.parse_args()
	main(args)


