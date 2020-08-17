import os
import pdb
import json
import torch
import numpy as np

def generate_clsData_from_dataset(path_in, path_out):

	with open(path_out, 'w') as f_out:
		print('Create a file at: {}'.format(path_out))

	f_out = open(path_out, 'a')

	files = os.listdir(path_in)

	for i in range(len(files)):
		print('{}/{} ({:.2f}%) processed\r'.format(i, len(files), i*100/len(files)), end='')
		with open(os.path.join(path_in, files[i])) as f:
			data = json.load(f)
			index = data['extracted_headline'][0]
			headline = data['article'][index]
		
		line = headline+"\t0\t0\n"
		f_out.write(line)

	f_out.close()

#path_in = "/home/yunzhu/Headline/Datasets/CNNDM/finished_files_cleaned_single/test"
#path_out = "/home/yunzhu/Headline/FASum/FASRL/model/classifier/cls_data/infer_ext_test.tsv"
#generate_clsData_from_dataset(path_in, path_out)


def precaculate_topic_dis(dir_topic_sents, dir_topic_head, dir_refer_index, path_out, split):
	if not os.path.exists(path_out):
		os.makedirs(path_out)

	num_document = {'train':281208, 'val':12727, 'test':10577}
	num = num_document[split]

	with open(dir_refer_index) as f:
		refer_index = torch.tensor(np.loadtxt(f), dtype=torch.uint8).cuda()

	for i in range(num):
		print("{}/{} ({:.2f}%)\r".format(i, num, i*100/num), end='')
		# Find the topic distribution according to index
		index = refer_index[i]

		filename_head = os.path.join(dir_topic_head, "{}.doc-topics".format(index))
		T_h = torch.tensor(np.loadtxt(filename_head)).cuda()[:,1]

		# Read the topic distribution of sentences
		filename_sents = os.path.join(dir_topic_sents, "{}.sents-topics".format(i))
		T_si = torch.tensor(np.loadtxt(filename_sents)).cuda()

		if len(T_si.shape) == 1:
			T_si = T_si.unsqueeze(0)

		data = T_h * T_si # element wise multiply, method2
		#data = T_h + T_si 
		# Save the multiplication result back
		np.savetxt(os.path.join(path_out, '{}.txt'.format(i)), data)


def check_topic_sent_num():
	split = "train"
	json_path = "/home/yunzhu/Headline/Datasets/CNNDM/finished_files_cleaned_single"
	topic_path = "/data1/home2/Headline/Dataset/topic_dis/method2"

	json_dir = os.path.join(json_path, split)
	topic_dir = os.path.join(topic_path, split)

	num_document = {'train':281208, 'val':12727, 'test':10577}
	num = num_document[split]


	wrong_data = []
	#wrong_data = np.loadtxt('index_wrong_data.npy')
	#num = len(wrong_data)
	#ckpt_list = [59998, 62715]
	for i in range(num):
	#for i in ckpt_list:
		with open(os.path.join(json_dir, "{}.json".format(i))) as f:
			article = json.load(f)['article']
		sent_num_art = len(article)
		
		topic = np.loadtxt(os.path.join(topic_dir, "{}.txt".format(i)))
		sent_num_topic = len(topic)

		print(sent_num_art)
		print(sent_num_topic)
		if sent_num_art != sent_num_topic:
			wrong_data.append(i)
			print("Wrong: {}".format(i))

		if i % 1000 == 0:
			print("{}/{}".format(i, num))

	#np.savetxt('index_wrong_data.npy',wrong_data)

	print("Total data: {}".format(len(wrong_data)))
#check_topic_sent_num()

def combine_topic_dis_to_json(json_dir, topic_dir, json_out_dir, split):

	if not os.path.exists(json_out_dir):
		os.makedirs(json_out_dir)

	num_document = {'train':281208, 'val':12727, 'test':10577}
	num = num_document[split]

	wrong_data = []
	for i in range(num):
		with open(os.path.join(json_dir, "{}.json".format(i))) as f:
			data = json.load(f)
		topic = np.loadtxt(os.path.join(topic_dir, "{}.txt".format(i)))

		# Check sent number equal
		if len(data['article']) != len(topic):
			wrong_data.append(i)
			print('------{}'.format(i))

		data['topic_method'] = topic.tolist()

		with open(os.path.join(json_out_dir, "{}.json".format(i)), 'w') as f:
			json.dump(data, f, indent=4)
		if i%1000 == 0:
			print("{}/{}".format(i,num))

#combine_topic_dis_to_json()

def combine_topic_index_to_json(json_dir, topic_dir, json_out_dir, dir_refer_index=None, dir_topic_head=None,split="test"):
	print("Combining topic value into CNN/DM json dataset")

	if not os.path.exists(json_out_dir):
		os.makedirs(json_out_dir)

	num_document = {'train':281208, 'val':12727, 'test':10577}
	num = num_document[split]

	wrong_data = []


	add_topic_label = False
	if add_topic_label == True:
		with open(dir_refer_index) as f:
			refer_index = torch.tensor(np.loadtxt(f), dtype=torch.uint8).cuda()

	for i in range(num):
		with open(os.path.join(json_dir, "{}.json".format(i))) as f:
			data = json.load(f)
		topic = np.loadtxt(os.path.join(topic_dir, "{}.txt".format(i)))


		if len(topic.shape) == 1: # Some data only contain single sentence article
			topic = topic.reshape(1,-1)

		if len(data['article']) != len(topic):
			print('--------{}'.format(i))

		num_sent = len(topic)
		output = [None]*num_sent

		for j in range(num_sent):
			check_value = 0
			threshold = 1e-5
			value = 0.98*sum(topic[j])
			while check_value < value:
				sent_index = np.where(topic[j]>threshold)[0].tolist()
				sent_value = topic[j][sent_index].tolist()
				check_value = sum(sent_value)
				threshold /= 2
				#print('value: {}, scale down for {}'.format(check_value, i))
		
			output[j] = (sent_index, sent_value)	

		data['topic_method'] = output

		if add_topic_label == True:
			index = refer_index[i]
			filename_head = os.path.join(dir_topic_head, "{}.doc-topics".format(index))
			T_h = torch.FloatTensor(np.loadtxt(filename_head)).cuda()[:,1]
			
			check_value, threshold = 0, 1e-2
			value = 0.98*sum(T_h)
			while check_value < value:
				head_index = np.where(T_h>threshold)[0].tolist()
				head_value = T_h[head_index].tolist()
				check_value = sum(head_value)
				threshold /= 2
		
			data['topic_label'] = [head_index, head_value]

		with open(os.path.join(json_out_dir, "{}.json".format(i)), 'w') as f:
			json.dump(data, f, indent=4)

		print("{}/{}\r".format(i,num), end='')

#combine_topic_index_to_json()	

def reconstruct_topic_dis(data_topic, batch_size=None):
	"""
	json_out_path = "/data1/home2/Headline/Dataset/CNNDM/finished_files_cleaned_single_m2_i/train"

	with open(os.path.join(json_out_path, '0.json')) as f:
		data = json.load(f)
	topics = data['topic_method2']"""
	if batch_size == None:
		num_sent = len(data_topic)
		topic_dis = torch.zeros((num_sent, 512), dtype=torch.float).cuda()
		for i in range(num_sent):
			index = (torch.tensor(data_topic[i][0]),)
			value = torch.FloatTensor(data_topic[i][1]).cuda()

			topic_dis[i].index_put_(index, value)

		return topic_dis
	else:
		# Deal with batch data
		batch_topic = data_topic
		batch_topic_dis = []
		for b in range(batch_size):
			data_topic = batch_topic[b]
			topic_dis = reconstruct_topic_dis(data_topic)
			batch_topic_dis.append(topic_dis)

		return batch_topic_dis

def ckpt_data():
	split = "test"
	data_path = "/data1/home2/Headline/Dataset/CNNDM/finished_files_cleaned_single_m5_i"
	data_dir = os.path.join(data_path, split)
	
	num_document = {'train':281208, 'val':12727, 'test':10577}
	num = num_document[split]

	for i in range(num):
		with open(os.path.join(data_dir, '{}.json'.format(i))) as f:
			try:
				data = json.loads(f.read())
				if len(data['article']) < 2:
					print('-------------{}'.format(i))
				if len(data['topic_method'][0][0]) == 0:
					print('-------------{}'.format(i))	
				if i%1000 == 0:
					print('{}/{}'.format(i, num))
			except:
				print('-----------{}'.format(i))


#ckpt_data()
def tackle_huge_data():
	train_data = [100790]

	split = "train"
	json_path = "/data1/home2/Headline/Dataset/CNNDM/finished_files_cleaned_single"
	topic_path = "/data1/home2/Headline/Dataset/topic_dis/method2"
	json_out_path = "/data1/home2/Headline/Dataset/CNNDM/finished_files_cleaned_single_m2"

	json_dir = os.path.join(json_path, split)
	topic_dir = os.path.join(topic_path, split)
	json_out_dir = os.path.join(json_out_path, split)

	for i in train_data:
		with open(os.path.join(json_dir, '{}.json'.format(i))) as f:
			data = json.load(f)

		topic = np.loadtxt(os.path.join(topic_dir, '{}.txt'.format(i))).tolist()
		data['topic_method2'] = topic
		
		with open(os.path.join(json_out_path, '{}.json'.format(i)), 'w') as f:
			json.dump(data, f, indent=4)
		
#tackle_huge_data()
