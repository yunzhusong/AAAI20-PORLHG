from function import precaculate_topic_dis, combine_topic_dis_to_json, combine_topic_index_to_json
import os


split ="train"
corpus_dir = "/data1/home2/Headline/XSum-master/XSum-Dataset/cnndm-256-iter1000"

dir_topic_sents = os.path.join(corpus_dir, "sents-{}-topics".format(split))
dir_topic_head  = os.path.join(corpus_dir, "headline-{}-topics".format(split))
dir_refer_index = "/data1/home2/Headline/Anal_Topic/refer_index_method3_{}.txt".format(split) # val
dir_out = "/data1/home2/Headline/Dataset/topic_dis/method6"

file_out = os.path.join(dir_out, split)
print("Precaculating the topic multiplication...")
#precaculate_topic_dis(dir_topic_sents, dir_topic_head, dir_refer_index, file_out, split)

# 
json_path = "/data1/home2/Headline/Dataset/CNNDM/finished_files_cleaned_single"
topic_path = dir_out
json_out_path = "/data1/home2/Headline/Dataset/CNNDM/finished_files_cleaned_single_m6_i"


json_dir = os.path.join(json_path, split)
topic_dir = os.path.join(topic_path, split)
json_out_dir = os.path.join(json_out_path, split)

#combine_topic_dis_to_json(json_dir, topic_dir, json_out_dir, split)

#

print("Combine the precaculating result to json file...")
combine_topic_index_to_json(json_dir, topic_dir, json_out_dir, dir_refer_index, dir_topic_head, split)
