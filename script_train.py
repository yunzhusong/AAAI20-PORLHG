export DATA=/data1/home2/Headline/Dataset/CNNDM/finished_files_cleaned_single_m2_i
export ROUGE=/home/yunzhu/pyrouge-master/tools/ROUGE-1.5.5
export METEOR=/home/yunzhu/Tool/meteor-1.5/meteor-1.5.jar

## #0. get a pop word index
#python make_pop.py
## #1. pretrained a word2vec word embedding
#python train_word2vec.py --path=/home/yunzhu/Headline/FASum/FASRL/word2vec

### 2. make the pseudo-labels
#python make_extraction_labels.py


### 3. train abstractor and extractir using ML objective
#python train_abstractor.py --path=/home/yunzhu/Headline/FASum/PORLHG_v3/save_trained_model/exp_0901/abstractor --w2v=/home/yunzhu/Headline/FASum/FASRL/word2vec/word2vec.128d.226k.bin

#python train_extractor_ml.py --path=/home/yunzhu/Headline/FASum/PORLHG_v3/save_trained_model/exp_0912/extractor --w2v=/home/yunzhu/Headline/FASum/FASRL/word2vec/word2vec.128d.226k.bin


### 4. train the full RL model
#python train_full_rl.py --path=/home/yunzhu/Headline/FASum/PORLHG_v3/save_trained_model/exp_0823/rl_3 --abs_dir=/home/yunzhu/Headline/FASum/PORLHG_v3/save_trained_model/exp_0823/abstractor --ext_dir=/home/yunzhu/Headline/FASum/PORLHG_v3/save_trained_model/exp_0822/extractor


### 5. decode summary from pretrained model
# Decode the abs result
#python decode_full_model.py --path=/home/yunzhu/Headline/FASum/PORLHG_v3/save_decode_result/exp_0823/rl_3/test --model_dir=/home/yunzhu/Headline/FASum/PORLHG_v3/save_trained_model/exp_0823/rl_3 --beam=5 --test
# Decode the ext result 
#python decode_full_model.py --path=/home/yunzhu/Headline/FASum/PORLHG_v3/save_decode_result/exp_0830/rl/test_ext --model_dir=/home/yunzhu/Headline/FASum/PORLHG_v3/save_trained_model/exp_0830/rl --beam=1 --test
#python decode_baselines.py --path=/home/yunzhu/Headline/FASum/PORLHG_v3/save_decode_result/exp_ext/extractor/test --ext_dir=/home/yunzhu/Headline/FASum/PORLHG_v3/save_trained_model/exp_ext/extractor --test

### 6. make the reference files for evaluation
#python make_eval_references.py
 
### 7. run evaluation
export DATA=/data1/home2/Headline/Dataset/CNNDM/finished_files_cleaned_single_m2_i

python eval_full_model.py --rouge --decode_dir=/home/yunzhu/Headline/FASum/PORLHG_v3/save_decode_result/exp_0830/rl/test

python eval_full_model.py --meteor --decode_dir=/home/yunzhu/Headline/FASum/PORLHG_v3/save_decode_result/exp_0823/rl_3/test


#python eval_baselines.py --rouge --decode_dir=/home/yunzhu/Headline/FASum/PORLHG_v3/save_decode_result/exp_0822/extractor/test --n_ext=1
#python eval_baselines.py --meteor --decode_dir=/home/yunzhu/Headline/FASum/PORLHG_v3/save_decode_result/exp_0822/extractor/test --n_ext=1

#python demo_test.py --result_path=/home/yunzhu/Headline/FASum/FASRL/save_decode_result/exp_0224/test/output --num=2
