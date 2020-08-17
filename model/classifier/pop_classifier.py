import pdb
import os
import time
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, f1_score
import load_data
import torch
import torch.nn.functional as F
from torchtext import data
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import numpy as np
import dill
from models.LSTM import LSTMClassifier

do_train = True

corpusdir = "/data1/home2/Headline/Dataset/dataset_cls/2_class_m24s10"
#save_path = "/home/yunzhu/Headline/FASum/PORLHG_v3/model/classifier/trained_model/lstm_all_4.pt"
save_path = "/home/yunzhu/Headline/FASum/PORLHG_v3/model/classifier/trained_model/try.pt"
#save_path = "trained_model/lstm_c35s32_without0_songlin.pt"
#save_path = "/home/yunzhu/Headline/FASum/FASRL/model/classifier/trained_model/lstm_without0.pt" # c36s29
#save_path = "/home/yunzhu/Headline/FASum/FASRL/model/classifier/trained_model/lstm_c35s32_without0.pt" # c35s32

#TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()
#loss_fn = F.cross_entropy

learning_rate = 2e-5
batch_size = 32
output_size = 1
hidden_size = 256
embedding_length = 300
conv_hidden = 300

def write_file(path_in, path_out):
    num = len(os.listdir(path_in))
    #num = 10577
    with open(path_out, 'w') as f:
        print("Create the inference data for classifier at: {}".format(path_out))
        f.write("headline\tcomment\tshare\n")
    f_tsv = open(path_out, 'a')
    for i in range(num):
        with open(os.path.join(path_in, str(i)+'.dec')) as f:
            headline = f.read().replace("\n", " .")
            if headline != None:
                line = headline + "\t0\t0\n"
            else:
                line = " \t0\t0\n"
        f_tsv.write(line)



def read_data(path, index, src):
    with open(os.path.join(path, str(index)+src)) as f:
        return f.read()

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch, loss_fn, optim):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    #optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.headline[0]
        target = batch.comment
        #target = batch.share
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not 32):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        #loss = loss_fn(prediction, target)
        loss = loss_fn(prediction.reshape(-1), target.float())
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        #if steps % 100 == 0:
            #print('Epoch: {}, Idx: {}, Training Loss: {}, Training Accuracy: {}'.format(epoch+1, idx+1, loss.item(), acc.item()))
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter, loss_fn):
    total_epoch_loss = 0
    total_epoch_acc = 0
    gt = []
    pred = []
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.headline[0]
            if (text.size()[0] is not 32):
                continue
            target = batch.comment
            #target = batch.share
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                model.cuda()
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction.reshape(-1), target.float())
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            gt += list(torch.max(prediction, 1)[1].view(target.size()).data.cpu())
            pred += list(target.data.cpu())
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()
        total_epoch_uar = recall_score(gt, pred, average='macro')
        print(confusion_matrix(gt, pred))
    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter), total_epoch_uar

def test_model(model, val_iter):
    total_epoch_acc = 0
    model.eval()
    model.cuda()
    prediction = torch.Tensor(np.zeros((len(val_iter),2)))
    with torch.no_grad():
        #for idx, batch in enumerate(val_iter):
        for idx, batch in enumerate(val_iter):
            text = batch.headline[0]
            if (text.size()[0] is not 32):
                continue
            target = batch.comment
            #target = batch.share
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
    return prediction
    """
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100*num_corrects/len(batch)
            total_epoch_acc += acc.item()
    return  total_epoch_acc/len(val_iter)
    """

	

def do_inference(sentences, TEXT, vocab_size, word_embeddings):
    ## Load mode for inference
    batch_size = len(sentences)
    model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings, conv_hidden, 0.0)
    model.cuda()
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict)
    model.eval()

    data_field = [('headline', TEXT)]
    ## prepare data
    score = None
    examples = []
    for text in sentences:
        examples.append(data.Example.fromlist([text], data_field))
    infer_data = data.Dataset(examples, data_field, filter_pred=None)
    infer_iter = data.Iterator(dataset=infer_data, batch_size=batch_size, train=False, sort=False, device=0)
    for idx, batch in enumerate(infer_iter):
        text = batch.headline[0]
        #if (text.size()[0] is not 32):
        #    continue
        prediction = model(text)
    score = torch.max(prediction, 1)[1].float().mean().item()
    return score

def classifier():
    #################################################################################
    # Write the output data into the infer data for the classifier
    """
    #path = "/home/yunzhu/Headline/FASum/FASRL/save_decode_result/BM25/test"
    #path = "/home/yunzhu/Headline/FASum/FASRL/save_decode_result/PREFIX/test"
    #path = "/home/yunzhu/Headline/FASum/FASRL/save_decode_result/random/test"
    #path = "/home/yunzhu/Headline/FASum/FASRL/save_decode_result/seq2seq/withatt/test"
    #path = "/data1/home2/Headline/PointerSumm/log/decode_model_95000_1555784722/test"
    #path = "/home/yunzhu/Headline/FASum/FASRL/save_decode_result/exp_0223/test"
    #path = "/home/yunzhu/Headline/FASum/PORLHG_v3/save_decode_result/exp_0907/extractor/test"
    #path = "/data1/home2/Headline/Dataset/CNNDM/finished_files_cleaned_single_m2/refs/test"
    #path = "/home/yunzhu/Headline/FASum/PORLHG_v3/save_decode_result/exp_0912/rl/test"
    #path = "/home/yunzhu/Headline/FASum/PORLHG_v3/save_decode_result/exp_0823_v4/test"
    path = "/home/yunzhu/Headline/FASum/PORLHG_v3/save_decode_result/exp_0823/rl_3/test"

    #path_in = path
    path_in = os.path.join(path, "output")
    print('We are testing:{}'.format(path))
    filename = "temp.tsv"
    path_out= "/home/yunzhu/Headline/FASum/FASRL/model/classifier/cls_data/{}".format(filename)

    write_file(path_in, path_out)

    TEXT, vocab_size, word_embeddings, _, _, test_iter = load_data.load_dataset(corpusdir, batch_size, filename=path_out)
    model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings, conv_hidden, 0.0)
    print('Loading the pretrained model: {}'.format(save_path.split('/')[-1]))
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict)

    loss_fn = F.cross_entropy

    test_loss, test_acc, test_uar = eval_model(model, test_iter, loss_fn)

    print('Inference popularity predictor for: {}'.format(path_in))
    print('Test Loss: {:.2f}, Test Acc: {:.2f}%, Test Uar: {:.2f}'.format(test_loss, test_acc, test_uar))
    print('There are {:.2f}% are classified as positive'.format(100-test_acc))
    with open(os.path.join(path, "popularity.txt"), 'w') as f:
        f.write("Inference by: {}".format(save_path))
        f.write("model: {}".format(path))
        f.write("score: {}".format(100-test_acc))

    """
    #########################################################
    """
    TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()
    loss_fn = F.cross_entropy
    
    with open('TEXT.Field', 'rb') as f:
        TEXT = dill.load(f)

    #path = "/home/yunzhu/Headline/FASum/FASRL/save_decode_result/exp_0224/test/output"
    path = "/home/yunzhu/Headline/Datasets/CNNDM/finished_files_cleaned/refs/test"
    num = len(os.listdir(path))

    total_score = 0
    for i in range(num):
        sentence = read_data(path, i, '.ref')
        
        score = do_inference(sentence, TEXT, vocab_size, word_embeddings)
        total_score += score
        print("{}/{} finished, score:{}".format(i, num, score))
    print("total_score: {}".format(total_score))
    print("avg score: {}".format(total_score/num))
     """

    #####################################################
    
    TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset(corpusdir, batch_size)
    loss_fn = F.binary_cross_entropy_with_logits

    model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings, conv_hidden, 0.1)

    val_acc_best=0.
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))  ## move the optim from train() to here
    scheduler = ReduceLROnPlateau(optim, 'min', verbose=True, patience=2)

    for epoch in range(10):
        train_loss, train_acc = train_model(model, train_iter, epoch, loss_fn, optim)
        val_loss, val_acc, _ = eval_model(model, valid_iter, loss_fn)
        scheduler.step(val_loss)
        if val_acc_best < val_acc:
            torch.save(model.state_dict(), save_path)
            test_loss, test_acc, _ = eval_model(model, test_iter, loss_fn)
            print('[info] Epoch{} Test Loss: {:.2f}, Test Acc: {:.2f}%'.format(epoch, test_loss, test_acc))
            val_acc_best = val_acc
        print('Epoch: {}, Train Loss: {:.2f}, Train Acc: {:.2f}%, Val Loss: {:.2f}, Val Acc: {:.2f}%'.format(epoch+1,  train_loss, train_acc, val_loss, val_acc)) 

    #test_loss, test_acc = eval_model(model, test_iter, loss_fn)

    #print('Test Loss: {}, Test Acc: {}'.format(test_loss, test_acc))
    
    ##################################################################    
    return TEXT, vocab_size, word_embeddings

if __name__=="__main__":
    classifier()
    



