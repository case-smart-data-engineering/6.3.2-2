'''
    LSTM-based Deep Learning Models for Non-factoid Answer Selection
    Ming Tan, Cicero dos Santos, Bing Xiang, Bowen Zhou, ICLR 2016
    https://arxiv.org/abs/1511.04108
'''
import os
import random
import argparse
from tqdm import tqdm
import numpy as np
import torch
from gensim.models.keyedvectors import KeyedVectors
from utils import load_data, load_data2, load_vocabulary, Config, load_embd_weights,load_valid_data
from utils import make_vector
from models import QA_LSTM

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--start_epoch', type=int, default=0, help='resume epoch count, default=0')
parser.add_argument('--n_epochs', type=int, default=4, help='input batch size')
parser.add_argument('--embd_size', type=int, default=300, help='word embedding size')
parser.add_argument('--hidden_size', type=int, default=141, help='hidden size of one-directional LSTM')
parser.add_argument('--max_sent_len', type=int, default=200, help='max sentence length')
parser.add_argument('--margin', type=float, default=0.2, help='margin for loss function')
parser.add_argument('--use_pickle', type=int, default=0, help='load dataset from pickles')
parser.add_argument('--test', type=int, default=0, help='1 for test, or for training')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--resume', default='./checkpoints/Epoch-3.model', type=str, metavar='PATH', help='path saved params')    #model_best.tar
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

PAD = '<PAD>'
id_to_word, label_to_ans, label_to_ans_text = load_vocabulary('./datas/V2/vocabulary', './datas/V2/InsuranceQA.label2answer.token.encoded')
w2i = {w: i for i, w in enumerate(id_to_word.values(), 1)}
w2i[PAD] = 0
vocab_size = len(w2i)
print('vocab_size:', vocab_size)

# path = './datas/V2/InsuranceQA.question.anslabel.token.500.pool.solr.test.encoded'

train_data = load_data('./datas/V2/InsuranceQA.question.anslabel.token.500.pool.solr.train.encoded', id_to_word, label_to_ans_text)
test_data = load_data2('./datas/V2/InsuranceQA.question.anslabel.token.500.pool.solr.test.encoded', id_to_word, label_to_ans_text)
print('n_train:', len(train_data))
print('n_test:', len(test_data))

args.vocab_size = vocab_size
args.pre_embd   = None

print('loading a word2vec binary...')
model_path = './GoogleNews-vectors-negative300.bin'
word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)
print('loaded!')
pre_embd = load_embd_weights(word2vec, vocab_size, args.embd_size, w2i)
# save_pickle(pre_embd, 'pre_embd.pickle')
args.pre_embd = pre_embd

# 保存模型
def save_checkpoint(state, filename):
    print('save model!', filename)
    torch.save(state, filename)

# 计算损失函数
def loss_fn(pos_sim, neg_sim):
    loss = args.margin - pos_sim + neg_sim
    if loss.data[0] < 0:
        loss.data[0] = 0
    return loss

# 训练模型
def train(model, data, test_data, optimizer, n_epochs=4, batch_size=256):
    '''
    :param model: LSTM
    :param data: 训练数据，也就是train_data
    :param test_data: 测试数据，也就是test_data
    :param optimizer: 随机梯度下降，使用AMD
    :param n_epochs:
    :param batch_size:
    :return:
    '''
    for epoch in range(n_epochs):
        model.train()
        print('epoch', epoch)
        random.shuffle(data) # TODO use idxies
        losses = []
        for i, d in enumerate(tqdm(data)):
            q, pos, negs = d[0], d[1], d[2]
            vec_q = make_vector([q], w2i, len(q))
            vec_pos = make_vector([pos], w2i, len(pos))
            pos_sim = model(vec_q, vec_pos)

            for _ in range(50):
                neg = random.choice(negs)
                vec_neg = make_vector([neg], w2i, len(neg))
                neg_sim = model(vec_q, vec_neg)
                loss = loss_fn(pos_sim, neg_sim)
                if loss.data[0] != 0:
                    losses.append(loss)
                    break

            if len(losses) == batch_size or i == len(data) - 1:
                loss = torch.mean(torch.stack(losses, 0).squeeze(), 0)
                # print(loss.data[0])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses = []

        filename = '{}/Epoch-{}.model'.format('./checkpoints', epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, filename=filename)
        test(model, test_data)

# 测试模型
def test(model, data):
    acc, total = 0, 0
    for d in data:
        q = d[0]
        qq = q
        print('question:', ' '.join(q))    # 输出问题Q
        labels = d[1]     # 答案标签
        cands = d[2]     # 答案候选池

        # preprare answer labels
        label_indices = [cands.index(l) for l in labels if l in cands]
        print("answerlabel：",label_indices)

        # build data
        q = make_vector([q], w2i, len(q))
        cands = [label_to_ans_text[c] for c in cands] # id to text
        # print("init：",cands)
        max_cand_len = min(args.max_sent_len, max([len(c) for c in cands]))
        cands = make_vector(cands, w2i, max_cand_len)

        # predict
        scores = [model(q, c.unsqueeze(0)).data[0] for c in cands]  # 计算问题和答案的相似度
        pred_idx = np.argmax(scores)
        if pred_idx in label_indices:
            print('correct',pred_idx,label_indices)
            answer = label_to_ans_text[str(pred_idx+1)]
            Q = " ".join(qq)
            answer = " ".join(answer)
            print("answer:",answer)
            # 将测试数据中正确的写入文件中
            with open('correct_qa.txt',"a+",encoding='utf-8') as wf:
                wf.write(str(Q) + '\n')
                wf.write(str(answer) + '\n\n')
            acc += 1
        else:
            print('wrong')
        total += 1
    print('Test Acc:', 100*acc/total, '%')

def predict():

    # 测试数据
    test_data = load_data2('./datas/V2/InsuranceQA.question.anslabel.token.500.pool.solr.test.encoded', id_to_word,
                           label_to_ans_text)

    # valid_data = load_valid_data('./datas/V2/InsuranceQA.question.anslabel.token.500.pool.solr.test.encoded', id_to_word,
    #                        label_to_ans_text)
    model = QA_LSTM(args)
    # 加载训练好的模型
    checkpoint = torch.load(args.resume)
    # 获取训练好的模型参数
    model.load_state_dict(checkpoint['state_dict'])
    args.start_epoch = checkpoint['epoch']
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()
    test(model,test_data)
    print('测试完毕！')


if __name__ == '__main__':
    model = QA_LSTM(args)
    # 如果GPU可用，则使用GPu来进行模型的训练
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    # args.resume 加载最优模型
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])  # TODO ?
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # 开始训练模型
    # train(model, train_data, test_data, optimizer)

    # 测试模型
    # test(model, test_data)

    # 预测
    predict()


