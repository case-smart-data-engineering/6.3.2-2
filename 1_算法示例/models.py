import torch
import torch.nn as nn

# 定义词嵌入
class WordEmbedding(nn.Module):
    def __init__(self, args, is_train_embd=True): # In QA-LSTM model, embedding weights is fine-tuned
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embd_size)
        if args.pre_embd is not None:
            print('pre embedding weight is set')
            self.embedding.weight = nn.Parameter(args.pre_embd, requires_grad=is_train_embd)

    def forward(self, x):
        return self.embedding(x)

# 定义LSTM模型
class QA_LSTM(nn.Module):
    def __init__(self, args):
        super(QA_LSTM, self).__init__()
        self.word_embd = WordEmbedding(args)   # 词嵌入
        # LSTM模型
        self.shared_lstm = nn.LSTM(args.embd_size, args.hidden_size, batch_first=True, bidirectional=True)
        # 使用cos计算问题和答案的相似度
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, q, a):
        # embedding
        q = self.word_embd(q) # (bs, L, E) 对问题进行嵌入
        a = self.word_embd(a) # (bs, L, E) 对答案进行嵌入

        # LSTM
        q, _h = self.shared_lstm(q) # (bs, L, 2H) 问题经过LSTM后提取特征
        a, _h = self.shared_lstm(a) # (bs, L, 2H) 答案经过LSTM后提取特征

        # 平均池化
        # q = torch.mean(q, 1) # (bs, 2H)
        # a = torch.mean(a, 1) # (bs, 2H)

        # maxpooling  最大池化
        q = torch.max(q, 1)[0] # (bs, 2H)
        a = torch.max(a, 1)[0] # (bs, 2H)
        # 返回问题和答案的相似度
        return self.cos(q, a) # (bs,)
