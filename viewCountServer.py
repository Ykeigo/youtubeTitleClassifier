#!/usr/bin/env python
# coding: utf-8

import socket;
import select;
import threading

# torchtext
import torchtext
from torchtext import data
from torchtext import datasets

# pytorch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

import numpy as np

# その他もろもろ
from sklearn.model_selection import train_test_split

import MeCab
import gensim

import socket


# In[2]:

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 接続待ちするサーバのホスト名とポート番号を指定
host = "127.0.0.1";
#host =
port = 7001
argument = (host, port)
sock.bind(argument)
# 100 ユーザまで接続を許可
sock.listen(100)
clients = []

print("conenction created")


me = MeCab.Tagger ("-Owakati")
def mecab_tokenizer(text):
    return me.parse(text).split()


# In[3]:

print("loading Embeddings")
embedding = gensim.models.KeyedVectors.load_word2vec_format("model.vec")


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

BATCH_SIZE = 100 # バッチサイズ
EMBEDDING_DIM = 300 # 単語の埋め込み次元数
LSTM_DIM = 128 # LSTMの隠れ層の次元数
#VOCAB_SIZE = len(embedding.index2word) # 全単語数
TAG_SIZE = 2 # 今回はネガポジ判定を行うのでネットワークの最後のサイズは2
DA = 128 # AttentionをNeural Networkで計算する際の重み行列のサイズ
R = 1 # Attentionの数


# In[5]:


class BiLSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, lstm_dim):
        super(BiLSTMEncoder, self).__init__() #親の__init__で一回初期化する
        self.lstm_dim = lstm_dim
        
        # bidirectional=Trueでお手軽に双方向のLSTMにできる
        self.bilstm = nn.LSTM(embedding_dim, lstm_dim, batch_first=True, bidirectional=True)

    def forward(self, sentences):
        #print("forward")
        #分かち書きテキストが来るのでembeddingにする
        embeds = []
        for s in sentences:
            #print(s)
            vecs = []
            for w in s:
                if w in embedding:
                    vecs.append(embedding[w])
                else:
                    vecs.append(np.zeros(EMBEDDING_DIM))
            #vecs = torch.tensor(vecs)
            #print(vecs.shape)
            embeds.append(vecs)
            
        embeds = torch.tensor(embeds).float()
        #print("embeddings shape {}".format(embeds.shape))
        
        # 各隠れ層のベクトルがほしいので第１戻り値を受け取る
        out, _ = self.bilstm(embeds)

        # 前方向と後ろ方向の各隠れ層のベクトルを結合したままの状態で返す
        return out


# In[6]:


class SelfAttention(nn.Module):
  def __init__(self, lstm_dim, da, r):
    super(SelfAttention, self).__init__()
    self.lstm_dim = lstm_dim
    self.da = da
    self.r = r
    self.main = nn.Sequential(
        # Bidirectionalなので各隠れ層のベクトルの次元は２倍のサイズになってます。
        nn.Linear(lstm_dim * 2, da), 
        nn.Tanh(),
        nn.Linear(da, r)
    )
  def forward(self, out):
    return F.softmax(self.main(out), dim=1)


# In[7]:


class SelfAttentionClassifier(nn.Module):
  def __init__(self, lstm_dim, da, r, tagset_size):
    super(SelfAttentionClassifier, self).__init__()
    self.lstm_dim = lstm_dim
    self.r = r
    self.attn = SelfAttention(lstm_dim, da, r)
    self.main = nn.Linear(lstm_dim * 2 * r, tagset_size)

  def forward(self, out):
    attention_weight = self.attn(out)
    
    feats = torch.tensor([], device=device)
    for i in range(self.r):
        m1 = (out * attention_weight[:,:,i].unsqueeze(2)).sum(dim=1)
        feats = torch.cat([feats, m1], dim = 1)
    #m1 = (out * attention_weight[:,:,0].unsqueeze(2)).sum(dim=1)
    #m2 = (out * attention_weight[:,:,1].unsqueeze(2)).sum(dim=1)
    #m3 = (out * attention_weight[:,:,2].unsqueeze(2)).sum(dim=1)
    #feats = torch.cat([m1, m2, m3], dim=1)
    return F.log_softmax(self.main(feats)), attention_weight


# In[11]:


encoder = BiLSTMEncoder(EMBEDDING_DIM, LSTM_DIM).to(device)
encoder.load_state_dict(torch.load("encoder.trained"))
classifier = SelfAttentionClassifier(LSTM_DIM, DA, R, TAG_SIZE).to(device)
classifier.load_state_dict(torch.load("classifier.trained"))

print("all loaded")
# In[17]:

def calcViewCount(text):
    #text = input()
    text_tensor = np.array([mecab_tokenizer(text)])
    print(text_tensor)

    
    encoder_outputs = encoder(text_tensor)
    output, attn = classifier(encoder_outputs)
    
    pred = output.data.max(1, keepdim=True)[1]

    viewCount = -300 * np.power(float(output[0][0]),3)

    #id2ans = {1: 'positive', 0:'negative'}

    #print(output[0])
    #print("{:.02f}".format(np.power(np.e , float(output[0][0]))))
    #print("再生回数：{:.0f}回".format(-300 * np.power(float(output[0][0]),3) ))
    #print(pred)
    #print(id2ans[int(pred)])
    
    return viewCount


# 接続済みクライアントは読み込みおよび書き込みを繰り返す
def loop_handler(connection, address):
    while True:
        try:
            #クライアント側から受信する
            res = connection.recv(4096)
            print(res.decode("utf-8"))
            vc = calcViewCount(res.decode("utf-8"))
            connection.send(str(vc).encode("utf-8"))
            #break
        except Exception as e:
            print(e)
            break

print("connection waiting")
while True:
    try:
        # 接続要求を受信
        conn, addr = sock.accept()

    except KeyboardInterrupt:
        sock.close()
        exit()
        break
    # アドレス確認
    print("[アクセス元アドレス]=>{}".format(addr[0]))
    print("[アクセス元ポート]=>{}".format(addr[1]))
    print("\r\n");
    # 待受中にアクセスしてきたクライアントを追加
    clients.append((conn, addr))
    # スレッド作成
    thread = threading.Thread(target=loop_handler, args=(conn, addr), daemon=True)
    # スレッドスタート
    thread.start()
