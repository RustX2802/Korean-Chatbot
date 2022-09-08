import math
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchtext
import torch.optim as optim

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

class PositionalEncoder(nn.Module):
    '''입력된 단어의 위치를 원백터정보에 더한다'''
    def __init__(self, position, d_model):
        super().__init__()

        self.d_model = d_model  # 단어 백터의 원래 차원 수

        # 입력 문장에서의 임베딩 벡터의 위치（pos）임베딩 벡터 내의 차원의 인덱스（i）
        pe = torch.zeros(position, d_model)

        # 학습시에는 GPU 사용
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)

        for pos in range(position):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos /
                                          (10000 ** ((2 * i)/d_model)))

        # pe의 선두에 미니배치 차원을 추가한다
        self.pe = pe.unsqueeze(0)

        self.pe.requires_grad = False

    def forward(self, x):
        # 입력x와 Positonal Encoding을 더한다
        ret = math.sqrt(self.d_model)*x + self.pe[:, :x.size(1)]
        return ret

    def scaled_dot_product_attention(query, key, value, mask):
        matmul_qk = torch.matmul(query, torch.transpose(key,2,3))

        depth = key.shape[-1]
        logits = matmul_qk / math.sqrt(depth)

        if mask is not None:
            logits += (mask * -1e9)

        attention_weights = F.softmax(logits, dim=-1)

        output = torch.matmul(attention_weights, value)

        return output, attention_weights

class MultiheadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % self.num_heads == 0
        # d_model을 num_heads로 나눈 값.
        self.depth = int(d_model/self.num_heads)

        # WQ, WK, WV에 해당하는 밀집층 정의
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        # WO에 해당하는 밀집층 정의
        self.out = nn.Linear(d_model, d_model)


    # num_heads 개수만큼 q, k, v를 split하는 함수
    def split_heads(self, inputs, batch_size):
      inputs = torch.reshape(
          inputs, (batch_size, -1, self.num_heads, self.depth))
      return torch.transpose(inputs, 1,2)

    def forward(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = query.shape[0]
        # 1. WQ, WK, WV에 해당하는 밀집층 지나기
        # q : (batch_size, query의 문장 길이, d_model)
        # k : (batch_size, key의 문장 길이, d_model)
        # v : (batch_size, value의 문장 길이, d_model)
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)


        # 2. 헤드 나누기
        # q : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        # k : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
        # v : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)


        # 3. 스케일드 닷 프로덕트 어텐션. 앞서 구현한 함수 사용.
        # (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        # (batch_size, query의 문장 길이, num_heads, d_model/num_heads)
        scaled_attention = torch.transpose(scaled_attention, 1,2)

        # 4. 헤드 연결(concatenate)하기
        # (batch_size, query의 문장 길이, d_model)
        concat_attention = torch.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # 5. WO에 해당하는 밀집층 지나기
        # (batch_size, query의 문장 길이, d_model)
        outputs = self.out(concat_attention)
        return outputs

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, attention):
        outputs = self.linear_1(attention)
        outputs = F.relu(outputs)
        outputs = self.linear_2(outputs)
        return outputs

class EncoderBlock(nn.Module):
  def __init__(self, d_ff, d_model, num_heads, dropout):
    super(EncoderBlock, self).__init__()
    
    self.attn = MultiheadAttention(d_model, num_heads)
    self.dropout_1 = nn.Dropout(dropout)
    self.norm_1 = nn.LayerNorm(d_model)
    self.ff = FeedForward(d_model, d_ff)
    self.dropout_2 = nn.Dropout(dropout)
    self.norm_2 = nn.LayerNorm(d_model)

  def forward(self, inputs, padding_mask):
    attention = self.attn({'query': inputs, 'key': inputs, 'value': inputs, 'mask': padding_mask})
    attention = self.dropout_1(attention)
    attention = self.norm_1(inputs + attention)
    outputs = self.ff(attention)
    outputs = self.dropout_2(outputs)
    outputs = self.norm_2(attention + outputs)

    return outputs

class Encoder(nn.Module):
  def __init__(self,text_embedding_vectors, vocab_size, num_layers, d_ff, d_model, num_heads, dropout):
    super(Encoder, self).__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.num_layers = num_layers
    self.embb = nn.Embedding(text_embedding_vectors, d_model)
    self.dropout_1 = nn.Dropout(dropout)
    self.PE = PositionalEncoder(vocab_size, d_model)
    self.encoder_block = EncoderBlock(d_ff, d_model, num_heads, dropout)
  def forward(self, x, padding_mask):
    emb = self.embb(x)
    emb *= math.sqrt(self.d_model)
    emb = self.PE(emb)
    output = self.dropout_1(emb)

    for i in range(self.num_layers):
      output = self.encoder_block(output, padding_mask)

    return output

class DecoderBlock(nn.Module):
  def __init__(self, d_ff, d_model, num_heads, dropout):
    super(DecoderBlock, self).__init__()
    
    self.attn = MultiheadAttention(d_model, num_heads)
    self.attn_2 = MultiheadAttention(d_model, num_heads)
    self.dropout_1 = nn.Dropout(dropout)
    self.norm_1 = nn.LayerNorm(d_model)
    self.ff = FeedForward(d_model, d_ff)
    self.dropout_2 = nn.Dropout(dropout)
    self.dropout_3 = nn.Dropout(dropout)
    self.norm_2 = nn.LayerNorm(d_model)
    self.norm_3 = nn.LayerNorm(d_model)

  def forward(self, inputs, enc_outputs, padding_mask, look_ahead_mask):
    attention1 = self.attn({'query': inputs, 'key': inputs, 'value': inputs, 'mask': look_ahead_mask})
    attention1 = self.norm_1(inputs + attention1)
    attention2 = self.attn_2({'query': attention1, 'key': enc_outputs, 'value': enc_outputs, 'mask': padding_mask})
    attention2 = self.dropout_1(attention2)
    attention2 = self.norm_2(attention1 + attention2)

    outputs = self.ff(attention2)
    outputs = self.dropout_3(outputs)
    outputs = self.norm_3(attention2 + outputs)

    return outputs

class Decoder(nn.Module):
  def __init__(self,text_embedding_vectors,  vocab_size, num_layers, d_ff, d_model, num_heads, dropout):
    super(Decoder, self).__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.num_layers = num_layers
    self.embb = nn.Embedding(text_embedding_vectors, d_model)
    self.dropout_1 = nn.Dropout(dropout)
    self.PE = PositionalEncoder(vocab_size, d_model)
    self.decoder_block = DecoderBlock(d_ff, d_model, num_heads, dropout)
  def forward(self, enc_output, dec_input, padding_mask, look_ahead_mask):
    emb = self.embb(dec_input)
    emb *= math.sqrt(self.d_model)
    emb = self.PE(emb)
    output = self.dropout_1(emb)
    for i in range(self.num_layers):
      output = self.decoder_block(output, enc_output, padding_mask, look_ahead_mask)

    return output

class transformer(nn.Module):
    def __init__(self, text_embedding_vectors, vocab_size, num_layers, d_ff, d_model, num_heads, dropout):
        self.vocab_size = vocab_size
        super(transformer, self).__init__()
        self.enc_outputs = Encoder(text_embedding_vectors, vocab_size, num_layers, d_ff, d_model, num_heads, dropout)
        self.dec_outputs = Decoder(text_embedding_vectors, vocab_size, num_layers, d_ff, d_model, num_heads, dropout)
        self.output = nn.Linear(d_model, text_embedding_vectors)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, dec_input):
        enc_input = input
        dec_input = dec_input
        enc_padding_mask = create_padding_mask(enc_input)
        dec_padding_mask = create_padding_mask(enc_input)
        look_ahead_mask = create_look_ahead_mask(dec_input)
    
        enc_output = self.enc_outputs(enc_input, enc_padding_mask)
        dec_output = self.dec_outputs(enc_output, dec_input, dec_padding_mask, look_ahead_mask)
        output = self.output(dec_output)
        return output

import pandas as pd
import re
import urllib.request
import time

#urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData%20.csv", filename="ChatBotData.csv")
import os
file_list = []
for dirname, _, filenames in os.walk('ChatbotData.csv'):
    for filename in filenames:
        file_list.append(os.path.join(dirname, filename))
    train_data = pd.read_csv(file_list[0])
    train_data.head()

from torchtext import data, datasets
import os