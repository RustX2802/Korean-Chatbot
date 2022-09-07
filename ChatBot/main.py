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