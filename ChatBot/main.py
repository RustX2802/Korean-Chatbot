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