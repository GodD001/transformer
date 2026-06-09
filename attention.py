import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)   
        scores = torch.matmul(Q, K.transpose(-2, -1))/ math.sqrt(d_k)
        if mask is not None:
            scores=scores.masked_fill(mask==0, float('-inf'))
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        return out, attn
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout =0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = SelfAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        batch_size =q.size(0)
        Q =self.W_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        K =self.W_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        V =self.W_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        out, attn = self.attention(Q, K, V, mask)
        out = out.transpose(1,2).contiguous().view(batch_size,-1, self.n_heads * self.d_k)
        out= self.fc(out)
        out = self.dropout(out)
        return self.norm(out+q), attn

class FeedForward(nn.module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super.__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        out = self.fc2(self.dropout(torch.relu(self.fc1(x))))
        return self.norm(out+x)
class EncoderLayer(nn.module):
    def __init__(self, d_model, n_heads, d_ff, dropout =0.1):
        super.__init__()
        self.self_attn = MultiheadAttention(d_model, n_heads,dropout)
        self.fnn = FeedForward(d_model, d_ff,dropout)

    def forward(self, src, src_mask = None):
        out,_ = self.self_attn(src,src,src,src_mask)
        out = self.ffn(out)
        return out
    
class DecoderLayer(nn.module):
    def __init__(self, d_models, n_heads, d_ff, dropout =0.1):
        super().__init__()
        self.self_attn =MultiheadAttention(d_models, n_heads, dropout)
        self.cross_attn = MultiheadAttention(d_models, n_heads, dropout)
        self.ffn =FeedForward(d_models, d_ff, dropout)
    def forward(self, tgt, memory, tgt_mask=None, memory_mask= None):
        out,_ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        out,_ = self. cross_attn(out, memory, memory, memory_mask)
        out = self.ffn(out)
        return out
