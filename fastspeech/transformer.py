import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FFTMultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super(FFTMultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_attention = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model).to(device)
        self.wk = nn.Linear(d_model, d_model).to(device)
        self.wv = nn.Linear(d_model, d_model).to(device)

        self.softmax = nn.Softmax(dim=2)

        self.postlayer = nn.Linear(d_model, d_model).to(device)
        self.dropout = nn.Dropout(0.2)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, batch, mask=None):
        # batch is initially (batch size x sequence length x d_model)
        # mask is initially (batch size x sequence length)
        # I implement Scaled Dot-Product Attention per Attention is All You Need (https://arxiv.org/abs/1706.03762)
        # as softmax((QK^T)/sqrt(d_model))V.
        batch_size, sequence_length, _ = batch.shape

        # Apply initial linear projection. In self-attention, queries, keys, and values are all the same sequence
        queries = self.wq(batch)
        keys = self.wk(batch)
        values = self.wv(batch)

        # Split the projected vectors among the heads.
        # The first part separates each embedding into num_heads partitions;
        # the second part rearranges the tensor so that the first dimension is (num_heads x batch_size).
        q = queries.view(batch_size, sequence_length, self.num_heads, self.d_attention)
        k = keys.view(batch_size, sequence_length, self.num_heads, self.d_attention)
        v = values.view(batch_size, sequence_length, self.num_heads, self.d_attention)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, sequence_length, self.d_attention)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, sequence_length, self.d_attention)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, sequence_length, self.d_attention)


        # Compute attention
        attention_weights = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.d_attention)
        if mask is not None:
            attention_mask = mask.unsqueeze(1).repeat(self.num_heads, sequence_length, 1)
            attention_weights.masked_fill(attention_mask, np.NINF)
        attention_softmax = self.softmax(attention_weights)
        attention = torch.bmm(attention_softmax, v)

        # Rearrange the data back to (batch size x sequence length x d_model)
        attention = attention.view(self.num_heads, batch_size, sequence_length, self.d_attention)
        attention = attention.permute(1, 2, 0, 3).contiguous()
        attention = attention.view(batch_size, sequence_length, -1)

        # Post-processing
        residual = self.dropout(self.postlayer(attention))
        return self.layernorm(batch + residual)


class FFTConvolution(nn.Module):
    def __init__(self, d_model, d_hidden, kernel_size):
        super(FFTConvolution, self).__init__()
        self.layer1 = nn.Conv1d(d_model, d_hidden, kernel_size=kernel_size, padding=(kernel_size - 1) // 2).to(device)
        self.layer2 = nn.Conv1d(d_hidden, d_model, kernel_size=1, padding=0).to(device)

        self.dropout = nn.Dropout(0.2)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, batch):
        hidden = self.layer1(batch.transpose(1, 2))
        output = self.layer2(F.relu(hidden)).transpose(1, 2)

        residual = self.dropout(output)
        return self.layernorm(batch + residual)


class FeedForwardTransformer(nn.Module):
    def __init__(self, num_heads=2, d_model=256, conv_d_hidden=1024, kernel_size=9):
        super(FeedForwardTransformer, self).__init__()
        self.multihead_attention = FFTMultiHeadSelfAttention(num_heads, d_model).to(device)
        self.convolution = FFTConvolution(d_model, conv_d_hidden, kernel_size).to(device)

    def forward(self, batch, mask):
        if mask is not None:
            past_end_of_sequence_mask = mask.unsqueeze(-1)
            sequence = self.multihead_attention(batch, mask=mask)
            sequence.masked_fill(past_end_of_sequence_mask, 0)
            sequence = self.convolution(sequence)
            sequence.masked_fill(past_end_of_sequence_mask, 0)
        else:
            sequence = self.multihead_attention(batch, mask=mask)
            sequence = self.convolution(sequence)

        return sequence
    
