import torch
import torch.nn as nn

from transformer import FeedForwardTransformer
import parameters as Parameters

def positional_encoding_table(k, num_dimensions):
    i = torch.arange(num_dimensions // 2)
    exponential = (10000**((2 * i)/num_dimensions)).expand(k, -1)
    positions = torch.arange(k).unsqueeze(1)
    
    sines = torch.sin(positions / exponential)
    cosines = torch.cos(positions / exponential)

    # trick to interleave sines and cosines without a for loop
    table = torch.stack((sines, cosines), dim=1).transpose(1, 2).contiguous().view(k, num_dimensions)

    return torch.FloatTensor(table)


class Encoder(nn.Module):
    def __init__(self, d_model, num_blocks=3):
        super(Encoder, self).__init__()
        
        self.phoneme_embedding = nn.Embedding(1, d_model)
        self.transformer = nn.ModuleList(
            [FeedForwardTransformer(num_heads=2, d_model=d_model) for _ in range(num_blocks)]
        )
        self.positional_encoding = positional_encoding_table(Parameters.MAX_SEQUENCE_LENGTH + 1, d_model)

    def forward(self, batch, mask):
        positional_encoding = self.positional_encoding[:batch.shape[1], :].expand(batch.shape[0], -1, -1)

        seq = self.phoneme_embedding(batch)
        seq = seq + positional_encoding
        for fft in self.transformer:
            seq = fft(seq, mask)
        
        return seq

class Decoder(nn.Module):
    def __init__(self, d_model, num_blocks=3):
        super(Decoder, self).__init__()

        self.transformer = nn.ModuleList(
            [FeedForwardTransformer(num_heads=2, d_model=d_model) for _ in range(num_blocks)]
        )
        self.positional_encoding = positional_encoding_table(Parameters.MAX_SEQUENCE_LENGTH, d_model)

    def forward(self, batch, mask):
        seq_length = min(batch.shape[1], Parameters.MAX_SEQUENCE_LENGTH)
        positional_encoding = self.positional_encoding[:seq_length, :].expand(batch.shape[0], -1, -1)

        seq = batch + positional_encoding
        for fft in self.transformer:
            seq = fft(seq, mask)
        
        return seq

        
class VarianceAdaptor(nn.Module):
    def __init__(self):
        super(VarianceAdaptor, self).__init__()
        pass

    def forward(self, batch):
        pass


class VariancePredictor(nn.Module):
    def __init__(self):
        super(VarianceAdaptor, self).__init__()
        self.layers = nn.Sequential(
            
        )

    def forward(self, batch):
        pass