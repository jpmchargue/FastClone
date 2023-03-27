import torch
import torch.nn as nn

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
    def __init__(self):
        super(Encoder, self).__init__()
        pass

    def forward(self, batch):
        pass

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        pass

    def forward(self, batch):
        pass

class VarianceAdaptor(nn.Module):
    def __init__(self):
        super(VarianceAdaptor, self).__init__()
        pass

    def forward(self, batch):
        pass