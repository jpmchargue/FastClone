import numpy as np
import torch
import time

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


def my_encoding_table(k, num_dimensions):
    i = torch.arange(num_dimensions // 2)
    exponential = (10000**((2 * i)/num_dimensions)).expand(k, -1)
    positions = torch.arange(k).unsqueeze(1)
    
    sines = torch.sin(positions / exponential)
    cosines = torch.cos(positions / exponential)

    table = torch.stack((sines, cosines), dim=1).transpose(1, 2).contiguous().view(k, num_dimensions)

    return torch.FloatTensor(table)

print(get_sinusoid_encoding_table(5, 10))
print(my_encoding_table(5, 10))

start = time.time()
a = get_sinusoid_encoding_table(100, 1000)
print(f"MING24 TIME: {round(time.time() - start, 5)}")
start = time.time()
b = my_encoding_table(100, 1000)
print(f"MY TIME: {round(time.time() - start, 5)}")

print(torch.norm(a))
print(torch.norm(a - b).item())