import json
import numpy as np
import torch

import hifigan


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def blockify(sequences):
    """
    Create a tensor from a list of sequences.
    Every row in the tensor represents one sequence in the input list,
    and shorter sequences are padded with zeros to match the length 
    of the longest sequence.

    Args:
        sequences (List(ndarray)): a list of sequences
    """
    max_input_length = max([len(s) for s in sequences])
    new_sequences = np.array([np.pad(s, (0, max_input_length - len(s))) for s in sequences])
    return torch.from_numpy(new_sequences)

def blockify3D(matrices):
    """
    Create a tensor from a list of matrices.
    All matrices are assumed to have the same length in the second dimension,
    but may have different lengths in the first dimension.
    All matrices are padded in the first dimension to match the length
    of the longest matrix.

    Args:
        matrices (List(ndarray)): a list of matrices
    """
    max_matrix_length = max([m.shape[0] for m in matrices])
    new_matrices = np.array([np.pad(m, (0, 0, 0, max_matrix_length - m.shape[0])) for m in matrices])
    return torch.from_numpy(new_matrices)

def prepare_vocoder():
    """
    Get a pretrained HiFiGAN vocoder.
    """
    with open("hifigan/config.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
    ckpt = torch.load("hifigan/generator_universal.pth.tar")
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)

    return vocoder