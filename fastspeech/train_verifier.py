import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from verifier import Verifier
from dataio import SpeakerDataset, UtteranceDataset
from scheduled_optimizer import ScheduledOptimizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def step_gel(input_batch, verifier, optimizer):
    """
    Performs one training step for the Generalized E2E Loss verification network.

    Args:
        input_batch: an (N, M, L, D)-dimensional tensor, where
            N is the number of speakers (64),
            M is the number of utterances per speaker (10),
            L is the length of each utterance in frames (140-180), and
            D is the dimensionality of one frame of data (40).
    """
    optimizer.zero_grad()
    
    N = input_batch.size(0)
    M = input_batch.size(1)
    L = input_batch.size(2)
    D = input_batch.size(3)
    outD = 256

    flattened = torch.flatten(input_batch, start_dim=0, end_dim=1)
    num_batches = flattened.size(0)

    encoded = verifier(flattened)
    encoded = encoded.view((640, -1))

    loss = verifier.loss(encoded)
    loss.backward()
    optimizer.step()
    return loss

def train_gel(verifier, optimizer, num_iters, learning_rate=0.01, print_every=100):
    start = time.time()

    #criterion = GE2ELoss(init_w=10.0, init_b=-5.0, loss_method='contrast')
    criterion = nn.L1Loss()
    optimizer_v = optim.SGD(verifier.parameters(), lr=learning_rate)
    loss_records = []
    loss_total = 0
    
    # generate input tensor
    dataset = SpeakerDataset("C:/Users/James/Desktop/toyprojects/prosody/data/speakers")

    # run step_gel()
    for iter in tqdm(range(1, num_iters + 1)):
        input_batch = torch.from_numpy(dataset[iter]).to(device)
        loss = step_gel(input_batch, verifier, optimizer_v) 
        #(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        loss_total += loss
        loss_records.append(loss.item())
        tqdm.write(f"{iter} | LOSS: {loss}")

        if iter % print_every == 0:
            loss_avg = loss_total / print_every
            tqdm.write('Step %d (%d%%) %.4f' % (iter, iter / num_iters * 100, loss_avg))
            loss_total = 0
    
    plt.plot(loss_records)
    plt.show()

def save_state(network, optimizer, filename):
    print(network.state_dict())
    torch.save(
        {
            "model": network.state_dict(),
            "optimizer": optimizer._optimizer.state_dict(),
        },
        filename
    )

def load_state(network, optimizer, filename):
    ckpt = torch.load(filename)
    network.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

verifier = Verifier()
optimizer = ScheduledOptimizer(verifier, 0, 0.01, warmup_duration=250)

load_state(verifier, optimizer, "transformer_verifier_mean5000.pth.tar")

train_gel(verifier, optimizer, 5000)
save_state(verifier, optimizer, "transformer_verifier_mean10000.pth.tar")