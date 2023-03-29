import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import FeedForwardTransformer

import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Verifier(nn.Module):
    def __init__(self, input_frame_size=40, hidden_size=256, fingerprint_size=256):
        super(Verifier, self).__init__()
        self.num_blocks = 3
        self.num_speakers = 64
        self.num_utterances_per_speaker = 10
        self.window_length = 160
        self.fingerprint_size = fingerprint_size

        self.preprocess = nn.Linear(input_frame_size, hidden_size).to(device)
        self.transformer = nn.ModuleList([FeedForwardTransformer(num_heads=2, d_model=hidden_size, conv_d_hidden=256) for _ in range(self.num_blocks)])
        self.project = nn.Linear(hidden_size, fingerprint_size).to(device)

        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(device)

        self.loss_function = nn.CrossEntropyLoss().to(device)

    def forward(self, batch):
        # batch is initially (num utterances x 160 x 40)
        seq = self.preprocess(batch) # (num utterances x 160 x 256)
        for i in range(self.num_blocks): # (num utterances x 160 x 256)
            seq = self.transformer[i](seq, mask=None)
        #finals = seq[:, -1] # (num utterances x 256)
        finals = torch.mean(seq, 1)
        fingerprint = self.project(finals) # (num utterances x fingerprint size)

        return fingerprint

    def get_similarity_matrix(self, batch):
        # batch is initially (num utterances * 256)
        ns = self.num_speakers
        nu = self.num_utterances_per_speaker

        new_view = batch.view(ns, nu, -1) # (64 x 10 x 256)
        speaker_sums = new_view.sum(dim=1, keepdim=True) # (64 x 1 x 256)
        centroids = F.normalize((speaker_sums / nu).squeeze(), dim=1).to(device) # (64 x 256)
        centroids_minus_i = F.normalize(((speaker_sums - new_view) / (nu - 1)), dim=2).view(ns * nu, -1).to(device) # (640 x 256)
        batch = F.normalize(batch, dim=1).to(device) # Fun fact: CorentinJ's implementation is incorrect, since it's missing this normalization.

        similarity_matrix = torch.matmul(batch, centroids.t()) # (640 x 64)
        for j in range(ns):
            for i in range(nu):
                ji = (j * nu) + i
                similarity_matrix[ji, j] = torch.dot(batch[ji], centroids_minus_i[ji])

        similarity_matrix = (similarity_matrix * self.similarity_weight) + self.similarity_bias

        return similarity_matrix
    
    def loss(self, batch):
        ns = self.num_speakers
        nu = self.num_utterances_per_speaker

        similarity_matrix = self.get_similarity_matrix(batch)
        #similarity_matrix_clone = similarity_matrix.clone().cpu().detach()
        #plt.imshow(similarity_matrix_clone)
        #plt.show()

        target = torch.arange(ns).repeat_interleave(nu).to(device)
        loss = self.loss_function(similarity_matrix, target)

        return loss

    def get_embedding(self, spectrogram):
        """
        Compute the overall embedding vector for a sound sample,
        as inferred by the verifier network.
        This is calculated by finding the embedding vector for each point
        of a sliding window of length 160 with 50% overlap.
        Then, all embedding vectors are L2-normalized and averaged.
        """
        L = len(spectrogram)
        d_vector = torch.zeros(self.fingerprint_size).to(device)
        count = 0
        i = 0
        while (i + self.window_length < L):
            input = torch.from_numpy(spectrogram[i:i+self.window_length]).to(device)
            input = torch.unsqueeze(input, dim=0)
            embedding = self.forward(input)
            norm = torch.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                d_vector = d_vector + embedding
            count += 1
            i += self.window_length // 2
        d_vector = d_vector.view(-1)
        return d_vector.detach().cpu() / count

        