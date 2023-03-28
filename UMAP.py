import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import torch
import umap

from dataio import SpeakerDataset
from verifier import Verifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cwd = os.getcwd()

LOAD_MODEL = True
NUM_SPEAKERS_PLOTTED = 10
NUM_UTTERANCES = 10
DATA_PATH = "C:/Users/James/Desktop/toyprojects/prosody/data/speakers"

verifier = Verifier()
if LOAD_MODEL:
    print("Loading model...")
    checkpoint_name = "transformer_verifier7500.pt"
    if os.path.exists(os.path.join(cwd, checkpoint_name)):
        verifier.load_state_dict(torch.load(checkpoint_name))

print("Loading random sample...")
speakers = random.sample(os.listdir(DATA_PATH), NUM_SPEAKERS_PLOTTED)
all_spectrograms = []
for speaker in speakers:
    names = random.sample(os.listdir(os.path.join(DATA_PATH, speaker)), NUM_UTTERANCES)
    paths = [os.path.join(DATA_PATH, speaker, name) for name in names]
    for p in paths:
        print(p)
    spectrograms = [np.load(path) for path in paths]
    all_spectrograms.extend(spectrograms)

print("Embedding...")
all_embeddings = torch.stack([verifier.get_embedding(spectrogram) for spectrogram in all_spectrograms])

for embedding in all_embeddings:
    print(all_embeddings)

print("Mapping...")
scaled = StandardScaler().fit_transform(all_embeddings)
umapped = umap.UMAP().fit_transform(scaled)

print("Plotting...")
colors = ['r', "orange", 'y', 'g', "aqua", 'b', "indigo", 'm', "slategray", "black"]
labels = np.repeat(colors, 10)
mp = [mpatches.Patch(color=c, label='Speaker ' + speakers[i]) for i, c in enumerate(colors)]
plt.scatter(
    umapped[:, 0],
    umapped[:, 1],
    c=labels)
plt.legend(handles=mp)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP Projection of Speaker Embeddings (20000)')
plt.show()