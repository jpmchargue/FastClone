from g2p_en import G2p
import numpy as np
from string import punctuation
import re
import fastspeech.parameters as Parameters
import torch
from scipy.io import wavfile

import cleantext

import fastspeech.parameters as Parameters
from fastspeech.fastspeech import FastSpeech
from fastspeech.utils import prepare_vocoder

import os
import sys
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def preprocess_english(text):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(Parameters.LEXICON_PATH)

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))

    phones = phones.replace("JH", "D G")

    sequence = np.array(
        cleantext.text_to_sequence(
            phones, ["english_cleaners"]
        )
    )

    return np.array([sequence])

def synthesize_new(text, model, vocoder, save_as=None):
    name = text if save_as is None else save_as
    embedded = preprocess_english(text)

    model.eval()
    with torch.no_grad():
        print("Synthesis")
        print(embedded.shape)
        print(embedded)
        
        start_time = time.time()
        predictions = model(embedded, ground_truth=None)
        end_time = time.time()
        print(f"Inference time: {round(end_time - start_time, 3)} s")

        mel_prediction = predictions[0][0]
        mel_prediction = mel_prediction.detach().transpose(0, 1).to(device)

        wav_prediction = vocoder(mel_prediction.unsqueeze(0)).squeeze()
        wav_prediction = (wav_prediction.cpu().numpy() * 32768.0).astype("int16")
    model.train()

    wavfile.write("results/examples/" + name + ".wav", Parameters.OUTPUT_SAMPLE_RATE, wav_prediction)

if __name__ == "__main__":
    model = FastSpeech()
    load_step = 50000
    if load_step is not None:
        checkpoint_name = f"results/checkpoints/checkpoint{load_step}.pth.tar"
        if os.path.exists(checkpoint_name):
            print("found checkpoint!")
            checkpoint = torch.load(checkpoint_name)
            model.load_state_dict(checkpoint["model"])

    vocoder = prepare_vocoder()
    synthesize_new(sys.argv[1], model, vocoder)