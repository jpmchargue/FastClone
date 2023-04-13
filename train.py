from json import load
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.io import wavfile

import fastspeech.parameters as Parameters
from fastspeech.fastspeech import FastSpeech
from fastspeech.scheduled_optimizer import ScheduledOptimizer
from fastspeech.datasets import FastSpeechDataset
from fastspeech.utils import prepare_vocoder

from synthesize import synthesize_new

import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def step_fastspeech(batch, model, optimizer):
    phonemes = [i[1] for i in batch]
    mel_truth = [i[2] for i in batch]
    pitch_truth = [i[3] for i in batch]
    energy_truth = [i[4] for i in batch]
    duration_truth = [i[5] for i in batch]
    ground_truth = (pitch_truth, energy_truth, duration_truth)

    predictions = model(phonemes, ground_truth=ground_truth)
    loss = model.loss(mel_truth, ground_truth, predictions)
    loss[0].backward()

    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step_and_update_lr()
    optimizer.zero_grad()

    return predictions, loss

def train_fastspeech(num_steps, load_step=None):
    iteration = load_step + 1 if load_step is not None else 1
    model = FastSpeech()
    optimizer = ScheduledOptimizer(model, iteration, Parameters.BASE_LEARNING_RATE)
    vocoder = prepare_vocoder()
    
    if load_step is not None:
        checkpoint = torch.load(f"results/checkpoints/checkpoint{load_step}.pth.tar")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    dataset = FastSpeechDataset(Parameters.DATASET_PATH, "train.txt")
    dataloader = DataLoader(dataset, batch_size=Parameters.BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn)

    progress_bar = tqdm(total=num_steps)
    synthesize_every = 1000
    save_every = 5000
    import matplotlib.pyplot as plt
    while True:
        for batch in dataloader:
            predictions, loss = step_fastspeech(batch, model, optimizer)
            (
                total_loss,
                pitch_loss,
                energy_loss,
                duration_loss,
                postnet_loss,
                prenet_loss,
            ) = loss

            loss_text = (f"Step {iteration} - Total Loss: {round(total_loss.item(), 3)}, "
                       f"Mel: {round(postnet_loss.item(), 3)}, "
                       f"Pitch: {round(pitch_loss.item(), 3)}, "
                       f"Energy: {round(energy_loss.item(), 3)}, "
                       f"Duration: {round(duration_loss.item(), 3)}")

            tqdm.write(loss_text)
            
            if iteration % synthesize_every == 0:
                synthesize_example(iteration, batch, predictions, vocoder)
                synthesize_new("At Case Western, we think beyond the possible.", model, vocoder, save_as="case_" + str(iteration))
                synthesize_new("Turn right, and the destination will be on your left.", model, vocoder, save_as="directions_" + str(iteration))

                with open('results/loss.txt', 'a') as log:
                    log.write(loss_text + "\n")

            if iteration % save_every == 0:
                create_checkpoint(iteration, model, optimizer)
            
            progress_bar.update(1)
            if iteration == num_steps:
                quit()
            iteration += 1
    
def synthesize_example(iteration, batch, output, vocoder):
    name = batch[0][0]
    mel_truth = torch.tensor(batch[0][2]).detach().transpose(0, 1).to(device)
    raw_duration_truth = batch[0][5]
    length_truth = int(sum(raw_duration_truth))
    mel_prediction = output[0][0][:length_truth].detach().transpose(0, 1).to(device)

    with torch.no_grad():
        wav_reconstruction = vocoder(mel_truth.unsqueeze(0)).squeeze()
        wav_prediction = vocoder(mel_prediction.unsqueeze(0)).squeeze()

    wav_reconstruction = (wav_reconstruction.cpu().numpy() * 32768.0).astype("int16")
    wav_prediction = (wav_prediction.cpu().numpy() * 32768.0).astype("int16")

    wavfile.write("results/examples/" + str(iteration) + "_" + name + "_reconstruction.wav", Parameters.OUTPUT_SAMPLE_RATE, wav_reconstruction)
    wavfile.write("results/examples/" + str(iteration) + "_" + name + "_prediction.wav", Parameters.OUTPUT_SAMPLE_RATE, wav_prediction)

def create_checkpoint(iteration, model, optimizer):
    model_dict = model.state_dict()
    optimizer_dict = optimizer._optimizer.state_dict()
    torch.save({
        "model": model_dict,
        "optimizer": optimizer_dict,
    }, f"results/checkpoints/checkpoint{iteration}.pth.tar")

print(f"Using device {device}")
train_fastspeech(50000)