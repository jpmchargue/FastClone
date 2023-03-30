from json import load
import torch
import torch.nn as nn
from torch.utils.data import Dataloader
from tqdm import tqdm
from scipy.io import wavfile

import fastspeech.parameters as Parameters
from fastspeech.fastspeech import FastSpeech
from fastspeech.scheduled_optimizer import ScheduledOptimizer
from fastspeech.datasets import FastSpeechDataset
from fastspeech.utils import prepare_vocoder


def step_fastspeech(batch, model, optimizer):
    #basenames = [i[0] for i in batch] # for logging, maybe
    phonemes = [i[1] for i in batch]
    mel_truths = [i[2] for i in batch]
    ground_truths = [i[3] for i in batch]

    predictions = model(phonemes, ground_truths)
    loss = model.loss(mel_truths, ground_truths, predictions)
    loss[0].backward()

    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step_and_update_lr()
    optimizer.zero_grad()

    return predictions, loss

def train_fastspeech(num_steps, load_step=None):
    model = FastSpeech()
    optimizer = ScheduledOptimizer(model, load_step, Parameters.BASE_LEARNING_RATE)
    vocoder = prepare_vocoder()

    if load_step is not None:
        checkpoint = torch.load(f"checkpoints/checkpoint{load_step}.pth.tar")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    dataset = FastSpeechDataset(Parameters.DATASET_PATH)
    dataloader = Dataloader(dataset, batch_size=Parameters.BATCH_SIZE, shuffle=True)

    iteration = 1
    progress_bar = tqdm(total=num_steps)
    synthesize_every = 100
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

            tqdm.write(f"Step {iteration} - Mel Loss: {postnet_loss}, Pitch Loss: {pitch_loss}, Energy Loss: {energy_loss}, Duration Loss: {duration_loss}")
            
            if iteration % synthesize_every == 0:
                synthesize_example(batch, predictions, vocoder)
            
            progress_bar.update(1)
            if iteration == num_steps:
                quit()
            iteration += 1
    
def synthesize_example(batch, output, vocoder):
    name = batch[0][0]
    mel_truth = batch[0][2]
    mel_prediction = output[0][0]

    wav_reconstruction = vocoder(mel_truth.unsqueeze(0)).squeeze(1)
    wav_prediction = vocoder(mel_prediction.unsqueeze(0)).squeeze(1)

    wav_reconstruction = wav_reconstruction.cpu().numpy() * 32768.0
    wav_prediction = wav_prediction.cpu().numpy() * 32768.0

    wavfile.write("examples/" + name + "_reconstruction.wav", Parameters.OUTPUT_SAMPLE_RATE, wav_reconstruction)
    wavfile.write("examples/" + name + "_prediction.wav", Parameters.OUTPUT_SAMPLE_RATE, wav_prediction)
