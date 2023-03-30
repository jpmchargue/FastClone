from json import load
import torch
from fastspeech.fastspeech import FastSpeech
from fastspeech.scheduled_optimizer import ScheduledOptimizer
import fastspeech.parameters as Parameters

def train_fastspeech(num_iters, load_step=None):
    model = FastSpeech()
    optimizer = ScheduledOptimizer(model, load_step, Parameters.BASE_LEARNING_RATE)
    if load_step is not None:
        checkpoint = torch.load(f"checkpoints/checkpoint{load_step}.pth.tar")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    