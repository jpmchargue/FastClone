import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cleantext
from fastspeech.transformer import FeedForwardTransformer
import fastspeech.parameters as Parameters
from fastspeech.utils import blockify, blockify3D


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, d_model, num_blocks=3):
        super(Encoder, self).__init__()
        
        self.phoneme_embedding = nn.Embedding(len(cleantext.symbols) + 1, d_model)
        self.transformer = nn.ModuleList(
            [FeedForwardTransformer(num_heads=2, d_model=d_model) for _ in range(num_blocks)]
        )
        self.positional_encoding = positional_encoding_table(Parameters.MAX_SEQUENCE_LENGTH + 1, d_model).to(device)

    def forward(self, batch, mask):
        positional_encoding = self.positional_encoding[:batch.shape[1], :].expand(batch.shape[0], -1, -1)

        seq = self.phoneme_embedding(batch)
        seq = seq + positional_encoding
        for fft in self.transformer:
            seq = fft(seq, mask)
        
        return seq


class Decoder(nn.Module):
    def __init__(self, d_model, num_blocks=3):
        super(Decoder, self).__init__()

        self.transformer = nn.ModuleList(
            [FeedForwardTransformer(num_heads=2, d_model=d_model) for _ in range(num_blocks)]
        )
        self.positional_encoding = positional_encoding_table(Parameters.MAX_SEQUENCE_LENGTH, d_model).to(device)

    def forward(self, batch, mask):
        seq_length = min(batch.shape[1], Parameters.MAX_SEQUENCE_LENGTH)
        positional_encoding = self.positional_encoding[:seq_length, :].expand(batch.shape[0], -1, -1)

        seq = batch + positional_encoding
        for fft in self.transformer:
            seq = fft(seq, mask)
        
        return seq

        
class VarianceAdaptor(nn.Module):
    def __init__(self, d_model, num_bins=256):
        super(VarianceAdaptor, self).__init__()
        self.pitch_predictor = VariancePredictor(d_model)
        self.energy_predictor = VariancePredictor(d_model)
        self.duration_predictor = VariancePredictor(d_model).to("cpu")

        # get statistics on minimum and maximum pitch/energy
        # values based on VCTK
        pitch_min, pitch_max = -3, 12
        energy_min, energy_max = -2, 9

        self.pitch_bins = torch.linspace(pitch_min, pitch_max, num_bins - 1).to(device)
        self.energy_bins = torch.linspace(energy_min, energy_max, num_bins - 1).to(device)

        self.pitch_embedding = nn.Embedding(num_bins, d_model)
        self.energy_embedding = nn.Embedding(num_bins, d_model)

    def length_regulator(self, batch, durations, true_max_mel_length=None):
        # batch and durations should have the same length in dimensions 0 and 1
        expanded = []
        masks = []
        for sequence, new_lengths in zip(batch, durations):
            new_sequence = sequence.repeat_interleave(new_lengths, dim=0)
            new_mask = torch.arange(true_max_mel_length).to(device) < torch.sum(new_lengths)
            expanded.append(new_sequence)
            masks.append(new_mask)

        pad_lengths_to = max([e.shape[0] for e in expanded]) if true_max_mel_length is None else true_max_mel_length
        padded = []
        for new_sequence in expanded:
            padded.append(F.pad(new_sequence, (0, 0, 0, pad_lengths_to - new_sequence.shape[0])))
        
        return torch.stack(padded), torch.stack(masks)

    def forward(self, batch, input_mask, ground_truth=None):
        # ground_truth should be of the form (pitch truth, energy truth, duration truth)
        
        # In this implementation, pitch and energy prediction are done at the phoneme level
        # and therefore before length regulation.
        pitch_predictions = self.pitch_predictor(batch, input_mask)
        pitch_embeddings = self.pitch_embedding(torch.bucketize(pitch_predictions, self.pitch_bins)) if ground_truth is None \
                else self.pitch_embedding(torch.bucketize(ground_truth[0], self.pitch_bins))

        energy_predictions = self.energy_predictor(batch, input_mask)
        energy_embeddings = self.energy_embedding(torch.bucketize(energy_predictions, self.energy_bins)) if ground_truth is None \
                else self.energy_embedding(torch.bucketize(ground_truth[1], self.energy_bins))

        batch = batch + pitch_embeddings + energy_embeddings

        raw_duration_predictions = self.duration_predictor(batch, input_mask)
        duration_predictions = torch.clamp(torch.round(torch.exp(raw_duration_predictions) - 1), min=0)
        output, output_masks = self.length_regulator(batch, duration_predictions) if ground_truth is None \
                else self.length_regulator(batch, ground_truth[2], max(torch.sum(ground_truth[2], dim=1)))

        return (
            output, 
            pitch_predictions, 
            energy_predictions, 
            raw_duration_predictions, 
            output_masks
        )


class VariancePredictor(nn.Module):
    def __init__(self, d_model, d_hidden=256, dropout=0.5):
        super(VariancePredictor, self).__init__()
        self.layers = nn.Sequential(
            TransposeConv1d(d_model, d_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            TransposeConv1d(d_hidden, d_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, batch, mask):
        output = self.layers(batch).squeeze(-1)
        output = output.masked_fill(mask, 0.0)
        return output
    
class TransposeConv1d(nn.Module):
    def __init__(self, d_input, d_output, kernel_size=3, padding=1):
        super(TransposeConv1d, self).__init__()
        self.conv = nn.Conv1d(d_input, d_output, kernel_size=kernel_size, padding=padding)
    
    def forward(self, batch):
        trans = batch.contiguous().transpose(1, 2)
        trans = self.conv(trans)
        output = trans.contiguous().transpose(1, 2)
        return output


class PostNet(nn.Module):
    def __init__(self, d_postnet, output_mel_channels, num_blocks=6, dropout=0.5):
        super(PostNet, self).__init__()
        self.blocks = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(output_mel_channels, d_postnet, kernel_size=5, padding=2),
                nn.BatchNorm1d(d_postnet),
                nn.Tanh(),
                nn.Dropout(dropout),
            )] +
            [nn.Sequential(
                nn.Conv1d(d_postnet, d_postnet, kernel_size=5, padding=2),
                nn.BatchNorm1d(d_postnet),
                nn.Tanh(),
                nn.Dropout(dropout),
            ) for i in range(num_blocks-2)] +
            [nn.Sequential(
                nn.Conv1d(d_postnet, output_mel_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(output_mel_channels),
                nn.Dropout(dropout),
            )]
        )

    def forward(self, batch):
        residual = batch.transpose(1, 2)
        for block in self.blocks:
            residual = block(residual)
        return batch + residual.transpose(1, 2)


class FastSpeech(nn.Module):
    def __init__(self):
        super(FastSpeech, self).__init__()
        d_model = Parameters.HIDDEN_DIMENSIONS
        d_postnet = Parameters.POSTNET_DIMENSIONS
        output_mel_channels = Parameters.OUTPUT_MEL_CHANNELS

        self.encoder = Encoder(d_model, num_blocks=Parameters.ENCODER_BLOCKS).to(device)
        self.variance_adaptor = VarianceAdaptor(d_model).to(device)
        self.decoder = Decoder(d_model, num_blocks=Parameters.DECODER_BLOCKS).to(device)
        self.mel_projection = nn.Linear(d_model, output_mel_channels).to(device)
        self.postnet = PostNet(d_postnet, output_mel_channels, num_blocks=Parameters.POSTNET_BLOCKS).to(device)

        self.variance_predictor_loss = nn.MSELoss().to(device)
        self.mel_loss = nn.L1Loss().to(device)

    def get_input_masks(self, phonemes):
        lengths = [len(input) for input in phonemes]
        max_input_length = max(lengths)
        indices = torch.arange(max_input_length).expand(len(phonemes), -1)
        return indices >= torch.tensor(lengths).unsqueeze(1)    

    def forward(self, phonemes, ground_truth=None):
        # phonemes is a batch: it's a list of lists, where every sublist represents a sequence of phonemes
        # ground_truth should be of the form (pitch truth, energy truth, duration truth)
        input_masks = self.get_input_masks(phonemes).to(device)
        phonemes = blockify(phonemes).type(torch.LongTensor).to(device)
        if ground_truth is not None:
            ground_truth = (
                blockify(ground_truth[0]).type(torch.LongTensor).to(device), # pitch by phoneme ground truth
                blockify(ground_truth[1]).type(torch.LongTensor).to(device), # energy by phoneme ground truth
                blockify(ground_truth[2]).type(torch.LongTensor).to(device), # duration by phoneme ground truth
            )

        seq = self.encoder(phonemes, input_masks)
        seq, pitch_prediction, energy_prediction, duration_prediction, output_masks = \
            self.variance_adaptor(seq, input_masks, ground_truth=ground_truth)
        seq = self.decoder(seq, output_masks)
        seq = self.mel_projection(seq)

        final = self.postnet(seq)

        return (final, seq, pitch_prediction, energy_prediction, duration_prediction, input_masks, output_masks)

    def loss(self, mel_truths, ground_truths, predictions):
        (
            pitch_truths, 
            energy_truths, 
            duration_truths
        ) = ground_truths
        (
            postnet_predictions, 
            prenet_predictions, 
            pitch_predictions,
            energy_predictions,
            duration_predictions,
            input_masks,
            output_masks
        ) = predictions

        torch.set_printoptions(threshold=10_000)



        pitch_predictions = pitch_predictions.masked_select(~input_masks)
        energy_predictions = energy_predictions.masked_select(~input_masks)
        duration_predictions = duration_predictions.masked_select(~input_masks)
        pitch_truths = torch.from_numpy(np.concatenate(pitch_truths)).float().to(device)
        energy_truths = torch.from_numpy(np.concatenate(energy_truths)).float().to(device)
        duration_truths = torch.from_numpy(np.log(np.concatenate(duration_truths)) + 1).float().to(device)
        pitch_loss = self.variance_predictor_loss(pitch_predictions, pitch_truths)
        energy_loss = self.variance_predictor_loss(energy_predictions, energy_truths)
        duration_loss = self.variance_predictor_loss(duration_predictions, duration_truths)

        postnet_predictions = postnet_predictions.masked_select(output_masks.unsqueeze(-1))
        prenet_predictions = prenet_predictions.masked_select(output_masks.unsqueeze(-1))
        mel_truths = torch.from_numpy(np.concatenate([m.flatten() for m in mel_truths])).to(device)
        postnet_loss = self.mel_loss(postnet_predictions, mel_truths)
        prenet_loss = self.mel_loss(prenet_predictions, mel_truths)

        total_loss = pitch_loss + energy_loss + duration_loss + postnet_loss + prenet_loss

        return (total_loss, pitch_loss, energy_loss, duration_loss, postnet_loss, prenet_loss)





