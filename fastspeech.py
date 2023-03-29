import torch
import torch.nn as nn

from transformer import FeedForwardTransformer
import parameters as Parameters

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
        
        self.phoneme_embedding = nn.Embedding(1, d_model)
        self.transformer = nn.ModuleList(
            [FeedForwardTransformer(num_heads=2, d_model=d_model) for _ in range(num_blocks)]
        )
        self.positional_encoding = positional_encoding_table(Parameters.MAX_SEQUENCE_LENGTH + 1, d_model)

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
        self.positional_encoding = positional_encoding_table(Parameters.MAX_SEQUENCE_LENGTH, d_model)

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
        self.pitch_predictor = VariancePredictor()
        self.energy_predictor = VariancePredictor()
        self.duration_predictor = VariancePredictor()

        # get statistics on minimum and maximum pitch/energy
        # values based on VCTK
        pitch_min, pitch_max = -3, 12
        energy_min, energy_max = -2, 9

        self.pitch_bins = torch.linspace(pitch_min, pitch_max, num_bins - 1)
        self.energy_bins = torch.linspace(energy_min, energy_max, num_bins - 1)

        self.pitch_embedding = nn.Embedding(num_bins, d_model)
        self.energy_embedding = nn.Embedding(num_bins, d_model)

    def length_regulator(self, batch, durations, max_mel_length):
        pass

    def forward(self, batch, input_mask, max_mel_length, ground_truth=None):
        # ground_truth should be of the form (pitch truth, energy truth, duration truth)
        
        # In this implementation, pitch and energy prediction are done at the phoneme level
        # and therefore before length regulation.
        pitch_predictions = self.pitch_predictor(batch, input_mask)
        pitch_embeddings = self.pitch_embedding(pitch_predictions) if ground_truth is None \
                else self.pitch_embedding(ground_truth[0])

        energy_predictions = self.energy_predictor(batch, input_mask)
        energy_embeddings = self.energy_embedding(energy_predictions) if ground_truth is None \
                else self.energy_embedding(ground_truth[1])

        batch = batch + pitch_embeddings + energy_embeddings

        raw_duration_predictions = self.duration_predictor(batch, input_mask)
        duration_predictions = torch.clamp(torch.round(torch.exp(raw_duration_predictions) - 1), min=0)
        output, output_masks = self.length_regulator(batch, duration_predictions) if ground_truth is None \
                else self.length_regulator(ground_truth[2])

        return (
            output, 
            pitch_predictions, 
            energy_predictions, 
            raw_duration_predictions, 
            output_masks
        )


class VariancePredictor(nn.Module):
    def __init__(self, d_model, d_hidden=256, dropout=0.5):
        super(VarianceAdaptor, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(d_model, d_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, batch, mask):
        output = self.layers(batch)
        output = output.masked_fill(mask, 0.0)
        return output