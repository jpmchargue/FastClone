import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import random

import cleantext

NUM_BATCH_SPEAKERS = 64
NUM_BATCH_UTTERANCES = 10
NUM_BATCH_FRAMES = 160

class SpeakerDataset(Dataset):
    """
    Creates a dataset for training GE2EL Speaker Verification,
    given a path to a folder containing training speaker data.
    The folder should contain a single subfolder for each speaker,
    and those subfolders should each contain .npy files representing
    40-bank log mel spectrograms of utterances from the respective speaker.
    Each 'index' of the dataset is a 64 * 10 * L * 40 ndarray of utterance data,
    with 64 speakers and 10 utterances per speaker.
    """
    def __init__(self, path):
        self.speaker_ids = os.listdir(path)
        self.speakers = [Speaker(path, id) for id in self.speaker_ids]
        self.deck = Deck(self.speakers)

    def __len__(self):
        return len(self.speaker_ids)

    def __getitem__(self, index):
        return np.array([s.fetchUtterances(NUM_BATCH_UTTERANCES, NUM_BATCH_FRAMES) for s in self.deck.deal(NUM_BATCH_SPEAKERS)])


class UtteranceDataset(Dataset):
    """
    Creates a dataset for training seq2seq speech reconstruction,
    given a path to a folder containing training speaker data.
    The folder should contain a single subfolder for each speaker,
    and those subfolders should each contain .npy files representing
    40-bank log mel spectrograms of utterances from the respective speaker.
    Each 'index' of the dataset is an L * 40 ndarray representing
    the mel spectrogram of a single random utterance.
    """
    def __init__(self, path):
        self.utterances = []
        for speaker in os.listdir(path):
            files = os.listdir(os.path.join(path, speaker))
            for file in files:
                self.utterances.append(os.path.join(path, speaker, file))
        self.deck = Deck(self.utterances)

    def __len__(self):
        return len(self.utterances)
    
    def __getitem__(self, index):
        spectrogram = np.load(self.deck.deal(1)[0])
        if len(spectrogram) > 500:
            return spectrogram[:500]
        return spectrogram


class Speaker():
    """
    Create an object representing a set of utterances from a single speaker,
    given the name of the data/speaker subfolder for that speaker.
    This object contains methods for gathering and randomly choosing
    batches of utterances from that speaker.
    """
    def __init__(self, path, id):    
        self.npys = self.getNpyReferences(path, id)
        self.deck = Deck(self.npys)

    def getNpyReferences(self, path, id):
        folder = os.path.join(path, id)
        return [os.path.join(folder, npy) for npy in os.listdir(folder)] 

    def trimToLength(self, utterance, length):
        # Special case if utterance length is less than target length
        while utterance.shape[0] < length:
            utterance = np.append(utterance, utterance, axis=0)
        
        start = random.randint(0, utterance.shape[0] - length)

        return utterance[start:start+length]

    def fetchUtterances(self, n, length):
        """
        Get a set of semi-random partial utterances from this speaker.
        By using a Deck, it's ensured that no utterance will be seen more than
        one more time than any other utterance.
        Args:
            n: the number of utterances to retrieve
            length: the length (in spectrogram frames) of the utterances returned
        Returns:
            a semi-random selection of n utterances with length frames from this speaker.
            The resulting array is (n * length * D), where D is the dimensionality of 
            one frame of the utterance spectrogram (generally 40).
        """
        chosen_npys = self.deck.deal(n)
        return [self.trimToLength(np.load(npy, mmap_mode='r'), length) for npy in chosen_npys]
        

class Deck():
    """
    A data structure which can be used to randomly 'deal' items 
    from a set of items without replacement.
    Once an item is dealt, it cannot be returned again until every item is dealt,
    ensuring that every item is dealt roughly the same number of times.
    """
    def __init__(self, items):
        if not isinstance(items, list):
            raise ValueError("Deck must be constructed with a list")

        self.items = items
        self.length = len(items)
        self.undealt = random.sample(self.items, self.length)

    def reshuffle(self):
        self.undealt = random.sample(self.items, self.length)

    def deal(self, n):
        if n <= len(self.undealt):
            dealt = self.undealt[:n]
            self.undealt = self.undealt[n:]
        else:
            dealt = self.undealt[:]
            n -= len(self.undealt)
            while n > self.length:
                self.reshuffle()
                dealt.extend(self.undealt[:])
                n -= self.length
            self.reshuffle()
            dealt.extend(self.undealt[:n])
            self.undealt = self.undealt[n:]
        return dealt


class FastSpeechDataset(Dataset):
    """
    A dataset for getting training batches for FastSpeech.
    """
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.names, self.speakers, self.text, self.raw_text = self.gather_all_utterances(dataset_path)

    def gather_all_utterances(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name, speaker, text, raw_text = [], [], [], []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        basename = self.names[index]
        speaker = self.speakers[index]
        phonemes = np.array(cleantext.text_to_sequence(self.text[index], ["english_cleaners"]))
        mel_path = os.path.join(
            self.dataset_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.dataset_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch_truth = np.load(pitch_path)
        energy_path = os.path.join(
            self.dataset_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy_truth = np.load(energy_path)
        duration_path = os.path.join(
            self.dataset_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration_truth = np.load(duration_path)
        #fingerprint_path = os.path.join(
        #    self.preprocessed_path,
        #    "fingerprint",
        #    "{}-fingerprint.npy".format(speaker),
        #)
        #fingerprint = np.load(fingerprint_path)

        return (
            basename,
            phonemes,
            mel,
            (
                pitch_truth,
                energy_truth,
                duration_truth,
            ),
        )