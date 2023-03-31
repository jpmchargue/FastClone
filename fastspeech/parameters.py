MAX_SEQUENCE_LENGTH = 1000
HIDDEN_DIMENSIONS = 256

ENCODER_BLOCKS = 3
DECODER_BLOCKS = 3

OUTPUT_MEL_CHANNELS = 80
POSTNET_DIMENSIONS = 512
POSTNET_BLOCKS = 6

# Training
DATASET_PATH = "data/LJSpeech"
BASE_LEARNING_RATE = 1/16
BATCH_SIZE = 16

# Synthesis
OUTPUT_SAMPLE_RATE = 22050