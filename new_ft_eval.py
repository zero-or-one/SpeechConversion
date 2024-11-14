import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, DatasetDict, Dataset
from datasets import Audio
from tqdm import tqdm
#from openai import OpenAI
from pydub import AudioSegment
from time import time
import re
from dotenv import load_dotenv
from metrics import ASRMetrics
import numpy as np

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# / START SETUP
# Load the dataset
test_name = 'test'
#test_name = 'valid'

speaker = 'MJY_Woman_40s'
target_dataset = os.path.join('dataset', speaker)
dataset_path = os.path.join(target_dataset, f"{test_name}.json")
dataset = load_dataset('json', data_files=dataset_path)
#model_path = 'openai/whisper-large-v3-turbo'
model_path = f'neoALI/STT-{speaker}'
model_name = model_path

TIME_WRAPPING = False
DELETION_RATE = 0.1
revision = None #'f4e56f1c4cdba8d1bd1740f39272bc6a23078d48'

# Split the test dataset into two halves (only for larger training data)
#test_data_split = dataset['train'].train_test_split(test_size=0.5, seed=42)
#dataset['train'] = test_data_split['train']
#dataset['train'] = test_data_split['test']
# /END SETUP 

def calculate_dataset_duration(dataset):
    total_duration = 0
    for item in dataset['train']:
        audio_path = item['audio']['path']
        try:
            audio = AudioSegment.from_file(audio_path)
            duration_seconds = len(audio) / 1000.0  # Convert milliseconds to seconds
            total_duration += duration_seconds
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    return total_duration
# Convert durations to hours, minutes, and seconds
def format_duration(duration_seconds):
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}-{int(minutes):02d}-{seconds:.2f}"

test_duration = calculate_dataset_duration(dataset)
test_duration = format_duration(test_duration)
print(f"Total duration of the test dataset: {test_duration} seconds")


dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.rename_column('sentence', 'transcription')
# take a second half of the dataset

# Create the validation dataset from the test indices
val_dataset = dataset['train']



feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path, token=os.getenv("HUGGINGFACE_TOKEN"), revision=revision)
tokenizer = WhisperTokenizer.from_pretrained(model_path, task="transcribe", language='ko', token=os.getenv("HUGGINGFACE_TOKEN"), revision=revision)
processor = WhisperProcessor.from_pretrained(model_path, task="transcribe", language='ko', token=os.getenv("HUGGINGFACE_TOKEN"), revision=revision)
model = WhisperForConditionalGeneration.from_pretrained(model_path, token=os.getenv("HUGGINGFACE_TOKEN"), revision=revision).to(device)
model = model.eval()

def spectrogram_processing(m, deletion_rate=0.1):
    """
    Augment a mel-spectrogram by randomly deleting frames with a probability inversely proportional
    to the frame-to-frame differences. Frames with smaller differences (less informative) have a higher
    chance of being deleted.

    Parameters:
    m (numpy.ndarray): Input spectrogram of shape (F, T).
    deletion_rate (float): Controls the overall rate of frame deletion (between 0 and 1).

    Returns:
    numpy.ndarray: Augmented spectrogram of shape (F, T').
    """
    # Transpose the matrix to have the shape (T, F)
    m = m.T
    T, F = m.shape

    # Compute frame-to-frame differences
    d = np.sum(np.abs(m[1:, :] - m[:-1, :]), axis=1)
    # Pad d with a zero at the beginning to make it length T
    d = np.concatenate(([0], d))

    # Normalize differences and invert to get deletion probabilities
    dmax = np.max(d)
    if dmax == 0:
        # Avoid division by zero if dmax is zero
        print("Warning: dmax is zero, returning unprocessed spectrogram")
        pd = np.ones(T) * deletion_rate
    else:
        # Invert differences: frames with smaller changes have higher deletion probability
        pd = (1 - (d / dmax)) * deletion_rate

    # Generate random numbers and determine which frames to delete
    random_values = np.random.rand(T)
    flags = random_values < pd

    # Ensure the first few frames are not deleted
    flags[:4] = False

    # Delete frames where flags are True
    m_processed = m[~flags, :]

    # Transpose back to original shape (F, T')
    m_processed = m_processed.T
    return m_processed

def process_and_transcribe(audio_sample):
    # Preprocess audio
    input_features = feature_extractor(audio_sample["array"], sampling_rate=audio_sample["sampling_rate"],
                                       ).input_features[0]
    if TIME_WRAPPING:
        input_features = spectrogram_processing(input_features, deletion_rate=DELETION_RATE)
        # pad to 3000 frames
        if input_features.shape[1] < 3000:
            input_features = np.pad(input_features, ((0, 0), (0, 3000 - input_features.shape[1])), mode='constant')

    input_features = torch.tensor(input_features).unsqueeze(0).to(device)
    
    # Generate transcription
    with torch.no_grad():
        input_ids = model.generate(input_features, max_length=225, num_beams=5, early_stopping=True)[0]
    
    # Decode the transcription
    prediction = processor.decode(input_ids, skip_special_tokens=True)
    
    return prediction.strip()

# Process samples and calculate metrics
all_predictions = []
all_references = []
total_time = 0
total_audio_duration = 0

for sample in tqdm(val_dataset):
    # Calculate audio duration
    audio_duration = len(sample['audio']['array']) / sample['audio']['sampling_rate']
    total_audio_duration += audio_duration
    
    # Process and transcribe
    start_time = time()
    prediction = process_and_transcribe(sample['audio'])
    end_time = time()
    
    # Calculate processing time
    processing_time = end_time - start_time
    total_time += processing_time
    
    # Store results
    all_predictions.append(prediction)
    all_references.append(sample['transcription'])

# Calculate and print metrics
print(f"Total audio duration: {total_audio_duration:.2f} seconds")
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Real-time factor: {total_time / total_audio_duration:.2f}")
print(f"Average processing time: {total_time / len(val_dataset):.2f} seconds")

# Evaluate the model
metrics = ASRMetrics()
wer = metrics.calculate_metrics_batched(all_predictions, all_references)


model_name = model_name.split("/")[-1]
# Save the predictions and ground truth text
with open(f"predictions_{model_name}_{test_name}.txt", "w") as f:
    f.write("\n".join(all_predictions))

with open(f"references_{model_name}_{test_name}.txt", "w") as f:
    f.write("\n".join(all_references))