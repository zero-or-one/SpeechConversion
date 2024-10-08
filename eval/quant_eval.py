import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, DatasetDict, Dataset
from datasets import Audio
import evaluate
from tqdm import tqdm
#from openai import OpenAI
from pydub import AudioSegment
from time import time
from stt import STT
import re

#client = OpenAI(api_key="sk-proj-CG3yXG41ZoKbweKmj3jET3BlbkFJDxj1S2Cazl7b0ATARbLy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the dataset
test_name = 'test'
dataset_path = f'./{test_name}.json'
dataset = load_dataset('json', data_files=dataset_path)


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


# Load the model and processor

#model_path = "VC-01-22-4.30-medium"
model_path = "whisper_turbo_adapter"
model_name = model_path

model = STT(model_name, verbose=False)



all_predictions = []
all_llm_predictions = []
all_references = []

start = time()
for batch in tqdm(dataset['train']):
    # Load the audio file
    audio_path = batch["audio"]["path"]
    reference = batch["transcription"]
    prediction = model.transcribe(audio_path)
    
    all_predictions.append(prediction)
    #all_llm_predictions.append(llm_predtion)
    all_references.append(reference)

print("Average time per sample: ", (time() - start) / len(all_references))

# Save the predictions and ground truth text
with open(f"predictions_{model_name}.txt", "w") as f:
    f.write("\n".join(all_predictions))

with open(f"references_{model_name}.txt", "w") as f:
    f.write("\n".join(all_references))

# Evaluate the model
metric = evaluate.load("wer")
metric_cer = evaluate.load("cer")


sub_references = [re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', ref.replace('\n', '')).strip() for ref in all_references]
sub_predictions = [re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', pred.replace('\n', '')).strip() for pred in all_predictions]


wer = 100 * metric.compute(predictions=sub_predictions, references=sub_references)
cer = 100 * metric_cer.compute(predictions=sub_predictions, references=sub_references)

print(f"WER: {wer:.2f}%")
print(f"CER: {cer:.2f}%")
