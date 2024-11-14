import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
from datasets import load_dataset, Audio
from tqdm import tqdm
from pydub import AudioSegment
from time import time
from dotenv import load_dotenv
from faster_whisper import WhisperModel  # Import Faster Whisper
from metrics import ASRMetrics

load_dotenv()

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"


# Dataset Setup
test_name = 'test'
speaker = 'IMJ_Woman'
dataset_path = os.path.join('dataset', speaker, f"{test_name}.json")
dataset = load_dataset('json', data_files=dataset_path)

# Model and Processor Setup
model_path = 'VC-20dB-IMJ_Woman-01-27-58.20'


# Initialize the Faster Whisper model
model = WhisperModel(model_path, device=device, compute_type="float16" if device == "cuda" else "int8"
                    )

# Function to calculate total duration of the dataset
def calculate_dataset_duration(dataset):
    total_duration = 0
    for item in dataset['train']:
        audio_path = item['audio']['path']
        try:
            audio = AudioSegment.from_file(audio_path)
            total_duration += len(audio) / 1000.0  # Convert milliseconds to seconds
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    return total_duration

def format_duration(duration_seconds):
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}-{int(minutes):02d}-{seconds:.2f}"

# Calculate dataset duration
test_duration = calculate_dataset_duration(dataset)
print(f"Total duration of the test dataset: {format_duration(test_duration)}")

# Process and Transcribe Audio using Faster Whisper
def process_and_transcribe(audio_path):
    segments, _ = model.transcribe(audio_path, beam_size=5, best_of=5)
    
    transcription = ""
    for segment in segments:
        transcription += segment.text.strip() + " "
    
    return transcription.strip()

# Prepare for evaluation
all_predictions, all_references = [], []
total_time = 0 

for sample in tqdm(dataset['train']):
    audio_path = sample['audio']['path']


    start_time = time()
    prediction = process_and_transcribe(audio_path)
    total_time += time() - start_time

    all_predictions.append(prediction)
    all_references.append(sample['sentence'])

# Evaluation
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Average processing time per sample: {total_time / len(dataset['train']):.2f} seconds")

# Calculate WER using ASRMetrics
metrics = ASRMetrics()
metrics.calculate_metrics_batched(all_predictions, all_references)

'''
# Save predictions and references
model_name_cleaned = model_path.split("/")[-1]
with open(f"predictions_{model_name_cleaned}_{test_name}.txt", "w") as f:
    f.write("\n".join(all_predictions))

with open(f"references_{model_name_cleaned}_{test_name}.txt", "w") as f:
    f.write("\n".join(all_references))
'''