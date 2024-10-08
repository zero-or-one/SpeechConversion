import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import evaluate
from tqdm import tqdm
from pydub import AudioSegment
from peft import PeftModel, PeftConfig
from time import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the dataset
test_name = 'test'
dataset_path = f'{test_name}.json'
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

# Create the validation dataset from the test indices
val_dataset = dataset['train']

# Load the processor and tokenizer
#model_path = 'seastar105/whisper-medium-ko-zeroth'
model_path = 'openai/whisper-large-v3'
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(model_path, task="transcribe", language='ko')
processor = WhisperProcessor.from_pretrained(model_path, task="transcribe", language='ko')

# Load the base Whisper model (without LoRA) and apply the LoRA adapter
base_model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)

# Load the adapter
#adapter_name = 'whisper-lora-01-22-0.77/checkpoint-576'
adapter_name = 'VC-01-22-4.30-large-lora-4/checkpoint-2009'
#adapter_name = 'kresnik/openai-whisper-large-v3-LORA-korean'
adapter_path = f'{adapter_name}'
peft_config = PeftConfig.from_pretrained(adapter_path)
model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
model = model.eval()

def prepare_dataset(batch):
    # Load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # Compute log-Mel input features from input audio array
    input_features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_features"] = torch.tensor(input_features)
    # Encode target text to label ids
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

column_names = val_dataset.column_names
val_dataset = val_dataset.map(prepare_dataset, remove_columns=column_names, num_proc=1)

all_predictions = []
all_references = []

start = time()
for batch in tqdm(val_dataset):
    input_features = batch["input_features"]
    input_features = torch.tensor(input_features).unsqueeze(0).to(device)
    
    # Generate the transcription using the model with LoRA adapter
    input_ids = model.generate(input_features, max_length=225, num_beams=5, early_stopping=True)[0]
    prediction = processor.decode(input_ids, skip_special_tokens=True)
    reference = processor.decode(batch["labels"], skip_special_tokens=True)
    
    all_predictions.append(prediction)
    all_references.append(reference)

print("Average time per sample: ", (time() - start) / len(val_dataset))
# Evaluate the model
metric = evaluate.load("wer")
metric_cer = evaluate.load("cer")
wer = 100 * metric.compute(predictions=all_predictions, references=all_references)
cer = 100 * metric_cer.compute(predictions=all_predictions, references=all_references)

with open(f'predictions_{adapter_name.split("/")[0]}.txt', 'w') as f:
    for prediction in all_predictions:
        f.write(f'{prediction}\n')

with open(f'references_{adapter_name.split("/")[0]}.txt', 'w') as f:
    for reference in all_references:
        f.write(f'{reference}\n')


print(f"WER: {wer:.2f}%")
print(f"CER: {cer:.2f}%")
