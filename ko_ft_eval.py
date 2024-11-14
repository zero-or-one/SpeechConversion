import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import json
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import evaluate
from tqdm import tqdm
from pydub import AudioSegment
from time import time
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the new dataset
dataset = load_dataset("Bingsu/zeroth-korean")

# Split the 'test' set into 'validation' and 'test' subsets
test_valid_split = dataset["test"].train_test_split(test_size=0.5, seed=42)
dataset["validation"] = test_valid_split["train"]
dataset["test"] = test_valid_split["test"]

# Define paths and model information
#model_path = 'whisper-large-v3-turbo-korean'
model_path = 'openai/whisper-large-v3-turbo'
model_name = model_path

# Load the Whisper model, tokenizer, and processor
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(model_path, task="transcribe", language='ko')
processor = WhisperProcessor.from_pretrained(model_path, task="transcribe", language='ko')
model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device).eval()

# Cast and prepare audio column
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.rename_column('text', 'transcription')

val_dataset = dataset["test"]



# Process and transcribe function
def process_and_transcribe(audio_sample):
    input_features = feature_extractor(audio_sample["array"], sampling_rate=audio_sample["sampling_rate"]).input_features[0]
    input_features = torch.tensor(input_features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        input_ids = model.generate(input_features, max_length=225, num_beams=5, early_stopping=True)[0]
    
    prediction = processor.decode(input_ids, skip_special_tokens=True)
    return prediction

# Evaluate the model
all_predictions = []
all_references = []
total_time = 0
total_audio_duration = 0

for sample in tqdm(val_dataset):
    audio_duration = len(sample['audio']['array']) / sample['audio']['sampling_rate']
    total_audio_duration += audio_duration

    start_time = time()
    prediction = process_and_transcribe(sample['audio'])
    end_time = time()
    
    total_time += end_time - start_time

    all_predictions.append(prediction)
    all_references.append(sample['transcription'])

print(f"Total audio duration: {total_audio_duration:.2f} seconds")
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Real-time factor: {total_time / total_audio_duration:.2f}")
print(f"Average processing time: {total_time / len(val_dataset):.2f} seconds")

# Calculate WER and CER
metric_wer = evaluate.load("wer")
metric_cer = evaluate.load("cer")

wer = 100 * metric_wer.compute(predictions=all_predictions, references=all_references)
cer = 100 * metric_cer.compute(predictions=all_predictions, references=all_references)
print(f"WER: {wer:.2f}%")
print(f"CER: {cer:.2f}%")

# Remove punctuation for secondary evaluation
def remove_punctuation(text):
    return re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', text).strip()

sub_references = [remove_punctuation(ref) for ref in all_references]
sub_predictions = [remove_punctuation(pred) for pred in all_predictions]

wer = 100 * metric_wer.compute(predictions=sub_predictions, references=sub_references)
cer = 100 * metric_cer.compute(predictions=sub_predictions, references=sub_references)
print(f"WER (no punctuation): {wer:.2f}%")
print(f"CER (no punctuation): {cer:.2f}%")

# Save predictions and references
model_name_clean = model_name.split("/")[-1]
with open(f"predictions_{model_name_clean}_validation.txt", "w") as f:
    f.write("\n".join(all_predictions))

with open(f"references_{model_name_clean}_validation.txt", "w") as f:
    f.write("\n".join(all_references))
