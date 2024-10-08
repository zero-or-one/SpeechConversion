import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, DatasetDict, Dataset
from datasets import Audio
import evaluate
from tqdm import tqdm
#from openai import OpenAI
from pydub import AudioSegment
from time import time
import re
#client = OpenAI(api_key="")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the dataset
test_name = 'test'
dataset_path = f'./{test_name}.json'
dataset = load_dataset('json', data_files=dataset_path)

# Split the test dataset into two halves (only for larger training data)
test_data_split = dataset['train'].train_test_split(test_size=0.5, seed=42)
dataset['train'] = test_data_split['test']

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

# Load the model and processor

#model_path = "VC-01-22-4.30-medium"
#model_path = "whisper_large-feature-merged"
#model_name = model_path

model_name = '_'
#model_path = 'seastar105/whisper-medium-ko-zeroth'
#model_path = "openai/whisper-large-v3"
#model_path = 'VC-01-22-4.30-turbo'
model_path = 'VC-01-32-22.37-turbo'

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(model_path, task="transcribe", language='ko')
processor = WhisperProcessor.from_pretrained(model_path, task="transcribe", language='ko')
model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
model = model.eval()

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # compute log-Mel input features from input audio array
    input_features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_features"] = torch.tensor(input_features)
    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

column_names = val_dataset.column_names
val_dataset = val_dataset.map(prepare_dataset, remove_columns=column_names, num_proc=1)



'''
system_prompt = "You are a helpful assistant for the company. Your task is to correct only spelling and minor grammatical errors in the transcribed Korean text. Ensure that the original meaning is preserved and avoid changing the words or phrasing unnecessarily. Focus on improving accuracy while making minimal alterations."
def generate_corrected_transcript(temperature, system_prompt, transcription):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content
'''


all_predictions = []
all_llm_predictions = []
all_references = []

start = time()
for batch in tqdm(val_dataset):
    input_features = batch["input_features"]
    input_features = torch.tensor(input_features).unsqueeze(0).to(device)
    #print(input_features.shape)
    input_ids = model.generate(input_features, max_length=225, num_beams=5, early_stopping=True)[0]
    prediction = processor.decode(input_ids, skip_special_tokens=True)
    reference = processor.decode(batch["labels"], skip_special_tokens=True)
    #llm_predtion = generate_corrected_transcript(0, system_prompt, prediction)
    
    all_predictions.append(prediction)
    #all_llm_predictions.append(llm_predtion)
    all_references.append(reference)

print("Average time per sample: ", (time() - start) / len(val_dataset))

# Save the predictions and ground truth text
with open(f"predictions_{model_name}.txt", "w") as f:
    f.write("\n".join(all_predictions))

with open(f"references_{model_name}.txt", "w") as f:
    f.write("\n".join(all_references))

# Evaluate the model
metric = evaluate.load("wer")
metric_cer = evaluate.load("cer")

print("With punctuation")
wer = 100 * metric.compute(predictions=all_predictions, references=all_references)
cer = 100 * metric_cer.compute(predictions=all_predictions, references=all_references)
print(f"WER: {wer:.2f}%")
print(f"CER: {cer:.2f}%")


print("Without punctuation")
sub_references = [re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', ref.replace('\n', '')).strip() for ref in all_references]
sub_predictions = [re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', pred.replace('\n', '')).strip() for pred in all_predictions]


wer = 100 * metric.compute(predictions=sub_predictions, references=sub_references)
cer = 100 * metric_cer.compute(predictions=sub_predictions, references=sub_references)

print(f"WER: {wer:.2f}%")
print(f"CER: {cer:.2f}%")
'''
llm_wer = 100 * metric.compute(predictions=all_llm_predictions, references=all_references)
llm_cer = 100 * metric_cer.compute(predictions=all_llm_predictions, references=all_references)

print(f"LLM WER: {llm_wer:.2f}%")
print(f"LLM CER: {llm_cer:.2f}%")
'''