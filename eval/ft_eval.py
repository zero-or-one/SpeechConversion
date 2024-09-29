import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, DatasetDict, Dataset
from datasets import Audio
import evaluate
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the test indices from the file
with open('../train/test_indices.txt', 'r') as f:
    test_indices = [int(x) for x in f.read().strip().split(',')]
    #print(test_indices)

# Load the dataset
dataset_path = '/home/sabina/korean_data/13_CUJ/dataset.json'
dataset = load_dataset('json', data_files=dataset_path)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.rename_column('sentence', 'transcription')

# Create the validation dataset from the test indices
val_dataset = dataset['train'].select(test_indices)

# Load the model and processor
model_name = 'whisper-small-voice-conversion-korean-10min'
model_path = os.path.join('/home/sabina/SpeechConversion/train/', model_name)
#model_path = 'jiwon65/whisper-small_korean-zeroth'
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(model_path, task="transcribe", language='ko')
processor = WhisperProcessor.from_pretrained(model_path, task="transcribe", language='ko')
model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)


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

# Evaluate the model
metric = evaluate.load("wer")
metric_cer = evaluate.load("cer")

all_predictions = []
all_references = []

for batch in tqdm(val_dataset):
    input_features = batch["input_features"]
    input_features = torch.tensor(input_features).unsqueeze(0).to(device)
    #print(input_features.shape)
    input_ids = model.generate(input_features, max_length=225, num_beams=5, early_stopping=True)[0]
    prediction = processor.decode(input_ids, skip_special_tokens=True)
    reference = processor.decode(batch["labels"], skip_special_tokens=True)
    all_predictions.append(prediction)
    all_references.append(reference)

wer = 100 * metric.compute(predictions=all_predictions, references=all_references)
cer = 100 * metric_cer.compute(predictions=all_predictions, references=all_references)

print(f"WER: {wer:.2f}%")
print(f"CER: {cer:.2f}%")

# Save the predictions and ground truth text
with open("predictions.txt", "w") as f:
    f.write("\n".join(all_predictions))

with open("references.txt", "w") as f:
    f.write("\n".join(all_references))