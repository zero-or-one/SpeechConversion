from datasets import load_dataset, concatenate_datasets
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor
import torch
from evaluate import load
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Load the Korean FLEURS dataset
dataset = load_dataset("google/fleurs", "ko_kr")
dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the Whisper model and processor

model_name = "openai/whisper-large-v3"
processor_name = model_name
#model_name = "spow12/whisper-medium-zeroth_korean" #"openai/whisper-large-v3"  # You can change this to other variants
#processor_name = "openai/whisper-medium"

processor = WhisperProcessor.from_pretrained(processor_name, language="ko")
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
feature_extractor = WhisperFeatureExtractor.from_pretrained(processor_name, language="ko")

# Load the WER metric
wer_metric = load("wer")
cer_metric = load("cer")

# Function to pad input features
def pad_input_features(input_features, target_length=3000):
    current_length = input_features.shape[-1]
    if current_length < target_length:
        padding_length = target_length - current_length
        padding = torch.zeros((input_features.shape[0], input_features.shape[1], padding_length))
        return torch.cat((input_features, padding), dim=-1)
    return input_features


# Evaluation loop
model.eval()
all_predictions = []
all_references = []

batch_size = 32
for i in range(0, len(dataset), batch_size):
    batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
    
    # Prepare inputs
    audio_samples = [sample['audio']['array'] for sample in batch]
    sampling_rate = batch[0]['audio']['sampling_rate']  # Assuming all samples have the same sampling rate
    
    # Extract features
    inputs = feature_extractor(audio_samples, sampling_rate=sampling_rate, return_tensors="pt")
    
    # Move input features to the correct device and pad
    inputs.input_features = pad_input_features(inputs.input_features.to(device))
    
    with torch.no_grad():
        generated_ids = model.generate(inputs=inputs.input_features)
    
    # Decode predictions
    predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    references = [sample['transcription'] for sample in batch]
    
    all_predictions.extend(predictions)
    all_references.extend(references)
    
    print(f"Processed batch {i//batch_size + 1}/{(len(dataset)-1)//batch_size + 1}")

# Compute WER for the entire dataset
wer = wer_metric.compute(predictions=all_predictions, references=all_references)
cer = cer_metric.compute(predictions=all_predictions, references=all_references)

print(model_name)
print(f"Average WER: {wer:.4f}")
print(f"Average CER: {cer:.4f}")