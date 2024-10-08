import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import evaluate
from tqdm import tqdm
from pydub import AudioSegment
import torch.nn as nn
from peft import PeftConfig, PeftModel
import re
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
model_path = 'VC-turbo-adapter'
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(model_path, task="transcribe", language='ko')
processor = WhisperProcessor.from_pretrained(model_path, task="transcribe", language='ko')

# Load the base Whisper model (without LoRA) and apply the LoRA adapter
class FeatureAdapter(nn.Module):
    def __init__(self, input_size, output_size, use_cnn=False):
        super(FeatureAdapter, self).__init__()
        hidden_size = 64  # Hidden size for the FC layers
        # Optionally choose between FC or CNN
        if use_cnn:
            self.layer1 = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=3, padding=1, bias=False)
            self.layer2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)
            self.layer3 = nn.Conv1d(in_channels=128, out_channels=output_size, kernel_size=3, padding=1, bias=False)
        else:
            self.layer1 = nn.Linear(input_size, hidden_size, bias=True)
            self.layer2 = nn.Linear(hidden_size, hidden_size, bias=True)
            self.layer3 = nn.Linear(hidden_size, output_size, bias=True)
            #self.layer4 = nn.Linear(96, 96, bias=False)
            #self.layer5 = nn.Linear(96, output_size, bias=False)
        # Residual connections and initialization with small values
        self.relu = nn.ReLU()
        self.initialize_weights()
        self.prev_weights = [layer.weight.data.clone() for layer in [self.layer1, self.layer2, self.layer3]]

    def initialize_weights(self):
        # Small random values for initialization
        for layer in [self.layer1, self.layer2, self.layer3]:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv1d):
                nn.init.normal_(layer.weight, mean=0.0, std=1e-2)
                nn.init.constant_(layer.weight, 0.0)

    def forward(self, x):
        x = x.transpose(1, 2)
        residual = x
        #print('x', x.shape)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        
        #x = self.relu(self.layer4(x))
        #x = self.relu(self.layer5(x))
        
        x = x + residual
        x = x.transpose(1, 2)
        return x

    def check_weight_update(self):
        current_weights = [layer.weight.data for layer in [self.layer1, self.layer2, self.layer3]]
        device = current_weights[0].device
        weight_changes = [torch.abs(curr.to(device) - prev.to(device)).mean().item() for curr, prev in zip(current_weights, self.prev_weights)]
        self.prev_weights = [w.clone() for w in current_weights]
        return weight_changes

# Modified Whisper model
class ModifiedWhisperModel(WhisperForConditionalGeneration):
    def __init__(self, config, use_cnn=False):
        super(ModifiedWhisperModel, self).__init__(config)
        
        #for param in self.parameters():
        #    param.requires_grad = False
        
        self.preprocessor = FeatureAdapter(input_size=128, output_size=128, use_cnn=use_cnn)
    
    def forward(self, **kwargs):
        input_features = kwargs.get("input_features")
        if input_features is None:
            return super(ModifiedWhisperModel, self).forward(**kwargs)
        
        preprocessed_features = self.preprocessor(input_features)
        kwargs["input_features"] = preprocessed_features
        outputs = super(ModifiedWhisperModel, self).forward(**kwargs)
        return outputs
    
    def generate(self, input_features=None, **kwargs):
        if input_features is not None:
            preprocessed_features = self.preprocessor(input_features)
            kwargs["input_features"] = preprocessed_features
        return super().generate(**kwargs)


# Training and Evaluation
model = ModifiedWhisperModel.from_pretrained(model_path, use_cnn=False).to(device)
'''
# Load the adapter
#adapter_name = 'whisper-lora-01-22-0.77/checkpoint-576'
#adapter_name = 'VC-all-adapter/checkpoint-54'
adapter_name = model_path
adapter_path = f'{adapter_name}'
peft_config = PeftConfig.from_pretrained(adapter_path)
model = PeftModel.from_pretrained(model, adapter_path).to(device)
'''
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

model_path = model_path.split("/")[0]
with open(f'predictions_{model_path}.txt', 'w') as f:
    for prediction in all_predictions:
        f.write(f'{prediction}\n')

with open(f'references_{model_path}.txt', 'w') as f:
    for reference in all_references:
        f.write(f'{reference}\n')

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