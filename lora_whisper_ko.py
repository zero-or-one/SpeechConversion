from dotenv import load_dotenv 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from huggingface_hub import login
import random
from tqdm import tqdm
random.seed(42)
from datasets import load_dataset, DatasetDict, Dataset
from datasets import Audio

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import GenerationConfig
from transformers import TrainerCallback
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

import json
from pydub import AudioSegment
from time import time, sleep

from collections import Counter
import math

# LoRA imports
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import TaskType

# Wait for 4.5 hours
#print("Waiting for 4.5 hours")
#sleep(4.5*60*60)

# Load environment variables
load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Function to calculate the dataset duration
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


# /START SETUP
speaker = 'KGH_Woman_30s'
target_dataset = os.path.join('dataset', speaker)
model_name = 'openai/whisper-large-v3'
processor_name = model_name
repo_name = f"VC-{speaker}-large-lora"
# /END SETUP 

# Load the dataset
train_dataset_path = os.path.join(target_dataset, 'train.json')
valid_dataset_path = os.path.join(target_dataset, 'valid.json')
test_dataset_path = os.path.join(target_dataset, 'test.json')

train_data = load_dataset('json', data_files=train_dataset_path)
valid_data = load_dataset('json', data_files=valid_dataset_path)
test_data = load_dataset('json', data_files=test_dataset_path)


atypical_voice = DatasetDict({
    "train": train_data['train'],
    "test": test_data['train'],
    "valid": valid_data['train']
})


# Calculate durations
train_duration = calculate_dataset_duration(train_data)
valid_duraction = calculate_dataset_duration(valid_data)
test_duration = calculate_dataset_duration(test_data)

# Convert durations to hours, minutes, and seconds
def format_duration(duration_seconds):
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}-{int(minutes):02d}-{seconds:.2f}"

train_duration_str = format_duration(train_duration)
print(f"Train set duration: {train_duration_str}")
print(f"Valid set duration: {format_duration(valid_duraction)}")
print(f"Test set duration: {format_duration(test_duration)}")

total_duration = train_duration + test_duration + valid_duraction
print(f"Total dataset duration: {format_duration(total_duration)}")


feature_extractor = WhisperFeatureExtractor.from_pretrained(processor_name)
tokenizer = WhisperTokenizer.from_pretrained(processor_name, task="transcribe", language='ko')
processor = WhisperProcessor.from_pretrained(processor_name, task="transcribe", language='ko')

atypical_voice = atypical_voice.cast_column("audio", Audio(sampling_rate=16000))
atypical_voice = atypical_voice.rename_column('sentence', 'transcription')


def calculate_token_weights(dataset, tokenizer):
    # Tokenize all sentences in the dataset
    print('Calculating token weights')
    all_tokens = []
    for item in dataset['train']:
        tokens = tokenizer.encode(item['transcription'])
        all_tokens.extend(tokens)

    # Count token frequencies
    token_counts = Counter(all_tokens)

    # Calculate total number of tokens
    total_tokens = len(all_tokens)

    # Calculate probabilities and weights
    token_weights = {}
    for token, count in token_counts.items():
        p_token = count / total_tokens
        weight = 1 / math.sqrt(p_token)
        token_weights[str(token)] = weight

    return token_weights

if not os.path.exists(f'stuff/token_weights_{train_duration_str}-turbo.json'):
    token_weights = calculate_token_weights(atypical_voice, tokenizer)

    # Save token weights to a JSON file
    with open(f'stuff/token_weights_{train_duration_str}-turbo.json', 'w') as f:
        json.dump(token_weights, f)

    print(f"Token weights have been calculated and saved to 'token_weights_{train_duration_str}-turbo.json'")


# Prepare dataset
def prepare_dataset(batch):
    # Extract audio features, assuming 16kHz sample rate
    audio = batch["audio"]

    # Compute log-Mel input features
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # Encode transcription text
    batch["labels"] = tokenizer(batch["transcription"]).input_ids

    return batch

# Apply the mapping
atypical_voice = atypical_voice.map(prepare_dataset, remove_columns=atypical_voice.column_names["train"], num_proc=1)

# Load Whisper model and set up LoRA
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# claude
gen_config = GenerationConfig.from_model_config(model.config)
gen_config.task = "transcribe"
#gen_config.language = "en"
gen_config.task_to_id = {
    "transcribe": 50359,
    "translate": 50358
  }

# Clear forced_decoder_ids and suppress_tokens
gen_config.forced_decoder_ids = None
gen_config.suppress_tokens = []

# Assign the generation config to the model
model.generation_config = gen_config


model = prepare_model_for_kbit_training(model)  # Make model ready for LoRA with int8 optimization

# LoRA configuration
lora_config = LoraConfig(
    r=32, 
    lora_alpha=64, 
    target_modules=["q_proj", "v_proj", "o_proj", "k_proj"],
    lora_dropout=0.05, 
    bias="none"
)

# Wrap the Whisper model with LoRA
model = get_peft_model(model, lora_config)
print("LoRA model initialized.")

# Data collator for speech-to-seq2seq tasks
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract audio input features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Process tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Mask padding tokens in labels
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove bos token if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# Metric calculation
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=repo_name,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=20,  # Adjusted for LoRA training
    fp16=True,
    evaluation_strategy="epoch",
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    save_strategy="epoch",
    save_total_limit=1,
    logging_strategy="epoch",
    report_to=["tensorboard"],
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_valid_loss",
    greater_is_better=False,
)

class HalfEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if (state.epoch + 1) % 0.5 == 0:
            control.should_save = True
            control.should_evaluate = True
            control.should_log = True

# overwrite new class
class CustomWhisperTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('Device:', self.args.device)
        # load from token_weights.json
        with open(f'stuff/token_weights_{train_duration_str}-turbo.json', 'r') as f:
            token_weights = json.load(f)

        weights_tensor = torch.ones(51866)  # Initialize a tensor of ones
        for token_index, weight in token_weights.items():
            #print(f"Token index: {token_index}, weight: {weight}")
            weights_tensor[int(token_index)] = weight  # Update the tensor with the weight
        self.class_weights = weights_tensor.to(self.args.device)  # Move tensor to the device

        # Initialize variables to track best metrics
        self.best_eval_loss = float('inf')
        self.best_eval_cer = float('inf')
        self.best_eval_wer = float('inf')

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Calculate weighted cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean')
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs):
        super().log(logs)
        # Update best metrics for validation set
        if 'eval_loss' in logs and logs['eval_loss'] < self.best_eval_loss:
            self.best_eval_loss = logs['eval_loss']
        if 'eval_cer' in logs and logs['eval_cer'] < self.best_eval_cer:
            self.best_eval_cer = logs['eval_cer']
        if 'eval_wer' in logs and logs['eval_wer'] < self.best_eval_wer:
            self.best_eval_wer = logs['eval_wer']

    def get_best_metrics(self):
        return {
            "best_eval_loss": self.best_eval_loss,
            "best_eval_cer": self.best_eval_cer,
            "best_eval_wer": self.best_eval_wer
        }

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Evaluate on validation set
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Evaluate on test set
        test_results = super().evaluate(self.eval_dataset['test'], ignore_keys, metric_key_prefix="test")
        
        # Combine results
        combined_results = {**eval_results, **test_results}
        return combined_results
    
trainer = CustomWhisperTrainer(
    args=training_args,
    model=model,
    train_dataset=atypical_voice["train"],
    eval_dataset={
        'valid': atypical_voice["valid"],
        'test': atypical_voice["test"]
    },
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[HalfEpochCallback()]
)

# Start training
train_start = time()
print("Training the model with LoRA")
trainer.train()

# Save the model and processor
model.push_to_hub("whisper-lora-korean")
processor.save_pretrained(training_args.output_dir)

train_end = time()
time_str = format_duration(train_end - train_start)
print(f"Training took {time_str} seconds")
