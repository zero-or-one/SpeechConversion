from dotenv import load_dotenv 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


from huggingface_hub import login
import random
from tqdm import tqdm
random.seed(42)
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from datasets import Audio

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import GenerationConfig
from transformers import TrainerCallback

import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import json
from pydub import AudioSegment
from time import sleep, time
import io
from collections import Counter

import math
#print('Waiting for 7 hours')
#sleep(7 * 60 * 60)


load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))


# /START SETUP
model_name = 'openai/whisper-large-v3-turbo'
processor_name = model_name
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

# Load the dataset

dataset = load_dataset("Bingsu/zeroth-korean")

# Split the test set into validation and test subsets
test_valid_split = dataset["test"].train_test_split(test_size=0.5)
dataset["validation"] = test_valid_split["train"]
dataset["test"] = test_valid_split["test"]

# split test into 2 and add to train
#test_data_split = test_data['train'].train_test_split(test_size=0.5, seed=42)
#train_data['train'] = concatenate_datasets([train_data['train'], test_data_split['test']])
#test_data['train'] = test_data_split['train']



print(f"Train set size: {len(dataset['train'])}")
print(f"Valid set size: {len(dataset['validation'])}")
print(f"Test set size: {len(dataset['test'])}")



# Convert durations to hours, minutes, and seconds
def format_duration(duration_seconds):
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}-{int(minutes):02d}-{seconds:.2f}"


repo_name = f'whisper-large-v3-turbo-korean'

feature_extractor = WhisperFeatureExtractor.from_pretrained(processor_name)
tokenizer = WhisperTokenizer.from_pretrained(processor_name, task="transcribe", language='ko')
processor = WhisperProcessor.from_pretrained(processor_name, task="transcribe", language='ko')


def prepare_dataset(batch):
    # Compute log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"]).input_features[0]

    # Encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

def calculate_duration(dataset):
    total_duration = 0
    for item in dataset:
        audio = AudioSegment.from_file(io.BytesIO(item['audio']['array']), format="raw", frame_rate=16000, channels=1, sample_width=2)
        total_duration += len(audio) / 1000.0  # duration in seconds
    hours = total_duration // 3600
    minutes = (total_duration % 3600) // 60
    return f"{int(hours)} hours and {int(minutes)} minutes"

print("------ Calculating dataset durations ------")
print(f"Train set duration: {calculate_duration(dataset['train'])}")
print(f"Validation set duration: {calculate_duration(dataset['validation'])}")
print(f"Test set duration: {calculate_duration(dataset['test'])}")
print("-------------------------------------------\n\n\n")

# Apply dataset processing
dataset = dataset.map(prepare_dataset, remove_columns=["audio", "text"], num_proc=1)



# Training and Evaluation
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
model.config.use_cache = False
model.config.suppress_tokens = [processor.tokenizer.encode(" ", add_special_tokens=False)[0]]

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)




wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Log shapes and types
    #print(f"pred_ids shape: {pred_ids.shape}, type: {type(pred_ids)}")
    #print(f"label_ids shape: {label_ids.shape}, type: {type(label_ids)}")
    
    # Handle Whisper model output
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
        #print("Whisper model detected, using first element of tuple")
    
    # Replace -100 with pad_token_id
    #original_label_ids = label_ids.copy()
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    #replaced_count = np.sum(original_label_ids != label_ids)
    #print(f"Replaced {replaced_count} instances of -100 with pad_token_id")
    
    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Log sample of decoded strings
    #print("Sample of decoded predictions:")
    #print(pred_str[:2])
    #print("Sample of decoded labels:")
    #print(label_str[:2])
    
    # Compute WER
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}


training_args = Seq2SeqTrainingArguments(
    output_dir=repo_name,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=30,  # Maximum number of epochs
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="epoch",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_strategy="epoch",
    save_total_limit=1,  # Keep only the last 2 checkpoints
    logging_strategy="epoch",
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_valid_loss",  # Use validation loss as metric
    greater_is_better=False,
    push_to_hub=True,
    logging_first_step=True,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    
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

        # Initialize variables to track best metrics
        self.best_eval_loss = float('inf')
        self.best_eval_cer = float('inf')
        self.best_eval_wer = float('inf')
    
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
    train_dataset=dataset["train"],
    eval_dataset={
        'valid': dataset["validation"],
        'test': dataset["test"]
    },
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[HalfEpochCallback()]
)
#processor.save_pretrained(training_args.output_dir)

checkpoint_dir = repo_name
latest_checkpoint = None

# Find the latest checkpoint folder
if os.path.exists(checkpoint_dir):
    checkpoints = [ckpt for ckpt in os.listdir(checkpoint_dir) if ckpt.startswith("checkpoint")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))

if latest_checkpoint:
    resume_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"Resuming from checkpoint: {resume_checkpoint_path}")
else:
    print("No checkpoint found. Training from scratch.")
    resume_checkpoint_path = None


train_start = time()
print(model.config)
#print('Evaluating the model before training')
#trainer.evaluate()
#exit()
print('Training the model')
trainer.train(resume_from_checkpoint=resume_checkpoint_path)

# Get the best metrics
best_metrics = trainer.get_best_metrics()
print(best_metrics)

train_end = time()
time_str = format_duration(train_end - train_start)
print(f"Training took {time_str} seconds")

kwargs = {
    'repo_id': repo_name,
    "dataset_tags": "korean handicap",
    "dataset": "Dysarthria voice recognition Korean",  # a 'pretty' name for the training dataset
    "dataset_args": "config: full, split: train",
    "language": "ko",
    "model_name": "Whisper Small Speech Conversion Finetuned",  # a 'pretty' name for our model
    "finetuned_from": model_name,
    "tasks": "automatic-speech-recognition",
}

model.push_to_hub(**kwargs)
