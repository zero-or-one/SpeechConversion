from dotenv import load_dotenv 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'


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

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import json
from pydub import AudioSegment
from time import sleep, time
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from collections import Counter

import math
#print('Waiting for 7 hours')
#sleep(7 * 60 * 60)

#os.environ["RANK"]="2"

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))



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
atypical_voice = DatasetDict()
train_dataset_path = 'train.json'
test_dataset_path = 'test.json'

train_data = load_dataset('json', data_files=train_dataset_path)
test_data = load_dataset('json', data_files=test_dataset_path)



print(f"Train set size: {len(train_data['train'])}")
print(f"Test set size: {len(test_data['train'])}")

atypical_voice = DatasetDict({
    "train": train_data['train'],
    "test": test_data['train'],
})

# Calculate durations
train_duration = calculate_dataset_duration(train_data)
test_duration = calculate_dataset_duration(test_data)

# Convert durations to hours, minutes, and seconds
def format_duration(duration_seconds):
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}-{int(minutes):02d}-{seconds:.2f}"

train_duration_str = format_duration(train_duration)
print(f"Train set duration: {train_duration_str}")

print(f"Test set duration: {format_duration(test_duration)}")

total_duration = train_duration + test_duration
print(f"Total dataset duration: {format_duration(total_duration)}")


#model_name = 'jiwon65/whisper-small_korean-zeroth'
#model_name = 'openai/whisper-large-v3'
#model_name = 'VC-01-22-0.77-medium'
model_name = 'VC-01-22-4.30-large-lora-4'
processor_name = model_name
repo_name = f'VC-large-adapter'

feature_extractor = WhisperFeatureExtractor.from_pretrained(processor_name)
tokenizer = WhisperTokenizer.from_pretrained(processor_name, task="transcribe", language='ko')
processor = WhisperProcessor.from_pretrained(processor_name, task="transcribe", language='ko')

# Prepare the dataset
print('sample', atypical_voice["train"][0])

# I don't need this, but let's play safe
atypical_voice = atypical_voice.cast_column("audio", Audio(sampling_rate=16000))
atypical_voice = atypical_voice.rename_column('sentence', 'transcription')

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch


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

if not os.path.exists(f'token_weights_{train_duration_str}-large.json'):
    token_weights = calculate_token_weights(atypical_voice, tokenizer)

    # Save token weights to a JSON file
    with open(f'token_weights_{train_duration_str}-large.json', 'w') as f:
        json.dump(token_weights, f)

    print(f"Token weights have been calculated and saved to 'token_weights_{train_duration_str}.json'")

atypical_voice = atypical_voice.map(prepare_dataset, remove_columns=atypical_voice.column_names["train"], num_proc=1)


# Feature Adapter
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
            self.layer1 = nn.Linear(input_size, hidden_size, bias=False)
            self.layer2 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.layer3 = nn.Linear(hidden_size, output_size, bias=False)
        # Residual connections and initialization with small values
        self.relu = nn.ReLU()
        self.initialize_weights()
        self.prev_weights = [layer.weight.data.clone() for layer in [self.layer1, self.layer2, self.layer3]]

    def initialize_weights(self):
        # Small random values for initialization
        for layer in [self.layer1, self.layer2, self.layer3]:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv1d):
                nn.init.normal_(layer.weight, mean=0.0, std=1e-2)
                #nn.init.constant_(layer.weight, 0.0)

    def forward(self, x):
        x = x.transpose(1, 2)
        residual = x
        #print('x', x.shape)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
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
    
    def generate(self, **kwargs):
        kwargs.pop('labels', None)
        return super(ModifiedWhisperModel, self).generate(**kwargs)
    

# Load pre-trained model with LoRA layers
base_model = ModifiedWhisperModel.from_pretrained(model_name, use_cnn=False)


# claude
gen_config = GenerationConfig.from_model_config(base_model.config)
gen_config.task = "transcribe"
gen_config.language = "ko"
gen_config.task_to_id = {
    "transcribe": 50359,
    "translate": 50358
  }
# Clear forced_decoder_ids and suppress_tokens
gen_config.forced_decoder_ids = None
gen_config.suppress_tokens = []
gen_config.language = "ko"
language_to_id_map = {
    "en": 50259,  # English
    "ko": 50263,  # Korean
    # Add more languages as needed
}
gen_config.lang_to_id = language_to_id_map
base_model.generation_config = gen_config

adapter_name = model_name
adapter_path = f'{adapter_name}'
peft_config = PeftConfig.from_pretrained(adapter_path)
model = PeftModel(base_model, peft_config)

'''
# LoRA config
lora_config = LoraConfig(
    r=16,  # Dimension of the low-rank approximation
    lora_alpha=32,  # LoRA scaling factor
    lora_dropout=0.05,  # Dropout for LoRA layers
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
)

# Load pre-trained model with LoRA layers
model = get_peft_model(base_model, lora_config)
'''

# Freeze all model parameters except LoRA layers and FeatureAdapter layers
for param in model.parameters():
    param.requires_grad = False
# Make LoRA layers trainable
for name, param in model.named_parameters():
    if 'lora' in name:  # LoRA layers are automatically named with 'lora'
        param.requires_grad = True

# Make Feature Adapter layers trainable as well
for name, param in model.preprocessor.named_parameters():
    param.requires_grad = True

# make output layer trainable as well
#for param in model.proj_out.parameters():
#    param.requires_grad = True

# Ensure that the LoRA layers and the Feature Adapter layers are trainable
trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
print(f"Trainable parameters: {trainable_params}")



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
    learning_rate=1e-5,
    num_train_epochs=60,  # Maximum number of epochs
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="epoch",
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    generation_max_length=225,
    save_strategy="epoch",
    save_total_limit=1,  # Keep only the last 2 checkpoints
    logging_strategy="epoch",
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # Use validation loss as metric
    greater_is_better=False,
    push_to_hub=True,
    logging_first_step=True,
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    #weight_decay=1e-2,
    #warmup_steps=100,
)

class HalfEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if (state.epoch + 1) % 0.5 == 0:
            control.should_save = True
            control.should_evaluate = True
            control.should_log = True

# overwrite new class
class FeatureAdapterMonitorCallback(TrainerCallback):
    def __init__(self, log_steps=1):
        self.log_steps = log_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_steps == 0:
            weight_changes = model.preprocessor.check_weight_update()
            print(f"Step {state.global_step}: Feature Adapter weight changes: {weight_changes}")

# Modified CustomWhisperTrainer
class CustomWhisperTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(f'token_weights_{train_duration_str}.json', 'r') as f:
            token_weights = json.load(f)

        weights_tensor = torch.ones(51866)
        for token_index, weight in token_weights.items():
            weights_tensor[int(token_index)] = weight
        self.class_weights = weights_tensor.to(self.args.device)

        self.best_cer = float('inf')
        self.best_wer = float('inf')
        
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        
        # Clip gradients for the preprocessor
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        return loss
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean')
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs):
        super().log(logs)
        if 'eval_cer' in logs and logs['eval_cer'] < self.best_cer:
            self.best_cer = logs['eval_cer']
        if 'eval_wer' in logs and logs['eval_wer'] < self.best_wer:
            self.best_wer = logs['eval_wer']

    def get_best_metrics(self):
        return {
            "best_cer": self.best_cer,
            "best_wer": self.best_wer
        }

    
trainer = CustomWhisperTrainer(
    args=training_args,
    model=model,
    train_dataset=atypical_voice["train"],
    eval_dataset=atypical_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[HalfEpochCallback()] #, FeatureAdapterMonitorCallback()],
)
processor.save_pretrained(training_args.output_dir)

train_start = time()
print(model.config)
print('Evaluating the model before training')
trainer.evaluate()
#exit()

print('Training the model')
trainer.train()

# Get the best metrics
best_metrics = trainer.get_best_metrics()
print(f"Best CER: {best_metrics['best_cer']}")
print(f"Best WER: {best_metrics['best_wer']}")

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