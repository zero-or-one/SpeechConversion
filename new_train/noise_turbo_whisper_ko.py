from dotenv import load_dotenv 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


from huggingface_hub import login
import random
from tqdm import tqdm
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

from collections import Counter
from metrics import ASRMetrics
from scipy.ndimage import gaussian_filter
import math
#print('Waiting for 7 hours')
#sleep(7 * 60 * 60)
random.seed(42)  
np.random.seed(42)  
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))


# /START SETUP
speaker = 'JHJ_Woman_40s'
target_dataset = os.path.join('dataset', speaker)
model_name = 'openai/whisper-large-v3-turbo'
processor_name = model_name
SNR = 20

DELETION_RATE = 0.05
#THRESHOLD = 0.99
#WRAP_RATE = 0.1
WARP_FACTOR = 0.1
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
atypical_voice = DatasetDict()

train_dataset_path = os.path.join(target_dataset, 'train.json')
valid_dataset_path = os.path.join(target_dataset, 'valid.json')
test_dataset_path = os.path.join(target_dataset, 'test.json')

train_data = load_dataset('json', data_files=train_dataset_path)
valid_data = load_dataset('json', data_files=valid_dataset_path)
test_data = load_dataset('json', data_files=test_dataset_path)

# split test into 2 and add to train
#test_data_split = test_data['train'].train_test_split(test_size=0.5, seed=42)
#train_data['train'] = concatenate_datasets([train_data['train'], test_data_split['test']])
#test_data['train'] = test_data_split['train']



print(f"Train set size: {len(train_data['train'])}")
print(f"Valid set size: {len(valid_data['train'])}")
print(f"Test set size: {len(test_data['train'])}")

# loop through each data, if sample does not exists, remove it
def filter_valid_samples(dataset):
    """Filter out samples whose audio files don't exist."""
    filtered_data = [
        sample for sample in dataset 
        if os.path.exists(sample['audio']['path'])
    ]
    removed_count = len(dataset) - len(filtered_data)
    if removed_count > 0:
        print(f"{removed_count} samples were removed.")
    return Dataset.from_list(filtered_data)

# Filter samples in all datasets and reassign
train_data = DatasetDict({'train': filter_valid_samples(train_data['train'])})
valid_data = DatasetDict({'train': filter_valid_samples(valid_data['train'])})
test_data = DatasetDict({'train': filter_valid_samples(test_data['train'])})


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

repo_name = f'VC-{WARP_FACTOR}mw-{speaker}-{train_duration_str}'

feature_extractor = WhisperFeatureExtractor.from_pretrained(processor_name)
tokenizer = WhisperTokenizer.from_pretrained(processor_name, task="transcribe", language='ko')
processor = WhisperProcessor.from_pretrained(processor_name, task="transcribe", language='ko')

# Prepare the dataset
print('sample', atypical_voice["train"][0])

# I don't need this, but let's play safe
atypical_voice = atypical_voice.cast_column("audio", Audio(sampling_rate=16000))
atypical_voice = atypical_voice.rename_column('sentence', 'transcription')

def add_noise(audio, target_snr_db):
    """
    Add Gaussian noise to the audio signal based on the target SNR (dB).
    """
    audio_signal = np.array(audio["array"])
    signal_power = np.mean(audio_signal ** 2)
    
    # Calculate the target noise power based on the SNR formula
    target_noise_power = signal_power / (10 ** (target_snr_db / 10))
    
    # Generate Gaussian noise
    noise = np.random.normal(0, np.sqrt(target_noise_power), audio_signal.shape)
    
    # Add noise to the original signal
    noisy_audio = audio_signal + noise
    return noisy_audio

def apply_specaugment(Sxx, time_mask=0.05, freq_mask=0.05):
    """
    Apply time and frequency masking to a spectrogram (SpecAugment).
    """
    num_time_steps, num_freq_bins = Sxx.shape

    # Time masking
    time_mask_len = int(num_time_steps * time_mask)
    time_start = np.random.randint(0, num_time_steps - time_mask_len)
    Sxx[time_start:time_start + time_mask_len, :] = 0

    # Frequency masking
    freq_mask_len = int(num_freq_bins * freq_mask)
    freq_start = np.random.randint(0, num_freq_bins - freq_mask_len)
    Sxx[:, freq_start:freq_start + freq_mask_len] = 0

    return Sxx


def spectrogram_processing_dr(m, deletion_rate):
    """
    Augment a mel-spectrogram by randomly deleting frames with a probability inversely proportional
    to the frame-to-frame differences. Frames with smaller differences (less informative) have a higher
    chance of being deleted.

    Parameters:
    m (numpy.ndarray): Input spectrogram of shape (F, T).
    deletion_rate (float): Controls the overall rate of frame deletion (between 0 and 1).

    Returns:
    numpy.ndarray: Augmented spectrogram of shape (F, T').
    """
    # Transpose the matrix to have the shape (T, F)
    m = m.T
    T, F = m.shape

    # Compute frame-to-frame differences
    d = np.sum(np.abs(m[1:, :] - m[:-1, :]), axis=1)
    # Pad d with a zero at the beginning to make it length T
    d = np.concatenate(([0], d))

    # Normalize differences and invert to get deletion probabilities
    dmax = np.max(d)
    if dmax == 0:
        # Avoid division by zero if dmax is zero
        print("Warning: dmax is zero, returning unprocessed spectrogram")
        pd = np.ones(T) * deletion_rate
    else:
        # Invert differences: frames with smaller changes have higher deletion probability
        pd = (1 - (d / dmax)) * deletion_rate

    # Generate random numbers and determine which frames to delete
    random_values = np.random.rand(T)
    flags = random_values < pd

    # Ensure the first few frames are not deleted
    flags[:4] = False

    # Delete frames where flags are True
    m_processed = m[~flags, :]

    # Transpose back to original shape (F, T')
    m_processed = m_processed.T
    return m_processed

def spectrogram_processing_tr(m, threshold=0.1):
    """
    Augment a spectrogram by randomly deleting frames with a probability inversely proportional
    to the frame-to-frame differences. Frames with smaller differences have a higher chance of being deleted.

    Parameters:
    m (numpy.ndarray): Input spectrogram of shape (F, T).
    deletion_rate (float): Controls the overall rate of frame deletion (between 0 and 1).

    Returns:
    numpy.ndarray: Augmented spectrogram of shape (F, T).
    """
    # Transpose the matrix to have the shape (T, F)
    m = m.T
    T, F = m.shape

    # Compute frame-to-frame differences
    d = np.sum(np.abs(m[1:, :] - m[:-1, :]), axis=1)
    # Pad d with a zero at the beginning to make it length T
    d = np.concatenate(([0], d))

    # Normalize differences and invert to get deletion probabilities
    dmax = np.max(d)
    if dmax == 0:
        # Avoid division by zero if dmax is zero
        print("Warning: dmax is zero, returning unprocessed spectrogram")
        pd = np.ones(T)
    else:
        # Invert differences: frames with smaller changes have higher deletion probability
        pd = (1 - (d / dmax)) 

    # Generate random numbers and determine which frames to delete
    #random_values = np.random.rand(T)
    flags = (pd > threshold) #& (random_values < pd)

    # Ensure the first and last few frames are not deleted
    flags[:4] = False
    flags[-4:] = False

    # Mask frames where flags are True
    #m_processed = m.copy()
    #m_processed[flags, :] = 0
    m_processed = m[~flags, :]
    
    # Transpose back to original shape (F, T)
    m_processed = m_processed.T
    return m_processed

def spectrogram_processing_wr(m, warp_rate=0.1, max_warp=10):
    """
    Augment a spectrogram by applying random time warping to frames with a probability
    inversely proportional to the frame-to-frame differences. Frames with smaller differences
    are more likely to be warped.

    Parameters:
    m (numpy.ndarray): Input spectrogram of shape (F, T).
    warp_rate (float): Controls the overall rate of frame warping (between 0 and 1).
    max_warp (int): Maximum number of frames to shift during warping.

    Returns:
    numpy.ndarray: Time-warped spectrogram of shape (F, T).
    """
    # Transpose the matrix to have shape (T, F) for easy frame manipulation
    m = m.T
    T, F = m.shape

    # Compute frame-to-frame differences
    d = np.sum(np.abs(m[1:, :] - m[:-1, :]), axis=1)
    # Pad d with a zero at the beginning to make it length T
    d = np.concatenate(([0], d))

    # Normalize differences and invert to get warping probabilities
    dmax = np.max(d)
    if dmax == 0:
        print("Warning: dmax is zero, returning unprocessed spectrogram")
        pw = np.ones(T) * warp_rate  # All frames have equal warp probability
    else:
        # Invert differences: frames with smaller changes have higher warping probability
        pw = (1 - (d / dmax)) * warp_rate

    # Generate random values to determine which frames to warp
    random_values = np.random.rand(T)
    flags = random_values < pw  # Frames to warp

    # Create an output matrix of the same shape
    warped_m = np.zeros_like(m)

    # Iterate through each frame and apply random warping if flagged
    for t in range(T):
        if flags[t]:
            # Apply random shift within the max_warp range
            shift = np.random.randint(-max_warp, max_warp + 1)
            new_t = min(max(0, t + shift), T - 1)  # Ensure within bounds
            warped_m[new_t] = m[t]
        else:
            # Copy the frame directly if not warped
            warped_m[t] = m[t]

    # Transpose back to the original shape (F, T)
    return warped_m.T

def magnitude_warping(Sxx, warp_factor=0.2, sigma=5):
    """
    Apply magnitude warping to a spectrogram by modifying the amplitude along the frequency axis.
    
    Parameters:
    - Sxx (numpy.ndarray): Input spectrogram of shape (frequency_bins, time_frames).
    - warp_factor (float): Amount of warping, controlling how much the amplitude is varied.
    - sigma (float): Standard deviation for Gaussian filter to smooth the amplitude modulation.

    Returns:
    - numpy.ndarray: Warped spectrogram.
    """
    freq_bins, time_frames = Sxx.shape
    
    # Create a random warp pattern
    warp_pattern = 1 + warp_factor * (np.random.randn(freq_bins) - 0.5)
    
    # Smooth the warp pattern along the frequency axis
    warp_pattern = gaussian_filter(warp_pattern, sigma=sigma)
    
    # Repeat pattern across the time axis
    warp_pattern = np.tile(warp_pattern, (time_frames, 1)).T
    
    # Apply the warp pattern to the spectrogram
    Sxx_warped = Sxx * warp_pattern
    
    return Sxx_warped


def prepare_dataset(batch, add_noise_flag=False, target_snr_db=20, 
                    add_mask_flag=False, mask_time=0.1, freq_mask=0.1, mask_prob=0.3,
                    apply_spectrogram_processing=False, deletion_rate=0.1,
                    apply_magnitude_warping=False, warp_factor=0.1):
    """
    Prepare dataset batch with optional noise augmentation.
    """
    audio = batch["audio"]
    audio_array = audio["array"]

    # Apply noise only if the flag is set
    if add_noise_flag:
        audio_array = add_noise({"array": audio_array}, target_snr_db)


    # Compute input features from (possibly noisy) audio
    batch["input_features"] = feature_extractor(
        audio_array, sampling_rate=audio["sampling_rate"],
        padding=False
    ).input_features[0]

    # Apply SpecAugment only if the flag is set
    if add_mask_flag and np.random.rand() < mask_prob:
        # Compute the spectrogram
        Sxx = np.abs(batch["input_features"])
        
        # Apply SpecAugment
        Sxx = apply_specaugment(Sxx, time_mask=mask_time, freq_mask=freq_mask)
        
        # Update the input features
        batch["input_features"] = Sxx

    if apply_spectrogram_processing: # and np.random.rand() < 0.3:
        # Apply the spectrogram processing function
        #print('before', batch["input_features"].shape)
        batch["input_features"] = spectrogram_processing_dr(batch["input_features"], deletion_rate)
        #print('after', batch["input_features"].shape)
        #exit()
    if apply_magnitude_warping and np.random.rand() < 0.3:
        # Apply magnitude warping to the spectrogram
        cur_type = batch["input_features"].dtype
        batch["input_features"] = magnitude_warping(batch["input_features"].astype(np.float32), warp_factor=warp_factor)
        # Convert back to original type
        batch["input_features"] = batch["input_features"].astype(cur_type)

    # Encode target text to label ids
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

if not os.path.exists(f'stuff/token_weights_{train_duration_str}-turbo.json'):
    token_weights = calculate_token_weights(atypical_voice, tokenizer)

    # Save token weights to a JSON file
    with open(f'stuff/token_weights_{train_duration_str}-turbo.json', 'w') as f:
        json.dump(token_weights, f)

    print(f"Token weights have been calculated and saved to 'token_weights_{train_duration_str}-turbo.json'")

atypical_voice = atypical_voice.map(prepare_dataset, num_proc=os.cpu_count())

# Training and Evaluation
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# claude
gen_config = GenerationConfig.from_model_config(model.config)
gen_config.task = "transcribe"
gen_config.task_to_id = {
    "transcribe": 50359,
    "translate": 50358
  }
gen_config.language = "ko"
language_to_id_map = {
    "en": 50259,  # English
    "ko": 50264,  # Korean
    # Add more languages as needed
}
gen_config.lang_to_id = language_to_id_map
# Clear forced_decoder_ids and suppress_tokens
gen_config.forced_decoder_ids = None
gen_config.suppress_tokens = []

model.generation_config = gen_config


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract input features and labels from the features list
        input_features_list = [feature["input_features"] for feature in features]  # List of lists or tensors
        labels = [feature["labels"] for feature in features]  # List of tokenized labels

        max_time_length = 3000  # Desired maximum time dimension

        # Process each input feature to have a fixed time dimension of 3000
        processed_features = []
        for feature in input_features_list:
            # Convert feature to tensor if it's a list
            if isinstance(feature, list):
                feature = torch.tensor(feature)
            elif isinstance(feature, np.ndarray):
                feature = torch.from_numpy(feature)
            # Now feature should be a tensor
            # Ensure feature is of shape [feature_size, time_steps]
            if feature.dim() == 1:
                # Reshape to [1, time_steps] if necessary
                feature = feature.unsqueeze(0)
            elif feature.dim() > 2:
                # Flatten extra dimensions if necessary
                feature = feature.view(feature.size(0), -1)
            time_steps = feature.shape[-1]
            if time_steps > max_time_length:
                # Truncate the feature to the first 3000 time steps
                feature = feature[:, :max_time_length]
            elif time_steps < max_time_length:
                # Pad the feature with zeros to reach 3000 time steps
                pad_size = max_time_length - time_steps
                feature = torch.nn.functional.pad(feature, (0, pad_size), "constant", 0)
            processed_features.append(feature)

        # Stack the processed features into a batch tensor
        batch_input_features = torch.stack(processed_features)

        # Prepare the batch dictionary
        batch = {"input_features": batch_input_features}

        # Pad the labels to the maximum length in the batch
        label_features = [{"input_ids": label} for label in labels]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt"
        )

        # Replace padding token id's of the labels by -100 so they are ignored by the loss function
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove the decoder start token if it's included
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


#wer_metric = evaluate.load("wer")
#cer_metric = evaluate.load("cer")
metrics = ASRMetrics()

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
    
    # Compute SER and CER
    ser, cer = metrics.calculate_metrics_batched(pred_str, label_str)
    ser = ser * 100
    cer = cer * 100

    return {
        "cer": cer,
        "ser": ser,
    }


training_args = Seq2SeqTrainingArguments(
    output_dir=repo_name,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,
    num_train_epochs=25,  # Maximum number of epochs
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
    load_best_model_at_end=False,
    metric_for_best_model="valid_loss",  # Use validation loss as metric
    greater_is_better=False,
    push_to_hub=True,
    logging_first_step=True,
    lr_scheduler_type="cosine",
    
)

class HalfEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if (state.epoch + 1) % 0.5 == 0:
            control.should_evaluate = True
            control.should_log = True
        if state.epoch % 3 == 0:
            control.should_save = True
        else:
            control.should_save = False

class NoiseAugmentationCallback(TrainerCallback):
    def __init__(self, target_snr_db=20):
        self.target_snr_db = target_snr_db

    def on_epoch_begin(self, args, state, control, **kwargs):
        """
        Add noise to the training dataset at the beginning of each epoch.
        """
        print(f"Adding noise with {self.target_snr_db}dB SNR to the training data.")

        # Apply noise augmentation dynamically to the training set
        atypical_voice["train"] = atypical_voice["train"].map(
            lambda x: prepare_dataset(x, add_noise_flag=True, target_snr_db=self.target_snr_db, 
                            add_mask_flag=False, apply_spectrogram_processing=False, deletion_rate=DELETION_RATE,
                            apply_magnitude_warping=True, warp_factor=WARP_FACTOR),
                                    num_proc=1,
                                )

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
        self.best_eval_ser = float('inf')

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
        if 'eval_ser' in logs and logs['eval_ser'] < self.best_eval_ser:
            self.best_eval_wer = logs['eval_ser']

    def get_best_metrics(self):
        return {
            "best_eval_loss": self.best_eval_loss,
            "best_eval_cer": self.best_eval_cer,
            "best_eval_ser": self.best_eval_ser
        }

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Evaluate on validation set
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Evaluate on test set
        #test_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix="test")
        
        return eval_results
    
    
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
    callbacks=[HalfEpochCallback(), NoiseAugmentationCallback(target_snr_db=SNR)],
)
processor.save_pretrained(training_args.output_dir)

train_start = time()
print(model.config)
#print('Evaluating the model before training')
#trainer.evaluate()
#exit()
print('Training the model')
trainer.train()

# Get the best metrics
best_metrics = trainer.get_best_metrics()
print("Best metrics:", best_metrics)

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