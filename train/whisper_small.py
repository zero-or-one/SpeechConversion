from dotenv import load_dotenv 
import os
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

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer


os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Load the dataset
atypical_voice = DatasetDict()
dataset = load_dataset("jmaczan/TORGO", split="train", use_auth_token=True)

'''
# split randomly
def split_dataset(dataset, test_size=300):
    # Convert to a list of indices
    indices = list(range(len(dataset)))
    
    # Ensure the first 10 samples are in the test set
    test_indices = indices[:10]
    
    # Shuffle the remaining indices
    remaining_indices = indices[10:]
    random.shuffle(remaining_indices)
    
    # Add the shuffled indices to complete the test set
    test_indices.extend(remaining_indices[:test_size - 10])
    
    # The rest goes to the train set
    train_indices = remaining_indices[test_size - 10:]
    
    # Use the select method to create new datasets
    test_set = dataset.select(test_indices)
    train_set = dataset.select(train_indices)
    
    return train_set, test_set

# Assuming your dataset is named 'dataset'
train_data, test_data = split_dataset(dataset)
'''

'''
# split by taking speaker 1 as validation
test_data = dataset.select(range(118))
remaining_indices = list(range(118, len(dataset)))
random.shuffle(remaining_indices)
train_data = dataset.select(remaining_indices)
'''
# split by text
def split_dataset(dataset, test_texts):
    test_data = []
    train_data = []
    
    for item in dataset:
        if item['transcription'] in test_texts:
            test_data.append(item)
        else:
            train_data.append(item)
    
    return Dataset.from_list(train_data), Dataset.from_list(test_data)

test_texts = [ # I took them randomly
'stick',
'tear ',
'except in the winter when the ooze or snow or ice prevents',
'meat',
'know',
'he slowly takes a short walk in the open air each day',
'feet',
'train',
'when he speaks his voice is just a bit cracked and quivers a trifle',
'knee',
]

train_data, test_data = split_dataset(dataset, test_texts)

print(f"Train set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

atypical_voice["train"] = train_data
atypical_voice["test"] = test_data


# Load the model and tokenizer


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small.en")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small.en", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small.en", task="transcribe")

# Prepare the dataset
print('sample', atypical_voice["train"][0])

# I don't need this, but let's play safe
atypical_voice = atypical_voice.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

atypical_voice = atypical_voice.map(prepare_dataset, remove_columns=atypical_voice.column_names["train"], num_proc=128)
# safe to local


# Training and Evaluation

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en")

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


metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Whisper models return the last hidden state and the logits, we only need the ids
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}



training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-voice-conversion",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=1500,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=300,
    eval_steps=300,
    logging_steps=30,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    logging_first_step=True,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=atypical_voice["train"],
    eval_dataset=atypical_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
processor.save_pretrained(training_args.output_dir)

print(model.config)
print('Evaluating the model before training')
trainer.evaluate()
print('Training the model')
trainer.train()
print('Evaluating the model after training')
trainer.evaluate()


kwargs = {
    'repo_id': "whisper-small-voice-conversion",
    "dataset_tags": "jmaczan/TORGO",
    "dataset": "TORGO",  # a 'pretty' name for the training dataset
    "dataset_args": "config: full, split: train",
    "language": "en",
    "model_name": "Whisper Small Speech Conversion Finetuned",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
}

model.push_to_hub(**kwargs)