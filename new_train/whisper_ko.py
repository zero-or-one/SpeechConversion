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


from time import sleep
#print('Waiting for 7 hours')
#sleep(7 * 60 * 60)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))


# Load the dataset
atypical_voice = DatasetDict()

dataset_path = '13_CUJ/dataset.json'
dataset = load_dataset('json', data_files=dataset_path)
dataset = dataset.cast_column("audio", Audio())
# rename sentence to transcription
dataset = dataset.rename_column('sentence', 'transcription')


# split randomly
def split_dataset(dataset_dict, test_size=300):
    # Assume the main dataset is in the 'train' split of the DatasetDict
    dataset = dataset_dict['train']
    # Convert to a list of indices
    indices = list(range(len(dataset)))
    
    # Ensure the first 10 samples are in the test set
    with open('test_indices.txt', 'r') as f:
        test_indices = [int(x) for x in f.read().strip().split(',')]

    #exit()
    
    # The rest goes to the train set
    train_indices = [i for i in indices if i not in test_indices]

    # take 67% of the remaining indices for training
    #split = int(0.33 * len(remaining_indices))
    #train_indices = remaining_indices[:split]
    
    # Use the select method to create new datasets
    test_set = dataset.select(test_indices)
    train_set = dataset.select(train_indices)
    
    # Create a new DatasetDict with the split datasets
    return DatasetDict({
        'train': train_set,
        'test': test_set
    })

# Assuming your dataset is named 'dataset'
test_size = int(len(dataset['train']) * 0.2)  # 20% of the dataset for testing
atypical_voice = split_dataset(dataset, test_size=test_size)

# Now you can access the train and test splits like this:
train_data = atypical_voice['train']
test_data = atypical_voice['test']

print(f"Train set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")


# Load the model and tokenizer
model_name = 'openai/whisper-medium'
processor_name = model_name

#model_name = 'spow12/whisper-medium-zeroth_korean'
#processor_name = 'openai/whisper-medium'

#model_name = 'jiwon65/whisper-small_korean-zeroth'
#processor_name = model_name
processor = WhisperProcessor.from_pretrained(processor_name, task="transcribe", language='ko')
tokenizer = processor.tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained(processor_name)
#tokenizer = WhisperTokenizer.from_pretrained(processor_name, task="transcribe", language='ko')

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

atypical_voice = atypical_voice.map(prepare_dataset, remove_columns=atypical_voice.column_names["train"], num_proc=1)

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


wer_metrics = evaluate.load("wer")
cer_metrics = evaluate.load("cer")

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
    original_label_ids = label_ids.copy()
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
    wer = 100 * wer_metrics.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metrics.compute(predictions=pred_str, references=label_str)
    '''
    # Log individual sample WERs
    individual_wers = [100 * wer_metric.compute(predictions=[p], references=[l]) for p, l in zip(pred_str, label_str)]
    individual_cers = [100 * cer_metric.compute(predictions=[p], references=[l]) for p, l in zip(pred_str, label_str)]
    #print("Individual WERs for first 5 samples:")
    print(individual_wers[16:36])
    print(individual_cers[16:36])
    print(f"WER: {wer:.2f}, CER: {cer:.2f}")
    print(f"Mean WER: {np.mean(individual_wers):.2f}, Mean CER: {np.mean(individual_cers):.2f}")
    '''
    return {"wer": wer, "cer": cer}


training_args = Seq2SeqTrainingArguments(
    output_dir=model_name.split("/")[-1],
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=600,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=25,
    eval_steps=25,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",
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
    'repo_id': model_name.split("/")[-1],
    "dataset_tags": "korean handicap",
    "dataset": "Dysarthria voice recognition Korean",  # a 'pretty' name for the training dataset
    "dataset_args": "config: full, split: train",
    "language": "ko",
    "model_name": "Whisper Small Speech Conversion Finetuned",  # a 'pretty' name for our model
    "finetuned_from": model_name,
    "tasks": "automatic-speech-recognition",
}

model.push_to_hub(**kwargs)