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

import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import json



# FUNCTIONS AND METHODS
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


# overwrite new class
class CustomWhisperTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # load from token_weights.json
        with open('token_weights.json', 'r') as f:
            token_weights = json.load(f)

        weights_tensor = torch.ones(51865)  # Initialize a tensor of ones
        for token_index, weight in token_weights.items():
            #print(f"Token index: {token_index}, weight: {weight}")
            weights_tensor[int(token_index)] = weight  # Update the tensor with the weight
        self.class_weights = weights_tensor.to(self.args.device)  # Move tensor to the device

        # Initialize variables to track best metrics
        self.best_cer = float('inf')
        self.best_wer = float('inf')

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
        # Update best CER and WER
        if 'eval_cer' in logs and logs['eval_cer'] < self.best_cer:
            self.best_cer = logs['eval_cer']
        if 'eval_wer' in logs and logs['eval_wer'] < self.best_wer:
            self.best_wer = logs['eval_wer']

    def get_best_metrics(self):
        return {
            "best_cer": self.best_cer,
            "best_wer": self.best_wer
        }


# split randomly
def split_dataset(dataset, train_indices, test_indices):

    # Use the select method to create new datasets
    test_set = dataset.select(test_indices)
    train_set = dataset.select(train_indices)
    
    # Create a new DatasetDict with the split datasets
    return DatasetDict({
        'train': train_set,
        'test': test_set
    })





def train_model(dataset, train_indices, test_indices, fold_num, model_name, repo_name, results={}):
    atypical_voice = split_dataset(dataset, train_indices, test_indices)

    # Now you can access the train and test splits like this:
    train_data = atypical_voice['train']
    test_data = atypical_voice['test']

    print(f"Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

    # Load the model and tokenizer
    #model_name = 'spow12/whisper-medium-zeroth_korean'
    #processor_name = 'openai/whisper-medium'

    processor_name = model_name

    feature_extractor = WhisperFeatureExtractor.from_pretrained(processor_name)
    tokenizer = WhisperTokenizer.from_pretrained(processor_name, task="transcribe", language='ko')
    processor = WhisperProcessor.from_pretrained(processor_name, task="transcribe", language='ko')

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
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=50,
        max_steps=175,
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

    trainer = CustomWhisperTrainer(
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
    #exit()
    print('Training the model')
    trainer.train()
    print('Evaluating the model after training')
    trainer.evaluate()
    # Get the best metrics
    best_metrics = trainer.get_best_metrics()
    print(f"Fold: {fold_num}")
    print(f"Best CER: {best_metrics['best_cer']}")
    print(f"Best WER: {best_metrics['best_wer']}")
    results[fold_num] = [best_metrics['best_cer'], best_metrics['best_wer']]

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
    return results





if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--repo_name', type=str, default='whisper-small-voice-conversion-korean')
    parser.add_argument('--fold_num', type=int, default=0)
    args = parser.parse_args()

    load_dotenv()
    login(token=os.getenv("HUGGINGFACE_TOKEN"))


    # Load the dataset
    dataset_path = '/home/sabina/korean_data/13_CUJ/dataset.json'
    dataset = load_dataset('json', data_files=dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.rename_column('sentence', 'transcription')
    dataset = dataset['train']

    print('Dataset loaded', len(dataset))
    model_name = 'jiwon65/whisper-small_korean-zeroth'
    repo_name = args.repo_name
    num_folds = args.fold_num

    folds = []
    for fold in range(num_folds):
        fold = [int(x) for x in open(f'data/fold_{fold}.txt', 'r').read().strip().split('\n')]
        folds.append(fold)
    print('Folds loaded', len(folds))
    results = {}
    for i, fold in enumerate(folds):
        #if i == 0:
        #    continue
        print(f'------------------------------------Fold {i}------------------------------------')
        train_indices = []
        for j, fold_ in enumerate(folds):
            if j != i:
                train_indices.extend(fold_)
        test_indices = fold
        results = train_model(dataset, train_indices, test_indices, i, model_name, repo_name, results)
        exit()
    print("repo name: ", repo_name)
    for i, result in results.items():
        print(f"Fold {i}: CER: {result[0]:.2f}, WER: {result[1]:.2f}")
    # Save the results
    with open(f'data/results_{repo_name}.json', 'w') as f:
        json.dump(results, f)