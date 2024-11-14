import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, DatasetDict, Dataset
from datasets import Audio
from tqdm import tqdm
from pydub import AudioSegment
from time import time
import re
from dotenv import load_dotenv
from metrics import ASRMetrics
import numpy as np
import pandas as pd
from time import time

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


speakers = ['MJY_Woman_40s', 'MSY_Woman_40s', 'KGH_Woman_30s', 'JHJ_Woman_40s',
            'LJM_Man_30s', 'BMG_Woman_50s', 'YEJ_Woman_40s', 'CYR_Woman_50s',
            'LMH_Man_40s', 'INH_Man_30s', 'IMJ_Woman']
revisions = ['1612e343b5b30e31bb86c6d523939cf460bf42ac', 
             'c987fc3ece2b3f4301576d81772fe4c0573e4408', 
             '5ea66966210d3eeb0725a90ca32e146cb606e04c', 
             '7054a5829f42dfc3cb8f77bb94fd6f7985096850', 
             '64bd316a1073d4c43c7fd8a52bd53281ddf4329a', 
             '20e8b610b116e666de17de0e2fc5eaa564e00698', 
             '7423c75bbedf949e93abe42f5f026da495cc089a', 
             'e600996c058f88a861fd568e59a2c35b09ecba1c',
             '3e15c6629fe1a74ae6f6f9a2bb94b3261424c992', 
             '4a06860d49b3b7618f6b5870b05caf95f341bd1d', 
             '348fd0412ddac4e60d56dda36458e134d35f8d4d']
 
speakers = ['IMJ_Woman']


test_names = ['test', 'valid']

TIME_WRAPPING = False
DELETION_RATE = 0.1


def process_and_transcribe(audio_sample):
    # Preprocess audio
    input_features = feature_extractor(audio_sample["array"], sampling_rate=audio_sample["sampling_rate"]).input_features[0]

    input_features = torch.tensor(input_features).unsqueeze(0).to(device)

    # Generate transcription
    with torch.no_grad():
        input_ids = model.generate(input_features, max_length=225, num_beams=5, early_stopping=True)[0]

    # Decode the transcription
    prediction = processor.decode(input_ids, skip_special_tokens=True)
    return prediction.strip()

# Create a pandas DataFrame to store the results
results = pd.DataFrame(columns=['Name', 'valid WER (%)', 'valid WER w/o ins (%)', 'valid SER (%)', 'valid CER (%)',
                               'test WER (%)', 'test WER w/o ins (%)', 'test SER (%)', 'test CER (%)'])


def process(speaker, name):

    # Process samples and calculate metrics
    all_predictions = []
    all_references = []
    total_time = 0
    total_audio_duration = 0

    dataset_path = os.path.join('dataset', speaker, f"{name}.json")
    dataset = load_dataset('json', data_files=dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.rename_column('sentence', 'transcription')

    val_dataset = dataset['train']

    for sample in tqdm(val_dataset):
        audio_duration = len(sample['audio']['array']) / sample['audio']['sampling_rate']
        total_audio_duration += audio_duration

        start_time = time()
        prediction = process_and_transcribe(sample['audio'])
        end_time = time()

        processing_time = end_time - start_time
        total_time += processing_time

        all_predictions.append(prediction)
        all_references.append(sample['transcription'])

    metrics = ASRMetrics()
    total_ser, total_cer, total_wer_numerator, total_wer_no_ins_numerator = metrics.calculate_metrics_batched(all_predictions, all_references)
    return total_ser, total_cer, total_wer_numerator, total_wer_no_ins_numerator, all_references

model_path = f'seastar105/whisper-medium-ko-zeroth'
# Load the processor and tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path, token=os.getenv("HUGGINGFACE_TOKEN"))
tokenizer = WhisperTokenizer.from_pretrained(model_path, task="transcribe", language='ko', token=os.getenv("HUGGINGFACE_TOKEN"))
processor = WhisperProcessor.from_pretrained(model_path, task="transcribe", language='ko', token=os.getenv("HUGGINGFACE_TOKEN"))
model = WhisperForConditionalGeneration.from_pretrained(model_path, token=os.getenv("HUGGINGFACE_TOKEN")).to(device)
model = model.eval()

for speaker, revision in zip(speakers, revisions):
    '''
    model_path = f'neoALI/STT-{speaker}'
    # Load the processor and tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path, token=os.getenv("HUGGINGFACE_TOKEN"), revision=revision)
    tokenizer = WhisperTokenizer.from_pretrained(model_path, task="transcribe", language='ko', token=os.getenv("HUGGINGFACE_TOKEN"), revision=revision)
    processor = WhisperProcessor.from_pretrained(model_path, task="transcribe", language='ko', token=os.getenv("HUGGINGFACE_TOKEN"), revision=revision)
    model = WhisperForConditionalGeneration.from_pretrained(model_path, token=os.getenv("HUGGINGFACE_TOKEN"), revision=revision).to(device)
    model = model.eval()
    '''
    print(f"Processing {speaker}...")
    start = time()
    print('valid')
    valid_ser, valid_cer, valid_wer_numerator, valid_wer_no_ins_numerator, valid_all_references = process(speaker, 'valid')
    print('test')
    test_ser, test_cer, test_wer_numerator, test_wer_no_ins_numerator, all_references = process(speaker, 'test')
    end = time()

    avg_time = (end - start) / (len(valid_all_references) + len(all_references))
    print(f"Average inference time: {avg_time}")

    row = {
        'Name': speaker,
        f'valid WER (%)': valid_wer_numerator / len(valid_all_references) * 100,
        f'valid WER w/o ins (%)': valid_wer_no_ins_numerator / len(valid_all_references) * 100,
        f'valid SER (%)': valid_ser / len(valid_all_references) * 100,
        f'valid CER (%)': valid_cer / len(valid_all_references) * 100,
        f'test WER (%)': test_wer_numerator / len(all_references) * 100,
        f'test WER w/o ins (%)': test_wer_no_ins_numerator / len(all_references) * 100,
        f'test SER (%)': test_ser / len(all_references) * 100,
        f'test CER (%)': test_cer / len(all_references) * 100,
        f'Avg. Inference Time (s)': avg_time
    }
    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
    results.to_csv('speech_recognition_results_quant.csv', index=False)
