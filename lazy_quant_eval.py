import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
from datasets import load_dataset, Audio
from tqdm import tqdm
from pydub import AudioSegment
from time import time
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from metrics import ASRMetrics
import pandas as pd
from time import time

load_dotenv()

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Speakers and their respective model paths
speakers = ['MJY_Woman_40s', 'MSY_Woman_40s', 'KGH_Woman_30s', 'JHJ_Woman_40s',
            'LJM_Man_30s', 'BMG_Woman_50s', 'YEJ_Woman_40s', 'CYR_Woman_50s',
            'LMH_Man_40s', 'INH_Man_30s', 'IMJ_Woman']


test_names = ['test', 'valid']

# Create a pandas DataFrame to store the results
results = pd.DataFrame(columns=['Name', 'valid WER (%)', 'valid WER w/o ins (%)', 'valid SER (%)', 'valid CER (%)',
                               'test WER (%)', 'test WER w/o ins (%)', 'test SER (%)', 'test CER (%)', 'Avg. Inference Time (s)'])

def process_and_transcribe(audio_path):
    segments, _ = model.transcribe(audio_path, beam_size=5, best_of=5)
    
    transcription = ""
    for segment in segments:
        transcription += segment.text.strip() + " "
    
    return transcription.strip()

def process(speaker, test_name):
    # Prepare for evaluation
    all_predictions, all_references = [], []
    total_time = 0
    total_samples = 0

    dataset_path = os.path.join('dataset', speaker, f"{test_name}.json")
    dataset = load_dataset('json', data_files=dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.rename_column('sentence', 'transcription')

    for sample in tqdm(dataset['train']):
        audio_path = sample['audio']['path']
        prediction = process_and_transcribe(audio_path)
        total_samples += 1
        all_predictions.append(prediction)
        all_references.append(sample['transcription'])
    metrics = ASRMetrics()
    total_ser, total_cer, total_wer_numerator, total_wer_no_ins_numerator = metrics.calculate_metrics_batched(all_predictions, all_references)
    return total_ser, total_cer, total_wer_numerator, total_wer_no_ins_numerator, all_references

for speaker in speakers:
    print(f"Processing speaker: {speaker}")
    model_path = f'neoALI/STT-{speaker}'
    model = WhisperModel(model_path, device=device, compute_type="float16" if device == "cuda" else "int8")
    
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