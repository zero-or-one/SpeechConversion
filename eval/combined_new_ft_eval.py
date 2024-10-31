import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from datasets import Audio
import evaluate
from tqdm import tqdm
from openai import OpenAI
from pydub import AudioSegment

from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the dataset
test_name = 'test'
dataset_path = f'/home/sabina/speech_handicap_dataset/imijeong/{test_name}.json'
dataset0 = load_dataset('json', data_files=dataset_path)



test_name = 'unused_train_data_0'
dataset_path = f'/home/sabina/speech_handicap_dataset/imijeong/{test_name}.json'
dataset1 = load_dataset('json', data_files=dataset_path)

test_name = 'unused_train_data_1'
dataset_path = f'/home/sabina/speech_handicap_dataset/imijeong/{test_name}.json'
dataset2 = load_dataset('json', data_files=dataset_path)

test_name = 'unused_train_data_2'
dataset_path = f'/home/sabina/speech_handicap_dataset/imijeong/{test_name}.json'
dataset3 = load_dataset('json', data_files=dataset_path)

def simplify_dataset(dataset):
    def simplify_example(example):
        return {
            'audio': {
                'path': example['audio']['path'],
                'sampling_rate': example['audio']['sampling_rate']
            },
            'sentence': example['sentence'],
            'speaker': example['speaker']
        }
    
    return dataset.map(simplify_example)

dataset0 = simplify_dataset(dataset0['train'])
dataset1 = simplify_dataset(dataset1['train'])
dataset2 = simplify_dataset(dataset2['train'])
dataset3 = simplify_dataset(dataset3['train'])

# combine the datasets
combined_dataset = concatenate_datasets([dataset0, dataset1, dataset2])

# Create a DatasetDict with the combined dataset
dataset = DatasetDict({'train': combined_dataset})

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

# Load the model and processor
model_name = 'VC-00-21-47.40'
model_path = os.path.join('/home/sabina/SpeechConversion/new_train/', model_name)
#model_path = 'jiwon65/whisper-small_korean-zeroth'
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(model_path, task="transcribe", language='ko')
processor = WhisperProcessor.from_pretrained(model_path, task="transcribe", language='ko')
model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # compute log-Mel input features from input audio array
    input_features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_features"] = torch.tensor(input_features)
    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

column_names = val_dataset.column_names
val_dataset = val_dataset.map(prepare_dataset, remove_columns=column_names, num_proc=1)




system_prompt = "You are a helpful assistant for the company. Your task is to correct only spelling and minor grammatical errors in the transcribed Korean text. Ensure that the original meaning is preserved and avoid changing the words or phrasing unnecessarily. Focus on improving accuracy while making minimal alterations."
def generate_corrected_transcript(temperature, system_prompt, transcription):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content



all_predictions = []
all_llm_predictions = []
all_references = []

for batch in tqdm(val_dataset):
    input_features = batch["input_features"]
    input_features = torch.tensor(input_features).unsqueeze(0).to(device)
    #print(input_features.shape)
    input_ids = model.generate(input_features, max_length=225, num_beams=5, early_stopping=True)[0]
    prediction = processor.decode(input_ids, skip_special_tokens=True)
    reference = processor.decode(batch["labels"], skip_special_tokens=True)
    #llm_predtion = generate_corrected_transcript(0, system_prompt, prediction)
    
    all_predictions.append(prediction)
    #all_llm_predictions.append(llm_predtion)
    all_references.append(reference)

# Save the predictions and ground truth text
with open(f"predictions_{model_name}_comb_3.txt", "w") as f:
    f.write("\n".join(all_predictions))




# Evaluate the model
metric = evaluate.load("wer")
metric_cer = evaluate.load("cer")
wer = 100 * metric.compute(predictions=all_predictions, references=all_references)
cer = 100 * metric_cer.compute(predictions=all_predictions, references=all_references)

print(f"WER: {wer:.2f}%")
print(f"CER: {cer:.2f}%")

llm_wer = 100 * metric.compute(predictions=all_llm_predictions, references=all_references)
llm_cer = 100 * metric_cer.compute(predictions=all_llm_predictions, references=all_references)

print(f"LLM WER: {llm_wer:.2f}%")
print(f"LLM CER: {llm_cer:.2f}%")

