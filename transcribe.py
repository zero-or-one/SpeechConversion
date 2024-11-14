import pandas as pd
import soundfile as sf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm.auto import tqdm

# Load the CSV file into a DataFrame
csv_path = '1026/CYR_Woman_50s.csv'
target_dir = '1026/CYR_Woman_50s'
target_col = '파일명(이름_성별_나이_날짜_일렬번호_순)'

df = pd.read_csv(csv_path)

# Initialize Whisper model and processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3", task='transcribe', language='ko')

model.config.task = 'transcribe'
model.config.language = 'ko'

# Move model to GPU if CUDA is available
if torch.cuda.is_available():
    model.to("cuda")

def transcribe_audio(file_path):
    # Load the audio file
    try:
        audio, sample_rate = sf.read(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return "No audio"
    # Make sure the audio is at 16,000 Hz
    if sample_rate != 16000:
        audio = sf.resample(audio, sample_rate, 16000)
        sample_rate = 16000
    # Process the audio
    input_features = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    # Move tensors to GPU if CUDA is available
    if torch.cuda.is_available():
        input_features = {key: value.to("cuda") for key, value in input_features.items()}
    # Perform transcription
    with torch.no_grad():  # Ensures memory is used efficiently
        generated_ids = model.generate(**input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

# Apply the transcription to each row in the DataFrame with progress monitoring
tqdm.pandas(desc="Transcribing audio files")
df['stt'] = df[target_col].progress_apply(lambda x: transcribe_audio(f"{target_dir}/{x}"))

# Save the DataFrame back to the CSV
df.to_csv(f'{csv_path.split()[0]}.csv', index=False)
