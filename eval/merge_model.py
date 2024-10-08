import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor
from peft import PeftModel

# Paths
model_path = 'openai/whisper-large-v3'
adapter_path = 'VC-01-22-4.30-large-lora-4'
save_path = "whisper_large-feature-merged"

# Load the base Whisper model
base_model = WhisperForConditionalGeneration.from_pretrained(model_path)

# Load the LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_path)

# Merge the LoRA adapter with the base model
model = model.merge_and_unload()

# Save the merged model
model.save_pretrained(save_path)

# Load the processor (includes tokenizer and feature extractor)
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(model_path, task="transcribe", language='ko')
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Save the processor to the same path
processor.save_pretrained(save_path)

print(f"Model and processor saved at {save_path}")
