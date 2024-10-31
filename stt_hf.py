import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor
from time import time

class WhisperTranscriber:
    def __init__(self, model_name: str, device=None):
        '''
        Args:
            model_name (str): The name of the model to use for speech-to-text.
            verbose (bool): Whether to print the transcribed text and other information
        '''
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        torch_dtype = torch.float16 # save memory
        print(f"Using device: {self.device}")
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(self.device)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            chunk_length_s=60,
            device=self.device,
            return_tensors=True
        )
    
    def transcribe(self, audio_path, language=None):
        """
        Transcribe the given audio file.
        
        Args:
        audio_path (str): Path to the audio file.
        language (str, optional): Language code for transcription. Default is None.
        
        Returns:
        str: Transcribed text.
        """
        result = self.pipe(audio_path, return_timestamps=False)
        print(result)
        return result["text"]

# Example usage
if __name__ == "__main__":
    # Initialize the transcriber
    model = WhisperTranscriber("ghost613/VC-01-22-4.30-turbo")
    
    # Set up the test
    audio_path = "test.wav"
    gt_text = "난 불효자인가. 불효자인가?"
    
    start = time()
    # Transcribe the audio
    transcription = model.transcribe(audio_path, language="ko")
    print(f"Time taken: {time() - start:.2f}s")

    # Print results
    print(f"Ground truth: {gt_text}")
    print(f"Transcription: {transcription}")
