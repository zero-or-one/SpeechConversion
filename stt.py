import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from faster_whisper import WhisperModel
import torch
import time
import re


def clean_text_for_tts(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.!?가-힣]', '', text) # some punctuation marks are kept
    # remove many spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text

SPEAKER_IDS = ['MJY_Woman_40s' , 'MSY_Woman_40s', 'KGH_Woman_30s', 'JHJ_Woman_40s', 'LJM_Man_30s', 
               'BMG_Woman_50s', 'YEJ_Woman_40s', 'CYR_Woman_50s', 'LMH_Man_40s', 'INH_Man_30s',
               'IMJ_Woman']


class STT(WhisperModel):
    def __init__(self, speaker_id, compute_type="float16", device='cuda', temperature=0.2, best_of=5, beam_size=5, verbose=False):
        '''
        Args:
            model_name (str): The name of the model to use for speech-to-text.
            verbose (bool): Whether to print the transcribed text and other information
        '''
        model_name = f"neoALI/STT-{speaker_id}"
        # remember current loaded model
        self.speaker_id = speaker_id
        if speaker_id not in SPEAKER_IDS:
            raise ValueError(f"Speaker id {speaker_id} not found. Choose from {SPEAKER_IDS}")

        super().__init__(model_name, compute_type=compute_type, device=device)
        if torch.cuda.device_count() > 1:
            print("두 개 gpu:", torch.cuda.device_count())
        self.device = device 
        self.compute_type = compute_type
        self.num_workers = 2

        self.verbose = verbose
        self.temperature = temperature
        self.best_of = best_of
        self.beam_size = beam_size

    
    def transcribe(self, audio_path, speaker_id=None):
        if speaker_id and speaker_id != self.speaker_id:
            if speaker_id not in SPEAKER_IDS:
                raise ValueError(f"Speaker id {speaker_id} not found. Choose from {SPEAKER_IDS}")
            self.speaker_id = speaker_id
            model_name = f"neoALI/STT-{speaker_id}"
            # reinitialize model
            super().__init__(model_name, compute_type=self.compute_type, device=self.device)
            print(f"Changed model to {model_name}")

        start = time.time()
        segments, info = super().transcribe(audio_path, temperature=self.temperature, best_of=self.best_of, 
                                            beam_size=self.beam_size)
        texts = [segment.text for segment in segments]
        combined_texts = "".join(texts)
        cleaned_text = clean_text_for_tts(combined_texts)
        if self.verbose:
            print(info)
            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

        transcription = ''.join(cleaned_text)
        print(f"Transcription took {time.time()-start:.2f} seconds")  
        return transcription

if __name__ == "__main__":
    speaker_id = "MJY_Woman_40s"
    model = STT(speaker_id, verbose=False, temperature=0.2, best_of=5, beam_size=5, device='cuda')

    audio_path = "test.wav"
    gt_text = "난 불효자인가. 불효자인가?"

    print('Test 1')
    transcription = model.transcribe(audio_path)
    print(f"Ground truth: {gt_text}")
    print(f"Transcription: {transcription}")
    
    print('Test 2')
    transcription = model.transcribe(audio_path, speaker_id=speaker_id)
    print(f"Ground truth: {gt_text}")
    print(f"Transcription: {transcription}")

    print('Test 3')
    speaker_id = SPEAKER_IDS[1]
    transcription = model.transcribe(audio_path, speaker_id=speaker_id)
    print(f"Ground truth: {gt_text}")
    print(f"Transcription: {transcription}")

    print('Test 4')
    speaker_id = 'RANDOM'
    try:
        transcription = model.transcribe(audio_path, speaker_id=speaker_id)
    except ValueError as e:
        print(e)
    print('All tests passed!')
