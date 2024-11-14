import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3" 
from faster_whisper import WhisperModel


class STT(WhisperModel):
    def __init__(self, model_name, temperature, best_of, beam_size, verbose=False):
        '''
        Args:
            model_name (str): The name of the model to use for speech-to-text.
            verbose (bool): Whether to print the transcribed text and other information

        Possible model names:
            "neoALI/whisper-medium-faster-imijeong"
            "neoALI/whisper-turbo-faster-imijeong"
        '''
        super().__init__(model_name)
        self.verbose = verbose
        self.temperature = temperature
        self.best_of = best_of
        self.beam_size = beam_size
    
    def transcribe(self, audio_path, initial_prompt=None, condition_on_previous_text=True):
        segments, info = super().transcribe(audio_path, temperature=self.temperature, best_of=self.best_of, 
                                            beam_size=self.beam_size, initial_prompt=initial_prompt,
                                            condition_on_previous_text=condition_on_previous_text)
        texts = [segment.text for segment in segments]
        if self.verbose:
            print(info)
            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

        transcription = ' '.join(texts)
        return transcription


if __name__ == "__main__":
    model_name = "whisper-turbo-faster-imijeong"
    model = STT(model_name, verbose=False, temperature=0.2, best_of=5, beam_size=5)

    audio_path = "test.wav"
    gt_text = "난 불효자인가. 불효자인가?"

    transcription = model.transcribe(audio_path)

    print(f"Ground truth: {gt_text}")
    print(f"Transcription: {transcription}")
