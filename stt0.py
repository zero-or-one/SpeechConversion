import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3" 
from faster_whisper import WhisperModel


class STT
    def __init__(self, model_name, temperature, best_of, num_alternatives, verbose=False):
        '''
        Args:
            model_name (str): The name of the model to use for speech-to-text.
            verbose (bool): Whether to print the transcribed text and other information

        Possible model names:
            "neoALI/whisper-medium-quanted-handicapped"
        '''

        self.model = WhisperModel(model_name)
        self.verbose = verbose
        self.temperature = temperature
        self.best_of = best_of
        self.num_alternatives = num_alternatives
    
    def transcribe(self, audio_path):
        transcriptions = []
        for _ in range(self.num_alternatives):
            segments, info = self.model.transcribe(audio_path, temperature=self.temperature, best_of=self.best_of)
            texts = [segment.text for segment in segments]
            transcriptions.append(" ".join(texts))

            if self.verbose:
                print(info)
                for segment in segments:
                    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        
        return transcriptions


if __name__ == "__main__":
    model_name = "neoALI/whisper-medium-quanted-handicapped"
    model = STT(model_name, verbose=True, temperature=0.3, best_of=3, num_alternatives=3)

    audio_path = "test.wav"
    gt_text = "난 불효자인가. 불효자인가?"

    transcriptions = model.transcribe(audio_path)

    print(f"Ground truth: {gt_text}")
    for i, transcription in enumerate(transcriptions):
        print(f"Transcription {i+1}: {transcription}")
