import whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa


def asr_infer_whisper_fast(model, audio_path):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}") # ignore

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print the recognized text
    return result.text


def asr_infer_whisper_hf(model, processor, audio_path):
    audio_input, sampling_rate = librosa.load(audio_path, sr=16000)

    input_features = processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="base", help="Path to the model")
    parser.add_argument("--audio_path", default="audio/n1.wav", type=str, help="Path to the audio file")
    args = parser.parse_args()

    '''
    model = whisper.load_model("base")
    result = asr_infer_whisper_fast(model, args.audio_path)
    '''
    processor = WhisperProcessor.from_pretrained("/home/sabina/SpeechConversion/train/whisper-small-voice-conversion")
    model = WhisperForConditionalGeneration.from_pretrained("/home/sabina/SpeechConversion/train/whisper-small-voice-conversion")
    model.config.forced_decoder_ids = None
    
    result = asr_infer_whisper_hf(model, processor, args.audio_path)
    print(result)