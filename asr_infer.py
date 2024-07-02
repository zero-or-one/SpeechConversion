import whisper


def asr_infer_whisper(model, audio_path):
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


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="base", help="Path to the model")
    parser.add_argument("--audio_path", default="audio/n1.wav", type=str, help="Path to the audio file")
    args = parser.parse_args()

    model = whisper.load_model("base")
    result = asr_infer_whisper(model, args.audio_path)
    print(result)
