from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import sys
sys.path.append('MeloTTS')
from melo.api import TTS
from stt import STT
import os


def tts_infer_openvoice(tts, tone_color_converter, text, reference, speed=1, output_dir='results', file_name='output', device='cuda:0'):
    os.makedirs(output_dir, exist_ok=True)
    src_path = f'{output_dir}/tmp_{file_name}.wav'
    speaker_ids = tts.hps.data.spk2id
    assert len(speaker_ids) == 1
    speaker_key = list(speaker_ids.keys())[0]
    speaker_id = speaker_ids[speaker_key]

    # produce speech
    tts.tts_to_file(text, speaker_id, src_path, speed=speed)
    if reference is None:
        return
    
    # convert speech
    speaker_key = speaker_key.lower().replace('_', '-')
    target_se, audio_name = se_extractor.get_se(reference, tone_color_converter, vad=False)
    source_se = torch.load(f'checkpoint/{speaker_key}.pth', map_location=device)
    
    save_path = f'{output_dir}/{file_name}.wav'

    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=save_path,
        message=encode_message)


if __name__ == "__main__":
    from argparse import ArgumentParser
    import torch
    
    parser = ArgumentParser()
    parser.add_argument("--audio_path", default="audio/n1.wav", type=str, help="Path to the audio file")
    parser.add_argument("--tts", type=str, default="KR", help="model to use for TTS")
    parser.add_argument("--speed", default=1, type=float, help="Speed of the TTS")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device to run the model on")
    parser.add_argument("--speaker_id", default="MJY_Woman_40s", type=str, help="Speaker ID for STT model")

    args = parser.parse_args()

    # Initialize STT model with the specified speaker
    stt_model = STT(args.speaker_id, device=args.device)
    
    # Perform speech-to-text
    text = stt_model.transcribe(args.audio_path)
    print("ASR Result:", text)

    # Initialize TTS and tone color converter
    tone_color_converter = ToneColorConverter(f'checkpoint/config.json', device=args.device)
    tone_color_converter.load_ckpt(f'checkpoint/converter.pth')
    tts = TTS(language=args.tts, device=args.device)
    file_name = text.split()[0].lower()

    # Perform TTS and voice conversion
    tts_infer_openvoice(tts, tone_color_converter, text, args.audio_path, speed=args.speed, device=args.device, file_name=file_name)
    print(f"Output saved to results/{file_name}.wav")