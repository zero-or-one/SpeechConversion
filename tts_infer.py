import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import sys
sys.path.append('MeloTTS')
from melo.api import TTS


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
    parser = ArgumentParser()
    parser.add_argument("--tts", type=str, default="EN_NEWEST", help="model to use for TTS")
    parser.add_argument("--text", type=str, default="Hello, how are you?", help="Text to convert to speech")
    parser.add_argument("--audio_path", default=None, type=str, help="Path to the audio file")
    parser.add_argument("--speed", default=1, type=float, help="Speed of the TTS")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device to run the model on")
    args = parser.parse_args()

    tone_color_converter = ToneColorConverter(f'checkpoint/config.json', device=args.device)
    tone_color_converter.load_ckpt(f'checkpoint/converter.pth')
    tts = TTS(language=args.tts, device=args.device)
    file_name = args.text.split()[0].lower()

    tts_infer_openvoice(tts, tone_color_converter, args.text, args.audio_path, speed=args.speed, device=args.device, file_name=file_name)
    print(f"Output saved to results/{file_name}.wav")
