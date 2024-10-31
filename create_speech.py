import json
import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import sys
sys.path.append('MeloTTS')
from melo.api import TTS
from tqdm import tqdm

def tts_infer_openvoice(tts, tone_color_converter, text, reference, 
                        speed=1, output_dir='results', file_name='output', device='cuda:0', default_reference=None):
    os.makedirs(output_dir, exist_ok=True)
    src_path = f'{output_dir}/tmp_{file_name}.wav'
    speaker_ids = tts.hps.data.spk2id
    assert len(speaker_ids) == 1
    speaker_key = list(speaker_ids.keys())[0]
    speaker_id = speaker_ids[speaker_key]

    # Generate TTS speech
    tts.tts_to_file(text, speaker_id, src_path, speed=speed)

    if reference is None:
        return

    # Perform voice conversion
    speaker_key = speaker_key.lower().replace('_', '-')
    try:
        target_se, audio_name = se_extractor.get_se(reference, tone_color_converter, vad=False)
    except Exception as e:
        print(f"{e}, using default reference instead")
        target_se, audio_name = se_extractor.get_se(default_reference, tone_color_converter, vad=False)

    source_se = torch.load(f'checkpoint/{speaker_key}.pth', map_location=device)

    save_path = f'{output_dir}/{file_name}.wav'

    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=save_path,
        message=encode_message)
    # remove temporary file
    os.remove(src_path)

def process_json_and_generate_speech(json_file, tts, tone_color_converter, speed=1, device='cuda:0', default_reference=None):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in tqdm(data):
        try:
            audio_path = entry['audio']['path']
            transcription = entry['prediction']
            output_dir = os.path.dirname(audio_path)
            file_name = os.path.splitext(os.path.basename(audio_path))[0] + "_converted"

            print(f"Generating speech for: {transcription}")

            # Generate speech and perform voice conversion
            tts_infer_openvoice(
                tts, tone_color_converter, transcription, audio_path,
                speed=speed, output_dir=output_dir, file_name=file_name, device=device,
                default_reference=default_reference
            )
            print(f"Saved converted audio to: {output_dir}/{file_name}.wav")
        except Exception as e:
            print(f"Error while processing entry: {e}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    #filename = 'test'
    filename = 'valid'
    speaker = 'LJM_Man_30s'

    parser.add_argument("--json_file", type=str, default=f"/home/sabina/project_samples/{speaker}/{filename}/{filename}.json", help="Path to the JSON file")
    parser.add_argument("--tts", type=str, default="KR", help="Model to use for TTS")
    parser.add_argument("--speed", default=1, type=float, help="Speed of the TTS")
    parser.add_argument("--device", default="cuda:1", type=str, help="Device to run the model on")
    parser.add_argument("--default_reference", default="/home/sabina/project_samples/LJM_Man_30s/test/LJM_Man_30s_240812_1150_236.wav", 
                        type=str, help="Default reference audio for voice conversion")

    args = parser.parse_args()

    # Initialize TTS and ToneColorConverter models
    tone_color_converter = ToneColorConverter('checkpoint/config.json', device=args.device)
    tone_color_converter.load_ckpt('checkpoint/converter.pth')
    tts = TTS(language=args.tts, device=args.device)

    # Process JSON and generate speech with voice conversion
    process_json_and_generate_speech(args.json_file, tts, tone_color_converter, speed=args.speed, device=args.device, default_reference=args.default_reference)
