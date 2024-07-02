import whisper

import argparse
import logging
import os
from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torchaudio
from icefall.utils import AttributeDict, str2bool

from valle.data import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from valle.data.collation import get_text_token_collater
from valle.models import get_model

from asr_infer import asr_infer


# TTS functions
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio-prompts",
        type=str,
        default="audio/n1.wav",
        help="Audio prompts which are separated by | and should be aligned with --text-prompts.",
    )

    parser.add_argument(
        "--text-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint/best-valid-loss-stage2-base.pt",
        help="Path to the saved checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Path to the tokenized files.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=-100,
        help="Whether AR Decoder do top_k(if > 0) sampling.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature of AR Decoder top_k sampling.",
    )

    parser.add_argument(
        "--continual",
        type=str2bool,
        default=False,
        help="Do continual task.",
    )

    return parser.parse_args()


def load_model(checkpoint, device):
    if not checkpoint:
        return None

    checkpoint = torch.load(checkpoint, map_location=device)

    args = AttributeDict(checkpoint)
    model = get_model(args)

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys
    model.to(device)
    model.eval()
    text_tokens = args.text_tokens

    return model, text_tokens


def tts_infer(model, text_tokenizer, audio_tokenizer, text_collater, device, args):
    text_prompts = " ".join(args.text_prompts.split("|"))

    audio_prompts = []
    if args.audio_prompts:
        for n, audio_file in enumerate(args.audio_prompts.split("|")):
            encoded_frames = tokenize_audio(audio_tokenizer, audio_file)
            audio_prompts.append(encoded_frames[0][0])

        assert len(args.text_prompts.split("|")) == len(audio_prompts)
        audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
        audio_prompts = audio_prompts.to(device)
    results = []
    for text in args.text.split("|"):
        logging.info(f"synthesize text: {text}")
        text_tokens, text_tokens_lens = text_collater(
            [
                tokenize_text(
                    text_tokenizer, text=f"{text_prompts} {text}".strip()
                )
            ]
        )
        # synthesis
        if args.continual:
            assert text == ""
            encoded_frames = model.continual(
                text_tokens.to(device),
                text_tokens_lens.to(device),
                audio_prompts,
            )
        else:
            enroll_x_lens = None
            if text_prompts:
                _, enroll_x_lens = text_collater(
                    [
                        tokenize_text(
                            text_tokenizer, text=f"{text_prompts}".strip()
                        )
                    ]
                )
            encoded_frames = model.inference(
                text_tokens.to(device),
                text_tokens_lens.to(device),
                audio_prompts,
                enroll_x_lens=enroll_x_lens,
                top_k=args.top_k,
                temperature=args.temperature,
            )

        if audio_prompts != []:
            samples = audio_tokenizer.decode(
                [(encoded_frames.transpose(2, 1), None)]
            )
            # store
            results.append(samples[0].cpu())
        else:  # Transformer
            results.append(None)
    return results


@torch.no_grad()
def main():
    args = get_args()

    # ASR
    model = whisper.load_model("base")
    text = asr_infer(model, args.audio_prompts)
    print('Recognized text:', text)

    args.text_prompts = text
    args.text = text

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    #  TTS
    text_tokenizer = TextTokenizer(backend=args.text_extractor)
    model, text_tokens = load_model(args.checkpoint, device)
    text_collater = get_text_token_collater(text_tokens)
    audio_tokenizer = AudioTokenizer()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    results = tts_infer(
        model,
        text_tokenizer,
        audio_tokenizer,
        text_collater,
        device,
        args,
    )
    # it's only one result
    result = results[0]
    audio_path = args.output_dir / f"{args.audio_prompts.split('/')[-1].split('.')[0]}.wav"
    torchaudio.save(str(audio_path), result, 22050)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()