import argparse
import datetime as dt
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Matcha imports
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse

# Vocos imports
from vocos import Vocos

# Vocos imports
# from matcha.vocos import Vocos

MULTIACCENT_MODEL = "projecte-aina/matxa-tts-cat-multiaccent"
DEFAULT_CLEANER = "catalan_cleaners"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def matxa_alvocat():
    # matxa = "projecte-aina/matxa-tts-cat-multispeaker"
    matxa = "projecte-aina/matxa-tts-cat-multiaccent"
    alvocat = "projecte-aina/alvocat-vocos-22khz"
    matcha_model = MatchaTTS.from_pretrained(matxa, device=device).to(device)
    vocoder = Vocos.from_pretrained(alvocat, device=device).to(device)
    print("Matxa i alVoCat llestos")
    return matcha_model, vocoder


@torch.inference_mode()
def process_text(text: str, cleaner: str):
    x = torch.tensor(
        intersperse(text_to_sequence(text, [cleaner]), 0), dtype=torch.long, device=device
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": x_phones}


@torch.inference_mode()
def synthesise(matcha_model, text, spks, n_timesteps, temperature, length_scale, cleaner):
    text_processed = process_text(text, cleaner)
    start_t = dt.datetime.now()
    output = matcha_model.synthesise(
        text_processed["x"],
        text_processed["x_lengths"],
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spks,
        length_scale=length_scale,
    )
    # merge everything to one dict
    output.update({"start_t": start_t, **text_processed})
    return output


@torch.inference_mode()
def to_vocos_waveform(mel, vocoder):
    audio = vocoder.decode(mel).cpu().squeeze()
    return audio


def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    np.save(folder / f"{filename}", output["mel"].cpu().numpy())
    sf.write(folder / f"{filename}.wav", output["waveform"], 22050, "PCM_24")


def tts(
    matcha_model,
    vocos_vocoder,
    text,
    spk_id,
    n_timesteps=80,
    length_scale=0.85,
    temperature=0.70,
    output_path=None,
    cleaner="auto",
    prints=False,
):
    if spk_id < 0 or spk_id > 7:
        raise ValueError("Speaker ID must be between 0 and 7.")
    if cleaner == "auto":
        cleaner = get_cleaner_for_speaker_id(spk_id)

    n_spk = torch.tensor([spk_id], device=device, dtype=torch.long) if spk_id >= 0 else None
    rtfs, rtfs_w = [], []

    output = synthesise(matcha_model, text, n_spk, n_timesteps, temperature, length_scale, cleaner)
    output["waveform"] = to_vocos_waveform(output["mel"], vocos_vocoder)

    # Compute Real Time Factor (RTF) with Vocoder
    t = (dt.datetime.now() - output["start_t"]).total_seconds()
    rtf_w = t * 22050 / (output["waveform"].shape[-1])

    # Pretty print
    if prints:
        print(output["mel"].shape)
        print(f"{'*' * 53}")
        print("Input text")
        print(f"{'-' * 53}")
        print(output["x_orig"])
        print(f"{'*' * 53}")
        print("Phonetised text")
        print(f"{'-' * 53}")
        print(output["x_phones"])
        print(f"{'*' * 53}")
        print(f"RTF:\t\t{output['rtf']:.6f}")
        print(f"RTF Waveform:\t{rtf_w:.6f}")
        rtfs.append(output["rtf"])
        rtfs_w.append(rtf_w)

        print(f"Number of ODE steps: {n_timesteps}")
        print(f"Mean RTF:\t\t\t\t{np.mean(rtfs):.6f} ± {np.std(rtfs):.6f}")
        print(f"Mean RTF Waveform (incl. vocoder):\t{np.mean(rtfs_w):.6f} ± {np.std(rtfs_w):.6f}")

    # Save the generated waveform
    if output_path is not None:
        save_to_folder("synth", output, os.path.join(output_path, "spk_" + str(spk_id)))
    else:
        return output["waveform"]


def get_cleaner_for_speaker_id(speaker_id):
    speaker_cleaner_mapping = {
        0: "catalan_balear_cleaners",
        1: "catalan_balear_cleaners",
        2: "catalan_cleaners",
        3: "catalan_cleaners",
        4: "catalan_occidental_cleaners",
        5: "catalan_occidental_cleaners",
        6: "catalan_valencia_cleaners",
        7: "catalan_valencia_cleaners",
    }

    return speaker_cleaner_mapping.get(speaker_id, DEFAULT_CLEANER)


if __name__ == "__main__":
    # default_cleaner = "auto" if matxa == MULTIACCENT_MODEL else DEFAULT_CLEANER
    default_cleaner = "auto"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None, help="Path to output the files.")
    parser.add_argument(
        "--text_input",
        type=str,
        default="Això és una prova de síntesi de veu.",
        help="Text file to synthesize",
    )
    parser.add_argument("--temperature", type=float, default=0.70, help="Temperature")
    parser.add_argument("--length_scale", type=float, default=0.9, help="Speech rate")
    parser.add_argument("--speaker_id", type=int, default=2, help="Speaker ID")
    parser.add_argument("--cleaner", type=str, default=default_cleaner, help="Text cleaner to use")
    args = parser.parse_args()
    cleaner = (
        get_cleaner_for_speaker_id(args.speaker_id)
        if default_cleaner == "auto" and args.cleaner == "auto"
        else args.cleaner
    )

    model, vocos_vocoder = matxa_alvocat()
    tts(
        model,
        vocos_vocoder,
        args.text_input,
        spk_id=args.speaker_id,
        n_timesteps=80,
        length_scale=args.length_scale,
        temperature=args.temperature,
        output_path=args.output_path,
        cleaner=cleaner,
    )
