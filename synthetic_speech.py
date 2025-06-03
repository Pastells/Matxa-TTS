import os

HOME = "/media/clic/tts"
os.environ["PATH"] += f":{HOME}/espeak-ng/bin"
os.environ["LD_LIBRARY_PATH"] = f"{HOME}/espeak-ng/lib"
os.environ["ESPEAK_DATA_PATH"] = f"{HOME}/espeak-ng/espeak-ng-data"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_DISABLED"] = "true"

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from matcha_vocos_inference import matxa_alvocat, tts_batch  # noqa: E402

print("Loading models...")
matcha_model, vocos_vocoder = matxa_alvocat()


def load_texts_from_file(file: str) -> list[dict]:
    """Load texts from various file formats"""
    file_path = Path(file)

    if file_path.suffix == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            texts = [
                {"text": line.strip(), "id": i}
                for i, line in enumerate(f.readlines())
                if line.strip()
            ]

    elif file_path.suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                if isinstance(data[0], str):
                    texts = [{"text": text, "id": i} for i, text in enumerate(data)]
                else:
                    texts = data
            else:
                raise ValueError("JSON should contain a list of texts or text objects")

    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
        if "text" not in df.columns:
            raise ValueError("CSV file must have a 'text' column")
        texts = df.to_dict("records")
        for i, text_data in enumerate(texts):
            if "id" not in text_data:
                text_data["id"] = i

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    return texts


def create_batches(items: list, batch_size: int) -> list[list]:
    """Create batches from a list of items"""
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i : i + batch_size])
    return batches


def create_synthetic_corpus_parallel(
    texts_input: str | list[dict],
    output_dir: str,
    speaker_ids: list[int] | None = None,
    batch_size: int = 32,
    n_timesteps: int = 80,
    length_scale: float = 0.85,
    temperature: float = 0.70,
    sample_rate: int = 22050,
    audio_format: str = "wav",
    same_speaker_batching: bool = True,
) -> list[dict]:
    """
    Create a synthetic corpus from batch texts using parallel processing

    Args:
        texts_input: Path to text file or list of text dictionaries
        output_dir: Directory to save the synthetic corpus
        speaker_ids: List of speaker IDs to use (default: [0,1,2,3,4,5,6,7])
        batch_size: Number of texts to process simultaneously
        same_speaker_batching: If True, batch texts with same speaker together for efficiency
    """

    # Load texts
    if isinstance(texts_input, str):
        texts = load_texts_from_file(texts_input)
    else:
        texts = texts_input

    if speaker_ids is None:
        speaker_ids = list(range(8))

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Prepare text-speaker combinations
    text_speaker_pairs = []
    for text_data in texts:
        for spk_id in speaker_ids:
            text_speaker_pairs.append((text_data, spk_id))

    # Group by speaker for efficient batching if requested
    if same_speaker_batching:
        # Group pairs by speaker
        speaker_groups = {}
        for text_data, spk_id in text_speaker_pairs:
            if spk_id not in speaker_groups:
                speaker_groups[spk_id] = []
            speaker_groups[spk_id].append((text_data, spk_id))

        # Create batches within each speaker group
        all_batches = []
        for spk_id, pairs in speaker_groups.items():
            speaker_batches = create_batches(pairs, batch_size)
            all_batches.extend(speaker_batches)
    else:
        # Create mixed batches
        all_batches = create_batches(text_speaker_pairs, batch_size)

    # Storage for metadata
    corpus_metadata = []

    # Process batches
    print(f"Processing {len(text_speaker_pairs)} items in {len(all_batches)} batches...")

    for batch in tqdm(all_batches, desc="Processing batches"):
        try:
            # Prepare batch data
            batch_texts = [pair[0]["text"] for pair in batch]
            batch_spk_ids = [pair[1] for pair in batch]
            batch_text_data = [pair[0] for pair in batch]

            # Generate TTS for batch
            results = tts_batch(
                matcha_model,
                vocos_vocoder,
                batch_texts,
                batch_spk_ids,
                n_timesteps=n_timesteps,
                length_scale=length_scale,
                temperature=temperature,
            )

            # Save results
            for i, result in enumerate(results):
                text_data = batch_text_data[i]
                spk_id = batch_spk_ids[i]
                text_id = text_data.get("id", "unknown")

                # Create filename
                filename = f"text{text_id}_speaker{spk_id}_scale{round(length_scale, 2)}_temp{round(temperature, 2)}"
                audio_file = f"{filename}.{audio_format}"

                # Save audio
                audio_path = output_path / audio_file
                waveform = result["waveform"]
                if torch.is_tensor(waveform):
                    waveform = waveform.cpu().numpy()

                sf.write(str(audio_path), waveform, sample_rate, "PCM_24")

                # Collect metadata
                metadata_entry = {
                    "text_id": text_id,
                    "speaker_id": spk_id,
                    "text": result["text"],
                    "phonemes": result["phonemes"],
                    "audio_file": str(audio_path.relative_to(output_path)),
                    "n_timesteps": n_timesteps,
                    "length_scale": length_scale,
                    "temperature": temperature,
                }

                # Add any additional metadata from original text data
                for key, value in text_data.items():
                    if key not in ["text", "id"]:
                        metadata_entry[f"original_{key}"] = value

                corpus_metadata.append(metadata_entry)

        except Exception as e:
            print(f"Error processing batch: {e}")
            continue

    print(f"Synthetic corpus created with {len(corpus_metadata)} audio files in {output_dir}")

    # Print summary statistics
    if corpus_metadata:
        print("\n=== Corpus Statistics ===")
        print(f"Total texts: {len(texts)}")
        print(f"Total speakers: {len(speaker_ids)}")
        print(f"Total audio files: {len(corpus_metadata)}")
        print(f"Batch size used: {batch_size}")

    return corpus_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create synthetic TTS corpus with parallel batch processing"
    )
    parser.add_argument(
        "--texts_file",
        type=str,
        required=True,
        help="Path to file containing texts (.txt, .json, or .csv)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the synthetic corpus"
    )
    parser.add_argument(
        "--speaker_ids",
        type=int,
        nargs="+",
        default=None,
        help="Speaker IDs to use (default: all speakers 0-7)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Number of texts to process in parallel"
    )
    parser.add_argument("--temperature", type=float, default=0.70, help="Temperature")
    parser.add_argument("--length_scale", type=float, default=0.85, help="Speech rate")
    parser.add_argument("--n_timesteps", type=int, default=80, help="Number of ODE steps")
    parser.add_argument(
        "--audio_format",
        type=str,
        default="wav",
        help="Output audio format",
    )
    parser.add_argument(
        "--no_same_speaker_batching",
        action="store_true",
        help="Disable grouping texts by same speaker for batching",
    )

    args = parser.parse_args()

    def create_synthetic_corpus(length_scale):
        return create_synthetic_corpus_parallel(
            texts_input=args.texts_file,
            output_dir=args.output_dir,
            speaker_ids=args.speaker_ids,
            batch_size=args.batch_size,
            n_timesteps=args.n_timesteps,
            length_scale=length_scale,
            temperature=args.temperature,
            audio_format=args.audio_format,
            same_speaker_batching=not args.no_same_speaker_batching,
        )

    corpus_metadata = []
    metadata = create_synthetic_corpus(length_scale=args.length_scale)
    corpus_metadata.extend(metadata)
    metadata = create_synthetic_corpus(length_scale=args.length_scale + 0.15)
    corpus_metadata.extend(metadata)
    metadata = create_synthetic_corpus(length_scale=args.length_scale - 0.15)
    corpus_metadata.extend(metadata)

    output_path = Path(args.output_dir)
    metadata_csv = output_path / "corpus_metadata.csv"
    pd.DataFrame(corpus_metadata).to_csv(metadata_csv, index=False)
    print(f"Corpus metadata saved {metadata_csv}")
