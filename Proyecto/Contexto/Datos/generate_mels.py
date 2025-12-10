import os
import librosa
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/mel_generation.log"),
        logging.StreamHandler()
    ]
)

# Configuration
INPUT_DIR = Path("data/audio/processed")
OUTPUT_DIR = Path("data/audio/mels")
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
def generate_mel(file_path, output_path):
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

        # Generate Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

        # Convert to Tensor (1, 128, Time)
        tensor = torch.from_numpy(mel_spec_norm).float().unsqueeze(0)

        # Resize to fixed size (1, 128, 128)
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Save as .pt
        torch.save(tensor, output_path)
        return True
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate Mel Spectrograms from audio files.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of files to process")
    args = parser.parse_args()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get list of mp3 files
    files = sorted(list(INPUT_DIR.glob("*.mp3")))

    if not files:
        logging.warning(f"No .mp3 files found in {INPUT_DIR}")
        return

    # Apply limit if specified
    if args.limit:
        files_to_process = files[:args.limit]
        logging.info(f"Processing first {args.limit} files out of {len(files)} available.")
    else:
        files_to_process = files
        logging.info(f"Processing all {len(files)} files.")

    count = 0
    skipped = 0
    errors = 0

    for file_path in tqdm(files_to_process, desc="Generating Mels"):
        output_filename = file_path.stem + ".pt"
        output_path = OUTPUT_DIR / output_filename

        if output_path.exists():
            skipped += 1
            continue

        if generate_mel(file_path, output_path):
            count += 1
        else:
            errors += 1
    logging.info(f"Finished. Processed: {count}, Skipped: {skipped}, Errors: {errors}")
    print(f"\nDone! Processed: {count}, Skipped: {skipped}, Errors: {errors}")

if __name__ == "__main__":
    main()
