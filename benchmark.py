import csv
import time 
from pathlib import Path

from faster_whisper import WhisperModel
from jiwer import wer

PROJECT_DIR = Path(__file__).parent
EVAL_DIR = PROJECT_DIR / "data" / "eval"
REFERENCES_FILE = EVAL_DIR / "references.csv"

model = WhisperModel("base", device="cpu", compute_type="int8")

results = []

with REFERENCES_FILE.open("r", newline="", encoding="utf-8")as file:
    reader = csv.DictReader(file)

    for row in reader:
        audio_path = EVAL_DIR / row["filename"]
        reference = row["reference"]

        start_time = time.perf_counter()
        segments, info = model.transcribe(str(audio_path), language="en", beam_size=5,)
        hypothesis = " ".join(segment.text.strip() for segment in segments).strip()
        elapsed = time.perf_counter() - start_time
        error_rate = wer(reference, hypothesis)
        results.append((row["filename"], reference, hypothesis, elapsed, error_rate))

        print(f"\nfile: {row['filename']}")
        print(f"reference: {reference}")
        print(f"Whisper: {hypothesis}")
        print(f"Time: {elapsed:.2f} seconds | WER: {error_rate:.1%}")

# Add the overall WER and CSV-saving code here.
overall_wer = wer(
    [item[1] for item in results],
    [item[2] for item in results]
)

results_dir = PROJECT_DIR / "results"
results_dir.mkdir(exist_ok=True)
output_file = results_dir / "base_int8.csv"

with output_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "reference", "whisper_output", "time_seconds", "wer"])

    for filename, reference, hypothesis, elapsed, error_rate in results:
        writer.writerow([filename, reference, hypothesis, f"{elapsed:.2f}", f"{error_rate:.4f}"])

print(f"\nOverall WER: {overall_wer:.1%}")
print(f"Results saved to: {output_file}")