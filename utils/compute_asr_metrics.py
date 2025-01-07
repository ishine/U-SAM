import argparse
import json
import os
from jiwer import wer, cer

def load_predictions(pred_file):
    predictions = {}
    with open(pred_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                key, pred = parts
                # Remove special tokens and extra spaces, convert to lowercase
                predictions[key] = pred.replace('<|end_of_text|>', '').strip().lower()
    return predictions

def load_references(ref_file):
    references = {}
    with open(ref_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            # Convert reference to lowercase to match predictions
            references[item['key']] = item['label'].strip().lower()
    return references

def compute_asr_metrics_per_sample(predictions, references):
    sample_metrics = []
    total_wer = 0.0
    total_cer = 0.0
    count = 0

    for key, ref in references.items():
        if key in predictions:
            pred = predictions[key]
            wer_score = wer(ref, pred)  # Compute WER
            cer_score = cer(ref, pred)  # Compute CER
            sample_metrics.append({
                "key": key,
                "WER": wer_score,
                "CER": cer_score
            })
            total_wer += wer_score
            total_cer += cer_score
            count += 1
        else:
            print(f"Warning: Missing prediction for key {key}")

    avg_wer = total_wer / count if count > 0 else 0.0
    avg_cer = total_cer / count if count > 0 else 0.0

    return sample_metrics, avg_wer, avg_cer

def main(pred_dir, ref_file):
    for pred_file in os.listdir(pred_dir):
        if pred_file.endswith('.txt'):
            pred_file = os.path.join(pred_dir, pred_file)
            predictions = load_predictions(pred_file)
            references = load_references(ref_file)
            sample_metrics, avg_wer, avg_cer = compute_asr_metrics_per_sample(predictions, references)
            
            # Generate output filename based on prediction file name and average WER and CER
            pred_filename = os.path.basename(pred_file)
            output_dir = os.path.dirname(pred_file)
            output_file = os.path.join(output_dir, f"{pred_filename}_avg_wer_{avg_wer:.2f}_cer_{avg_cer:.2f}.json")
            
            # Save results to JSON file
            with open(output_file, 'w') as f:
                json.dump(sample_metrics, f, indent=4)
            print(f"Metrics saved to {output_file}")
        else:
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ASR Evaluation Metrics for each sample: WER and CER")
    parser.add_argument("--pred_dir", type=str, required=True, help="Path to the dir that contains prediction files")
    parser.add_argument("--ref_file", type=str, required=True, help="Path to the file containing reference transcriptions")
    
    args = parser.parse_args()
    main(args.pred_dir, args.ref_file)