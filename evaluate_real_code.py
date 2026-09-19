"""
Measures next-character accuracy on real, publicly available PyTorch code
(not hand-written snippets), using the same algorithm as
python_autocomplete/evaluate/eval_sample.py. Writes results to
real_accuracy_results.json.

Sources:
  - real_samples/mnist_main.py       -- pytorch/examples, mnist/main.py (BSD-3)
  - real_samples/nanogpt_model_trimmed.py -- karpathy/nanoGPT, model.py, first ~2900 chars (MIT)
"""
import json
import time
from pathlib import Path

import torch

from python_autocomplete.evaluate.beam_search import NextWordPredictionComplete
from python_autocomplete.evaluate.factory import get_predictor

SAMPLES = {
    'mnist_main': 'real_samples/mnist_main.py',
    'nanogpt_model': 'real_samples/nanogpt_model_trimmed.py',
}


def measure(predictor, text: str):
    correct = 0
    i = 0
    key_strokes = 0

    while i + 1 < len(text):
        prefix = text[:i + 1]
        stripped, prompt = predictor.rstrip(prefix)
        rest = prefix[len(stripped):]
        prediction_complete = NextWordPredictionComplete(rest, 5)
        prompt = torch.tensor(prompt, dtype=torch.long).unsqueeze(-1)

        predictions = predictor.get_next_word(prompt, None, rest, [1.], prediction_complete, 5)
        predictions.sort(key=lambda x: -x.prob)
        next_token = predictions[0].text[len(rest):] if predictions else ''

        if next_token and next_token == text[i + 1: i + 1 + len(next_token)]:
            correct += len(next_token)
        else:
            next_token = text[i + 1]

        i += len(next_token)
        key_strokes += 1

    return correct, key_strokes, len(text) - 1


def main():
    print('Loading model...')
    predictor = get_predictor()
    predictor.model.eval()
    print('Model loaded.\n')

    results = {}
    total_correct = 0
    total_length = 0
    total_time = 0.0

    for name, path in SAMPLES.items():
        text = Path(path).read_text()
        start = time.time()
        correct, key_strokes, length = measure(predictor, text)
        elapsed = time.time() - start
        accuracy = correct / length if length else 0

        results[name] = {
            'source_file': path,
            'accuracy': accuracy,
            'correct_chars': correct,
            'key_strokes': key_strokes,
            'length': length,
            'time_seconds': elapsed,
        }
        total_correct += correct
        total_length += length
        total_time += elapsed
        print(f"[{name:14s}] accuracy={accuracy:.1%}  length={length}  key_strokes={key_strokes}  time={elapsed:.1f}s")

    overall_weighted = total_correct / total_length if total_length else 0
    overall_simple_avg = sum(r['accuracy'] for r in results.values()) / len(results)

    output = {
        'samples': results,
        'summary': {
            'overall_weighted_accuracy': overall_weighted,
            'overall_simple_average_accuracy': overall_simple_avg,
            'total_length': total_length,
            'total_time_seconds': total_time,
        },
    }

    with open('real_accuracy_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Overall weighted accuracy: {overall_weighted:.1%}")
    print(f"Overall simple average accuracy: {overall_simple_avg:.1%}")
    print(f"Total time: {total_time:.1f}s")
    print("Results written to real_accuracy_results.json")


if __name__ == '__main__':
    main()
