"""
Runs the same real-code accuracy measurement as evaluate_real_code.py, but only
for the nanoGPT sample (mnist_main.py already completed separately). Writes to
nanogpt_result.json.
"""
import json
import time
from pathlib import Path

import torch

from python_autocomplete.evaluate.beam_search import NextWordPredictionComplete
from python_autocomplete.evaluate.factory import get_predictor

SAMPLE_PATH = 'real_samples/nanogpt_model_trimmed.py'


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
        if key_strokes % 50 == 0:
            print(f"  progress: {i}/{len(text)} chars, {key_strokes} predict calls, "
                  f"{time.time() - start_time:.1f}s elapsed", flush=True)

    return correct, key_strokes, len(text) - 1


start_time = time.time()


def main():
    global start_time
    print('Loading model...')
    predictor = get_predictor()
    predictor.model.eval()
    print('Model loaded.\n')

    text = Path(SAMPLE_PATH).read_text()
    start_time = time.time()
    correct, key_strokes, length = measure(predictor, text)
    elapsed = time.time() - start_time
    accuracy = correct / length if length else 0

    result = {
        'source_file': SAMPLE_PATH,
        'source_url': 'https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py',
        'license': 'MIT (karpathy/nanoGPT)',
        'accuracy': accuracy,
        'correct_chars': correct,
        'key_strokes': key_strokes,
        'length': length,
        'time_seconds': elapsed,
    }

    with open('nanogpt_result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print()
    print(f"nanogpt_model accuracy={accuracy:.1%}  length={length}  key_strokes={key_strokes}  time={elapsed:.1f}s")
    print("Result written to nanogpt_result.json")


if __name__ == '__main__':
    main()
