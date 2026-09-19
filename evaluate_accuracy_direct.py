"""
Measures autocomplete accuracy and latency by calling the model directly in-process
(same logic as python_autocomplete/serve.py's /autocomplete route), bypassing HTTP
entirely. Use this if the Flask server is unreachable from HTTP clients other than
the VSCode extension.

Usage:
    python evaluate_accuracy_direct.py [--input test_cases.json] [--output accuracy_results.json]
"""
import argparse
import json
import time

import torch

from python_autocomplete.evaluate.beam_search import NextWordPredictionComplete
from python_autocomplete.evaluate.factory import get_predictor


def is_match(prediction: str, expected: str) -> bool:
    prediction = prediction.strip()
    if not prediction:
        return False
    return expected.startswith(prediction) or prediction.startswith(expected)


def predict(predictor, prefix: str):
    stripped, prompt = predictor.rstrip(prefix)
    rest = prefix[len(stripped):]
    prediction_complete = NextWordPredictionComplete(rest, 15)
    prompt = torch.tensor(prompt, dtype=torch.long).unsqueeze(-1)

    predictions = predictor.get_next_word(prompt, None, rest, [1.], prediction_complete, 5)
    predictions.sort(key=lambda x: -x.prob)

    results = [pred.text[len(rest):] for pred in predictions]
    probs = [pred.prob for pred in predictions]
    return results, probs


def run_case(predictor, prompt: str, expected: str) -> dict:
    start = time.time()
    try:
        predictions, probs = predict(predictor, prompt)
    except Exception as e:
        elapsed = time.time() - start
        return {
            'prompt': prompt,
            'expected': expected,
            'error': str(e),
            'time_seconds': elapsed,
            'top1_match': False,
            'top5_match': False,
        }
    elapsed = time.time() - start

    top1_match = bool(predictions) and is_match(predictions[0], expected)
    top5_match = any(is_match(p, expected) for p in predictions[:5])

    return {
        'prompt': prompt,
        'expected': expected,
        'predictions': predictions,
        'probs': probs,
        'time_seconds': elapsed,
        'top1_match': top1_match,
        'top5_match': top5_match,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='test_cases.json')
    parser.add_argument('--output', default='accuracy_results.json')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        test_cases = json.load(f)

    print('Loading model...')
    predictor = get_predictor()
    predictor.model.eval()
    print('Model loaded. Running evaluation...\n')

    results = []
    total_start = time.time()
    for case in test_cases:
        result = run_case(predictor, case['prompt'], case['expected'])
        results.append(result)
        status = 'OK ' if result['top1_match'] else ('~  ' if result['top5_match'] else 'MISS')
        print(f"[{status}] {result['time_seconds']*1000:6.1f}ms  {case['prompt']!r} -> expected {case['expected']!r}"
              f"  got {result.get('predictions')}")
    total_time = time.time() - total_start

    n = len(results)
    n_errors = sum(1 for r in results if 'error' in r)
    top1_accuracy = sum(r['top1_match'] for r in results) / n if n else 0
    top5_accuracy = sum(r['top5_match'] for r in results) / n if n else 0
    avg_time = sum(r['time_seconds'] for r in results) / n if n else 0

    summary = {
        'total_cases': n,
        'errors': n_errors,
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'total_time_seconds': total_time,
        'avg_time_per_request_seconds': avg_time,
    }

    output = {'summary': summary, 'results': results}
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Cases: {n}  Errors: {n_errors}")
    print(f"Top-1 accuracy: {top1_accuracy:.1%}")
    print(f"Top-5 accuracy: {top5_accuracy:.1%}")
    print(f"Total time: {total_time:.2f}s  Avg per request: {avg_time*1000:.1f}ms")
    print(f"Results written to {args.output}")


if __name__ == '__main__':
    main()
