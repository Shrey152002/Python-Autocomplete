"""
Measures autocomplete accuracy and latency against a running `serve.py` instance.

Usage:
    python evaluate_accuracy.py [--input test_cases.json] [--output accuracy_results.json]
                                 [--url http://localhost:5000/autocomplete]

Input file format: a JSON list of {"prompt": "<code so far>", "expected": "<what should come next>"}.
"""
import argparse
import json
import time

import requests

# Ignore HTTP_PROXY/HTTPS_PROXY/NO_PROXY env vars, which can otherwise route
# localhost traffic through a proxy and hang until it times out.
session = requests.Session()
session.trust_env = False


def is_match(prediction: str, expected: str) -> bool:
    prediction = prediction.strip()
    if not prediction:
        return False
    return expected.startswith(prediction) or prediction.startswith(expected)


def run_case(url: str, prompt: str, expected: str) -> dict:
    start = time.time()
    try:
        response = session.post(url, json={'prompt': prompt}, timeout=30,
                                 headers={'Connection': 'close'})
        response.raise_for_status()
        data = response.json()
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

    if not data.get('success'):
        return {
            'prompt': prompt,
            'expected': expected,
            'error': 'server returned success=false',
            'time_seconds': elapsed,
            'top1_match': False,
            'top5_match': False,
        }

    predictions = data.get('prediction', [])
    probs = data.get('probs', [])
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
    parser.add_argument('--url', default='http://localhost:5000/autocomplete')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        test_cases = json.load(f)

    results = []
    total_start = time.time()
    for case in test_cases:
        result = run_case(args.url, case['prompt'], case['expected'])
        results.append(result)
        status = 'OK ' if result['top1_match'] else ('~  ' if result['top5_match'] else 'MISS')
        print(f"[{status}] {result['time_seconds']*1000:6.1f}ms  {case['prompt']!r} -> expected {case['expected']!r}")
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
