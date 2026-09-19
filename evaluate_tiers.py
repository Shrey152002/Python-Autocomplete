"""
Measures next-character accuracy across three difficulty/style tiers, using the
same algorithm as python_autocomplete/evaluate/eval_sample.py (accept the model's
top suggestion at each position; count it correct if it matches the real
continuation). Writes combined results to tiered_accuracy_results.json.
"""
import json
import time

import torch

from python_autocomplete.evaluate.beam_search import NextWordPredictionComplete
from python_autocomplete.evaluate.factory import get_predictor

TIERS = {
    'easy': '''
import numpy as np

def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
z = x + y
print(z)

for i in range(10):
    print(i)

data = {'a': 1, 'b': 2, 'c': 3}
for key, value in data.items():
    print(key, value)
''',
    'medium': '''
import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = MLP(784, 256, 10)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
''',
    'hard': '''
import functools
import contextlib
from dataclasses import dataclass, field


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    backoff_factor: float = 1.5
    exceptions: tuple = field(default_factory=lambda: (ConnectionError,))


def with_retry(policy):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = 1.0
            for attempt in range(policy.max_attempts):
                try:
                    return func(*args, **kwargs)
                except policy.exceptions as exc:
                    if attempt == policy.max_attempts - 1:
                        raise
                    delay *= policy.backoff_factor
            return None
        return wrapper
    return decorator
''',
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

    for name, code in TIERS.items():
        text = code.strip('\n') + '\n'
        start = time.time()
        correct, key_strokes, length = measure(predictor, text)
        elapsed = time.time() - start
        accuracy = correct / length if length else 0

        results[name] = {
            'accuracy': accuracy,
            'correct_chars': correct,
            'key_strokes': key_strokes,
            'length': length,
            'time_seconds': elapsed,
        }
        total_correct += correct
        total_length += length
        total_time += elapsed
        print(f"[{name:6s}] accuracy={accuracy:.1%}  length={length}  key_strokes={key_strokes}  time={elapsed:.1f}s")

    overall_weighted = total_correct / total_length if total_length else 0
    overall_simple_avg = sum(r['accuracy'] for r in results.values()) / len(results)

    output = {
        'tiers': results,
        'summary': {
            'overall_weighted_accuracy': overall_weighted,
            'overall_simple_average_accuracy': overall_simple_avg,
            'total_length': total_length,
            'total_time_seconds': total_time,
        },
    }

    with open('tiered_accuracy_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Overall weighted accuracy: {overall_weighted:.1%}")
    print(f"Overall simple average accuracy: {overall_simple_avg:.1%}")
    print(f"Total time: {total_time:.1f}s")
    print("Results written to tiered_accuracy_results.json")


if __name__ == '__main__':
    main()
