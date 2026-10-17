"""Independent train/validation/test benchmark with basic baselines and exports.

Use `python -m hdr_motor_ai.train --task group_contrast --epochs 40`.
Fixed seed and held-out, disjoint generated input examples; no test-set selection.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from .mechanics import GearNetwork
from .tasks import TASKS, make_dataset


class TinyMLP(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(6, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class TinyAttention(nn.Module):
    """Single-head learned self-attention, positional input plus mean pooling."""
    def __init__(self, width=12):
        super().__init__()
        self.embedding = nn.Linear(1, width)
        self.position = nn.Parameter(torch.randn(6, width) * 0.05)
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.readout = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, x):
        tokens = self.embedding(x.unsqueeze(-1)) + self.position
        weights = torch.softmax(self.q(tokens) @ self.k(tokens).transpose(1, 2) / (tokens.shape[-1] ** 0.5), -1)
        out = (weights @ self.v(tokens)).mean(1)
        return self.readout(out).squeeze(-1)


def accuracy(logits, labels):
    return float(((logits >= 0).float() == labels).float().mean())


def predict(model, x, steps):
    if isinstance(model, GearNetwork):
        return model(x, steps)[0]
    return model(x)


def experiment(task: str, output: Path, seed=7, epochs=32, steps=40,
               train_samples=512, val_samples=256, test_samples=512,
               batch_size=64, lr=0.012, models=('motor', 'mlp', 'attention'),
               channels=3, threads=2):
    torch.set_num_threads(threads)
    if any(v < 1 for v in [epochs, steps, train_samples, val_samples, test_samples, batch_size, channels]):
        raise ValueError('Training sizes, steps and epochs must be positive')
    output.mkdir(parents=True, exist_ok=True)
    # Same distributions, independent RNG draws & seeds, no test-driven checkpointing.
    x_train, y_train = make_dataset(train_samples, seed + 11, task)
    x_val, y_val = make_dataset(val_samples, seed + 137, task)
    x_test, y_test = make_dataset(test_samples, seed + 539, task)
    configs = {
        'motor': lambda: GearNetwork(channels=channels),
        'mlp': lambda: TinyMLP(),
        'attention': lambda: TinyAttention(),
    }
    results = {}
    histories = []
    for name in models:
        if name not in configs:
            raise ValueError(f'Unknown model {name}')
        torch.manual_seed(seed)
        model = configs[name]()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        best_loss = float('inf')
        best_state = None
        start = time.monotonic()
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(x_train))
            train_loss = 0.0
            for s in range(0, len(perm), batch_size):
                idx = perm[s:s + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = predict(model, x_train[idx], steps)
                loss = F.binary_cross_entropy_with_logits(logits, y_train[idx])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite training loss for {name} at epoch {epoch + 1}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                train_loss += float(loss.detach()) * len(idx)
            model.eval()
            with torch.no_grad():
                val_logits = predict(model, x_val, steps)
                val_loss = float(F.binary_cross_entropy_with_logits(val_logits, y_val))
                val_acc = accuracy(val_logits, y_val)
            histories.append({'model': name, 'epoch': epoch + 1,
                              'train_loss': round(train_loss / len(x_train), 6),
                              'validation_loss': round(val_loss, 6),
                              'validation_accuracy': round(val_acc, 6)})
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'{name}: {epoch+1}/{epochs} val_acc={val_acc:.3f} val_loss={val_loss:.3f}', flush=True)
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits = predict(model, x_test, steps)
            test_acc = accuracy(logits, y_test)
            test_loss = float(F.binary_cross_entropy_with_logits(logits, y_test))
        results[name] = {
            'best_epoch': best_epoch, 'validation_loss': round(best_loss, 6),
            'test_accuracy': round(test_acc, 6), 'test_loss': round(test_loss, 6),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'training_seconds': round(time.monotonic()-start, 2)
        }
        torch.save({'model_name': name, 'model_state': best_state, 'task': task,
                    'seed': seed, 'steps': steps, 'channels': channels}, output / f'{name}_best.pt')
        if name == 'motor':
            (output / 'motor_gears.json').write_text(json.dumps(model.browser_export(), indent=2), encoding='utf-8')
    summary = {
        'version': '0.3.0', 'task': task, 'seed': seed, 'epochs': epochs,
        'steps': steps, 'train_samples': train_samples,
        'validation_samples': val_samples, 'test_samples': test_samples,
        'model_results': results,
        'warning': 'Synthetic demonstrations only. This is a numerical mechanical simulation; training uses offline backpropagation.'
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (output / 'learning_curves.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['model','epoch','train_loss','validation_loss','validation_accuracy'])
        writer.writeheader(); writer.writerows(histories)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task', choices=TASKS, default='group_contrast')
    parser.add_argument('--output', type=Path, default=Path('runs/group_contrast'))
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--train-samples', type=int, default=512)
    parser.add_argument('--val-samples', type=int, default=256)
    parser.add_argument('--test-samples', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.012)
    parser.add_argument('--models', default='motor,mlp,attention')
    args=parser.parse_args()
    experiment(args.task, args.output, seed=args.seed, epochs=args.epochs,
               steps=args.steps, channels=args.channels,
               train_samples=args.train_samples, val_samples=args.val_samples,
               test_samples=args.test_samples, batch_size=args.batch_size, lr=args.lr,
               models=tuple(x.strip() for x in args.models.split(',') if x.strip()))

if __name__ == '__main__':
    main()
