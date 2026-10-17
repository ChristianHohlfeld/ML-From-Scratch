"""Controlled synthetic learning tasks with held-out seeds. Not claims about general AI."""
import torch

TASKS = ('group_contrast', 'nonlinear_relation', 'neighbor_contrast')


def make_dataset(samples: int, seed: int, task: str, n: int = 6):
    if task not in TASKS:
        raise ValueError(f'Unknown task {task}. Choose {TASKS}')
    if n != 6:
        raise ValueError('Benchmark task definitions currently require six motors')
    rng = torch.Generator().manual_seed(seed)
    x = torch.rand(samples, n, generator=rng) * 2 - 1
    if task == 'group_contrast':
        score = x[:, 0] + x[:, 2] + x[:, 4] - x[:, 1] - x[:, 3] - x[:, 5]
    elif task == 'neighbor_contrast':
        score = (x[:, 0] - x[:, 1]) + (x[:, 2] - x[:, 3]) + (x[:, 4] - x[:, 5])
    else:
        score = (x[:, 0] - x[:, 1]) * (x[:, 3] - x[:, 4]) + 0.2 * x[:, 2] * x[:, 5]
    return x, (score > 0).float()
