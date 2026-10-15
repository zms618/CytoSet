import argparse
import json
import os
import sys

import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig


def main():
    parser = argparse.ArgumentParser(description='Benchmark RT-DETR/CytoSet model-forward efficiency.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    cfg = YAMLConfig(args.config)
    model = cfg.model.to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    images = torch.randn(1, 3, 640, 640, device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            outputs = model(images)
        torch.cuda.synchronize(device)

        starts, ends = [], []
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs = model(images)
            end.record()
            starts.append(start)
            ends.append(end)
        torch.cuda.synchronize(device)

    latency_ms = sorted(start.elapsed_time(end) for start, end in zip(starts, ends))
    mean_ms = sum(latency_ms) / len(latency_ms)
    result = {
        'config': args.config,
        'device': torch.cuda.get_device_name(device),
        'input_shape': list(images.shape),
        'dtype': str(images.dtype),
        'parameters': params,
        'parameters_M': params / 1e6,
        'warmup_iterations': args.warmup,
        'timed_iterations': args.iters,
        'latency_mean_ms': mean_ms,
        'latency_median_ms': latency_ms[len(latency_ms) // 2],
        'latency_p95_ms': latency_ms[min(len(latency_ms) - 1, int(0.95 * len(latency_ms)))],
        'images_per_second': 1000.0 / mean_ms,
        'rate_method': 'reciprocal_of_mean_latency_not_independent_throughput',
        'measurement_scope': 'batch_one_cuda_model_forward_only',
        'output_shape': list(outputs['pred_logits'].shape),
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
