# Command-line tools

Run commands from the repository root after installing dependencies and configuring dataset paths.

## Training and evaluation

```bash
python tools/train.py -c configs/rtdetr/rtdetr_cytoset_seed42.yml --seed 42
python tools/train.py -c configs/rtdetr/rtdetr_cytoset_seed42.yml -r path/to/checkpoint.pth --test-only
```

Use `-r` to resume a checkpoint or `-t` to initialize compatible model parameters. These options are mutually exclusive.

## Model-forward latency

```bash
python tools/benchmark_efficiency.py --config configs/rtdetr/rtdetr_cytoset_seed42.yml --device cuda:0
```

The benchmark uses synthetic 640 x 640 inputs with batch size one on CUDA. It reports model-forward latency and its reciprocal image rate. Data loading, preprocessing and postprocessing are excluded. The image rate is derived from latency, rather than an independently measured throughput.

## Additional utilities

- `infer.py`: image visualization and experimental sliced inference.
- `export_onnx.py`: experimental ONNX export. The `--simplify` option requires `onnxsim`.

See [configuration and usage](../REPRODUCTION.md) for the supported workflow.
