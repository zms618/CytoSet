# Configuration and usage

## Dataset layout

The default configuration uses the following layout:

```text
dataset/tct/
  images/
  annotations/
    instances_train.json
    instances_val.json
```

Set `img_folder` and `ann_file` in `configs/dataset/tct_5class.yml` to match your dataset. Each annotation's `file_name` must resolve relative to the image directory. Category IDs and ordinal ranks are listed in the root README.

Public datasets are available from the [Tianchi competition](https://tianchi.aliyun.com/competition/entrance/231757/introduction) and the [CRIC Cervix database](https://database.cric.com.br/). Use 640 × 640 patches with COCO-format bounding-box annotations.

## Model configuration

The CytoSet configuration uses PResNet-50vd, 640 x 640 inputs, 50 detection queries and two decoder layers. DCSR, SGQS and OCAQ are enabled. Model-level multi-scale resizing is disabled, and postprocessing returns the top 50 detections.

The three seed configurations share these model settings. Pass the corresponding random seed explicitly with `--seed`; the filename does not set the random seed.

The architecture figure uses SGOS and SGQS to denote the spatial grouping component.

## Training and checkpoints

Activate your Python environment before running commands from the repository root.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_cytoset_seed42.yml --seed 42
```

Use `-r path/to/checkpoint.pth` to resume training or `-t path/to/checkpoint.pth` to initialize compatible model parameters. Resume and tuning are mutually exclusive.

Outputs are written to the configuration's `output_dir`. Select a saved checkpoint explicitly when evaluating:

```bash
python tools/train.py -c configs/rtdetr/rtdetr_cytoset_seed42.yml -r path/to/checkpoint.pth --test-only
```

## Supported workflow

The documented workflow uses single-GPU FP32 cell detection and COCO-format evaluation. Other backbone configurations, mixed precision, distributed training, sliced-image inference and export utilities are experimental. The paper's WSI analysis is outside the scope of this package.
