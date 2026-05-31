# CytoSet

**CytoSet: Continuum-Aware Set Prediction for Fine-Grained Cervical Cytology Cell Detection**

> Published in **IEEE Transactions on Medical Imaging (TMI)**

## Abstract

Squamous cell detection in cervical ThinPrep Cytologic Test (TCT) images must localize abnormal cells and assign Bethesda grades consistent with lesion severity. Unlike generic object detection with independent categories, the five squamous Bethesda grades (ASC-US, LSIL, ASC-H, HSIL, SCC) form an ordered severity progression with weak morphological boundaries between adjacent levels.

We propose **CytoSet**, a task-structured adaptation of RT-DETR that aligns its set-prediction pipeline with the ordered lesion structure through three coupled corrections:

- **DCSR** (Dual-Codebook Semantic Refinement): Refines encoder semantics by separating foreground-like and background-like cytological responses
- **SGOS** (Spatial-Grouped One-to-One Selector): Groups multi-scale proposals from the same cell into a single spatially consistent query
- **CCAQ** (Continuum-Constrained Alignment and Quality Calibration): Regularizes matched decoder queries using ordered Bethesda class centers and severity-aware calibration

## Results

On the five-class **TCT-MS** benchmark:

| Metric | RT-DETR | CytoSet |
|:---:|:---:|:---:|
| mAP | 37.16 | **39.08** |
| AP@0.5 | 51.18 | **53.72** |
| AP@0.75 | 41.58 | **43.52** |
| SCR (cross-grade error) | 7.01 | **5.78** |
| HR-DG (high-risk downgrade) | 10.37 | **7.82** |

CytoSet reduces high-risk downgrading from 32.1% to 21.4% on the internal WSI cohort and from 28.5% to 22.8% on Tianchi.

## Architecture

CytoSet preserves the RT-DETR pipeline and introduces three task-structured corrections:

```
Input Image
    |
[Backbone (PResNet-50vd)]
    |
    +-- S3, S4, S5 features
            |
        [DCSR on S5] --> Foreground/Background semantic refinement
            |
        [CCFF] --> Cross-scale feature fusion
            |
        [SGOS] --> Spatial-grouped proposal selection
            |
        [CCAQ] --> Ordered Bethesda grade supervision
            |
        Detection Output (5 Bethesda classes)
```

## Code

**Coming Soon!**

We are preparing the code for public release. Please stay tuned.

## Citation

If you find this work useful, please cite:

```bibtex
@article{he2026cytoset,
  title={CytoSet: Continuum-Aware Set Prediction for Fine-Grained Cervical Cytology Cell Detection},
  author={He, Cheng and Chen, Fufeng and Chen, Fuhai and Guo, Weiwei and Yang, Chunxu and Chen, Zhiwei and Zhou, Qingqing},
  journal={IEEE Transactions on Medical Imaging},
  year={2026}
}
```

## Acknowledgements

This work was supported by:
- Fujian Provincial Department of Education Youth Project (Grant No. JZ230006)
- Engineering Research Center of Big Data Intelligence, Ministry of Education, China
- Fujian Key Laboratory of Network Computing and Intelligent Information Processing (Fuzhou University)
- National Natural Science Foundation of China (Grant No. 62306306)
- Guangdong Basic and Applied Basic Research Foundation (Grant No. 2022A1515110128)

## Contact

- Fuhai Chen (Corresponding Author): chenfuhai3c@163.com
- GitHub: [zms618/CytoSet](https://github.com/zms618/CytoSet)
