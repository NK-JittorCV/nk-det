# JittorDet

## introduction

JittorDet is an object detection benchmark based on [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/).

## Supported Models

JittorDet supports commonly used datasets (COCO, VOC) and models (RetinaNet, Faster R-CNN) out of box.

Currently supported models are as below:

- RetinaNet
- Faster R-CNN
- GFocalLoss
- PKD
- ⭐ CrossKD: [https://github.com/jbwang1997/CrossKD](https://github.com/jbwang1997/CrossKD)

New state-of-the-art models are being implemented.

## Getting Started

### Install

Please first follow the [tutorial](https://github.com/Jittor/jittor) to install jittor.
Here, we recommend using jittor==1.3.6.10, which we have tested on.

Then, install the `jittordet` by running:
```
pip install -v -e .
```

If you want to use multi-gpu training or testing, please install OpenMPI
```
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
```

### Training

We support single-gpu, multi-gpu training.
```
#Single-GPU
python tools/train.py {CONFIG_PATH}

# Multi-GPU
bash tools/dist_train.sh {CONFIG_PATH} {NUM_GPUS}
```

### Testing

We support single-gpu, multi-gpu testing.
```
#Single-GPU
python tools/test.py {CONFIG_PATH}

# Multi-GPU
bash tools/dist_test.sh {CONFIG_PATH} {NUM_GPUS}
```



## Citation


```bibtex
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}

@InProceedings{wang2024crosskd,
    author    = {Wang, Jiabao and Chen, Yuming and Zheng, Zhaohui and Li, Xiang and Cheng, Ming-Ming and Hou, Qibin},
    title     = {CrossKD: Cross-Head Knowledge Distillation for Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {16520-16530}
}
```
