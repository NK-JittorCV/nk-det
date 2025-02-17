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

New state-of-the-art models are also being implemented:

- SARDet100K

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

# Citation

If this work is helpful for your research, please consider citing the following entry.

```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}

@inproceedings{wang2024crosskd,
  title={CrossKD: Cross-head knowledge distillation for object detection},
  author={Wang, Jiabao and Chen, Yuming and Zheng, Zhaohui and Li, Xiang and Cheng, Ming-Ming and Hou, Qibin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16520--16530},
  year={2024}
}

@article{li2024sardet,
  title={Sardet-100k: Towards open-source benchmark and toolkit for large-scale sar object detection},
  author={Li, Yuxuan and Li, Xiang and Li, Weijie and Hou, Qibin and Liu, Li and Cheng, Ming-Ming and Yang, Jian},
  journal={arXiv preprint arXiv:2403.06534},
  year={2024}
}
```

# Acknowledge

Our code is developed on top of following open source codebase:

- [Jittor](https://github.com/Jittor/jittor)
- [JDet](https://github.com/Jittor/JDet)
- [MMCV](https://github.com/open-mmlab/mmcv)
- [MMDetection](https://github.com/open-mmlab/mmdetection)

We sincerely appreciate their amazing works.
