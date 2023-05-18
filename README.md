# Multi-Models for Images in pyTorch (MMIT)

<div align="center">

 **mmit** is a python library with pretrained building blocks for Computer Vision models.

[![License badge](https://img.shields.io/github/license/abcamiletto/mmit?style=for-the-badge)](https://github.com/abcamiletto/mmit/blob/master/LICENSE)
![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.10+-red?style=for-the-badge&logo=pytorch)
![Python - Version](https://img.shields.io/badge/PYTHON-3.8+-red?style=for-the-badge&logo=python&logoColor=white)

</div>

## Main Features

**mmit** is engineered with the objective of streamlining the construction of Computer Vision models. It offers a consistent interface for all encoders and decoders, thus enabling effortless integration of any desired combination.

In terms of encoders, mmit is compatible with all encoders from [timm](https://github.com/huggingface/pytorch-image-models) utilizing a standardized API. However, it is noteworthy that timm does not accommodate feature extraction for transformer encoders. To resolve this, we have adopted a principled stance and introduced support for a select number of these encoders.

Regarding decoders, mmit currently facilitates UNet and UNet++ decoders, again employing a unified API. Our roadmap includes plans for incorporating additional decoders in the future.

One of the distinctive features of mmit is its ability to automatically construct the decoder to correspond with the output shape of any given encoder. This ensures seamless compatibility and interoperability, enhancing the user experience and the efficiency of model building.

Wrapping up, our main features are:

- Feature Extraction with Transformer Backbones
- Modular decoders that works with any encoders
- Pretrained encoder+decoders modules

## Installation

We can simply install mmit using pip:

```bash
pip install mmit
```

## Quick Start

Let's look at a super simple example of how to use mmit:

```python
import torch
import mmit

encoder = mmit.create_encoder('resnet18')
decoder = mmit.create_decoder('unetplusplus') # automatically matches encoder output shape!

x = torch.randn(2, 3, 256, 256)
features = encoder(x)
out = decoder(*features)
```

## To Do List

In the future, we plan to add support for:

- [x] timm encoders
- [ ] some of timm transformers encoders
- [ ] torchvision / torchub models
- [x] UNet and UNet++ decoders
- [ ] FPN
- [ ] DeepLabV3
- [ ] DeepLabV3+
- [ ] API for building end-to-end models
- [ ] multiple heads
- [ ] popular loss function
- [ ] popular datasets
- [ ] popular metrics

## Awesome Sources

- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Timm](https://github.com/huggingface/pytorch-image-models)
- [SMP](https://github.com/qubvel/segmentation_models.pytorch)
