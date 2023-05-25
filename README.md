![LogoTitle](docs/source/_static/logo/logo_title.png)

<!--Introduction-->

<div align="center">

 **mmit** is a python library to build any encoder matched with any decoder for any Computer Vision model.

[![License badge](https://img.shields.io/github/license/abcamiletto/mmit?style=for-the-badge)](https://github.com/abcamiletto/mmit/blob/master/LICENSE)
![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.10+-red?style=for-the-badge&logo=pytorch)
![Python - Version](https://img.shields.io/badge/PYTHON-3.8+-red?style=for-the-badge&logo=python&logoColor=white)

</div>
<!--End Introduction-->

For a quick overview of **mmit**, check out the [documentation](https://mmit.readthedocs.io/en/latest/).

Let's take a look at what we have here!

- [Main Features](#main-features-)
- [Installation](#installation-)
- [Quick Start](#quick-start-)
- [To Do List](#to-do-list)

## [Main Features](#main-features) <!--Main Features-->

**mmit** is engineered with the objective of streamlining the construction of Computer Vision models. It offers a consistent interface for all encoders and decoders, thus enabling effortless integration of any desired combination.

Here are just a few of the things that mmit does well:

- **Any encoder works with any decoder at any input size**
- **Unified interface** for all decoders
- Support for all pretrained **encoders from timm**
- Pretrained encoder+decoders modules 🚧
- PEP8 compliant (unified code style)
- Tests, high code coverage and type hints
- Clean code

<!--End Main Features-->
## [Installation](#installation) <!--Installation-->

To install mmit:

```console
pip install mmit
```
<!--End Installation-->

## [Quick Start](#quick-start) <!--Quick Start-->

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
<!--End Quick Start-->
## [To Do List](#to-do-list)

In the future, we plan to add support for:

- [x] timm encoders
- [ ] some of timm transformers encoders with feature extraction
- [ ] torchvision / torchub models
- [ ] more decoders
- [ ] lightning script to train models
- [x] multiple heads
- [ ] popular loss function
- [ ] popular datasets
- [ ] popular metrics

## [Awesome Sources](#awesome-sources) <!-- omit in toc -->

This project is inspired by, and would not be possible without, the following amazing libraries

- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Timm](https://github.com/huggingface/pytorch-image-models)
- [SMP](https://github.com/qubvel/segmentation_models.pytorch)
