# Examples <!-- omit in toc -->

Let's take a look at what we have here!

- [Build a Segmentation Model](#build-a-segmentation-model)
- [Build a Decoder](#build-a-decoder)
- [Customize a Decoder](#customize-a-decoder)

## [Build a Segmentation Model](#build-a-segmentation-model)

Let's say you want to build a segmentation model. You can do it like this:

```python
import mmit
import torch

model = mmit.create_model('resnet18', 'unet', num_classes=2)

x = torch.randn(2, 3, 256, 256)
out = model(x)
```

## [Build a Decoder](#build-a-decoder)

Let's say you want to build a decoder for a given encoder. You can do it like this:

```python
import mmit
import torch

encoder = mmit.create_encoder('resnet18', out_indices=(0, 1, 4), output_stride=8)
decoder = mmit.create_decoder('unet') # automatically matches encoder characteristics!

x = torch.randn(2, 3, 256, 256)
features = encoder(x)
out = decoder(*features)
```

Written like this, it feels like a lot of magic is going on. The explicit way to do this is:

```python
encoder = mmit.create_encoder('resnet18', out_indices=(0, 1, 4), output_stride=8)
decoder = mmit.create_decoder('unetplusplus', encoder.out_channels, encoder.out_reductions)
```

## [Customize a Decoder](#customize-a-decoder)

The available customization options are a lot, but here are an example:

```python
import mmit
import torch

encoder = mmit.create_encoder('resnet18', out_indices=(0, 1, 4), output_stride=8)
decoder = mmit.create_decoder('unetplusplus', upsample_layer='interpolate', activation_layer='leaky-relu')

x = torch.randn(2, 3, 256, 256)
features = encoder(x)
out = decoder(*features)
```
