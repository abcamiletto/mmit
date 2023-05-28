# Welcome to mmit's documentation

```{include} ../../README.md
:relative-images:
:heading-offset: 2
:start-after: <!--Introduction-->
:end-before: <!--End Introduction-->
```

```{include} ../../README.md
:relative-images:
:heading-offset: 2
:start-after: <!--Main Features-->
:end-before: <!--End Main Features-->
```

```{toctree}
:caption: 'User Guide'
:maxdepth: 1

guide/install
guide/quickstart
guide/examples
```

```{toctree}
:caption: 'API to ...'
:maxdepth: 1

api/create_model
api/create_decoder
api/create_encoder
```

```{toctree}
:caption: 'Decoders'
:maxdepth: 1

modules/decoders/unet
modules/decoders/unet++
modules/decoders/fpn
modules/decoders/deeplabv3
modules/decoders/deeplabv3+
modules/decoders/pspnet
modules/decoders/linknet
modules/decoders/pan
```

```{toctree}
:caption: 'Encoders'
:maxdepth: 1

modules/encoders/timm
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
