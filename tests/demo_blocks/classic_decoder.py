import torch

from mmit.decoders.basedecoder import BaseDecoder


class ClassicDecoder(BaseDecoder):
    def __init__(self, input_channels=None, input_reductions=None, out_classes=1, **kwargs):
        super().__init__(input_channels, input_reductions)
        self.input_channels = input_channels
        self.input_reductions = input_reductions
        self._out_classes = out_classes

    def forward(self, *features):
        self._validate_forward(*features)
        img_h, img_w = self.get_input_image_size(features[0])
        batch_size = features[0].shape[0]

        return torch.rand(batch_size, self._out_classes, img_h, img_w)

    @property
    def out_classes(self):
        return self._out_classes

    def _validate_forward(self, *features):
        img_h, img_w = self.get_input_image_size(features[0])
        for channel, feature in zip(self.input_channels, features):
            wrong_channel = channel != feature.shape[1]
            if wrong_channel:
                raise ValueError(
                    f"Expected {channel} channels in the input, got {feature.shape[1]}"
                )

        for reduction, feature in zip(self.input_reductions, features):
            error_h = abs(img_h - feature.shape[2] * reduction)
            error_w = abs(img_w - feature.shape[3] * reduction)
            if error_h > 2 or error_w > 2:
                raise ValueError(
                    f"Expected {img_h}x{img_w} input image size, got {feature.shape[2]}x{feature.shape[3]}"
                )

    def get_input_image_size(self, first_feature):
        feat_h, feat_w = first_feature.shape[2:]
        init_reduction = self.input_reductions[0]
        return feat_h * init_reduction, feat_w * init_reduction
