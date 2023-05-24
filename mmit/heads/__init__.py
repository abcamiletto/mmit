from .classification import ClassificationHead
from .segmentation import SegmentationHead

heads_builder = {"classification": ClassificationHead, "segmentation": SegmentationHead}
