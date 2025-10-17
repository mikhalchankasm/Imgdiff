"""
Imgdiff - Инструмент для сравнения изображений с высокой производительностью
"""

__version__ = "2.0.0"

from .core.diff import diff_mask_fast, coarse_to_fine, ssim_mask
from .core.colors import bgr_to_lab_diff
from .core.morph import bboxes_from_mask, filter_small_components
from .core.overlay import draw_diff_overlay, create_heatmap

__all__ = [
    "diff_mask_fast",
    "coarse_to_fine",
    "ssim_mask",
    "bgr_to_lab_diff",
    "bboxes_from_mask",
    "filter_small_components",
    "draw_diff_overlay",
    "create_heatmap",
]

