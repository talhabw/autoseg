"""
ML module - SAM segmentation, encoder embeddings, and propagation
"""

from ml.segment import SegmentService, get_segment_service
from ml.embed import EmbedService, get_embed_service
from ml.propagate import PropagateService, generate_candidates, propose_bboxes

__all__ = [
    "SegmentService", "get_segment_service",
    "EmbedService", "get_embed_service",
    "PropagateService", "generate_candidates", "propose_bboxes",
]
