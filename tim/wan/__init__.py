"""Wan2.1 training vs inference mismatch toolkit."""

from tim.wan.metrics import TimVideoReport, compare_videos
from tim.wan.profiles import INFER_PROFILES, InferProfile, get_profile

__all__ = [
    "InferProfile",
    "INFER_PROFILES",
    "get_profile",
    "TimVideoReport",
    "compare_videos",
]
