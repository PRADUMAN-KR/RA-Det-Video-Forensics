"""
Detectors module for RA-Det Production Cascade.

Includes:
- Layer 1: C2PA Provenance Checker
- Layer 2: VideoMAE Anomaly Detector
"""

from .provenance import check_provenance
from .anomaly_detector import VideoAnomalyDetector

__all__ = ["check_provenance", "VideoAnomalyDetector"]
