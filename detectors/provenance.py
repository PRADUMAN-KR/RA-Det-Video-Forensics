"""
Layer 1: C2PA Digital Provenance Metadata Reader.

Parses digital birth certificates and metadata manifests in media files.
Returns instant real/fake verdicts for C2PA-compliant AI generators or camera hardware.
"""

import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("ra_det_provenance")

# Try importing official Rust-backed C2PA Python SDK
C2PA_SDK_AVAILABLE = False
try:
    import c2pa
    C2PA_SDK_AVAILABLE = True
except ImportError:
    logger.warning("c2pa-python SDK not installed. Falling back to binary manifest inspection.")


KNOWN_AI_SIGNATURES = [
    b"c2pa", b"com.adobe.c2pa", b"openai.sora", b"runway.gen2", b"runway.gen3",
    b"pika.art", b"luma.dreammachine", b"kling.ai", b"google.veo", b"alibaba.wan",
    b"minimax.hailuo", b"lightricks.ltx", b"stability.sdv", b"midjourney", b"dall-e"
]

KNOWN_CAMERA_SIGNATURES = [
    b"sony_alpha", b"nikon_z", b"canon_eos", b"leica_c2pa", b"apple_iphone"
]


def check_provenance(file_path: str) -> Dict[str, Any]:
    """
    Check video file for C2PA digital provenance metadata.

    Returns:
        dict containing:
            verdict: "ai_generated", "camera_captured", or "unknown"
            confidence: float (1.0 for hard C2PA match)
            generator: str or None
            manifest_found: bool
            checked: True
    """
    if not os.path.exists(file_path):
        return {
            "verdict": "unknown",
            "confidence": 0.0,
            "generator": None,
            "manifest_found": False,
            "checked": True,
            "error": "File not found"
        }

    # Method A: Try official c2pa library
    if C2PA_SDK_AVAILABLE:
        try:
            reader = c2pa.Reader.from_file(file_path)
            manifest_json_str = reader.json()
            if manifest_json_str:
                manifest = json.loads(manifest_json_str)
                active_manifest = manifest.get("active_manifest", {})
                assertions = active_manifest.get("assertions", [])

                generator_name = active_manifest.get("claim_generator", "Unknown C2PA Generator")
                
                # Check assertions for AI generation actions
                is_ai = False
                for assertion in assertions:
                    label = assertion.get("label", "")
                    data = assertion.get("data", {})
                    if "c2pa.actions" in label:
                        actions = data.get("actions", [])
                        for action in actions:
                            if action.get("action") in ["c2pa.created", "c2pa.placed", "c2pa.edited"]:
                                digital_source_type = action.get("digitalSourceType", "")
                                if "trainedAlgorithmicMedia" in digital_source_type or "compositeWithTrainedAlgorithmicMedia" in digital_source_type:
                                    is_ai = True

                if is_ai:
                    return {
                        "verdict": "ai_generated",
                        "confidence": 1.0,
                        "generator": generator_name,
                        "manifest_found": True,
                        "checked": True
                    }
                else:
                    return {
                        "verdict": "camera_captured",
                        "confidence": 0.95,
                        "generator": None,
                        "manifest_found": True,
                        "checked": True
                    }
        except Exception as e:
            # Manifest not present or corrupted
            logger.debug(f"C2PA reader exception for {file_path}: {e}")

    # Method B: Direct binary chunk inspection (fallback when c2pa python is not present)
    try:
        with open(file_path, "rb") as f:
            header_bytes = f.read(5 * 1024 * 1024)  # Read first 5MB for JUMBF box or MP4 metadata
            
            # Check for AI generator signatures
            for sig in KNOWN_AI_SIGNATURES:
                if sig in header_bytes.lower():
                    gen_str = sig.decode("utf-8", errors="ignore")
                    return {
                        "verdict": "ai_generated",
                        "confidence": 0.99,
                        "generator": gen_str,
                        "manifest_found": True,
                        "checked": True
                    }

            # Check for Camera C2PA signatures
            for sig in KNOWN_CAMERA_SIGNATURES:
                if sig in header_bytes.lower():
                    return {
                        "verdict": "camera_captured",
                        "confidence": 0.95,
                        "generator": None,
                        "manifest_found": True,
                        "checked": True
                    }
    except Exception as e:
        logger.debug(f"Binary inspection exception: {e}")

    return {
        "verdict": "unknown",
        "confidence": 0.0,
        "generator": None,
        "manifest_found": False,
        "checked": True
    }
