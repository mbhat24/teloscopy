"""Image type classification: distinguish FISH microscopy from regular photos.

Provides a heuristic classifier that examines image properties (channel
distribution, intensity histogram, spatial frequency content, bit depth)
to determine whether an uploaded image is a qFISH fluorescence microscopy
image or a regular photograph (e.g. selfie, portrait).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

try:
    from enum import StrEnum
except ImportError:  # Python < 3.11
    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """Backport of StrEnum for Python 3.9/3.10."""

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageType(StrEnum):
    """Detected image type."""

    FISH_MICROSCOPY = "fish_microscopy"
    FACE_PHOTO = "face_photo"
    UNKNOWN_PHOTO = "unknown_photo"


@dataclass
class ClassificationResult:
    """Result of image-type classification."""

    image_type: ImageType
    confidence: float  # 0.0–1.0
    face_detected: bool
    is_fluorescence: bool
    detail: str


def _is_fluorescence_like(img: np.ndarray) -> tuple[bool, float]:
    """Check if image has fluorescence microscopy characteristics.

    Fluorescence images typically have:
    - Very dark background (majority of pixels near zero)
    - Sparse bright spots on dark background
    - High dynamic range concentrated in one or two channels
    - Skewed intensity histogram (most pixels dark, few very bright)
    """
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalise to 0-255 for analysis
    if gray.dtype == np.uint16:
        gray = (gray / 256).astype(np.uint8)
    elif gray.dtype == np.float64 or gray.dtype == np.float32:
        gray = (gray * 255).clip(0, 255).astype(np.uint8)

    total_pixels = gray.size
    score = 0.0

    # Check 1: Dark background dominance (>60% of pixels below 20% intensity)
    dark_fraction = np.sum(gray < 50) / total_pixels
    if dark_fraction > 0.60:
        score += 0.3
    if dark_fraction > 0.80:
        score += 0.2

    # Check 2: Histogram skewness (fluorescence is heavily right-skewed)
    mean_val = np.mean(gray)
    if mean_val < 60:
        score += 0.2

    # Check 3: High contrast ratio (bright spots vs background)
    p99 = np.percentile(gray, 99)
    p50 = np.percentile(gray, 50)
    if p50 > 0 and p99 / max(p50, 1) > 5:
        score += 0.2

    # Check 4: Channel separation (fluorescence often has signal in 1-2 channels)
    if img.ndim == 3 and img.shape[2] >= 3:
        b, g, r = cv2.split(img[:, :, :3])
        if b.dtype == np.uint16:
            b, g, r = b / 256, g / 256, r / 256
        channel_means = [np.mean(b), np.mean(g), np.mean(r)]
        max_ch = max(channel_means)
        min_ch = min(channel_means)
        if max_ch > 0 and min_ch / max_ch < 0.3:
            score += 0.1

    is_fluor = score >= 0.5
    return is_fluor, min(score, 1.0)


def _detect_face(img: np.ndarray) -> tuple[bool, list[tuple[int, int, int, int]]]:
    """Detect faces using multiple OpenCV classifiers for robustness.

    Tries frontal Haar cascade first, then profile cascade, then LBP
    cascade, each with progressively relaxed parameters to maximise
    recall on real-world photos (varying lighting, angles, occlusion).

    Returns (found, list_of_bounding_boxes).
    """
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if gray.dtype != np.uint8:
        if gray.dtype == np.uint16:
            gray = (gray / 256).astype(np.uint8)
        else:
            gray = (gray * 255).clip(0, 255).astype(np.uint8)

    # Histogram equalisation for improved detection under poor lighting.
    gray_eq = cv2.equalizeHist(gray)

    face_list: list[tuple[int, int, int, int]] = []

    # Strategy 1: Frontal Haar cascade (default parameters)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) > 0:
        face_list = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        return True, face_list

    # Strategy 2: Frontal Haar cascade with relaxed parameters
    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) > 0:
        face_list = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        return True, face_list

    # Strategy 3: Frontal Haar alt cascade
    alt_path = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
    alt_cascade = cv2.CascadeClassifier(alt_path)
    faces = alt_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(40, 40),
    )
    if len(faces) > 0:
        face_list = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        return True, face_list

    # Strategy 4: Alt2 cascade (tree-based)
    alt2_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    alt2_cascade = cv2.CascadeClassifier(alt2_path)
    faces = alt2_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(40, 40),
    )
    if len(faces) > 0:
        face_list = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        return True, face_list

    # Strategy 5: Profile face cascade
    profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
    profile_cascade = cv2.CascadeClassifier(profile_path)
    faces = profile_cascade.detectMultiScale(
        gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40)
    )
    if len(faces) > 0:
        face_list = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        return True, face_list

    # Strategy 6: LBP frontal cascade (faster, different feature space)
    lbp_path = cv2.data.lbpcascades + "lbpcascade_frontalface_improved.xml"
    lbp_cascade = cv2.CascadeClassifier(lbp_path)
    if not lbp_cascade.empty():
        faces = lbp_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40)
        )
        if len(faces) > 0:
            face_list = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
            return True, face_list

    return False, []


def classify_image(image_path: str) -> ClassificationResult:
    """Classify an uploaded image as FISH microscopy or regular photo.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    ClassificationResult
        Classification with confidence score and detection details.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return ClassificationResult(
            image_type=ImageType.UNKNOWN_PHOTO,
            confidence=0.0,
            face_detected=False,
            is_fluorescence=False,
            detail="Could not read image file.",
        )

    is_fluor, fluor_score = _is_fluorescence_like(img)
    face_found, face_boxes = _detect_face(img)

    logger.info(
        "Image classification: fluorescence_score=%.2f, face_detected=%s, faces=%d",
        fluor_score,
        face_found,
        len(face_boxes),
    )

    if face_found and not is_fluor:
        return ClassificationResult(
            image_type=ImageType.FACE_PHOTO,
            confidence=min(0.6 + 0.4 * (1.0 - fluor_score), 1.0),
            face_detected=True,
            is_fluorescence=False,
            detail=f"Detected {len(face_boxes)} face(s) in regular photograph.",
        )
    elif is_fluor and not face_found:
        return ClassificationResult(
            image_type=ImageType.FISH_MICROSCOPY,
            confidence=fluor_score,
            face_detected=False,
            is_fluorescence=True,
            detail="Image has fluorescence microscopy characteristics.",
        )
    elif face_found and is_fluor:
        # Unlikely but possible — face takes priority
        return ClassificationResult(
            image_type=ImageType.FACE_PHOTO,
            confidence=0.6,
            face_detected=True,
            is_fluorescence=True,
            detail="Face detected in image with fluorescence-like properties.",
        )
    else:
        return ClassificationResult(
            image_type=ImageType.UNKNOWN_PHOTO,
            confidence=0.5,
            face_detected=False,
            is_fluorescence=False,
            detail="Image type uncertain: no face detected and not fluorescence microscopy.",
        )
