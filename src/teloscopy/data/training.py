"""
teloscopy.data.training — Training dataset generation and management.

Provides tools for generating annotated training datasets from both
synthetic and real microscopy images.  Supports:

* **Synthetic dataset generation** with ground-truth labels for spot
  detection, segmentation, and telomere length estimation.
* **Training manifests** — JSON metadata describing each sample's
  annotations, split assignment (train/val/test), and provenance.
* **Data augmentation** — configurable transforms (rotation, flip, noise,
  intensity scaling) for training robustness.
* **Dataset loading** — iterable loaders with on-the-fly augmentation
  and optional caching.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrainingSample:
    """Descriptor for a single training image with annotations."""

    image_id: str
    image_path: str
    annotation_path: str
    split: str  # "train", "val", or "test"
    image_type: str  # "synthetic", "real_qfish", "real_fluorescence"
    width: int = 0
    height: int = 0
    n_chromosomes: int = 0
    n_spots: int = 0
    mean_telomere_bp: float = 0.0
    snr_db: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingManifest:
    """Metadata describing a complete training dataset."""

    name: str
    version: str
    created_at: str
    generator: str = "teloscopy.data.training"
    description: str = ""
    total_samples: int = 0
    splits: dict[str, int] = field(default_factory=dict)
    image_types: dict[str, int] = field(default_factory=dict)
    samples: list[TrainingSample] = field(default_factory=list)

    def save(self, path: str) -> None:
        """Save manifest as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        logger.info("Manifest saved: %s (%d samples)", path, self.total_samples)

    @classmethod
    def load(cls, path: str) -> TrainingManifest:
        """Load manifest from JSON."""
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        samples = [TrainingSample(**s) for s in d.pop("samples", [])]
        return cls(**d, samples=samples)


# ---------------------------------------------------------------------------
# Augmentation transforms
# ---------------------------------------------------------------------------


@dataclass
class AugmentationConfig:
    """Configuration for on-the-fly data augmentation."""

    random_rotation: bool = True
    rotation_range: float = 180.0  # degrees
    horizontal_flip: bool = True
    vertical_flip: bool = True
    intensity_scale_range: tuple[float, float] = (0.8, 1.2)
    additive_noise_std: float = 50.0
    gaussian_blur_sigma: float = 0.5
    random_crop: bool = False
    crop_fraction: float = 0.9


def augment_image(
    image: np.ndarray,
    spots: np.ndarray | None = None,
    config: AugmentationConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply random augmentations to an image and its spot coordinates.

    Parameters
    ----------
    image : np.ndarray
        2-D or 3-D image array (H, W) or (H, W, C).
    spots : np.ndarray | None
        Spot coordinates as (N, 2) array of (y, x) positions, or ``None``.
    config : AugmentationConfig | None
        Augmentation settings.  Defaults to ``AugmentationConfig()``.
    rng : np.random.Generator | None
        NumPy random generator for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        Augmented image and transformed spot coordinates.
    """
    if config is None:
        config = AugmentationConfig()
    if rng is None:
        rng = np.random.default_rng()

    img = image.astype(np.float64).copy()
    pts = spots.copy() if spots is not None else None
    h, w = img.shape[:2]

    # Horizontal flip
    if config.horizontal_flip and rng.random() > 0.5:
        img = np.flip(img, axis=1).copy()
        if pts is not None:
            pts[:, 1] = w - 1 - pts[:, 1]

    # Vertical flip
    if config.vertical_flip and rng.random() > 0.5:
        img = np.flip(img, axis=0).copy()
        if pts is not None:
            pts[:, 0] = h - 1 - pts[:, 0]

    # Random rotation
    if config.random_rotation:
        angle = rng.uniform(-config.rotation_range, config.rotation_range)
        rad = math.radians(angle)
        cy, cx = h / 2, w / 2
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        # Rotate image (using scipy if available, else skip)
        try:
            from scipy.ndimage import rotate as ndi_rotate

            img = ndi_rotate(img, angle, reshape=False, order=1, mode="constant")
        except ImportError:
            pass
        # Rotate spot coordinates around centre
        if pts is not None and len(pts) > 0:
            dy = pts[:, 0] - cy
            dx = pts[:, 1] - cx
            pts[:, 0] = cy + dy * cos_a - dx * sin_a
            pts[:, 1] = cx + dy * sin_a + dx * cos_a
            # Filter out-of-bounds spots
            valid = (
                (pts[:, 0] >= 0) & (pts[:, 0] < h) & (pts[:, 1] >= 0) & (pts[:, 1] < w)
            )
            pts = pts[valid]

    # Intensity scaling
    lo, hi = config.intensity_scale_range
    scale = rng.uniform(lo, hi)
    img = img * scale

    # Additive noise
    if config.additive_noise_std > 0:
        noise = rng.normal(0, config.additive_noise_std, img.shape)
        img = img + noise

    # Gaussian blur
    if config.gaussian_blur_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter

            if img.ndim == 2:
                img = gaussian_filter(img, sigma=config.gaussian_blur_sigma)
            else:
                for c in range(img.shape[2]):
                    img[:, :, c] = gaussian_filter(img[:, :, c], sigma=config.gaussian_blur_sigma)
        except ImportError:
            pass

    img = np.clip(img, 0, 65535)
    return img, pts


# ---------------------------------------------------------------------------
# Training dataset generator
# ---------------------------------------------------------------------------


class TrainingDatasetGenerator:
    """Generate annotated training datasets for Teloscopy models.

    Produces labelled microscopy images suitable for training spot
    detectors, segmentation networks, and telomere length estimators.

    Parameters
    ----------
    output_dir : str
        Root directory for generated data.
    name : str
        Dataset name (used in manifest).
    version : str
        Dataset version string.
    """

    def __init__(
        self,
        output_dir: str = "./training_data",
        name: str = "teloscopy_training_v1",
        version: str = "1.0.0",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.name = name
        self.version = version
        self._samples: list[TrainingSample] = []

    def generate(
        self,
        n_train: int = 200,
        n_val: int = 40,
        n_test: int = 40,
        image_size: tuple[int, int] = (512, 512),
        n_chromosomes_range: tuple[int, int] = (15, 46),
        snr_range: tuple[float, float] = (5.0, 20.0),
        seed: int = 42,
    ) -> TrainingManifest:
        """Generate a complete training dataset with train/val/test splits.

        Parameters
        ----------
        n_train, n_val, n_test : int
            Number of images per split.
        image_size : tuple[int, int]
            Generated image dimensions (H, W).
        n_chromosomes_range : tuple[int, int]
            Min/max chromosomes per image.
        snr_range : tuple[float, float]
            Signal-to-noise range in dB.
        seed : int
            Base random seed for reproducibility.

        Returns
        -------
        TrainingManifest
            Manifest describing all generated samples.
        """
        from teloscopy.telomere.synthetic import generate_metaphase_spread

        rng = np.random.default_rng(seed)
        self._samples = []

        splits = {"train": n_train, "val": n_val, "test": n_test}
        idx = 0

        for split_name, n_images in splits.items():
            split_dir = self.output_dir / split_name
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "annotations").mkdir(parents=True, exist_ok=True)

            for i in range(n_images):
                n_chrom = int(rng.integers(n_chromosomes_range[0], n_chromosomes_range[1] + 1))
                snr = float(rng.uniform(snr_range[0], snr_range[1]))
                noise_std = 200.0 / (10 ** (snr / 20))

                data = generate_metaphase_spread(
                    image_size=image_size,
                    n_chromosomes=n_chrom,
                    noise_std=noise_std,
                    seed=seed + idx,
                )

                image_id = f"{split_name}_{idx:05d}"
                img_path = str(split_dir / "images" / f"{image_id}.npz")
                ann_path = str(split_dir / "annotations" / f"{image_id}.json")

                # Save image as compressed numpy archive
                np.savez_compressed(
                    img_path,
                    dapi=data["dapi"].astype(np.float32),
                    cy3=data["cy3"].astype(np.float32),
                    labels=data["labels"].astype(np.int16),
                )

                # Save annotations as JSON
                spots = data["spots"]
                annotation = {
                    "image_id": image_id,
                    "n_chromosomes": n_chrom,
                    "n_spots": len(spots),
                    "spots": [
                        {
                            "y": float(s["y"]),
                            "x": float(s["x"]),
                            "intensity": float(s["intensity"]),
                            "length_bp": int(s["length_bp"]),
                            "chromosome_id": int(s.get("chromosome_id", 0)),
                        }
                        for s in spots
                    ],
                    "mean_telomere_bp": float(np.mean([s["length_bp"] for s in spots]))
                    if spots
                    else 0.0,
                    "snr_db": snr,
                    "image_size": list(image_size),
                    "seed": seed + idx,
                }

                with open(ann_path, "w", encoding="utf-8") as fh:
                    json.dump(annotation, fh, indent=2)

                self._samples.append(
                    TrainingSample(
                        image_id=image_id,
                        image_path=img_path,
                        annotation_path=ann_path,
                        split=split_name,
                        image_type="synthetic",
                        width=image_size[1],
                        height=image_size[0],
                        n_chromosomes=n_chrom,
                        n_spots=len(spots),
                        mean_telomere_bp=annotation["mean_telomere_bp"],
                        snr_db=snr,
                    )
                )
                idx += 1
                if idx % 50 == 0:
                    logger.info("Generated %d / %d images", idx, sum(splits.values()))

        manifest = TrainingManifest(
            name=self.name,
            version=self.version,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            description=(
                f"Synthetic qFISH microscopy training dataset. "
                f"{n_train} train / {n_val} val / {n_test} test images at "
                f"{image_size[0]}x{image_size[1]}px with {n_chromosomes_range[0]}-"
                f"{n_chromosomes_range[1]} chromosomes per image."
            ),
            total_samples=idx,
            splits={k: v for k, v in splits.items()},
            image_types={"synthetic": idx},
            samples=self._samples,
        )

        manifest.save(str(self.output_dir / "manifest.json"))
        logger.info(
            "Training dataset generated: %d images in %s", idx, self.output_dir
        )
        return manifest

    def add_real_images(
        self,
        image_dir: str,
        annotation_dir: str,
        split: str = "train",
        image_type: str = "real_qfish",
    ) -> int:
        """Register real microscopy images into the training dataset.

        Scans *image_dir* for TIFF files and *annotation_dir* for matching
        JSON annotations.  Appends to the internal sample list.

        Parameters
        ----------
        image_dir : str
            Directory containing TIFF images.
        annotation_dir : str
            Directory containing JSON annotation files.
        split : str
            Split assignment for all added images.
        image_type : str
            Image provenance type (e.g. ``"real_qfish"``).

        Returns
        -------
        int
            Number of images successfully registered.
        """
        img_dir = Path(image_dir)
        ann_dir = Path(annotation_dir)
        count = 0

        for img_path in sorted(img_dir.glob("*.tif")) + sorted(img_dir.glob("*.tiff")):
            ann_path = ann_dir / f"{img_path.stem}.json"
            if not ann_path.exists():
                logger.warning("No annotation for %s, skipping", img_path.name)
                continue

            with open(ann_path, encoding="utf-8") as fh:
                ann = json.load(fh)

            self._samples.append(
                TrainingSample(
                    image_id=img_path.stem,
                    image_path=str(img_path),
                    annotation_path=str(ann_path),
                    split=split,
                    image_type=image_type,
                    width=ann.get("image_size", [0, 0])[1],
                    height=ann.get("image_size", [0, 0])[0],
                    n_chromosomes=ann.get("n_chromosomes", 0),
                    n_spots=ann.get("n_spots", 0),
                    mean_telomere_bp=ann.get("mean_telomere_bp", 0.0),
                    snr_db=ann.get("snr_db", 0.0),
                )
            )
            count += 1

        logger.info("Added %d real images from %s", count, image_dir)
        return count


# ---------------------------------------------------------------------------
# Training data loader
# ---------------------------------------------------------------------------


class TrainingDataLoader:
    """Iterable loader for training datasets with on-the-fly augmentation.

    Parameters
    ----------
    manifest_path : str
        Path to a ``manifest.json`` file.
    split : str
        Which split to load (``"train"``, ``"val"``, or ``"test"``).
    augmentation : AugmentationConfig | None
        Augmentation settings.  ``None`` disables augmentation.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Randomly shuffle samples each epoch.
    seed : int
        Random seed for shuffling and augmentation.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        augmentation: AugmentationConfig | None = None,
        batch_size: int = 8,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.manifest = TrainingManifest.load(manifest_path)
        self.split = split
        self.augmentation = augmentation if split == "train" else None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        self._samples = [s for s in self.manifest.samples if s.split == split]
        logger.info(
            "Loaded %d %s samples from %s",
            len(self._samples),
            split,
            manifest_path,
        )

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return math.ceil(len(self._samples) / self.batch_size)

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        """Yield batches of loaded and augmented samples."""
        indices = np.arange(len(self._samples))
        if self.shuffle:
            self.rng.shuffle(indices)

        batch: list[dict[str, Any]] = []
        for idx in indices:
            sample = self._samples[idx]
            loaded = self._load_sample(sample)
            if loaded is not None:
                batch.append(loaded)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _load_sample(self, sample: TrainingSample) -> dict[str, Any] | None:
        """Load a single sample from disk with optional augmentation."""
        try:
            if sample.image_path.endswith(".npz"):
                data = np.load(sample.image_path)
                image = data["cy3"]  # use Cy3 channel for spot detection
                dapi = data["dapi"]
                labels = data["labels"]
            else:
                # Real TIFF images
                try:
                    from skimage.io import imread

                    image = imread(sample.image_path).astype(np.float64)
                    dapi = image
                    labels = None
                except ImportError:
                    logger.warning("scikit-image not available; skipping %s", sample.image_id)
                    return None

            # Load spot annotations
            with open(sample.annotation_path, encoding="utf-8") as fh:
                ann = json.load(fh)
            spots = np.array(
                [[s["y"], s["x"]] for s in ann.get("spots", [])],
                dtype=np.float64,
            )
            if spots.size == 0:
                spots = np.empty((0, 2), dtype=np.float64)

            # Apply augmentation
            if self.augmentation is not None:
                image, spots = augment_image(image, spots, self.augmentation, self.rng)

            return {
                "image_id": sample.image_id,
                "image": image,
                "dapi": dapi,
                "labels": labels,
                "spots": spots,
                "n_spots": len(spots),
                "mean_telomere_bp": sample.mean_telomere_bp,
                "annotation": ann,
            }
        except Exception:
            logger.warning("Failed to load sample %s", sample.image_id, exc_info=True)
            return None
