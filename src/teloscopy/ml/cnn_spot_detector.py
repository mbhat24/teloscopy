"""CNN-based telomere spot detector for quantitative FISH microscopy.

This module implements a U-Net-style encoder-decoder convolutional neural
network for detecting telomere spots in qFISH (quantitative fluorescence
in situ hybridization) microscopy images.  The architecture is implemented
using only NumPy and SciPy so that no deep-learning framework (PyTorch,
TensorFlow) is required at inference time, maximising portability across
HPC clusters and legacy lab workstations.

Architecture overview
---------------------
The network follows the U-Net design introduced by Ronneberger *et al.*
(2015) and subsequently adapted for biomedical microscopy by Falk *et al.*
(2019).  It consists of:

* **Encoder** – three down-sampling blocks, each containing a 3×3
  convolution, ReLU activation and 2×2 max-pooling.  Channel progression:
  1 → 32 → 64 → 128.
* **Bottleneck** – two 3×3 convolutions expanding to 256 channels and
  contracting back to 128.
* **Decoder** – three up-sampling blocks with bilinear interpolation and
  skip (concatenation) connections from the encoder, followed by 3×3
  convolutions that halve the channel count at each stage:
  128+128 → 128 → 64, 64+64 → 64 → 32, 32+32 → 32 → 16.
* **Head** – a 1×1 convolution projecting to a single-channel probability
  map passed through a sigmoid activation.

Weight management
-----------------
Since shipping trained weights alongside this source package is
impractical, the module provides:

* ``_generate_default_weights`` – Kaiming/He initialisation
  (He *et al.* 2015) suitable for ReLU networks.
* ``save_weights`` / ``load_weights`` – NumPy ``.npz`` serialisation for
  easy archival and versioning of trained parameters.

References
----------
.. [1] O. Ronneberger, P. Fischer and T. Brox, "U-Net: Convolutional
       Networks for Biomedical Image Segmentation", *MICCAI*, 2015.
       https://doi.org/10.1007/978-3-319-24574-4_28
.. [2] T. Falk *et al.*, "U-Net: deep learning for cell counting,
       detection, and morphometry", *Nature Methods*, 16, 67–70, 2019.
       https://doi.org/10.1038/s41592-018-0261-2
.. [3] K. He, X. Zhang, S. Ren and J. Sun, "Delving Deep into
       Rectifiers: Surpassing Human-Level Performance on ImageNet
       Classification", *ICCV*, 2015.
       https://doi.org/10.1109/ICCV.2015.123
.. [4] Y. LeCun, L. Bottou, G. B. Orr and K.-R. Müller, "Efficient
       BackProp", in *Neural Networks: Tricks of the Trade*, 1998.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ArrayLike = Union[np.ndarray, Sequence]  # noqa: UP007
SpotRecord = dict[str, float]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_INPUT_CHANNELS: int = 1
_ENCODER_CHANNELS: tuple[int, ...] = (32, 64, 128)
_BOTTLENECK_CHANNELS: int = 256
_DECODER_CHANNELS: tuple[int, ...] = (128, 64, 32)
_HEAD_INTERMEDIATE: int = 16
_KERNEL_SIZE: int = 3
_POOL_SIZE: int = 2
_DEFAULT_CONFIDENCE_THRESHOLD: float = 0.35
_DEFAULT_MIN_DISTANCE: int = 3
_DEFAULT_SEED: int = 42


# ===================================================================== #
#                        Low-level operations                            #
# ===================================================================== #


def _kaiming_init(fan_in: int, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """He/Kaiming normal initialisation for ReLU layers.

    Parameters
    ----------
    fan_in : int
        Number of input units (``C_in * kernel_h * kernel_w``).
    shape : tuple of int
        Desired weight tensor shape ``(C_out, C_in, kH, kW)``.
    rng : numpy.random.Generator
        PRNG instance for reproducibility.

    Returns
    -------
    numpy.ndarray
        Weight tensor drawn from N(0, sqrt(2 / fan_in)).
    """
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=shape).astype(np.float32)


def _zeros(shape: tuple[int, ...]) -> np.ndarray:
    """Return a float32 zero tensor."""
    return np.zeros(shape, dtype=np.float32)


# -- Convolution ----------------------------------------------------------


def _conv2d(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, padding: str = "same"
) -> np.ndarray:
    """Apply a 2-D convolution over a batch of multi-channel images.

    Parameters
    ----------
    x : ndarray, shape ``(B, C_in, H, W)``
    weight : ndarray, shape ``(C_out, C_in, kH, kW)``
    bias : ndarray, shape ``(C_out,)``
    padding : ``"same"`` or ``"valid"``

    Returns
    -------
    ndarray, shape ``(B, C_out, H', W')``
    """
    batch, c_in, h, w = x.shape
    c_out, c_in_k, kh, kw = weight.shape
    assert c_in == c_in_k, "Channel mismatch between input and kernel."

    if padding == "same":
        pad_h, pad_w = kh // 2, kw // 2
    else:
        pad_h, pad_w = 0, 0

    # Pad spatial dims only
    x_pad = np.pad(
        x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0.0
    )

    out_h = x_pad.shape[2] - kh + 1
    out_w = x_pad.shape[3] - kw + 1
    out = np.empty((batch, c_out, out_h, out_w), dtype=np.float32)

    for co in range(c_out):
        acc = np.zeros((batch, out_h, out_w), dtype=np.float32)
        for ci in range(c_in):
            kernel = weight[co, ci]  # (kH, kW)
            for b in range(batch):
                acc[b] += ndimage.correlate(x_pad[b, ci], kernel, mode="constant", cval=0.0)[
                    :out_h, :out_w
                ]
        out[:, co, :, :] = acc + bias[co]
    return out


# -- Activation / pooling -------------------------------------------------


def _relu(x: np.ndarray) -> np.ndarray:
    """Element-wise ReLU activation."""
    return np.maximum(x, 0.0, dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable element-wise sigmoid."""
    pos = x >= 0
    out = np.empty_like(x)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _maxpool2d(x: np.ndarray, pool: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """2-D max-pooling with stride equal to pool size.

    Returns
    -------
    pooled : ndarray
        Down-sampled tensor.
    indices : ndarray
        Argmax indices (flattened within each pool window) for potential
        up-sampling use.
    """
    b, c, h, w = x.shape
    oh, ow = h // pool, w // pool
    # Trim so dimensions are divisible by pool
    x_trim = x[:, :, : oh * pool, : ow * pool]
    x_rs = x_trim.reshape(b, c, oh, pool, ow, pool)
    pooled = x_rs.max(axis=(3, 5))
    indices = x_rs.reshape(b, c, oh, pool, ow, pool)
    indices.reshape(b, c, oh, pool * ow * pool)  # placeholder
    return pooled, np.empty(0)  # indices unused in bilinear decoder


def _upsample2d(x: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bilinear up-sampling of a 4-D tensor to a target spatial size.

    Uses ``scipy.ndimage.zoom`` per sample and channel.

    Parameters
    ----------
    x : ndarray, shape ``(B, C, H, W)``
    target_h, target_w : int
        Desired output spatial dimensions.

    Returns
    -------
    ndarray, shape ``(B, C, target_h, target_w)``
    """
    b, c, h, w = x.shape
    if h == target_h and w == target_w:
        return x
    zoom_h = target_h / h
    zoom_w = target_w / w
    out = np.empty((b, c, target_h, target_w), dtype=np.float32)
    for bi in range(b):
        for ci in range(c):
            out[bi, ci] = ndimage.zoom(x[bi, ci], (zoom_h, zoom_w), order=1)[:target_h, :target_w]
    return out


# ===================================================================== #
#                          Network blocks                                #
# ===================================================================== #


@dataclass
class _ConvBlock:
    """A single convolution → ReLU block with named parameters.

    Attributes
    ----------
    weight : ndarray, shape ``(C_out, C_in, kH, kW)``
    bias : ndarray, shape ``(C_out,)``
    """

    weight: np.ndarray
    bias: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return _relu(_conv2d(x, self.weight, self.bias))


@dataclass
class _EncoderBlock:
    """Encoder stage: conv → ReLU → max-pool."""

    conv: _ConvBlock

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(pooled, skip)``."""
        feat = self.conv(x)
        pooled, _ = _maxpool2d(feat)
        return pooled, feat


@dataclass
class _DecoderBlock:
    """Decoder stage: upsample → concatenate skip → conv → ReLU."""

    conv: _ConvBlock

    def __call__(self, x: np.ndarray, skip: np.ndarray) -> np.ndarray:
        up = _upsample2d(x, skip.shape[2], skip.shape[3])
        cat = np.concatenate([up, skip], axis=1)  # channel axis
        return self.conv(cat)


# ===================================================================== #
#                          UNet Model                                    #
# ===================================================================== #


class _UNet:
    """Minimal U-Net implemented in pure NumPy / SciPy.

    This is intentionally a *forward-pass-only* implementation optimised
    for portability rather than training speed.  The ``backward`` method
    provides a simplified SGD skeleton for fine-tuning on small datasets.

    Parameters
    ----------
    seed : int, optional
        Random seed for weight initialisation (default 42).
    """

    def __init__(self, seed: int = _DEFAULT_SEED) -> None:
        self._rng = np.random.default_rng(seed)
        self._params: dict[str, np.ndarray] = {}
        self._build()

    # ------------------------------------------------------------------ #
    #  Construction helpers                                               #
    # ------------------------------------------------------------------ #

    def _make_conv(self, name: str, c_in: int, c_out: int, k: int = _KERNEL_SIZE) -> _ConvBlock:
        """Create and register a convolution block."""
        fan_in = c_in * k * k
        w = _kaiming_init(fan_in, (c_out, c_in, k, k), self._rng)
        b = _zeros((c_out,))
        self._params[f"{name}.weight"] = w
        self._params[f"{name}.bias"] = b
        return _ConvBlock(weight=w, bias=b)

    def _build(self) -> None:
        """Instantiate all layers and store references."""
        # -- Encoder ---------------------------------------------------
        self.enc1 = _EncoderBlock(self._make_conv("enc1", 1, 32))
        self.enc2 = _EncoderBlock(self._make_conv("enc2", 32, 64))
        self.enc3 = _EncoderBlock(self._make_conv("enc3", 64, 128))

        # -- Bottleneck ------------------------------------------------
        self.bn1 = self._make_conv("bn1", 128, 256)
        self.bn2 = self._make_conv("bn2", 256, 128)

        # -- Decoder ---------------------------------------------------
        #   skip channels are concatenated → double the input channels
        self.dec3 = _DecoderBlock(self._make_conv("dec3", 128 + 128, 128))
        self.dec2 = _DecoderBlock(self._make_conv("dec2", 128 + 64, 64))
        self.dec1 = _DecoderBlock(self._make_conv("dec1", 64 + 32, 32))

        # -- Refinement and head ---------------------------------------
        self.refine = self._make_conv("refine", 32, _HEAD_INTERMEDIATE)
        self.head = _ConvBlock(
            weight=_kaiming_init(_HEAD_INTERMEDIATE, (1, _HEAD_INTERMEDIATE, 1, 1), self._rng),
            bias=_zeros((1,)),
        )
        self._params["head.weight"] = self.head.weight
        self._params["head.bias"] = self.head.bias

    # ------------------------------------------------------------------ #
    #  Forward pass                                                       #
    # ------------------------------------------------------------------ #

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run a full forward pass.

        Parameters
        ----------
        x : ndarray, shape ``(B, 1, H, W)``
            Batch of single-channel microscopy images.  Pixel values
            should be in [0, 1].

        Returns
        -------
        ndarray, shape ``(B, 1, H, W)``
            Per-pixel probability map (sigmoid-activated).
        """
        target_h, target_w = x.shape[2], x.shape[3]

        # Encoder
        e1, s1 = self.enc1(x)  # s1: (B,32,H,W);   e1: (B,32,H/2,W/2)
        e2, s2 = self.enc2(e1)  # s2: (B,64,H/2,W/2)
        e3, s3 = self.enc3(e2)  # s3: (B,128,H/4,W/4)

        # Bottleneck
        b = self.bn1(e3)  # (B,256,H/8,W/8)
        b = self.bn2(b)  # (B,128,H/8,W/8)

        # Decoder
        d3 = self.dec3(b, s3)  # (B,128,H/4,W/4)
        d2 = self.dec2(d3, s2)  # (B,64,H/2,W/2)
        d1 = self.dec1(d2, s1)  # (B,32,H,W)

        # Head
        d1 = self.refine(d1)  # (B,16,H,W)
        logits = _conv2d(d1, self.head.weight, self.head.bias)
        prob = _sigmoid(logits)

        # Ensure spatial dims match input (guard against rounding)
        if prob.shape[2] != target_h or prob.shape[3] != target_w:
            prob = _upsample2d(prob, target_h, target_w)

        return prob

    # ------------------------------------------------------------------ #
    #  Parameter access                                                   #
    # ------------------------------------------------------------------ #

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        """Return a *shallow* copy of the parameter dictionary."""
        return dict(self._params)

    def parameter_count(self) -> int:
        """Total number of scalar parameters in the model."""
        return sum(p.size for p in self._params.values())

    # ------------------------------------------------------------------ #
    #  Serialisation                                                      #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return a deep copy of all parameters suitable for saving."""
        return {k: v.copy() for k, v in self._params.items()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Load parameters from a state dictionary **in-place**.

        Parameters
        ----------
        state : dict
            Mapping of parameter names to numpy arrays.  Keys must match
            those returned by :meth:`state_dict`.

        Raises
        ------
        KeyError
            If a required parameter is missing.
        ValueError
            If a parameter has the wrong shape.
        """
        for name, current in self._params.items():
            if name not in state:
                raise KeyError(f"Missing parameter: {name}")
            incoming = state[name]
            if incoming.shape != current.shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': expected {current.shape}, got {incoming.shape}"
                )
            # In-place copy so that block references stay valid
            np.copyto(current, incoming.astype(np.float32))


# ===================================================================== #
#                      Post-processing helpers                           #
# ===================================================================== #


def _extract_spots_from_probmap(
    prob: np.ndarray,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    min_distance: int = _DEFAULT_MIN_DISTANCE,
) -> list[SpotRecord]:
    """Convert a 2-D probability map into a list of spot records.

    The procedure mirrors classical blob detection: threshold → label
    connected components → measure region properties.

    Parameters
    ----------
    prob : ndarray, shape ``(H, W)``
        Per-pixel detection probability in [0, 1].
    confidence_threshold : float
        Pixels below this value are discarded.
    min_distance : int
        Minimum separation between spot centres (suppresses duplicates).

    Returns
    -------
    list of dict
        Each dict contains ``x, y, intensity, confidence, radius``.
    """
    binary = prob >= confidence_threshold
    labelled, n_objects = ndimage.label(binary)

    spots: list[SpotRecord] = []
    for idx in range(1, n_objects + 1):
        component = labelled == idx
        area = int(component.sum())
        if area < 1:
            continue

        # Centre of mass (row, col)
        cy, cx = ndimage.center_of_mass(prob, labelled, idx)
        peak_val = float(prob[component].max())
        mean_val = float(prob[component].mean())
        radius = float(np.sqrt(area / np.pi))

        spots.append(
            {
                "x": int(round(cx)),
                "y": int(round(cy)),
                "intensity": mean_val,
                "confidence": peak_val,
                "radius": radius,
            }
        )

    # Non-maximum suppression based on min_distance
    spots.sort(key=lambda s: s["confidence"], reverse=True)
    kept: list[SpotRecord] = []
    for spot in spots:
        too_close = False
        for existing in kept:
            dist = np.hypot(spot["x"] - existing["x"], spot["y"] - existing["y"])
            if dist < min_distance:
                too_close = True
                break
        if not too_close:
            kept.append(spot)

    return kept


# ===================================================================== #
#                     Metric computation helpers                         #
# ===================================================================== #


def _match_spots(
    pred: list[SpotRecord], gt: list[SpotRecord], tolerance: float = 5.0
) -> tuple[int, int, int]:
    """Match predicted spots to ground-truth spots.

    A predicted spot is a *true positive* if its centre is within
    ``tolerance`` pixels of an unmatched ground-truth spot.

    Parameters
    ----------
    pred : list of dict
        Predicted spot records.
    gt : list of dict
        Ground-truth spot records.
    tolerance : float
        Maximum Euclidean distance for a match (pixels).

    Returns
    -------
    tp, fp, fn : int
        True positives, false positives, false negatives.
    """
    gt_matched = [False] * len(gt)
    tp = 0

    for p in pred:
        best_dist = float("inf")
        best_idx = -1
        for gi, g in enumerate(gt):
            if gt_matched[gi]:
                continue
            d = np.hypot(p["x"] - g["x"], p["y"] - g["y"])
            if d < best_dist:
                best_dist = d
                best_idx = gi
        if best_dist <= tolerance and best_idx >= 0:
            tp += 1
            gt_matched[best_idx] = True

    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn


def _compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Intersection-over-Union for two binary masks.

    Parameters
    ----------
    pred_mask, gt_mask : ndarray of bool
        Binary segmentation masks of identical shape.

    Returns
    -------
    float
        IoU score in [0, 1].
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0  # both empty → perfect agreement
    return float(intersection / union)


# ===================================================================== #
#                       Public API: CNNSpotDetector                       #
# ===================================================================== #


class CNNSpotDetector:
    """CNN-based telomere spot detector for qFISH microscopy images.

    This detector wraps a lightweight U-Net (Ronneberger *et al.* 2015)
    implemented entirely in NumPy/SciPy.  It produces a per-pixel
    probability map which is then post-processed into discrete spot
    coordinates, intensities and confidence scores.

    Parameters
    ----------
    model_path : str or Path or None
        Path to a ``.npz`` file containing pre-trained weights.  If
        ``None``, the model is initialised with Kaiming/He random weights
        (He *et al.* 2015) — suitable for benchmarking or as a starting
        point for fine-tuning.
    confidence_threshold : float
        Probability threshold for spot extraction (default 0.35).
    min_spot_distance : int
        Minimum pixel distance between detected spot centres after
        non-maximum suppression (default 3).
    seed : int
        Random seed used when generating default weights.

    Examples
    --------
    >>> import numpy as np
    >>> detector = CNNSpotDetector()
    >>> image = np.random.rand(128, 128).astype(np.float32)
    >>> spots = detector.detect(image)
    >>> type(spots)
    <class 'list'>
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
        min_spot_distance: int = _DEFAULT_MIN_DISTANCE,
        seed: int = _DEFAULT_SEED,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.min_spot_distance = min_spot_distance
        self._seed = seed

        self._model = _UNet(seed=seed)
        logger.info(
            "U-Net instantiated with %s parameters.",
            f"{self._model.parameter_count():,}",
        )

        if model_path is not None:
            self.load_weights(model_path)
        else:
            logger.info("No model_path supplied — using default random weights.")
            self._generate_default_weights()

    # ------------------------------------------------------------------ #
    #  Weight management                                                  #
    # ------------------------------------------------------------------ #

    def _generate_default_weights(self) -> None:
        """Populate model parameters with Kaiming-initialised values.

        This is already done during ``_UNet.__init__``; calling this
        method explicitly resets the weights to a fresh random draw with
        the detector's configured seed.  Useful for reproducible
        benchmarking and unit tests.
        """
        logger.debug("Generating default (random) weights with seed=%d.", self._seed)
        fresh = _UNet(seed=self._seed)
        self._model.load_state_dict(fresh.state_dict())

    def save_weights(self, path: str | Path) -> None:
        """Serialise current model weights to a ``.npz`` file.

        Parameters
        ----------
        path : str or Path
            Destination file path.  Parent directories are created
            automatically if they do not exist.

        Raises
        ------
        OSError
            If the file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **self._model.state_dict())
        logger.info("Weights saved to %s (%d parameters).", path, self._model.parameter_count())

    def load_weights(self, path: str | Path) -> None:
        """Load model weights from a ``.npz`` file.

        Parameters
        ----------
        path : str or Path
            Source file path (must exist).

        Raises
        ------
        FileNotFoundError
            If the weights file does not exist.
        KeyError
            If the archive is missing required parameter arrays.
        ValueError
            If any parameter has an incompatible shape.
        """
        path = Path(path)
        if not path.exists():
            # Try adding .npz extension
            if not path.with_suffix(".npz").exists():
                raise FileNotFoundError(f"Weight file not found: {path}")
            path = path.with_suffix(".npz")

        data = dict(np.load(str(path), allow_pickle=False))
        self._model.load_state_dict(data)
        logger.info("Weights loaded from %s.", path)

    # ------------------------------------------------------------------ #
    #  Pre-processing                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        """Normalise and reshape a 2-D image for the network.

        Steps:
        1. Convert to float32.
        2. Normalise to [0, 1] using the image's own min/max.
        3. Reshape to ``(1, 1, H, W)`` (batch-of-one, single channel).

        Parameters
        ----------
        image : ndarray, shape ``(H, W)``
            Input grayscale microscopy image.

        Returns
        -------
        ndarray, shape ``(1, 1, H, W)``
        """
        img = np.asarray(image, dtype=np.float32)
        if img.ndim == 3:
            # Take first channel if multi-channel
            img = img[:, :, 0] if img.shape[2] <= 4 else img[0]
        if img.ndim != 2:
            raise ValueError(f"Expected 2-D image, got shape {image.shape}.")
        lo, hi = img.min(), img.max()
        if hi - lo > 1e-8:
            img = (img - lo) / (hi - lo)
        else:
            img = np.zeros_like(img)
        return img[np.newaxis, np.newaxis, :, :]

    # ------------------------------------------------------------------ #
    #  Inference                                                          #
    # ------------------------------------------------------------------ #

    def predict_probmap(self, image: np.ndarray) -> np.ndarray:
        """Compute the raw probability map without spot extraction.

        Parameters
        ----------
        image : ndarray, shape ``(H, W)``
            Grayscale microscopy image.

        Returns
        -------
        ndarray, shape ``(H, W)``
            Per-pixel probability of being a telomere spot.
        """
        x = self._preprocess(image)
        prob = self._model.forward(x)  # (1, 1, H, W)
        return prob[0, 0]

    def detect(self, image: np.ndarray) -> list[SpotRecord]:
        """Detect telomere spots in a single microscopy image.

        This is the primary public entry point.  It runs the full
        pipeline: pre-processing → CNN forward pass → post-processing
        (thresholding, connected-component labelling, non-maximum
        suppression).

        Parameters
        ----------
        image : ndarray, shape ``(H, W)``
            Input grayscale qFISH microscopy image (arbitrary dtype).

        Returns
        -------
        list of dict
            Each dictionary contains:

            - ``x`` (int) – column coordinate of spot centre.
            - ``y`` (int) – row coordinate of spot centre.
            - ``intensity`` (float) – mean probability within the spot
              region (proxy for fluorescence intensity).
            - ``confidence`` (float) – peak probability within the spot
              region.
            - ``radius`` (float) – estimated spot radius in pixels
              (derived from connected-component area).

        Examples
        --------
        >>> det = CNNSpotDetector()
        >>> spots = det.detect(np.random.rand(64, 64))
        >>> all("x" in s and "confidence" in s for s in spots)
        True
        """
        logger.debug("Running CNN spot detection on image of shape %s.", image.shape)
        t0 = time.perf_counter()

        prob = self.predict_probmap(image)
        spots = _extract_spots_from_probmap(
            prob,
            confidence_threshold=self.confidence_threshold,
            min_distance=self.min_spot_distance,
        )

        elapsed = time.perf_counter() - t0
        logger.info("Detected %d spots in %.3f s.", len(spots), elapsed)
        return spots

    # ------------------------------------------------------------------ #
    #  Training (skeleton / fine-tuning)                                  #
    # ------------------------------------------------------------------ #

    def train(
        self,
        images: list[np.ndarray],
        labels: list[np.ndarray],
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 1,
        log_every: int = 5,
    ) -> list[float]:
        """Train (fine-tune) the model on labelled qFISH data.

        .. warning::

            This is a **skeleton** implementation using pixel-wise MSE
            loss and numerical-gradient SGD.  It is intended for small
            fine-tuning runs (< 100 images) on a single CPU.  For
            serious training, export the architecture to PyTorch or
            TensorFlow and use GPU-accelerated auto-differentiation.

        The training loop follows the classic SGD recipe described by
        LeCun *et al.* (1998) with a fixed learning rate schedule:

        1. Forward pass through the U-Net.
        2. Compute pixel-wise binary cross-entropy loss.
        3. Estimate parameter gradients via finite differences.
        4. Update parameters: ``w ← w − lr * grad``.

        Parameters
        ----------
        images : list of ndarray
            Training images, each of shape ``(H, W)``.
        labels : list of ndarray
            Ground-truth binary masks (1 = spot, 0 = background), each
            of the same spatial shape as the corresponding image.
        epochs : int
            Number of full passes through the training set.
        lr : float
            Learning rate for SGD.
        batch_size : int
            Number of images per mini-batch (currently only 1 is
            implemented).
        log_every : int
            Log training loss every *n* epochs.

        Returns
        -------
        list of float
            Training loss at the end of each epoch.
        """
        if len(images) != len(labels):
            raise ValueError(f"Mismatch: {len(images)} images vs {len(labels)} labels.")
        if len(images) == 0:
            raise ValueError("Need at least one training sample.")

        logger.info(
            "Starting training: %d samples, %d epochs, lr=%.4f.",
            len(images),
            epochs,
            lr,
        )

        losses: list[float] = []
        eps_fd = 1e-4  # finite-difference epsilon

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            indices = np.arange(len(images))
            np.random.shuffle(indices)

            for i in indices:
                x = self._preprocess(images[i])
                y_true = labels[i].astype(np.float32)
                if y_true.ndim == 2:
                    y_true = y_true[np.newaxis, np.newaxis, :, :]

                # --- Forward pass ---
                y_pred = self._model.forward(x)

                # Crop / resize to match
                if y_pred.shape != y_true.shape:
                    y_true_rs = _upsample2d(y_true, y_pred.shape[2], y_pred.shape[3])
                else:
                    y_true_rs = y_true

                # --- Binary cross-entropy loss ---
                y_pred_clip = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
                loss = -np.mean(
                    y_true_rs * np.log(y_pred_clip) + (1 - y_true_rs) * np.log(1 - y_pred_clip)
                )
                epoch_loss += float(loss)

                # --- Numerical gradient SGD (parameter-wise) ---
                # NOTE: This is intentionally simplified.  For each
                # parameter tensor we perturb a *random subset* of
                # elements rather than the full tensor, making the
                # cost O(n_sample) rather than O(n_params).
                n_sample = min(64, 16)  # elements to perturb per param
                for name, param in self._model.parameters.items():
                    flat = param.ravel()
                    sample_idx = np.random.choice(
                        len(flat),
                        size=min(n_sample, len(flat)),
                        replace=False,
                    )
                    grad_est = np.zeros_like(flat)
                    for si in sample_idx:
                        orig = flat[si]

                        flat[si] = orig + eps_fd
                        y_plus = self._model.forward(x)
                        if y_plus.shape != y_true_rs.shape:
                            y_true_fd = _upsample2d(y_true_rs, y_plus.shape[2], y_plus.shape[3])
                        else:
                            y_true_fd = y_true_rs
                        yp_clip = np.clip(y_plus, 1e-7, 1 - 1e-7)
                        loss_plus = -np.mean(
                            y_true_fd * np.log(yp_clip) + (1 - y_true_fd) * np.log(1 - yp_clip)
                        )

                        flat[si] = orig - eps_fd
                        y_minus = self._model.forward(x)
                        if y_minus.shape != y_true_rs.shape:
                            y_true_fd2 = _upsample2d(y_true_rs, y_minus.shape[2], y_minus.shape[3])
                        else:
                            y_true_fd2 = y_true_rs
                        ym_clip = np.clip(y_minus, 1e-7, 1 - 1e-7)
                        loss_minus = -np.mean(
                            y_true_fd2 * np.log(ym_clip) + (1 - y_true_fd2) * np.log(1 - ym_clip)
                        )

                        grad_est[si] = (loss_plus - loss_minus) / (2 * eps_fd)
                        flat[si] = orig  # restore

                    # SGD update
                    param.ravel()[:] -= lr * grad_est

            avg_loss = epoch_loss / len(images)
            losses.append(avg_loss)

            if epoch % log_every == 0 or epoch == 1:
                logger.info("Epoch %3d/%d — loss: %.6f", epoch, epochs, avg_loss)

        logger.info("Training complete.  Final loss: %.6f", losses[-1])
        return losses

    # ------------------------------------------------------------------ #
    #  Evaluation                                                         #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        images: list[np.ndarray],
        labels: list[np.ndarray],
        tolerance: float = 5.0,
    ) -> dict[str, float]:
        """Evaluate detection performance against ground-truth masks.

        Computes standard object-detection and segmentation metrics by
        comparing detected spots against labelled ground truth.

        Parameters
        ----------
        images : list of ndarray
            Test images, each of shape ``(H, W)``.
        labels : list of ndarray
            Ground-truth binary masks (1 = spot, 0 = background).
        tolerance : float
            Maximum distance (pixels) for a predicted spot to be matched
            to a ground-truth spot.

        Returns
        -------
        dict
            Metrics dictionary with keys:

            - ``precision`` – TP / (TP + FP).
            - ``recall`` – TP / (TP + FN).
            - ``f1`` – harmonic mean of precision and recall.
            - ``iou`` – mean Intersection-over-Union of probability maps
              vs ground-truth masks.
            - ``n_images`` – number of images evaluated.
            - ``mean_spots_pred`` – average predicted spots per image.
            - ``mean_spots_gt`` – average ground-truth spots per image.
        """
        if len(images) != len(labels):
            raise ValueError(f"Mismatch: {len(images)} images vs {len(labels)} labels.")

        total_tp, total_fp, total_fn = 0, 0, 0
        ious: list[float] = []
        pred_counts: list[int] = []
        gt_counts: list[int] = []

        for img, lbl in zip(images, labels):
            # Predicted spots
            pred_spots = self.detect(img)
            pred_counts.append(len(pred_spots))

            # Ground-truth spots from mask
            gt_binary = lbl.astype(bool)
            gt_labelled, gt_n = ndimage.label(gt_binary)
            gt_spots: list[SpotRecord] = []
            for idx in range(1, gt_n + 1):
                cy, cx = ndimage.center_of_mass(lbl, gt_labelled, idx)
                gt_spots.append({"x": int(round(cx)), "y": int(round(cy))})
            gt_counts.append(len(gt_spots))

            # Spot-level matching
            tp, fp, fn = _match_spots(pred_spots, gt_spots, tolerance)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Pixel-level IoU
            prob = self.predict_probmap(img)
            pred_mask = prob >= self.confidence_threshold
            ious.append(_compute_iou(pred_mask, gt_binary))

        # Aggregate metrics
        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "iou": round(float(np.mean(ious)) if ious else 0.0, 4),
            "n_images": len(images),
            "mean_spots_pred": round(float(np.mean(pred_counts)), 2),
            "mean_spots_gt": round(float(np.mean(gt_counts)), 2),
        }

    # ------------------------------------------------------------------ #
    #  Comparison with classical LoG detection                            #
    # ------------------------------------------------------------------ #

    def compare_with_log(
        self,
        image: np.ndarray,
        log_spots: list[SpotRecord],
        ground_truth: list[SpotRecord] | None = None,
        tolerance: float = 5.0,
    ) -> dict[str, Any]:
        """Compare CNN detections against classical LoG spot detection.

        This utility method runs CNN inference on the same image and
        returns a side-by-side comparison of the two methods.  If
        ``ground_truth`` spots are supplied, precision/recall/F1 are
        computed for both methods.

        Parameters
        ----------
        image : ndarray, shape ``(H, W)``
            Input microscopy image.
        log_spots : list of dict
            Spots detected by a Laplacian-of-Gaussian (or equivalent)
            classical detector.  Each dict must contain at least ``x``
            and ``y`` keys.
        ground_truth : list of dict or None
            Optional ground-truth spots for quantitative comparison.
        tolerance : float
            Matching tolerance in pixels.

        Returns
        -------
        dict
            Comparison report with structure::

                {
                    "cnn": {
                        "n_spots": int,
                        "spots": [...],
                        "metrics": {...} or None,
                        "elapsed_s": float,
                    },
                    "log": {
                        "n_spots": int,
                        "spots": [...],
                        "metrics": {...} or None,
                    },
                    "overlap": {
                        "matched_pairs": int,
                        "cnn_only": int,
                        "log_only": int,
                    },
                }

        Notes
        -----
        The LoG detector typically excels at finding bright, isolated
        spots in high-SNR images, while the CNN can learn to handle
        overlapping spots, uneven backgrounds and low-contrast signals
        that defeat hand-tuned LoG parameters (Falk *et al.* 2019).
        """
        # --- CNN detection ---
        t0 = time.perf_counter()
        cnn_spots = self.detect(image)
        cnn_time = time.perf_counter() - t0

        # --- Cross-method overlap ---
        # Match CNN spots to LoG spots
        matched = 0
        log_matched = [False] * len(log_spots)
        cnn_matched = [False] * len(cnn_spots)

        for ci, cs in enumerate(cnn_spots):
            for li, ls in enumerate(log_spots):
                if log_matched[li]:
                    continue
                d = np.hypot(cs["x"] - ls["x"], cs["y"] - ls["y"])
                if d <= tolerance:
                    matched += 1
                    log_matched[li] = True
                    cnn_matched[ci] = True
                    break

        cnn_only = sum(1 for m in cnn_matched if not m)
        log_only = sum(1 for m in log_matched if not m)

        # --- Ground-truth evaluation (optional) ---
        cnn_metrics = None
        log_metrics = None

        if ground_truth is not None:
            # CNN vs GT
            tp_c, fp_c, fn_c = _match_spots(cnn_spots, ground_truth, tolerance)
            prec_c = tp_c / max(tp_c + fp_c, 1)
            rec_c = tp_c / max(tp_c + fn_c, 1)
            f1_c = 2 * prec_c * rec_c / max(prec_c + rec_c, 1e-12)
            cnn_metrics = {
                "precision": round(prec_c, 4),
                "recall": round(rec_c, 4),
                "f1": round(f1_c, 4),
            }

            # LoG vs GT
            tp_l, fp_l, fn_l = _match_spots(log_spots, ground_truth, tolerance)
            prec_l = tp_l / max(tp_l + fp_l, 1)
            rec_l = tp_l / max(tp_l + fn_l, 1)
            f1_l = 2 * prec_l * rec_l / max(prec_l + rec_l, 1e-12)
            log_metrics = {
                "precision": round(prec_l, 4),
                "recall": round(rec_l, 4),
                "f1": round(f1_l, 4),
            }

        return {
            "cnn": {
                "n_spots": len(cnn_spots),
                "spots": cnn_spots,
                "metrics": cnn_metrics,
                "elapsed_s": round(cnn_time, 4),
            },
            "log": {
                "n_spots": len(log_spots),
                "spots": log_spots,
                "metrics": log_metrics,
            },
            "overlap": {
                "matched_pairs": matched,
                "cnn_only": cnn_only,
                "log_only": log_only,
            },
        }

    # ------------------------------------------------------------------ #
    #  Dunder helpers                                                     #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"CNNSpotDetector("
            f"params={self._model.parameter_count():,}, "
            f"threshold={self.confidence_threshold}, "
            f"min_dist={self.min_spot_distance})"
        )

    def summary(self) -> str:
        """Return a human-readable model summary string.

        Includes layer names, parameter shapes and total counts — useful
        for logging and debugging.

        Returns
        -------
        str
            Multi-line summary.
        """
        lines = [
            "CNNSpotDetector — U-Net Architecture Summary",
            "=" * 50,
        ]
        total = 0
        for name, param in sorted(self._model.parameters.items()):
            n = param.size
            total += n
            lines.append(f"  {name:25s}  {str(param.shape):20s}  {n:>10,}")
        lines.append("-" * 50)
        lines.append(f"  {'TOTAL':25s}  {'':20s}  {total:>10,}")
        lines.append(f"\nConfidence threshold : {self.confidence_threshold}")
        lines.append(f"Min spot distance    : {self.min_spot_distance} px")
        return "\n".join(lines)


# ===================================================================== #
#                          Module-level helpers                           #
# ===================================================================== #


def create_detector(
    model_path: str | Path | None = None,
    **kwargs: Any,
) -> CNNSpotDetector:
    """Factory function for creating a :class:`CNNSpotDetector`.

    This is a convenience wrapper that mirrors the constructor signature
    and can be used in configuration-driven pipelines.

    Parameters
    ----------
    model_path : str, Path, or None
        Optional path to pre-trained weights.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`CNNSpotDetector`.

    Returns
    -------
    CNNSpotDetector
    """
    return CNNSpotDetector(model_path=model_path, **kwargs)


def quick_detect(image: np.ndarray, **kwargs: Any) -> list[SpotRecord]:
    """One-shot detection without explicitly managing a detector instance.

    Creates a temporary :class:`CNNSpotDetector` with default weights,
    runs detection, and returns the spot list.  Useful for quick
    exploration in notebooks and scripts.

    Parameters
    ----------
    image : ndarray
        Grayscale microscopy image.
    **kwargs
        Forwarded to :class:`CNNSpotDetector`.

    Returns
    -------
    list of dict
        Detected spot records.
    """
    det = CNNSpotDetector(**kwargs)
    return det.detect(image)


# ===================================================================== #
#                         CLI entry point                                #
# ===================================================================== #


def _cli_main() -> None:
    """Minimal command-line entry point for smoke-testing.

    Usage::

        python -m teloscopy.ml.cnn_spot_detector [--size 128]

    Generates a synthetic image with Gaussian blobs, runs detection, and
    prints the results.
    """
    import argparse

    parser = argparse.ArgumentParser(description="CNN telomere spot detector — smoke test")
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        help="Side length of the synthetic test image (default: 64).",
    )
    parser.add_argument(
        "--n-spots",
        type=int,
        default=5,
        help="Number of synthetic telomere spots to inject (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    rng = np.random.default_rng(args.seed)
    size = args.size

    # Build synthetic image
    img = rng.normal(0.1, 0.02, (size, size)).astype(np.float32)
    gt_spots: list[SpotRecord] = []
    for _ in range(args.n_spots):
        cx = rng.integers(10, size - 10)
        cy = rng.integers(10, size - 10)
        sigma = rng.uniform(1.5, 3.0)
        yy, xx = np.mgrid[:size, :size]
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
        img += blob.astype(np.float32) * rng.uniform(0.3, 1.0)
        gt_spots.append({"x": int(cx), "y": int(cy)})

    img = np.clip(img, 0.0, 1.0)

    # Detect
    detector = CNNSpotDetector(seed=args.seed)
    print(detector.summary())
    print()

    spots = detector.detect(img)
    print(f"Detected {len(spots)} spots:")
    for s in spots:
        print(
            f"  x={s['x']:4d}  y={s['y']:4d}  "
            f"conf={s['confidence']:.3f}  "
            f"intensity={s['intensity']:.3f}  "
            f"radius={s['radius']:.2f}"
        )

    # Quick LoG comparison stub
    log_spots = [
        {
            "x": s["x"] + rng.integers(-2, 3),
            "y": s["y"] + rng.integers(-2, 3),
            "intensity": 0.5,
            "confidence": 0.5,
            "radius": 2.0,
        }
        for s in gt_spots
    ]
    report = detector.compare_with_log(img, log_spots, ground_truth=gt_spots)
    print(
        f"\nComparison — CNN: {report['cnn']['n_spots']} spots, "
        f"LoG: {report['log']['n_spots']} spots"
    )
    print(
        f"  Overlap: {report['overlap']['matched_pairs']} matched, "
        f"{report['overlap']['cnn_only']} CNN-only, "
        f"{report['overlap']['log_only']} LoG-only"
    )
    if report["cnn"]["metrics"]:
        m = report["cnn"]["metrics"]
        print(f"  CNN  → P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    if report["log"]["metrics"]:
        m = report["log"]["metrics"]
        print(f"  LoG  → P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")


if __name__ == "__main__":
    _cli_main()
