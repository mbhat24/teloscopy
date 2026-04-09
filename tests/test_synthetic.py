"""Tests for synthetic image generation and spot detection."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from teloscopy.telomere.synthetic import (
    generate_chromosome,
    generate_metaphase_spread,
    generate_telomere_spot,
)


class TestGenerateChromosome:
    def test_returns_2d_array(self):
        chrom = generate_chromosome()
        assert chrom.ndim == 2

    def test_shape_matches_requested(self):
        for shape in [(50, 20), (100, 30), (80, 15)]:
            chrom = generate_chromosome(shape=shape)
            assert chrom.shape == shape, f"Expected {shape}, got {chrom.shape}"

    def test_has_bright_pixels(self):
        """Chromosome should have pixels brighter than background."""
        chrom = generate_chromosome(intensity=40000.0)
        # Background is zero, chromosome interior should be bright
        assert chrom.max() > 0
        # After Gaussian smoothing, peak should still be substantial
        assert chrom.max() > 1000


class TestGenerateTelomereSpot:
    def test_returns_2d_array(self):
        spot = generate_telomere_spot()
        assert spot.ndim == 2

    def test_peak_at_center(self):
        """Peak intensity should be at or near center."""
        spot = generate_telomere_spot(size=21, sigma=2.0, intensity=50000)
        peak_idx = np.unravel_index(np.argmax(spot), spot.shape)
        center = (21 - 1) // 2
        # Peak should be within 1 pixel of the centre
        assert abs(peak_idx[0] - center) <= 1
        assert abs(peak_idx[1] - center) <= 1

    def test_gaussian_shape(self):
        """Intensity should decrease from center."""
        spot = generate_telomere_spot(size=21, sigma=2.0, intensity=50000)
        c = (21 - 1) // 2
        center_val = spot[c, c]
        edge_val = spot[0, 0]
        mid_val = spot[c, c // 2]

        assert center_val > mid_val > edge_val
        assert center_val == pytest.approx(50000, rel=0.01)
        # Edge should be near zero (far from centre with small sigma)
        assert edge_val < center_val * 0.01


class TestGenerateMetaphaseSpread:
    def test_returns_required_keys(self):
        """Should return dict with dapi, cy3, ground_truth."""
        data = generate_metaphase_spread(image_size=(256, 256), n_chromosomes=5, seed=42)
        assert "dapi" in data
        assert "cy3" in data
        assert "ground_truth" in data
        assert "chromosomes" in data["ground_truth"]
        assert "telomeres" in data["ground_truth"]

    def test_dapi_shape(self):
        data = generate_metaphase_spread(image_size=(512, 512), n_chromosomes=10, seed=42)
        assert data["dapi"].shape == (512, 512)

    def test_cy3_shape(self):
        data = generate_metaphase_spread(image_size=(512, 512), n_chromosomes=10, seed=42)
        assert data["cy3"].shape == (512, 512)

    def test_ground_truth_chromosomes_count(self):
        data = generate_metaphase_spread(n_chromosomes=10, seed=42)
        assert len(data["ground_truth"]["chromosomes"]) == 10

    def test_ground_truth_telomeres_count(self):
        """Each chromosome should have 2 telomeres."""
        data = generate_metaphase_spread(n_chromosomes=10, seed=42)
        assert len(data["ground_truth"]["telomeres"]) == 20

    def test_reproducible_with_seed(self):
        d1 = generate_metaphase_spread(n_chromosomes=5, seed=123)
        d2 = generate_metaphase_spread(n_chromosomes=5, seed=123)
        assert np.array_equal(d1["dapi"], d2["dapi"])
        assert np.array_equal(d1["cy3"], d2["cy3"])

    def test_dtype_uint16(self):
        data = generate_metaphase_spread(seed=42)
        assert data["dapi"].dtype == np.uint16
        assert data["cy3"].dtype == np.uint16

    def test_dapi_has_contrast(self):
        """DAPI channel should have bright chromosomes above dark background."""
        data = generate_metaphase_spread(image_size=(512, 512), n_chromosomes=10, seed=42)
        dapi = data["dapi"].astype(np.float64)
        assert dapi.max() > 5000
        assert dapi.min() < 1000

    def test_cy3_has_spots(self):
        """Cy3 channel should contain telomere spots above background."""
        data = generate_metaphase_spread(image_size=(512, 512), n_chromosomes=10, seed=42)
        cy3 = data["cy3"].astype(np.float64)
        assert cy3.max() > 5000

    def test_ground_truth_telomere_keys(self):
        """Each telomere ground truth entry should have y and x."""
        data = generate_metaphase_spread(n_chromosomes=5, seed=42)
        for t in data["ground_truth"]["telomeres"]:
            assert "y" in t
            assert "x" in t
            assert "arm" in t
            assert t["arm"] in ("p", "q")
