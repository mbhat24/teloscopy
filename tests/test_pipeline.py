"""Integration tests for the full telomere analysis pipeline."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from teloscopy.analysis.statistics import compute_cell_statistics, create_results_dataframe
from teloscopy.telomere.association import associate_spots_to_chromosomes, summarize_associations
from teloscopy.telomere.preprocessing import denoise, subtract_background
from teloscopy.telomere.quantification import (
    Calibration,
    measure_spot_intensity,
)
from teloscopy.telomere.segmentation import get_chromosome_properties, segment_otsu_watershed
from teloscopy.telomere.spot_detection import detect_spots
from teloscopy.telomere.synthetic import generate_metaphase_spread


@pytest.fixture
def synthetic_data():
    return generate_metaphase_spread(
        image_size=(512, 512),
        n_chromosomes=10,
        seed=42,
    )


class TestPreprocessing:
    def test_background_subtraction(self, synthetic_data):
        result = subtract_background(synthetic_data["dapi"])
        assert result.shape == synthetic_data["dapi"].shape
        assert result.dtype == synthetic_data["dapi"].dtype

    def test_denoise(self, synthetic_data):
        result = denoise(synthetic_data["dapi"].astype(np.float64))
        assert result.shape == synthetic_data["dapi"].shape
        assert result.dtype == np.float64


class TestSegmentation:
    def test_otsu_watershed_produces_labels(self, synthetic_data):
        dapi = denoise(synthetic_data["dapi"].astype(np.float64))
        labels = segment_otsu_watershed(dapi, min_area=100)
        assert labels.shape == dapi.shape
        assert labels.max() > 0  # At least one segment found

    def test_chromosome_properties(self, synthetic_data):
        dapi = denoise(synthetic_data["dapi"].astype(np.float64))
        labels = segment_otsu_watershed(dapi, min_area=100)
        props = get_chromosome_properties(labels, dapi)
        assert len(props) > 0
        assert "tip_p" in props[0]
        assert "tip_q" in props[0]
        assert "label" in props[0]
        assert "centroid" in props[0]
        assert "area" in props[0]


class TestSpotDetection:
    def test_detects_spots(self, synthetic_data):
        cy3 = synthetic_data["cy3"].astype(np.float64)
        cy3_norm = cy3 / cy3.max() if cy3.max() > 0 else cy3
        spots = detect_spots(cy3_norm, method="blob_log", threshold=0.02)
        assert len(spots) > 0
        assert "y" in spots[0]
        assert "x" in spots[0]
        assert "sigma" in spots[0]
        assert "radius" in spots[0]


class TestAssociation:
    def test_associate_spots(self, synthetic_data):
        dapi = denoise(synthetic_data["dapi"].astype(np.float64))
        labels = segment_otsu_watershed(dapi, min_area=100)
        chroms = get_chromosome_properties(labels, dapi)

        cy3 = synthetic_data["cy3"].astype(np.float64)
        cy3_norm = cy3 / cy3.max() if cy3.max() > 0 else cy3
        spots = detect_spots(cy3_norm, method="blob_log", threshold=0.02)

        if spots and chroms:
            associated = associate_spots_to_chromosomes(spots, chroms)
            assert len(associated) == len(spots)
            # Every spot should now have association metadata
            for spot in associated:
                assert "associated" in spot
                assert "chromosome_label" in spot
                assert "arm" in spot
                assert "tip_distance" in spot

            summary = summarize_associations(associated, chroms)
            assert "total_spots" in summary
            assert "associated_spots" in summary
            assert "association_rate" in summary
            assert summary["total_spots"] == len(spots)


class TestQuantification:
    def test_measure_single_spot(self, synthetic_data):
        cy3 = synthetic_data["cy3"].astype(np.float64)
        gt_telomeres = synthetic_data["ground_truth"]["telomeres"]
        if gt_telomeres:
            t = gt_telomeres[0]
            result = measure_spot_intensity(cy3, t["y"], t["x"])
            assert result["valid"]
            assert result["raw_intensity"] > 0
            assert "corrected_intensity" in result
            assert "background_level" in result
            assert "snr" in result


class TestCalibration:
    def test_linear_calibration(self):
        cal = Calibration(method="linear")
        cal.fit([1000, 10000], [5000, 50000])
        length = cal.predict(5500)
        assert 20000 < length < 35000

    def test_identity_calibration(self):
        cal = Calibration.identity()
        assert cal.predict(42.0) == 42.0

    def test_poly2_calibration(self):
        cal = Calibration(method="poly2")
        cal.fit([100, 500, 1000], [1000, 4000, 9000])
        result = cal.predict(750)
        assert result > 0


class TestStatistics:
    def test_compute_cell_stats(self):
        spots = [
            {"associated": True, "valid": True, "corrected_intensity": 1000},
            {"associated": True, "valid": True, "corrected_intensity": 2000},
            {"associated": True, "valid": True, "corrected_intensity": 3000},
            {"associated": False, "valid": True, "corrected_intensity": 500},
        ]
        stats = compute_cell_statistics(spots)
        assert stats["n_telomeres"] == 3
        assert stats["mean_intensity"] == 2000.0
        assert stats["total_spots"] == 4

    def test_empty_spots(self):
        stats = compute_cell_statistics([])
        assert stats["n_telomeres"] == 0
        assert stats["mean_intensity"] == 0.0


class TestResultsDataFrame:
    def test_creates_dataframe(self):
        spots = [
            {
                "y": 10,
                "x": 20,
                "sigma": 2.0,
                "radius": 2.83,
                "peak_intensity": 5000,
                "corrected_intensity": 4500,
                "background_level": 100,
                "snr": 45.0,
                "chromosome_label": 1,
                "arm": "p",
                "tip_distance": 3.0,
                "associated": True,
                "valid": True,
            },
        ]
        df = create_results_dataframe(spots, image_name="test.tif")
        assert len(df) == 1
        assert "image" in df.columns
        assert df.iloc[0]["image"] == "test.tif"
        assert df.iloc[0]["y"] == 10

    def test_empty_spots_dataframe(self):
        df = create_results_dataframe([], image_name="empty.tif")
        assert len(df) == 0
        assert "image" in df.columns
