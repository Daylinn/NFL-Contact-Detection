"""
Tests for src/detection_utils.py

Covers:
  - Shared find_contacts() function
  - field_to_pixel() coordinate conversion
  - Input validation
"""

import numpy as np
import pandas as pd
import pytest

from src.detection_utils import FIELD_LENGTH, FIELD_WIDTH, field_to_pixel, find_contacts

# ===================================================================
# Constants
# ===================================================================

class TestConstants:
    def test_field_dimensions(self):
        assert FIELD_LENGTH == 120
        assert FIELD_WIDTH == 53.3


# ===================================================================
# find_contacts
# ===================================================================

class TestFindContacts:
    def test_basic_contact(self):
        df = pd.DataFrame([
            {"pid": "A", "x": 50, "y": 25, "d": 1.0},
            {"pid": "B", "x": 51, "y": 25, "d": 0.8},
        ])
        result = find_contacts(
            df, x_col="x", y_col="y", decel_col="d", player_col="pid",
            distance_threshold=2.0, decel_threshold=0.5,
        )
        assert len(result) == 1
        assert result[0]["player1"] == "A"
        assert result[0]["player2"] == "B"
        assert result[0]["distance"] == pytest.approx(1.0)

    def test_no_contact_far(self):
        df = pd.DataFrame([
            {"pid": "A", "x": 0, "y": 0, "d": 1.0},
            {"pid": "B", "x": 50, "y": 50, "d": 1.0},
        ])
        result = find_contacts(
            df, x_col="x", y_col="y", decel_col="d", player_col="pid",
            distance_threshold=2.0, decel_threshold=0.5,
        )
        assert len(result) == 0

    def test_opposing_only(self):
        df = pd.DataFrame([
            {"pid": "A", "x": 50, "y": 25, "d": 1.0, "team": "home"},
            {"pid": "B", "x": 51, "y": 25, "d": 0.8, "team": "home"},
            {"pid": "C", "x": 50.5, "y": 25, "d": 0.9, "team": "away"},
        ])
        result = find_contacts(
            df, x_col="x", y_col="y", decel_col="d", player_col="pid",
            distance_threshold=2.0, decel_threshold=0.5,
            team_col="team", opposing_only=True,
        )
        # A-B are same team, so only A-C and B-C are candidates
        players = {(c["player1"], c["player2"]) for c in result}
        for p1, p2 in players:
            assert not (p1 in ("A", "B") and p2 in ("A", "B"))

    def test_opposing_only_requires_team_col(self):
        df = pd.DataFrame([{"pid": "A", "x": 50, "y": 25, "d": 1.0}])
        with pytest.raises(ValueError, match="team_col is required"):
            find_contacts(
                df, x_col="x", y_col="y", decel_col="d", player_col="pid",
                distance_threshold=2.0, decel_threshold=0.5,
                opposing_only=True,
            )

    def test_nan_dropped(self):
        df = pd.DataFrame([
            {"pid": "A", "x": 50, "y": 25, "d": 1.0},
            {"pid": "B", "x": np.nan, "y": 25, "d": 0.8},
        ])
        result = find_contacts(
            df, x_col="x", y_col="y", decel_col="d", player_col="pid",
            distance_threshold=2.0, decel_threshold=0.5,
        )
        assert len(result) == 0  # Only one valid player, no pairs

    def test_midpoint(self):
        df = pd.DataFrame([
            {"pid": "A", "x": 40, "y": 20, "d": 1.0},
            {"pid": "B", "x": 42, "y": 20, "d": 1.0},
        ])
        result = find_contacts(
            df, x_col="x", y_col="y", decel_col="d", player_col="pid",
            distance_threshold=3.0, decel_threshold=0.5,
        )
        assert result[0]["x"] == pytest.approx(41.0)
        assert result[0]["y"] == pytest.approx(20.0)

    def test_max_decel_reported(self):
        df = pd.DataFrame([
            {"pid": "A", "x": 50, "y": 25, "d": 3.0},
            {"pid": "B", "x": 51, "y": 25, "d": 0.8},
        ])
        result = find_contacts(
            df, x_col="x", y_col="y", decel_col="d", player_col="pid",
            distance_threshold=2.0, decel_threshold=0.5,
        )
        assert result[0]["max_decel"] == pytest.approx(3.0)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["pid", "x", "y", "d"])
        result = find_contacts(
            df, x_col="x", y_col="y", decel_col="d", player_col="pid",
            distance_threshold=2.0, decel_threshold=0.5,
        )
        assert result == []


# ===================================================================
# field_to_pixel
# ===================================================================

class TestFieldToPixel:
    def test_midfield_center(self):
        px, py = field_to_pixel(60, 26.65, 1920, 1080)
        assert px == 960
        assert py == 540

    def test_origin(self):
        px, py = field_to_pixel(0, 0, 1920, 1080)
        assert px == 0
        assert py == 0

    def test_far_corner(self):
        px, py = field_to_pixel(120, 53.3, 1920, 1080)
        assert px == 1920
        assert py == 1080

    def test_different_resolution(self):
        px, py = field_to_pixel(60, 26.65, 640, 480)
        assert px == 320
        assert py == 240


# ===================================================================
# Validation tests (for both modules)
# ===================================================================

class TestValidation:
    def test_detector_negative_distance_threshold(self):
        from src.nfl_contact_detector import NFLContactDetector
        with pytest.raises(ValueError, match="distance_threshold"):
            NFLContactDetector(distance_threshold=-1.0)

    def test_detector_negative_decel_threshold(self):
        from src.nfl_contact_detector import NFLContactDetector
        with pytest.raises(ValueError, match="decel_threshold"):
            NFLContactDetector(decel_threshold=-0.5)

    def test_overlay_negative_distance_threshold(self):
        from src.create_video_overlay import NFLVideoOverlay
        with pytest.raises(ValueError, match="distance_threshold"):
            NFLVideoOverlay(distance_threshold=-1.0)

    def test_overlay_negative_decel_threshold(self):
        from src.create_video_overlay import NFLVideoOverlay
        with pytest.raises(ValueError, match="decel_threshold"):
            NFLVideoOverlay(decel_threshold=-0.5)

    def test_detector_missing_columns(self):
        from src.nfl_contact_detector import NFLContactDetector
        detector = NFLContactDetector()
        bad_df = pd.DataFrame({"wrong_col": [1, 2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            detector.calculate_deceleration(bad_df)

    def test_overlay_missing_columns(self):
        from src.create_video_overlay import NFLVideoOverlay
        overlay = NFLVideoOverlay()
        bad_df = pd.DataFrame({"wrong_col": [1, 2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            overlay.process_tracking_data(bad_df, 1, 1)
