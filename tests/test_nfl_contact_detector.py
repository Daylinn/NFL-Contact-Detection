"""
Tests for src/nfl_contact_detector.py

Covers:
  - Deceleration calculation correctness and edge cases
  - Single-frame contact detection logic
  - Full-play contact detection pipeline
  - Visualization smoke tests
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from nfl_contact_detector import NFLContactDetector


# ===================================================================
# Initialization
# ===================================================================

class TestInit:
    def test_default_thresholds(self):
        detector = NFLContactDetector()
        assert detector.distance_threshold == 2.0
        assert detector.decel_threshold == 0.5

    def test_custom_thresholds(self):
        detector = NFLContactDetector(distance_threshold=3.0, decel_threshold=1.0)
        assert detector.distance_threshold == 3.0
        assert detector.decel_threshold == 1.0


# ===================================================================
# calculate_deceleration
# ===================================================================

class TestCalculateDeceleration:
    def test_deceleration_values(self, detector_tracking_df):
        """Verify deceleration = -(current_speed - previous_speed)."""
        detector = NFLContactDetector()
        result = detector.calculate_deceleration(detector_tracking_df)

        assert 'decel' in result.columns
        assert 'prev_speed' in result.columns
        assert 'speed_change' in result.columns

        # For H1: speed goes 5.0 -> 3.0 -> 4.0
        h1 = result[result['player'] == 'H1'].sort_values('time')
        decels = h1['decel'].tolist()
        assert np.isnan(decels[0])  # First frame has no previous
        assert decels[1] == pytest.approx(2.0)   # -(3.0 - 5.0) = 2.0
        assert decels[2] == pytest.approx(-1.0)  # -(4.0 - 3.0) = -1.0

    def test_first_frame_is_nan(self, detector_tracking_df):
        """First time step for each player should have NaN decel."""
        detector = NFLContactDetector()
        result = detector.calculate_deceleration(detector_tracking_df)

        for player in result['player'].unique():
            player_data = result[result['player'] == player].sort_values('time')
            assert np.isnan(player_data.iloc[0]['decel'])

    def test_no_cross_player_bleed(self):
        """Deceleration groupby must prevent values bleeding across players."""
        detector = NFLContactDetector()
        df = pd.DataFrame([
            {"gameKey": 1, "playID": 1, "player": "A", "time": 1.0, "s": 10.0},
            {"gameKey": 1, "playID": 1, "player": "B", "time": 1.0, "s": 2.0},
            {"gameKey": 1, "playID": 1, "player": "A", "time": 2.0, "s": 8.0},
            {"gameKey": 1, "playID": 1, "player": "B", "time": 2.0, "s": 3.0},
        ])
        result = detector.calculate_deceleration(df)

        a_t2 = result[(result['player'] == 'A') & (result['time'] == 2.0)].iloc[0]
        assert a_t2['decel'] == pytest.approx(2.0)  # -(8 - 10)

        b_t2 = result[(result['player'] == 'B') & (result['time'] == 2.0)].iloc[0]
        assert b_t2['decel'] == pytest.approx(-1.0)  # -(3 - 2)

    def test_does_not_mutate_input(self, detector_tracking_df):
        """Input DataFrame should not be modified."""
        detector = NFLContactDetector()
        original_cols = set(detector_tracking_df.columns)
        detector.calculate_deceleration(detector_tracking_df)
        assert set(detector_tracking_df.columns) == original_cols

    def test_sorted_output(self, detector_tracking_df):
        """Result should be sorted by gameKey, playID, player, time."""
        detector = NFLContactDetector()
        result = detector.calculate_deceleration(detector_tracking_df)
        sort_cols = ['gameKey', 'playID', 'player', 'time']
        expected = result.sort_values(sort_cols).reset_index(drop=True)
        actual = result.reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)


# ===================================================================
# detect_contacts_in_frame
# ===================================================================

class TestDetectContactsInFrame:
    def test_contact_detected(self, frame_with_contact):
        """Two close, decelerating players should produce a contact."""
        detector = NFLContactDetector(distance_threshold=2.0, decel_threshold=0.5)
        contacts = detector.detect_contacts_in_frame(frame_with_contact)

        assert len(contacts) == 1
        c = contacts[0]
        assert c['player1'] == 'H1'
        assert c['player2'] == 'V1'
        assert c['distance'] == pytest.approx(1.0)
        assert c['max_decel'] == pytest.approx(1.0)

    def test_contact_midpoint(self, frame_with_contact):
        """Contact location should be the midpoint between the two players."""
        detector = NFLContactDetector(distance_threshold=2.0, decel_threshold=0.5)
        contacts = detector.detect_contacts_in_frame(frame_with_contact)

        assert contacts[0]['x'] == pytest.approx(50.5)
        assert contacts[0]['y'] == pytest.approx(25.0)

    def test_no_contact_far_apart(self, frame_no_contact_far):
        """Distant players should not produce a contact."""
        detector = NFLContactDetector(distance_threshold=2.0, decel_threshold=0.5)
        contacts = detector.detect_contacts_in_frame(frame_no_contact_far)
        assert len(contacts) == 0

    def test_no_contact_low_decel(self, frame_no_contact_no_decel):
        """Close players without sufficient deceleration → no contact."""
        detector = NFLContactDetector(distance_threshold=2.0, decel_threshold=0.5)
        contacts = detector.detect_contacts_in_frame(frame_no_contact_no_decel)
        assert len(contacts) == 0

    def test_single_player_no_contacts(self, frame_single_player):
        """A frame with one player should return no contacts."""
        detector = NFLContactDetector()
        contacts = detector.detect_contacts_in_frame(frame_single_player)
        assert len(contacts) == 0

    def test_empty_frame(self):
        """An empty frame should return no contacts."""
        detector = NFLContactDetector()
        empty = pd.DataFrame(columns=['player', 'x', 'y', 'decel'])
        contacts = detector.detect_contacts_in_frame(empty)
        assert len(contacts) == 0

    def test_nan_rows_dropped(self, frame_with_nans):
        """Players with NaN values should be dropped; remaining valid pair checked."""
        detector = NFLContactDetector(distance_threshold=2.0, decel_threshold=0.5)
        contacts = detector.detect_contacts_in_frame(frame_with_nans)
        # V1 has NaN x, so only H1 and V2 remain (distance = 1.0, both decel > 0.5)
        assert len(contacts) == 1
        assert contacts[0]['player1'] == 'H1'
        assert contacts[0]['player2'] == 'V2'

    def test_no_duplicate_pairs(self):
        """Each pair should only appear once (A-B, not also B-A)."""
        detector = NFLContactDetector(distance_threshold=5.0, decel_threshold=0.1)
        df = pd.DataFrame([
            {"player": "A", "x": 50, "y": 25, "decel": 1.0},
            {"player": "B", "x": 51, "y": 25, "decel": 1.0},
            {"player": "C", "x": 52, "y": 25, "decel": 1.0},
        ])
        contacts = detector.detect_contacts_in_frame(df)
        pairs = {(c['player1'], c['player2']) for c in contacts}
        # No reversed duplicates
        for p1, p2 in pairs:
            assert (p2, p1) not in pairs

    def test_custom_thresholds(self, frame_with_contact):
        """Stricter threshold should reject previously valid contacts."""
        detector = NFLContactDetector(distance_threshold=0.5, decel_threshold=0.5)
        contacts = detector.detect_contacts_in_frame(frame_with_contact)
        assert len(contacts) == 0  # distance is 1.0, threshold is 0.5

    def test_only_one_player_needs_decel(self):
        """Contact should be detected if only one player exceeds decel threshold."""
        detector = NFLContactDetector(distance_threshold=2.0, decel_threshold=0.5)
        df = pd.DataFrame([
            {"player": "A", "x": 50, "y": 25, "decel": 1.0},
            {"player": "B", "x": 51, "y": 25, "decel": 0.1},
        ])
        contacts = detector.detect_contacts_in_frame(df)
        assert len(contacts) == 1


# ===================================================================
# detect_contacts_in_play
# ===================================================================

class TestDetectContactsInPlay:
    def test_returns_dataframe(self, detector_tracking_df):
        """Result should be a DataFrame."""
        detector = NFLContactDetector(distance_threshold=2.0, decel_threshold=0.5)
        result = detector.detect_contacts_in_play(detector_tracking_df)
        assert isinstance(result, pd.DataFrame)

    def test_contacts_have_time_column(self, detector_tracking_df):
        """Each contact row should have a 'time' column."""
        detector = NFLContactDetector(distance_threshold=2.0, decel_threshold=0.5)
        result = detector.detect_contacts_in_play(detector_tracking_df)
        if len(result) > 0:
            assert 'time' in result.columns

    def test_expected_schema(self, detector_tracking_df):
        """Contacts DataFrame should have the expected columns."""
        detector = NFLContactDetector(distance_threshold=2.0, decel_threshold=0.5)
        result = detector.detect_contacts_in_play(detector_tracking_df)
        if len(result) > 0:
            expected_cols = {'player1', 'player2', 'distance', 'max_decel', 'x', 'y', 'time'}
            assert expected_cols == set(result.columns)

    def test_contacts_detected_at_correct_time(self, detector_tracking_df):
        """H1 and V1 should have contact at time 2.0 (close + decelerating)."""
        detector = NFLContactDetector(distance_threshold=2.0, decel_threshold=0.5)
        result = detector.detect_contacts_in_play(detector_tracking_df)

        # At time 2.0, H1(50.5,25.1) and V1(50.8,25.3) are ~0.36 yards apart
        # and both decelerate by 2.0 yards/sec²
        time_2 = result[result['time'] == 2.0]
        assert len(time_2) >= 1
        players_involved = set(time_2['player1'].tolist() + time_2['player2'].tolist())
        assert 'H1' in players_involved
        assert 'V1' in players_involved

    def test_no_contacts_with_strict_thresholds(self, detector_tracking_df):
        """Very strict thresholds should produce no contacts."""
        detector = NFLContactDetector(distance_threshold=0.01, decel_threshold=100.0)
        result = detector.detect_contacts_in_play(detector_tracking_df)
        assert len(result) == 0


# ===================================================================
# visualize_contacts (smoke tests with mocked matplotlib)
# ===================================================================

class TestVisualizeContacts:
    def test_returns_figure(self, detector_tracking_df):
        """visualize_contacts should return a matplotlib Figure."""
        import matplotlib.pyplot as plt
        detector = NFLContactDetector()
        play_data = detector.calculate_deceleration(detector_tracking_df)
        contacts = detector.detect_contacts_in_play(detector_tracking_df)

        fig = detector.visualize_contacts(play_data, contacts)
        assert fig is not None
        plt.close(fig)

    def test_saves_to_file(self, detector_tracking_df, tmp_path):
        """Should save figure when output_path is provided."""
        import matplotlib.pyplot as plt
        detector = NFLContactDetector()
        play_data = detector.calculate_deceleration(detector_tracking_df)
        contacts = detector.detect_contacts_in_play(detector_tracking_df)

        out = str(tmp_path / "test_output.png")
        fig = detector.visualize_contacts(play_data, contacts, output_path=out)
        assert os.path.exists(out)
        plt.close(fig)

    def test_handles_empty_contacts(self, detector_tracking_df):
        """Should not crash with an empty contacts DataFrame."""
        import matplotlib.pyplot as plt
        detector = NFLContactDetector()
        play_data = detector.calculate_deceleration(detector_tracking_df)
        empty_contacts = pd.DataFrame()

        fig = detector.visualize_contacts(play_data, empty_contacts)
        assert fig is not None
        plt.close(fig)
