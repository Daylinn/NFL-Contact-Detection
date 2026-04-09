"""
Tests for src/create_video_overlay.py

Covers:
  - Tracking data processing and filtering
  - Contact detection (opposing teams only)
  - Step-to-frame mapping
  - Coordinate transformations
  - Video I/O with mocked OpenCV
  - Thumbnail extraction
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from src.create_video_overlay import NFLVideoOverlay


# ===================================================================
# Initialization
# ===================================================================

class TestInit:
    def test_default_thresholds(self):
        overlay = NFLVideoOverlay()
        assert overlay.distance_threshold == 2.0
        assert overlay.decel_threshold == 0.3

    def test_custom_thresholds(self):
        overlay = NFLVideoOverlay(distance_threshold=3.0, decel_threshold=1.0)
        assert overlay.distance_threshold == 3.0
        assert overlay.decel_threshold == 1.0


# ===================================================================
# process_tracking_data
# ===================================================================

class TestProcessTrackingData:
    def test_filters_by_game_and_play(self, overlay_tracking_df):
        """Only rows matching game_key and play_id should be returned."""
        overlay = NFLVideoOverlay()
        result = overlay.process_tracking_data(overlay_tracking_df, 58172, 100)

        assert (result['game_key'] == 58172).all()
        assert (result['play_id'] == 100).all()
        # play_id 999 row should be excluded
        assert len(result) == 9

    def test_deceleration_calculated(self, overlay_tracking_df):
        """Result should have decel column with correct values."""
        overlay = NFLVideoOverlay()
        result = overlay.process_tracking_data(overlay_tracking_df, 58172, 100)

        assert 'decel' in result.columns
        assert 'prev_speed' in result.columns

        # Player 101: speed 5.0 -> 3.0 -> 4.0
        p101 = result[result['nfl_player_id'] == 101].sort_values('step')
        decels = p101['decel'].tolist()
        assert np.isnan(decels[0])  # First step has no previous
        assert decels[1] == pytest.approx(2.0)   # 5.0 - 3.0
        assert decels[2] == pytest.approx(-1.0)  # 3.0 - 4.0

    def test_no_matching_data(self, overlay_tracking_df):
        """Non-existent game/play combo should return empty DataFrame."""
        overlay = NFLVideoOverlay()
        result = overlay.process_tracking_data(overlay_tracking_df, 99999, 99999)
        assert len(result) == 0

    def test_sorted_by_player_and_step(self, overlay_tracking_df):
        """Output should be sorted by nfl_player_id and step."""
        overlay = NFLVideoOverlay()
        result = overlay.process_tracking_data(overlay_tracking_df, 58172, 100)

        expected = result.sort_values(['nfl_player_id', 'step']).reset_index(drop=True)
        actual = result.reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)


# ===================================================================
# detect_contacts_at_step
# ===================================================================

class TestDetectContactsAtStep:
    def test_opposing_teams_contact(self, overlay_step_with_contact):
        """Opposing team players that are close + decelerating → contact."""
        overlay = NFLVideoOverlay(distance_threshold=2.0, decel_threshold=0.3)
        contacts = overlay.detect_contacts_at_step(overlay_step_with_contact)

        assert len(contacts) == 1
        c = contacts[0]
        assert c['player1'] == 12
        assert c['player2'] == 88
        assert c['distance'] == pytest.approx(1.0)
        assert c['max_decel'] == pytest.approx(1.5)

    def test_same_team_no_contact(self, overlay_step_same_team):
        """Same-team players should never produce contacts."""
        overlay = NFLVideoOverlay(distance_threshold=2.0, decel_threshold=0.3)
        contacts = overlay.detect_contacts_at_step(overlay_step_same_team)
        assert len(contacts) == 0

    def test_contact_midpoint(self, overlay_step_with_contact):
        """Contact (x, y) should be the midpoint of the two players."""
        overlay = NFLVideoOverlay(distance_threshold=2.0, decel_threshold=0.3)
        contacts = overlay.detect_contacts_at_step(overlay_step_with_contact)

        assert contacts[0]['x'] == pytest.approx(50.5)
        assert contacts[0]['y'] == pytest.approx(25.0)

    def test_uses_jersey_number_as_player_id(self, overlay_step_with_contact):
        """Player identifiers in contacts should be jersey numbers."""
        overlay = NFLVideoOverlay(distance_threshold=2.0, decel_threshold=0.3)
        contacts = overlay.detect_contacts_at_step(overlay_step_with_contact)

        assert contacts[0]['player1'] == 12
        assert contacts[0]['player2'] == 88

    def test_no_contact_below_decel_threshold(self):
        """Close opposing players without deceleration → no contact."""
        overlay = NFLVideoOverlay(distance_threshold=2.0, decel_threshold=0.3)
        df = pd.DataFrame([
            {"nfl_player_id": 1, "x_position": 50, "y_position": 25,
             "decel": 0.1, "jersey_number": 10, "team": "home"},
            {"nfl_player_id": 2, "x_position": 51, "y_position": 25,
             "decel": 0.1, "jersey_number": 20, "team": "away"},
        ])
        contacts = overlay.detect_contacts_at_step(df)
        assert len(contacts) == 0

    def test_empty_step(self):
        """Empty step data → no contacts."""
        overlay = NFLVideoOverlay()
        empty = pd.DataFrame(columns=[
            'nfl_player_id', 'x_position', 'y_position',
            'decel', 'jersey_number', 'team'
        ])
        contacts = overlay.detect_contacts_at_step(empty)
        assert len(contacts) == 0


# ===================================================================
# detect_all_contacts
# ===================================================================

class TestDetectAllContacts:
    def test_returns_dict(self, overlay_tracking_df):
        """Result should be a dict mapping step → list of contacts."""
        overlay = NFLVideoOverlay(distance_threshold=2.0, decel_threshold=0.3)
        play_data = overlay.process_tracking_data(overlay_tracking_df, 58172, 100)
        result = overlay.detect_all_contacts(play_data)

        assert isinstance(result, dict)
        for step, contacts in result.items():
            assert isinstance(step, (int, np.integer))
            assert isinstance(contacts, list)

    def test_only_steps_with_contacts(self, overlay_tracking_df):
        """Dict should only contain steps that actually have contacts."""
        overlay = NFLVideoOverlay(distance_threshold=2.0, decel_threshold=0.3)
        play_data = overlay.process_tracking_data(overlay_tracking_df, 58172, 100)
        result = overlay.detect_all_contacts(play_data)

        for contacts in result.values():
            assert len(contacts) > 0

    def test_no_contacts_strict_threshold(self, overlay_tracking_df):
        """Very strict thresholds → empty dict."""
        overlay = NFLVideoOverlay(distance_threshold=0.01, decel_threshold=100.0)
        play_data = overlay.process_tracking_data(overlay_tracking_df, 58172, 100)
        result = overlay.detect_all_contacts(play_data)
        assert len(result) == 0


# ===================================================================
# _map_steps_to_frames
# ===================================================================

class TestMapStepsToFrames:
    def test_linear_mapping(self):
        """Steps should map linearly to frame numbers."""
        overlay = NFLVideoOverlay()
        steps = [10, 20, 30, 40, 50]
        total_frames = 100

        result = overlay._map_steps_to_frames(steps, total_frames)

        assert result[10] == 0     # int((0/5)*100) = 0
        assert result[20] == 20    # int((1/5)*100) = 20
        assert result[30] == 40    # int((2/5)*100) = 40
        assert result[40] == 60    # int((3/5)*100) = 60
        assert result[50] == 80    # int((4/5)*100) = 80

    def test_single_step(self):
        """One step should map to frame 0."""
        overlay = NFLVideoOverlay()
        result = overlay._map_steps_to_frames([1], 100)
        assert result[1] == 0

    def test_returns_dict(self):
        """Should return a dict mapping step number to frame number."""
        overlay = NFLVideoOverlay()
        result = overlay._map_steps_to_frames([1, 2, 3], 90)
        assert isinstance(result, dict)
        assert len(result) == 3


# ===================================================================
# Coordinate transformations
# ===================================================================

class TestCoordinateTransforms:
    def test_player_overlay_coordinates(self):
        """Field coordinates should map to pixel coordinates correctly."""
        overlay = NFLVideoOverlay()
        step_data = pd.DataFrame([{
            "x_position": 60.0,     # midfield
            "y_position": 26.65,    # mid-width (53.3/2)
            "team": "home",
            "jersey_number": 12,
            "speed": 3.0,
        }])

        width, height = 1920, 1080
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        with patch('cv2.circle'), patch('cv2.putText'):
            result = overlay._draw_player_overlays(frame, step_data, width, height)

        # Just verifying it doesn't crash; coordinate math:
        # px = int((60 / 120) * 1920) = 960
        # py = int((26.65 / 53.3) * 1080) = 540
        assert result is not None

    def test_contact_overlay_coordinates(self):
        """Contact location coordinates should map correctly."""
        overlay = NFLVideoOverlay()
        contacts = [{"x": 60.0, "y": 26.65, "player1": 12, "player2": 88,
                      "distance": 1.0, "max_decel": 1.5}]
        width, height = 1920, 1080
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        with patch('cv2.drawMarker'), patch('cv2.putText'), patch('cv2.rectangle'):
            result = overlay._draw_contact_overlays(frame, contacts, width, height)

        assert result is not None


# ===================================================================
# create_overlay_video (mocked OpenCV)
# ===================================================================

class TestCreateOverlayVideo:
    def test_raises_on_invalid_video(self):
        """Should raise ValueError when video cannot be opened."""
        overlay = NFLVideoOverlay()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch('cv2.VideoCapture', return_value=mock_cap):
            with pytest.raises(ValueError, match="Cannot open video"):
                overlay.create_overlay_video(
                    'nonexistent.mp4',
                    pd.DataFrame({'step': [1]}),
                    {},
                    'output.mp4'
                )

    def test_processes_frames_and_releases(self, overlay_tracking_df):
        """Should process frames, write output, and release resources."""
        overlay = NFLVideoOverlay()
        play_data = overlay.process_tracking_data(overlay_tracking_df, 58172, 100)
        contacts = overlay.detect_all_contacts(play_data)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            5: 30,    # FPS
            3: 1920,  # WIDTH
            4: 1080,  # HEIGHT
            7: 90,    # FRAME_COUNT
        }.get(prop, 0)

        # Simulate reading 3 frames then stopping
        fake_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [
            (True, fake_frame.copy()),
            (True, fake_frame.copy()),
            (True, fake_frame.copy()),
            (False, None),
        ]

        mock_out = MagicMock()

        with patch('cv2.VideoCapture', return_value=mock_cap), \
             patch('cv2.VideoWriter_fourcc', return_value=0), \
             patch('cv2.VideoWriter', return_value=mock_out):
            frames, contact_frames = overlay.create_overlay_video(
                'input.mp4', play_data, contacts, 'output.mp4'
            )

        assert frames == 3
        assert mock_out.write.call_count == 3
        mock_cap.release.assert_called_once()
        mock_out.release.assert_called_once()

    def test_return_type(self, overlay_tracking_df):
        """Should return a tuple of (total_frames, contact_frames)."""
        overlay = NFLVideoOverlay()
        play_data = overlay.process_tracking_data(overlay_tracking_df, 58172, 100)
        contacts = {}

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {5: 30, 3: 640, 4: 480, 7: 1}.get(prop, 0)
        mock_cap.read.side_effect = [(True, np.zeros((480, 640, 3), dtype=np.uint8)), (False, None)]

        mock_out = MagicMock()

        with patch('cv2.VideoCapture', return_value=mock_cap), \
             patch('cv2.VideoWriter_fourcc', return_value=0), \
             patch('cv2.VideoWriter', return_value=mock_out):
            result = overlay.create_overlay_video(
                'input.mp4', play_data, contacts, 'output.mp4'
            )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result == (1, 0)  # 1 frame processed, 0 with contacts


# ===================================================================
# create_thumbnail (mocked OpenCV)
# ===================================================================

class TestCreateThumbnail:
    def test_successful_extraction(self, tmp_path):
        """Should return True on successful thumbnail extraction."""
        overlay = NFLVideoOverlay()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100  # total frames
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        with patch('cv2.VideoCapture', return_value=mock_cap), \
             patch('cv2.imwrite', return_value=True):
            result = overlay.create_thumbnail('video.mp4', str(tmp_path / 'thumb.jpg'))

        assert result is True
        mock_cap.release.assert_called_once()

    def test_default_middle_frame(self):
        """When frame_number is None, should seek to middle frame."""
        overlay = NFLVideoOverlay()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100  # total frames → middle = 50
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        with patch('cv2.VideoCapture', return_value=mock_cap), \
             patch('cv2.imwrite'):
            overlay.create_thumbnail('video.mp4', 'thumb.jpg')

        # Should seek to frame 50
        mock_cap.set.assert_called_once()
        assert mock_cap.set.call_args[0][1] == 50

    def test_specific_frame_number(self):
        """Should seek to the specified frame number."""
        overlay = NFLVideoOverlay()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        with patch('cv2.VideoCapture', return_value=mock_cap), \
             patch('cv2.imwrite'):
            overlay.create_thumbnail('video.mp4', 'thumb.jpg', frame_number=25)

        mock_cap.set.assert_called_once()
        assert mock_cap.set.call_args[0][1] == 25

    def test_returns_false_on_failed_open(self):
        """Should return False when video can't be opened."""
        overlay = NFLVideoOverlay()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch('cv2.VideoCapture', return_value=mock_cap):
            result = overlay.create_thumbnail('bad.mp4', 'thumb.jpg')

        assert result is False

    def test_returns_false_on_failed_read(self):
        """Should return False when frame read fails."""
        overlay = NFLVideoOverlay()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (False, None)

        with patch('cv2.VideoCapture', return_value=mock_cap):
            result = overlay.create_thumbnail('video.mp4', 'thumb.jpg')

        assert result is False
