"""
Shared pytest fixtures for NFL Contact Detection tests.

Provides synthetic DataFrames that match each module's expected schema.
"""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures for nfl_contact_detector.py
# (columns: gameKey, playID, player, time, x, y, s, dir)
# ---------------------------------------------------------------------------

@pytest.fixture
def detector_tracking_df():
    """Multi-player, multi-frame tracking data for NFLContactDetector.

    Layout at time 1.0:
        H1 at (50, 25) speed 5.0
        H2 at (60, 30) speed 4.0
        V1 at (51, 25.5) speed 5.0   <-- close to H1 (dist ~1.12 yards)
        V2 at (80, 40) speed 3.0     <-- far from everyone

    At time 2.0 all players have slowed (positive deceleration).
    """
    rows = [
        # time 1.0
        {"gameKey": 1, "playID": 100, "player": "H1", "time": 1.0, "x": 50, "y": 25, "s": 5.0, "dir": 90},
        {"gameKey": 1, "playID": 100, "player": "H2", "time": 1.0, "x": 60, "y": 30, "s": 4.0, "dir": 90},
        {"gameKey": 1, "playID": 100, "player": "V1", "time": 1.0, "x": 51, "y": 25.5, "s": 5.0, "dir": 270},
        {"gameKey": 1, "playID": 100, "player": "V2", "time": 1.0, "x": 80, "y": 40, "s": 3.0, "dir": 180},
        # time 2.0  – H1 and V1 still close; both decelerate sharply
        {"gameKey": 1, "playID": 100, "player": "H1", "time": 2.0, "x": 50.5, "y": 25.1, "s": 3.0, "dir": 90},
        {"gameKey": 1, "playID": 100, "player": "H2", "time": 2.0, "x": 60.5, "y": 30.1, "s": 3.5, "dir": 90},
        {"gameKey": 1, "playID": 100, "player": "V1", "time": 2.0, "x": 50.8, "y": 25.3, "s": 3.0, "dir": 270},
        {"gameKey": 1, "playID": 100, "player": "V2", "time": 2.0, "x": 80.2, "y": 40.1, "s": 2.5, "dir": 180},
        # time 3.0 – players separate
        {"gameKey": 1, "playID": 100, "player": "H1", "time": 3.0, "x": 52, "y": 26, "s": 4.0, "dir": 90},
        {"gameKey": 1, "playID": 100, "player": "H2", "time": 3.0, "x": 61, "y": 30.5, "s": 4.0, "dir": 90},
        {"gameKey": 1, "playID": 100, "player": "V1", "time": 3.0, "x": 48, "y": 24, "s": 4.0, "dir": 270},
        {"gameKey": 1, "playID": 100, "player": "V2", "time": 3.0, "x": 80.5, "y": 40.2, "s": 3.0, "dir": 180},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def frame_with_contact():
    """Single-frame data where two players are close and decelerating."""
    return pd.DataFrame([
        {"player": "H1", "x": 50.0, "y": 25.0, "decel": 1.0},
        {"player": "V1", "x": 51.0, "y": 25.0, "decel": 0.8},
    ])


@pytest.fixture
def frame_no_contact_far():
    """Single-frame data where players are too far apart."""
    return pd.DataFrame([
        {"player": "H1", "x": 10.0, "y": 10.0, "decel": 1.0},
        {"player": "V1", "x": 50.0, "y": 40.0, "decel": 1.0},
    ])


@pytest.fixture
def frame_no_contact_no_decel():
    """Single-frame data where players are close but not decelerating."""
    return pd.DataFrame([
        {"player": "H1", "x": 50.0, "y": 25.0, "decel": 0.1},
        {"player": "V1", "x": 51.0, "y": 25.0, "decel": 0.0},
    ])


@pytest.fixture
def frame_single_player():
    """Single-frame data with only one player."""
    return pd.DataFrame([
        {"player": "H1", "x": 50.0, "y": 25.0, "decel": 1.0},
    ])


@pytest.fixture
def frame_with_nans():
    """Single-frame data containing NaN values that should be dropped."""
    return pd.DataFrame([
        {"player": "H1", "x": 50.0, "y": 25.0, "decel": 1.0},
        {"player": "V1", "x": np.nan, "y": 25.0, "decel": 0.8},
        {"player": "V2", "x": 51.0, "y": 25.0, "decel": 1.0},
    ])


# ---------------------------------------------------------------------------
# Fixtures for create_video_overlay.py
# (columns: game_key, play_id, nfl_player_id, step, x_position, y_position,
#           speed, jersey_number, team)
# ---------------------------------------------------------------------------

@pytest.fixture
def overlay_tracking_df():
    """Multi-player tracking data for NFLVideoOverlay.

    Step 1: Players 101(home) and 201(away) are close and decelerating.
    Step 2: Players move apart.
    """
    rows = [
        # Step 1
        {"game_key": 58172, "play_id": 100, "nfl_player_id": 101, "step": 1,
         "x_position": 50.0, "y_position": 25.0, "speed": 5.0,
         "jersey_number": 12, "team": "home"},
        {"game_key": 58172, "play_id": 100, "nfl_player_id": 201, "step": 1,
         "x_position": 51.0, "y_position": 25.0, "speed": 5.0,
         "jersey_number": 88, "team": "away"},
        {"game_key": 58172, "play_id": 100, "nfl_player_id": 301, "step": 1,
         "x_position": 80.0, "y_position": 40.0, "speed": 3.0,
         "jersey_number": 55, "team": "home"},
        # Step 2 – players decelerate
        {"game_key": 58172, "play_id": 100, "nfl_player_id": 101, "step": 2,
         "x_position": 50.5, "y_position": 25.1, "speed": 3.0,
         "jersey_number": 12, "team": "home"},
        {"game_key": 58172, "play_id": 100, "nfl_player_id": 201, "step": 2,
         "x_position": 50.8, "y_position": 25.2, "speed": 3.0,
         "jersey_number": 88, "team": "away"},
        {"game_key": 58172, "play_id": 100, "nfl_player_id": 301, "step": 2,
         "x_position": 80.5, "y_position": 40.1, "speed": 2.5,
         "jersey_number": 55, "team": "home"},
        # Step 3 – players separate
        {"game_key": 58172, "play_id": 100, "nfl_player_id": 101, "step": 3,
         "x_position": 52.0, "y_position": 26.0, "speed": 4.0,
         "jersey_number": 12, "team": "home"},
        {"game_key": 58172, "play_id": 100, "nfl_player_id": 201, "step": 3,
         "x_position": 48.0, "y_position": 24.0, "speed": 4.0,
         "jersey_number": 88, "team": "away"},
        {"game_key": 58172, "play_id": 100, "nfl_player_id": 301, "step": 3,
         "x_position": 81.0, "y_position": 40.2, "speed": 3.5,
         "jersey_number": 55, "team": "home"},
        # Different play (should be filtered out)
        {"game_key": 58172, "play_id": 999, "nfl_player_id": 101, "step": 1,
         "x_position": 10.0, "y_position": 10.0, "speed": 2.0,
         "jersey_number": 12, "team": "home"},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def overlay_step_with_contact():
    """Single step where opposing players are close and decelerating."""
    return pd.DataFrame([
        {"nfl_player_id": 101, "x_position": 50.0, "y_position": 25.0,
         "decel": 1.5, "jersey_number": 12, "team": "home"},
        {"nfl_player_id": 201, "x_position": 51.0, "y_position": 25.0,
         "decel": 0.8, "jersey_number": 88, "team": "away"},
    ])


@pytest.fixture
def overlay_step_same_team():
    """Single step where close players are on the same team (no contact expected)."""
    return pd.DataFrame([
        {"nfl_player_id": 101, "x_position": 50.0, "y_position": 25.0,
         "decel": 1.5, "jersey_number": 12, "team": "home"},
        {"nfl_player_id": 102, "x_position": 51.0, "y_position": 25.0,
         "decel": 0.8, "jersey_number": 44, "team": "home"},
    ])
