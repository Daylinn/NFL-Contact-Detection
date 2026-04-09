"""
Shared utilities for contact detection.

Provides the core contact detection algorithm used by both
NFLContactDetector and NFLVideoOverlay.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional

# NFL field dimensions (yards)
FIELD_LENGTH = 120
FIELD_WIDTH = 53.3


def find_contacts(
    players: pd.DataFrame,
    x_col: str,
    y_col: str,
    decel_col: str,
    player_col: str,
    distance_threshold: float,
    decel_threshold: float,
    team_col: Optional[str] = None,
    opposing_only: bool = False,
) -> List[Dict]:
    """
    Detect contacts among a set of players in a single frame/step.

    This is the shared algorithm used by both detector modules. It compares
    all valid player pairs using Euclidean distance and deceleration thresholds.

    Args:
        players: DataFrame with at least [player_col, x_col, y_col, decel_col].
        x_col: Column name for x position.
        y_col: Column name for y position.
        decel_col: Column name for deceleration.
        player_col: Column name for player identifier.
        distance_threshold: Maximum distance (yards) for contact.
        decel_threshold: Minimum deceleration for impact.
        team_col: Column name for team (required if opposing_only=True).
        opposing_only: If True, only check pairs on different teams.

    Returns:
        List of contact dicts with keys:
            player1, player2, distance, max_decel, x, y
    """
    required = [player_col, x_col, y_col, decel_col]
    if opposing_only:
        if team_col is None:
            raise ValueError("team_col is required when opposing_only=True")
        required.append(team_col)

    valid = players[required].dropna()

    contacts = []
    for i, p1 in valid.iterrows():
        for j, p2 in valid.iterrows():
            if i >= j:
                continue
            if opposing_only and p1[team_col] == p2[team_col]:
                continue

            distance = np.sqrt(
                (p1[x_col] - p2[x_col]) ** 2 + (p1[y_col] - p2[y_col]) ** 2
            )

            if distance <= distance_threshold:
                if p1[decel_col] > decel_threshold or p2[decel_col] > decel_threshold:
                    contacts.append({
                        'player1': p1[player_col],
                        'player2': p2[player_col],
                        'distance': distance,
                        'max_decel': max(p1[decel_col], p2[decel_col]),
                        'x': (p1[x_col] + p2[x_col]) / 2,
                        'y': (p1[y_col] + p2[y_col]) / 2,
                    })

    return contacts


def field_to_pixel(x: float, y: float, width: int, height: int) -> tuple:
    """
    Convert NFL field coordinates (yards) to pixel coordinates.

    Args:
        x: Field x position in yards (0 to 120).
        y: Field y position in yards (0 to 53.3).
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        (px, py) integer pixel coordinates.
    """
    px = int((x / FIELD_LENGTH) * width)
    py = int((y / FIELD_WIDTH) * height)
    return px, py
