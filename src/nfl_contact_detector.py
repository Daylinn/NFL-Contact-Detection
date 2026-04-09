"""
NFL Player Contact Detection System
====================================

Detects player-to-player contact using NFL tracking data.

This module provides functions to:
1. Process tracking data and calculate contact features
2. Detect contacts based on proximity and deceleration
3. Visualize results on field diagrams

Author: Daylin Hart
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.detection_utils import find_contacts, FIELD_LENGTH, FIELD_WIDTH


class NFLContactDetector:
    """
    Contact detection system for NFL player tracking data.
    
    Uses proximity and deceleration thresholds to identify player-to-player
    contact events during gameplay.
    """
    
    def __init__(self, distance_threshold: float = 2.0, decel_threshold: float = 0.5):
        """
        Initialize contact detector.
        
        Args:
            distance_threshold: Maximum distance (yards) for contact
            decel_threshold: Minimum deceleration (yards/sec²) for impact
        """
        if distance_threshold < 0:
            raise ValueError("distance_threshold must be non-negative")
        if decel_threshold < 0:
            raise ValueError("decel_threshold must be non-negative")
        self.distance_threshold = distance_threshold
        self.decel_threshold = decel_threshold

    def calculate_deceleration(self, tracking_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate player deceleration from tracking data.
        
        Args:
            tracking_df: DataFrame with columns [gameKey, playID, player, time, s]
        
        Returns:
            DataFrame with added 'decel' column
        """
        required = {'gameKey', 'playID', 'player', 'time', 's'}
        missing = required - set(tracking_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = tracking_df.copy()
        df = df.sort_values(['gameKey', 'playID', 'player', 'time'])
        
        # Calculate velocity change
        df['prev_speed'] = df.groupby(['gameKey', 'playID', 'player'])['s'].shift(1)
        df['speed_change'] = df['s'] - df['prev_speed']
        df['decel'] = -df['speed_change']  # Positive deceleration = slowing down
        
        return df
    
    def detect_contacts_in_frame(self, frame_data: pd.DataFrame) -> List[Dict]:
        """
        Detect contacts in a single frame.

        Args:
            frame_data: DataFrame with player positions for one time step
                       Required columns: [player, x, y, decel]

        Returns:
            List of contact dictionaries with:
                - player1, player2: Player identifiers
                - distance: Distance between players (yards)
                - max_decel: Maximum deceleration of pair (yards/sec²)
                - x, y: Contact location coordinates
        """
        return find_contacts(
            frame_data,
            x_col='x', y_col='y', decel_col='decel', player_col='player',
            distance_threshold=self.distance_threshold,
            decel_threshold=self.decel_threshold,
        )
    
    def detect_contacts_in_play(self, play_tracking: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all contacts in a play.
        
        Args:
            play_tracking: DataFrame with tracking data for entire play
        
        Returns:
            DataFrame with detected contacts and metadata
        """
        # Calculate deceleration
        play_data = self.calculate_deceleration(play_tracking)
        
        # Detect contacts for each time step
        all_contacts = []
        for time, frame_data in play_data.groupby('time', sort=True):
            frame_contacts = self.detect_contacts_in_frame(frame_data)
            
            for contact in frame_contacts:
                contact['time'] = time
                all_contacts.append(contact)
        
        return pd.DataFrame(all_contacts)
    
    def visualize_contacts(self, 
                          play_data: pd.DataFrame, 
                          contacts_df: pd.DataFrame,
                          output_path: str = None) -> plt.Figure:
        """
        Create visualization of contacts on field diagram.
        
        Args:
            play_data: Tracking data for play
            contacts_df: Detected contacts
            output_path: Optional path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('white')
        
        # Select frame to visualize
        times = sorted(play_data['time'].unique())
        if len(contacts_df) > 0:
            contact_time = contacts_df.groupby('time').size().idxmax()
        else:
            contact_time = times[len(times)//2]
        
        frame_data = play_data[play_data['time'] == contact_time]
        frame_contacts = contacts_df[contacts_df['time'] == contact_time] if len(contacts_df) > 0 else pd.DataFrame()
        
        # Plot 1: Field view with player positions
        ax1 = axes[0]
        self._draw_field(ax1)
        self._draw_players(ax1, frame_data)
        self._draw_contacts(ax1, frame_contacts)
        
        # Plot 2: Contact timeline
        ax2 = axes[1]
        self._draw_contact_timeline(ax2, contacts_df, contact_time)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def _draw_field(self, ax):
        """Draw NFL field background."""
        ax.set_facecolor('#2C5F2D')
        ax.set_xlim([0, FIELD_LENGTH])
        ax.set_ylim([0, FIELD_WIDTH])
        ax.set_xlabel('Yards', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel('Width (yards)', fontsize=12, fontweight='bold', color='white')
        ax.set_title('Player Positions & Contact Detection', fontsize=14, fontweight='bold', color='white')
        
        # Field lines
        for yard in range(0, FIELD_LENGTH + 1, 10):
            ax.axvline(yard, color='white', alpha=0.3, linewidth=0.8)
        ax.axhline(FIELD_WIDTH / 2, color='white', alpha=0.5, linewidth=1.5)
        ax.tick_params(colors='white')
    
    def _draw_players(self, ax, frame_data):
        """Draw players as circles with velocity vectors."""
        for _, player in frame_data.iterrows():
            # Determine color by team (H=home, V=visitor)
            color = '#FF4444' if player['player'].startswith('H') else '#4444FF'
            
            # Draw player circle
            circle = Circle((player['x'], player['y']), 1.5, 
                          color=color, alpha=0.7, edgecolor='white', linewidth=2)
            ax.add_patch(circle)
            
            # Draw velocity vector if moving
            if player['s'] > 0.5:
                dx = 2 * player['s'] * np.cos(np.radians(player['dir']))
                dy = 2 * player['s'] * np.sin(np.radians(player['dir']))
                ax.arrow(player['x'], player['y'], dx, dy,
                        head_width=0.8, head_length=0.5, 
                        fc=color, ec='white', linewidth=1.5)
    
    def _draw_contacts(self, ax, frame_contacts):
        """Draw contact markers."""
        if len(frame_contacts) > 0:
            for idx, contact in frame_contacts.iterrows():
                ax.scatter(contact['x'], contact['y'], 
                          s=500, c='yellow', marker='*', 
                          edgecolors='red', linewidth=2, zorder=10,
                          label='Contact Detected' if idx == frame_contacts.index[0] else '')
            ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    def _draw_contact_timeline(self, ax, contacts_df, current_time):
        """Draw contact frequency over time."""
        if len(contacts_df) > 0:
            contact_counts = contacts_df.groupby('time').size()
            ax.plot(range(len(contact_counts)), contact_counts.values, 
                   color='#FF4444', linewidth=2.5, marker='o', markersize=6)
            ax.fill_between(range(len(contact_counts)), contact_counts.values, 
                           alpha=0.3, color='#FF4444')
            ax.set_ylabel('Number of Contacts', fontsize=12, fontweight='bold')
            ax.set_title('Contact Detection Over Time', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No contacts detected\nin this play', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Contact Detection Over Time', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Frame Index', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')


def main():
    """
    Example usage demonstrating contact detection on sample data.
    """
    print("="*60)
    print("NFL PLAYER CONTACT DETECTION - DEMO")
    print("="*60)
    
    # Load sample data
    print("\n[1/3] Loading tracking data...")
    tracking = pd.read_csv('/mnt/user-data/uploads/test_player_tracking.csv')
    print(f"  ✓ Loaded {len(tracking):,} tracking records")
    
    # Select a sample play
    sample_game = tracking['gameKey'].iloc[0]
    sample_plays = tracking[tracking['gameKey'] == sample_game]['playID'].unique()
    sample_play = sample_plays[1] if len(sample_plays) > 1 else sample_plays[0]
    
    play_data = tracking[(tracking['gameKey'] == sample_game) & 
                        (tracking['playID'] == sample_play)].copy()
    
    print(f"\n[2/3] Detecting contacts...")
    print(f"  Game: {sample_game}, Play: {sample_play}")
    print(f"  Duration: {play_data['time'].nunique()} frames")
    print(f"  Players: {play_data['player'].nunique()}")
    
    # Run detection
    detector = NFLContactDetector(distance_threshold=2.0, decel_threshold=0.5)
    contacts_df = detector.detect_contacts_in_play(play_data)
    
    print(f"  ✓ Detected {len(contacts_df)} contacts")
    if len(contacts_df) > 0:
        print(f"  ✓ Average distance: {contacts_df['distance'].mean():.2f} yards")
        print(f"  ✓ Average deceleration: {contacts_df['max_decel'].mean():.2f} yards/sec²")
    
    # Visualize
    print("\n[3/3] Creating visualization...")
    detector.visualize_contacts(play_data, contacts_df, 'contact_detection_demo.png')
    print("  ✓ Saved: contact_detection_demo.png")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()