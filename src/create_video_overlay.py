"""
NFL Contact Detection - Video Overlay System
============================================

Overlays contact detection results on NFL game video footage.

This module synchronizes tracking data with video frames and creates
annotated output showing:
- Player positions with jersey numbers
- Detected contact events
- Real-time alerts

Note: Video overlay is optional visualization - contact detection
works with tracking data alone.

Author: Daylin Hart
Date: January 2026
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class NFLVideoOverlay:
    """
    Video processing system for visualizing contact detection results.
    
    Synchronizes tracking data with game footage and overlays:
    - Player position markers
    - Jersey number labels
    - Contact event highlights
    - Alert banners
    """
    
    def __init__(self, distance_threshold: float = 2.0, decel_threshold: float = 0.3):
        """
        Initialize video overlay system.
        
        Args:
            distance_threshold: Contact distance threshold (yards)
            decel_threshold: Contact deceleration threshold (yards/sec²)
        """
        self.distance_threshold = distance_threshold
        self.decel_threshold = decel_threshold
        
    def process_tracking_data(self, tracking_df: pd.DataFrame, 
                             game_key: int, play_id: int) -> pd.DataFrame:
        """
        Filter and prepare tracking data for specific play.
        
        Args:
            tracking_df: Full tracking DataFrame
            game_key: Game identifier
            play_id: Play identifier
        
        Returns:
            Filtered and processed tracking data with deceleration
        """
        # Filter to specific play
        play_data = tracking_df[
            (tracking_df['game_key'] == game_key) & 
            (tracking_df['play_id'] == play_id)
        ].copy()
        
        # Calculate deceleration
        play_data = play_data.sort_values(['nfl_player_id', 'step'])
        play_data['prev_speed'] = play_data.groupby('nfl_player_id')['speed'].shift(1)
        play_data['decel'] = play_data['prev_speed'] - play_data['speed']
        
        return play_data
    
    def detect_contacts_at_step(self, step_data: pd.DataFrame) -> List[Dict]:
        """
        Detect contacts for a single tracking time step.
        
        Args:
            step_data: Tracking data for one time step
        
        Returns:
            List of contact dictionaries
        """
        contacts = []
        
        players = step_data[[
            'nfl_player_id', 'x_position', 'y_position', 
            'decel', 'jersey_number', 'team'
        ]].dropna()
        
        # Check all opposing player pairs
        for i, p1 in players.iterrows():
            for j, p2 in players.iterrows():
                if i >= j or p1['team'] == p2['team']:
                    continue
                
                # Calculate distance
                dist = np.sqrt(
                    (p1['x_position'] - p2['x_position'])**2 + 
                    (p1['y_position'] - p2['y_position'])**2
                )
                
                # Check contact conditions
                if dist <= self.distance_threshold:
                    if p1['decel'] > self.decel_threshold or p2['decel'] > self.decel_threshold:
                        contacts.append({
                            'player1': p1['jersey_number'],
                            'player2': p2['jersey_number'],
                            'distance': dist,
                            'max_decel': max(p1['decel'], p2['decel']),
                            'x': (p1['x_position'] + p2['x_position']) / 2,
                            'y': (p1['y_position'] + p2['y_position']) / 2
                        })
        
        return contacts
    
    def detect_all_contacts(self, play_data: pd.DataFrame) -> Dict[int, List[Dict]]:
        """
        Detect contacts for all steps in play.
        
        Args:
            play_data: Tracking data for entire play
        
        Returns:
            Dictionary mapping step numbers to contact lists
        """
        all_contacts = {}
        
        for step in sorted(play_data['step'].unique()):
            step_data = play_data[play_data['step'] == step]
            contacts = self.detect_contacts_at_step(step_data)
            
            if contacts:
                all_contacts[step] = contacts
        
        return all_contacts
    
    def create_overlay_video(self,
                            video_path: str,
                            tracking_data: pd.DataFrame,
                            contacts: Dict[int, List[Dict]],
                            output_path: str) -> Tuple[int, int]:
        """
        Create annotated video with contact overlays.
        
        Args:
            video_path: Path to input video file
            tracking_data: Processed tracking data
            contacts: Dictionary of detected contacts by step
            output_path: Path for output video
        
        Returns:
            Tuple of (total_frames_processed, frames_with_contacts)
        
        Example:
            overlay = NFLVideoOverlay()
            play_data = overlay.process_tracking_data(tracking, 58172, 3247)
            contacts = overlay.detect_all_contacts(play_data)
            frames, contact_frames = overlay.create_overlay_video(
                'input.mp4', play_data, contacts, 'output.mp4'
            )
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Map tracking steps to video frames
        steps = sorted(tracking_data['step'].unique())
        step_to_frame = self._map_steps_to_frames(steps, total_frames)
        
        # Process each frame
        frame_count = 0
        contact_frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Find corresponding tracking step
            closest_step = min(step_to_frame.keys(), 
                             key=lambda x: abs(step_to_frame[x] - frame_count))
            
            # Draw overlays
            step_data = tracking_data[tracking_data['step'] == closest_step]
            frame = self._draw_player_overlays(frame, step_data, width, height)
            
            # Draw contacts if present
            if closest_step in contacts:
                frame = self._draw_contact_overlays(
                    frame, contacts[closest_step], width, height
                )
                contact_frame_count += 1
            
            # Frame counter
            frame = self._draw_frame_counter(frame, frame_count, total_frames, width, height)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"  Processing: {frame_count}/{total_frames} frames...", end='\r')
        
        cap.release()
        out.release()
        
        print(f"\n  ✓ Processed {frame_count} frames")
        print(f"  ✓ {contact_frame_count} frames with contact alerts")
        
        return frame_count, contact_frame_count
    
    def _map_steps_to_frames(self, steps: List[int], total_frames: int) -> Dict[int, int]:
        """Map tracking time steps to video frame numbers."""
        step_to_frame = {}
        for i, step in enumerate(steps):
            frame_num = int((i / len(steps)) * total_frames)
            step_to_frame[step] = frame_num
        return step_to_frame
    
    def _draw_player_overlays(self, frame: np.ndarray, step_data: pd.DataFrame,
                             width: int, height: int) -> np.ndarray:
        """Draw player position circles and jersey numbers."""
        for _, player in step_data.iterrows():
            # Convert field coordinates to pixel coordinates
            px = int((player['x_position'] / 120) * width)
            py = int((player['y_position'] / 53.3) * height)
            
            # Team color (red=home, blue=away)
            color = (0, 0, 255) if player['team'] == 'home' else (255, 0, 0)
            
            # Draw player circle
            cv2.circle(frame, (px, py), 8, color, -1)
            cv2.circle(frame, (px, py), 8, (255, 255, 255), 2)
            
            # Jersey number
            cv2.putText(frame, str(int(player['jersey_number'])), 
                       (px-10, py-15), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 2)
        
        return frame
    
    def _draw_contact_overlays(self, frame: np.ndarray, contacts: List[Dict],
                               width: int, height: int) -> np.ndarray:
        """Draw contact markers and alert banner."""
        for contact in contacts:
            # Contact location
            cx = int((contact['x'] / 120) * width)
            cy = int((contact['y'] / 53.3) * height)
            
            # Yellow star marker
            cv2.drawMarker(frame, (cx, cy), (0, 255, 255), 
                          cv2.MARKER_STAR, 30, 3)
            
            # "CONTACT!" label
            cv2.putText(frame, "CONTACT!", (cx-40, cy-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Alert banner
        cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 255), -1)
        cv2.putText(frame, f"CONTACTS DETECTED: {len(contacts)}", 
                   (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _draw_frame_counter(self, frame: np.ndarray, frame_count: int, 
                           total_frames: int, width: int, height: int) -> np.ndarray:
        """Draw frame counter in bottom-right corner."""
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                   (width-200, height-20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        return frame
    
    def create_thumbnail(self, video_path: str, output_path: str, 
                        frame_number: Optional[int] = None) -> bool:
        """
        Extract thumbnail image from video.
        
        Args:
            video_path: Path to video file
            output_path: Path for output image
            frame_number: Specific frame to extract (None = middle frame)
        
        Returns:
            True if successful, False otherwise
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return False
        
        if frame_number is None:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_number = total_frames // 2
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            cv2.imwrite(output_path, frame)
            return True
        
        return False


def main():
    """
    Demo script showing video overlay creation.
    """
    print("="*60)
    print("NFL CONTACT DETECTION - VIDEO OVERLAY DEMO")
    print("="*60)
    
    # Configuration
    VIDEO_PATH = '/mnt/user-data/uploads/58172_003247_Sideline.mp4'
    TRACKING_PATH = '/mnt/user-data/uploads/test_player_tracking.csv'
    OUTPUT_VIDEO = 'nfl_contact_overlay_demo.mp4'
    OUTPUT_THUMB = 'nfl_video_overlay_thumbnail.jpg'
    GAME_KEY = 58172
    PLAY_ID = 3247
    
    # Initialize overlay system
    print("\n[1/4] Loading data...")
    overlay = NFLVideoOverlay(distance_threshold=2.0, decel_threshold=0.3)
    
    tracking = pd.read_csv(TRACKING_PATH)
    play_data = overlay.process_tracking_data(tracking, GAME_KEY, PLAY_ID)
    
    print(f"  ✓ Play: {GAME_KEY}_{PLAY_ID}")
    print(f"  ✓ Players: {play_data['nfl_player_id'].nunique()}")
    print(f"  ✓ Time steps: {play_data['step'].nunique()}")
    
    # Detect contacts
    print("\n[2/4] Detecting contacts...")
    contacts = overlay.detect_all_contacts(play_data)
    
    total_contacts = sum(len(c) for c in contacts.values())
    print(f"  ✓ Contacts in {len(contacts)} time steps")
    print(f"  ✓ Total contact events: {total_contacts}")
    
    # Create overlay video
    print("\n[3/4] Processing video...")
    frames, contact_frames = overlay.create_overlay_video(
        VIDEO_PATH, play_data, contacts, OUTPUT_VIDEO
    )
    print(f"  ✓ Saved: {OUTPUT_VIDEO}")
    
    # Extract thumbnail
    print("\n[4/4] Creating thumbnail...")
    success = overlay.create_thumbnail(OUTPUT_VIDEO, OUTPUT_THUMB, contact_frames // 2)
    if success:
        print(f"  ✓ Saved: {OUTPUT_THUMB}")
    
    print("\n" + "="*60)
    print("VIDEO OVERLAY COMPLETE")
    print("="*60)
    print(f"\nResults:")
    print(f"  • Video: {OUTPUT_VIDEO} ({frames} frames)")
    print(f"  • Thumbnail: {OUTPUT_THUMB}")
    print(f"  • Contact alerts: {contact_frames} frames")


if __name__ == "__main__":
    main()