import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
import math
from datetime import datetime
from tqdm import tqdm

class CricketBowlingAnalyzer:
    def __init__(self, video_path, output_dir='./output/', athlete_height=None):
        """Initialize the Cricket Bowling Action Analyzer.
       
        Args:
            video_path (str): Path to the input cricket bowling video.
            output_dir (str): Directory to save output files.
            athlete_height (float, optional): Height of the athlete in meters for calibration.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
       
        # Calibration parameters
        self.athlete_height = athlete_height
        self.pixel_to_meter_ratio = None
        self.calibration_done = False
       
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
       
        # Data storage for analysis
        self.arm_positions = []  # Will store (frame_num, shoulder, elbow, wrist)
        self.arm_speeds = []  # Will store (frame_num, wrist_speed, wrist_speed_real)
        self.arm_angles = []  # Will store (frame_num, arm_angle)
       
        # Path visualization
        self.path_points = []  # Will store all wrist points for drawing the path
        self.color_map = plt.get_cmap('jet')  # Color map for speed visualization
       
        # Create output directory if it doesn't exist
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
   
    def calibrate_from_height(self, landmarks):
        """Perform space calibration using the athlete's height.
       
        This method uses the athlete's full body height to establish a conversion ratio
        between pixels and real-world units (meters).
       
        Args:
            landmarks: MediaPipe pose landmarks
       
        Returns:
            bool: True if calibration was successful, False otherwise
        """
        if not self.athlete_height or not landmarks:
            return False
           
        # Get ankle and head landmarks
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
       
        # Use average of ankles for better stability
        ankle_y = (left_ankle.y + right_ankle.y) / 2
       
        # Calculate height in pixels
        height_pixels = (ankle_y - nose.y) * self.frame_height
       
        # Safety check - make sure it's positive and reasonable
        if height_pixels <= 0:
            return False
       
        # Calculate ratio (meters per pixel)
        self.pixel_to_meter_ratio = self.athlete_height / height_pixels
       
        print(f"Calibration complete: 1 pixel = {self.pixel_to_meter_ratio:.5f} meters")
        print(f"Person height: {height_pixels:.1f} pixels = {self.athlete_height:.2f} meters")
        self.calibration_done = True
       
        return True
   
    def _calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
   
    def _calculate_speed(self, p1, p2, time_diff):
        """Calculate speed between two points in pixels per second and meters per second."""
        distance_px = self._calculate_distance(p1, p2)
        speed_px = distance_px / time_diff if time_diff > 0 else 0
       
        # Convert to real-world units if calibrated
        if self.calibration_done:
            distance_m = distance_px * self.pixel_to_meter_ratio
            speed_m = distance_m / time_diff if time_diff > 0 else 0
            return speed_px, speed_m
        else:
            return speed_px, None
   
    def _calculate_angle(self, shoulder, elbow, wrist):
        """Calculate the angle at the elbow joint."""
        vector1 = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
        vector2 = (elbow[0] - wrist[0], elbow[1] - wrist[1])
       
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
        magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
       
        if magnitude1 * magnitude2 == 0:
            return 0
           
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = max(-1, min(1, cos_angle))  # Ensure value is in [-1, 1]
        angle = math.degrees(math.acos(cos_angle))
        return angle
   
    def analyze_video(self):
        """Process the video frame by frame to detect and analyze bowling action."""
        prev_wrist = None
        frame_num = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
       
        # Output video setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = f"{self.output_dir}analyzed_bowling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
       
        print(f"Analyzing bowling action from video: {self.video_path}")
        print(f"Total frames to process: {total_frames}")
       
        # Calibration status tracker
        calibration_attempts = 0
        max_calibration_attempts = 30  # Try calibration on first 30 frames with landmarks
       
        for _ in tqdm(range(total_frames)):
            ret, frame = self.cap.read()
            if not ret:
                break
           
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           
            # Process the frame with MediaPipe Pose
            results = self.pose.process(rgb_frame)
           
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
               
                # Try to calibrate if not already done
                if not self.calibration_done and self.athlete_height and calibration_attempts < max_calibration_attempts:
                    if self.calibrate_from_height(landmarks):
                        # Display calibration info on frame
                        cv2.putText(frame, f"Calibration: 1 pixel = {self.pixel_to_meter_ratio:.5f} m",
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    calibration_attempts += 1
               
                # Get right side landmarks for a right-arm bowler
                # (adjust for left-arm bowlers if needed)
                shoulder = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * self.frame_width),
                            int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * self.frame_height))
               
                elbow = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x * self.frame_width),
                         int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y * self.frame_height))
               
                wrist = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x * self.frame_width),
                         int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y * self.frame_height))
               
                # Store the arm positions
                self.arm_positions.append((frame_num, shoulder, elbow, wrist))
               
                # Calculate arm angle
                angle = self._calculate_angle(shoulder, elbow, wrist)
                self.arm_angles.append((frame_num, angle))
               
                # Calculate speed if we have a previous point
                if prev_wrist:
                    time_diff = 1.0 / self.fps
                    speed_px, speed_m = self._calculate_speed(prev_wrist, wrist, time_diff)
                   
                    if speed_m is not None:
                        self.arm_speeds.append((frame_num, speed_px, speed_m))
                    else:
                        self.arm_speeds.append((frame_num, speed_px, 0))
                   
                    # Determine color based on speed
                    max_speed = 1000  # Adjust based on your data
                    color_val = min(speed_px / max_speed, 1.0)
                    color = tuple([int(255 * c) for c in self.color_map(color_val)[:3]])
                   
                    # Store path point with color information (for BGR format in OpenCV)
                    self.path_points.append((wrist, (color[2], color[1], color[0])))
               
                prev_wrist = wrist
               
                # Draw skeleton on the frame
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
               
                # Draw arm specific lines with thicker lines
                cv2.line(frame, shoulder, elbow, (0, 0, 255), 3)
                cv2.line(frame, elbow, wrist, (0, 0, 255), 3)
                cv2.circle(frame, shoulder, 5, (255, 0, 0), -1)
                cv2.circle(frame, elbow, 5, (255, 0, 0), -1)
                cv2.circle(frame, wrist, 5, (255, 0, 0), -1)
               
                # Display speed near the wrist
                if len(self.arm_speeds) > 0:
                    if self.calibration_done:
                        speed_text = f"Speed: {self.arm_speeds[-1][2]:.1f} m/s"
                    else:
                        speed_text = f"Speed: {self.arm_speeds[-1][1]:.1f} px/s"
                    cv2.putText(frame, speed_text, (wrist[0] + 10, wrist[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
               
                # Display angle near the elbow
                angle_text = f"Angle: {angle:.1f}°"
                cv2.putText(frame, angle_text, (elbow[0] + 10, elbow[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
           
            # Draw the motion path (with color gradient based on speed)
            for i in range(1, len(self.path_points)):
                cv2.line(frame, self.path_points[i-1][0], self.path_points[i][0],
                         self.path_points[i][1], 2)
           
            # Add frame number
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                       
            # Display calibration status
            if self.calibration_done:
                cv2.putText(frame, f"Calibrated: 1 px = {self.pixel_to_meter_ratio:.5f} m",
                           (10, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Not calibrated - using pixel measurements",
                           (10, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
           
            # Write the frame to output video
            out.write(frame)
           
            frame_num += 1
       
        # Release resources
        self.cap.release()
        out.release()
        print(f"Analysis complete. Output video saved to: {output_path}")
       
        # Save data to CSV
        self._save_data_to_csv()
       
        # Generate and save graphs
        self._generate_graphs()
       
        return output_path
   
    def _save_data_to_csv(self):
        """Save the collected data to CSV files for further analysis."""
        # Save arm positions
        positions_df = pd.DataFrame(
            [(f, *s, *e, *w) for f, s, e, w in self.arm_positions],
            columns=['frame', 'shoulder_x', 'shoulder_y', 'elbow_x', 'elbow_y', 'wrist_x', 'wrist_y']
        )
        positions_df.to_csv(f"{self.output_dir}arm_positions.csv", index=False)
       
        # Save arm speeds
        if self.arm_speeds:
            if self.calibration_done:
                speeds_df = pd.DataFrame(self.arm_speeds, columns=['frame', 'wrist_speed_px', 'wrist_speed_m'])
            else:
                speeds_df = pd.DataFrame([(f, s, 0) for f, s, _ in self.arm_speeds],
                                         columns=['frame', 'wrist_speed_px', 'wrist_speed_m'])
            speeds_df.to_csv(f"{self.output_dir}arm_speeds.csv", index=False)
       
        # Save arm angles
        if self.arm_angles:
            angles_df = pd.DataFrame(self.arm_angles, columns=['frame', 'elbow_angle'])
            angles_df.to_csv(f"{self.output_dir}arm_angles.csv", index=False)
           
        # Save calibration data
        calibration_df = pd.DataFrame([{
            'calibration_done': self.calibration_done,
            'pixel_to_meter_ratio': self.pixel_to_meter_ratio if self.calibration_done else 0,
            'athlete_height_m': self.athlete_height if self.athlete_height else 0
        }])
        calibration_df.to_csv(f"{self.output_dir}calibration_data.csv", index=False)
           
        print(f"Data saved to CSV files in {self.output_dir}")
   
    def _generate_graphs(self):
        """Generate and save analysis graphs."""
        # 1. Speed vs Time graph
        if self.arm_speeds:
            plt.figure(figsize=(10, 6))
            frames, speeds_px, speeds_m = zip(*self.arm_speeds)
            times = [f / self.fps for f in frames]  # Convert frames to seconds
           
            if self.calibration_done:
                plt.plot(times, speeds_m, 'b-', linewidth=2)
                plt.ylabel('Wrist Speed (meters/second)')
                # Mark the point of maximum speed
                max_speed_idx = speeds_m.index(max(speeds_m))
                plt.scatter(times[max_speed_idx], speeds_m[max_speed_idx],
                            color='red', s=100, zorder=5)
                plt.annotate(f"Max: {speeds_m[max_speed_idx]:.1f} m/s",
                             (times[max_speed_idx], speeds_m[max_speed_idx]),
                             xytext=(10, 10), textcoords='offset points')
            else:
                plt.plot(times, speeds_px, 'b-', linewidth=2)
                plt.ylabel('Wrist Speed (pixels/second)')
                # Mark the point of maximum speed
                max_speed_idx = speeds_px.index(max(speeds_px))
                plt.scatter(times[max_speed_idx], speeds_px[max_speed_idx],
                            color='red', s=100, zorder=5)
                plt.annotate(f"Max: {speeds_px[max_speed_idx]:.1f} px/s",
                             (times[max_speed_idx], speeds_px[max_speed_idx]),
                             xytext=(10, 10), textcoords='offset points')
           
            plt.xlabel('Time (seconds)')
            plt.title('Bowling Arm Speed vs Time')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}speed_vs_time.png", dpi=300)
            plt.close()
       
        # 2. Angle vs Time graph
        if self.arm_angles:
            plt.figure(figsize=(10, 6))
            frames, angles = zip(*self.arm_angles)
            times = [f / self.fps for f in frames]  # Convert frames to seconds
           
            plt.plot(times, angles, 'g-', linewidth=2)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Elbow Angle (degrees)')
            plt.title('Bowling Arm Angle vs Time')
            plt.grid(True)
           
            # Mark important points (min and max angles)
            min_angle_idx = angles.index(min(angles))
            max_angle_idx = angles.index(max(angles))
           
            plt.scatter(times[min_angle_idx], angles[min_angle_idx],
                        color='red', s=100, zorder=5)
            plt.annotate(f"Min: {angles[min_angle_idx]:.1f}°",
                         (times[min_angle_idx], angles[min_angle_idx]),
                         xytext=(10, 10), textcoords='offset points')
           
            plt.scatter(times[max_angle_idx], angles[max_angle_idx],
                        color='blue', s=100, zorder=5)
            plt.annotate(f"Max: {angles[max_angle_idx]:.1f}°",
                         (times[max_angle_idx], angles[max_angle_idx]),
                         xytext=(10, 10), textcoords='offset points')
           
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}angle_vs_time.png", dpi=300)
            plt.close()
       
        # 3. Motion trajectory visualization (2D plot)
        if self.arm_positions:
            plt.figure(figsize=(10, 10))
           
            # Extract wrist positions
            _, _, _, wrist_positions = zip(*self.arm_positions)
            wrist_x, wrist_y = zip(*wrist_positions)
           
            # Create scatter plot with color gradient based on time
            scatter = plt.scatter(wrist_x, wrist_y, c=range(len(wrist_x)), cmap='viridis',
                        s=30, alpha=0.7)
            plt.colorbar(scatter, label='Frame Number')
           
            # Connect points with lines
            plt.plot(wrist_x, wrist_y, 'k-', alpha=0.3, linewidth=1)
           
            # Mark start and end points
            plt.scatter(wrist_x[0], wrist_y[0], color='green', s=100, label='Start')
            plt.scatter(wrist_x[-1], wrist_y[-1], color='red', s=100, label='End')
           
            # Add scale if calibrated
            if self.calibration_done:
                # Add a 1-meter scale bar
                x_range = max(wrist_x) - min(wrist_x)
                y_range = max(wrist_y) - min(wrist_y)
                bar_length_px = 1.0 / self.pixel_to_meter_ratio  # 1 meter in pixels
               
                # Position the scale bar in the bottom right
                scale_x_start = min(wrist_x) + 0.7 * x_range
                scale_y = min(wrist_y) + 0.9 * y_range
               
                plt.plot([scale_x_start, scale_x_start + bar_length_px],
                         [scale_y, scale_y], 'k-', linewidth=3)
                plt.text(scale_x_start + bar_length_px/2, scale_y + 20,
                         "1 meter", ha='center')
           
            # Y-axis inverted to match image coordinates (top-left origin)
            plt.gca().invert_yaxis()
            plt.xlabel('X Position (pixels)')
            plt.ylabel('Y Position (pixels)')
            plt.title('Bowling Arm Trajectory (Wrist Position)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
           
            plt.savefig(f"{self.output_dir}trajectory_2d.png", dpi=300)
            plt.close()
           
        print(f"Analysis graphs saved to {self.output_dir}")


class CalibrationTool:
    """Interactive tool to calibrate the video based on athlete's height."""
   
    def __init__(self, video_path):
        """Initialize the calibration tool.
       
        Args:
            video_path (str): Path to the input video.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
       
        # Initialize MediaPipe Pose for automatic height detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
       
        # Calibration results
        self.athlete_height = None  # in meters
        self.pixel_to_meter_ratio = None
   
    def manual_calibration(self, height_meters):
        """Perform manual calibration by providing the athlete's height.
       
        Args:
            height_meters (float): Athlete's height in meters.
           
        Returns:
            float: The athlete's height in meters.
        """
        self.athlete_height = height_meters
        print(f"Manual calibration set: Athlete height = {height_meters} meters")
        return height_meters
   
    def run_calibration_wizard(self):
        """Run an interactive calibration wizard to help set up accurate measurements.
       
        This will:
        1. Ask for the athlete's height
        2. Show a frame from the video with pose detection
        3. Verify the calibration visually
       
        Returns:
            float: The athlete's height in meters.
        """
        print("\n=== Cricket Bowling Analysis Calibration Wizard ===")
       
        # Get user input for athlete's height
        while True:
            try:
                height_input = input("Enter the athlete's height (in meters, e.g. 1.85): ")
                self.athlete_height = float(height_input)
                if 1.0 <= self.athlete_height <= 2.5:  # Reasonable height range
                    break
                else:
                    print("Please enter a reasonable height (between 1.0 and 2.5 meters).")
            except ValueError:
                print("Invalid input. Please enter a number.")
       
        print(f"Athlete's height set to {self.athlete_height} meters.")
        print("\nLooking for a good frame to verify calibration...")
       
        # Find a good frame with full body visible
        found_good_frame = False
       
        # Skip to about 25% into the video to find the bowler
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.25))
       
        for _ in range(min(100, total_frames)):  # Try up to 100 frames
            ret, frame = self.cap.read()
            if not ret:
                break
               
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
           
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
               
                # Check if we can see ankles and head
                if (landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].visibility > 0.5 and
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].visibility > 0.5 and
                    landmarks[self.mp_pose.PoseLandmark.NOSE].visibility > 0.5):
                   
                    # Draw pose landmarks
                    annotated_frame = frame.copy()
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
                   
                    # Calculate height in pixels
                    left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                    right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
                    nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                   
                    # Use average of ankles for better stability
                    ankle_y = (left_ankle.y + right_ankle.y) / 2
                   
                    # Calculate height in pixels
                    height_pixels = (ankle_y - nose.y) * self.frame_height
                   
                    # Calculate ratio (meters per pixel)
                    if height_pixels > 0:
                        self.pixel_to_meter_ratio = self.athlete_height / height_pixels
                       
                        # Draw height line
                        nose_point = (int(nose.x * self.frame_width), int(nose.y * self.frame_height))
                        ankle_point = (int(nose.x * self.frame_width), int(ankle_y * self.frame_height))
                       
                        cv2.line(annotated_frame, nose_point, ankle_point, (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"{height_pixels:.1f} px = {self.athlete_height:.2f} m",
                                    (nose_point[0] + 10, nose_point[1] + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                       
                        # Add calibration info
                        cv2.putText(annotated_frame, f"1 pixel = {self.pixel_to_meter_ratio:.5f} meters",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                       
                        # Display calibration result
                        cv2.namedWindow("Calibration Verification", cv2.WINDOW_NORMAL)
                        cv2.imshow("Calibration Verification", annotated_frame)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                       
                        found_good_frame = True
                        break
       
        self.cap.release()
       
        if not found_good_frame:
            print("Could not find a good frame for calibration verification.")
            print("Will use the provided height for calibration: {self.athlete_height} meters")
        else:
            print(f"Calibration complete: 1 pixel = {self.pixel_to_meter_ratio:.5f} meters")
           
        return self.athlete_height


def main(video_path, output_dir='./output/', athlete_height=None):
    """Main function to run the cricket bowling analyzer with calibration.
   
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save output files.
        athlete_height (float, optional): Athlete's height in meters for calibration.
    """
    # Run calibration if needed
    if athlete_height is None:
        calibration_tool = CalibrationTool(video_path)
        athlete_height = calibration_tool.run_calibration_wizard()
   
    # Run the analyzer with calibration
    analyzer = CricketBowlingAnalyzer(video_path, output_dir, athlete_height)
    output_video_path = analyzer.analyze_video()
   
    print("\nAnalysis Summary:")
    print(f"- Input video: {video_path}")
    print(f"- Athlete height used for calibration: {athlete_height} meters")
    print(f"- Output video: {output_video_path}")
    print(f"- Data files and graphs saved to: {output_dir}")
    print("\nThe analysis includes:")
    print("1. Tracked arm motion path overlaid on the video")
    print("2. Real-world speed measurements (m/s) using height calibration")
    print("3. Elbow angle measurements")
    print("4. Visualization graphs for speed and angle profiles")
    print("5. CSV data files for further analysis")


if __name__ == "__main__":
    import argparse
   
    parser = argparse.ArgumentParser(description='Cricket Bowling Action Analyzer')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output', type=str, default='./output/', help='Directory to save output files')
    parser.add_argument('--height', type=float, help='Athlete\'s height in meters for calibration')
   
    args = parser.parse_args()
    main(args.video, args.output, args.height)