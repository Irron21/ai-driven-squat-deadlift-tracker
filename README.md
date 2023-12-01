# AI-Driven Deadlift Pose Counter

This AI-driven Deadlift Pose Counter is designed to analyze deadlift exercises using Mediapipe's pose estimation. The program tracks the user's body movements, specifically focusing on key body landmarks to count repetitions accurately and calculate the average speed per repetition.

## Overview

The project employs computer vision techniques and pose estimation algorithms to monitor and analyze a user's pose while performing deadlift exercises. The system identifies crucial body landmarks, such as shoulders, hips, and knees, to infer specific actions during the deadlift routine.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- Mediapipe (`mediapipe`)
- NumPy (`numpy`)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/irron21/deadlift-pose-counter.git
    cd deadlift-pose-counter
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the script:

    ```bash
    python deadlift_pose_counter.py
    ```

2. Select the deadlift type: `Sumo` or `Conventional`.
3. Provide the directory path to the video capturing the deadlift exercise.

## Functionalities

### Pose Estimation

- Utilizes Mediapipe's pose estimation to identify and track specific body landmarks during the deadlift.
- Landmarks include shoulders, hips, and knees, crucial for deadlift pose recognition.

### Repetition Counting

- Counts the number of repetitions completed during the deadlift routine.
- Identifies specific angles between body parts to determine the start, ongoing, and lockout stages of each repetition.

### Performance Analysis

- Calculates and displays the average speed per repetition in seconds based on the time taken between stages of the deadlift.

### Real-time Visualization

- Provides a live feed displaying the user's pose estimation, identified landmarks, and current stage of the deadlift.
- Renders visual cues like circles and lines to denote the detected body parts and their movements.

### Video Requirements

- The camera should be positioned to capture the entire body while ensuring the hips are visible at the lockout position.
- Adequate lighting and camera angle are essential for accurate pose estimation.

## Contributors

- [Irron21](https://github.com/irron21)

## License

This project is licensed under the [MIT License](LICENSE).
