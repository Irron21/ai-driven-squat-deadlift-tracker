# Squat & Deadlift Tracker

This repository consists of Python scripts designed for exercise movement tracking using computer vision techniques. The scripts facilitate the tracking of Squat and Deadlift exercises, allowing users to select an exercise for analysis.

## Files

### `main.py`

- Controls the exercise selection and initiates the tracking process.
- Enables users to choose between different exercises for tracking.

### `squat_tracker.py`

- Contains the `SquatTracker` class to monitor and track squat movements.
- Utilizes Mediapipe and OpenCV to estimate poses and visualize tracking information.
- Captures user-provided video input for squat exercise analysis.
- Calculates angles and tracks squat stages and repetitions.
#### Video Notes
- Ensure the subject is seen throughout the video (not too far ot too close)
- Rep is started once knee to hip is parallel or below parallel to the ground and counted once the knees are fully lockout

### `deadlift_tracker.py`

- Implements the `DeadliftTracker` class for monitoring and tracking deadlift exercise variations (Sumo and Conventional).
- Utilizes Mediapipe and OpenCV to estimate poses and visualize tracking information.
- Captures user-provided video input for deadlift exercise analysis.
- Calculates angles and tracks deadlift stages and repetitions based on the chosen deadlift type (Sumo or Conventional).
#### Video Notes
- Ensure the subject is seen throughout the video (not too far ot too close)
- Ensure the hips are seen on the top of the movement (lockout) for accurate measurements and not covered by the barbell plates
- Rep is counted once lockout is achieved (Back is fully extended)
(Sample)[https://cdn.discordapp.com/attachments/969505484224741397/1180167590996344873/image.png?ex=657c6fb8&is=6569fab8&hm=811e9602e7dd5c6fd3326a60c9246bf9edd3a7cdf7fc7a3599c3ae5efd807318&]
  
## Usage

1. Ensure Python 3.x is installed.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run `main.py`.
4. Choose the exercise to track:
   - `1` for Squat
   - `2` for Bench Press (Placeholder)
   - `3` for Deadlift
5. Follow the prompts to input the video file path and type of deadlift (Sumo/Conventional).

## Additional Notes

- The `main.py` script initiates exercise tracking threads based on user input.
- `squat_tracker.py` tracks squat exercises, providing real-time feedback on stages, repetitions, and average speed.
- `deadlift_tracker.py` monitors deadlift exercises, distinguishing between Sumo and Conventional types for accurate tracking.

## Contribution

Feel free to contribute to this project by improving tracking accuracy, extending functionality to support additional exercises, or enhancing the user interface.

---

**Disclaimer:** This project is for demonstration purposes and may require further enhancements for robustness in real-world scenarios.
