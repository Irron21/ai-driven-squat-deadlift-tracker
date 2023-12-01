import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import os
import threading

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main():
    deadlift_type = input("Which type of deadlift (Sumo/Conventional)? ")
    while deadlift_type.lower() not in ("sumo", "conventional"):
       deadlift_type = input("Which type of deadlift (Sumo/Conventional)? ")
    
    """ Video Notes (Crucial to work)
    1. Camera should be placed where hips should not be covered by the barbell at the lockout position
        - Side view then 1 step sideward to the front and you're good to go
    2. Camera must see the whole person
    3. Not too far and not too close"""
    
    video = input("Enter directory path to video: ")
    while not os.path.exists(video):
        video = input("Enter directory path to video: ")
    
    capture = cv.VideoCapture(video)
    # capture = cv.VideoCapture(0) 

    width = int(capture.get(3))  # Replace with the width of your video frames
    height = int(capture.get(4))  # Replace with the height of your video frames

    stage = None
    counter = 0
    rep_speeds = [] 
    current_rep_time = 0  
    last_rep_time = 0 
    waitKey_delay = 1

    thread1 = threading.Thread(target=process_image, args=(capture, stage, counter, rep_speeds, current_rep_time, last_rep_time, waitKey_delay, deadlift_type, width,height))
    thread1.start()

def process_image(capture, stage, counter, rep_speeds, current_rep_time, last_rep_time, waitKey_delay, deadlift_type, width,height):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while capture.isOpened():
            ret, frame = capture.read()

            if not ret:
                break  

            # frame = cv.resize(frame, (width//2, height//2))

            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True        
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]

                # if left_shoulder and left_hip and left_knee:  # Check if all landmarks are detected
                #     # Draw left shoulder
                #     cv.circle(image, (int(left_shoulder.x * width), int(left_shoulder.y * height)), 5, (255, 0, 0), -1)
                #     # Draw left hip
                #     cv.circle(image, (int(left_hip.x * width), int(left_hip.y * height)), 5, (0, 255, 0), -1)
                #     # Draw left knee
                #     cv.circle(image, (int(left_knee.x * width), int(left_knee.y * height)), 5, (0, 0, 255), -1)

                angle = calculate_angle(
                    [left_shoulder.x, left_shoulder.y],
                    [left_hip.x, left_hip.y],
                    [left_knee.x, left_knee.y]
                )

                cv.putText(image, str(round(angle)), 
                           tuple(np.multiply([left_hip.x, left_hip.y], [int(width), int(height)]).astype(int)), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)

                def process_deadlift(bottom_degree):
                    nonlocal stage, counter, rep_speeds, current_rep_time, last_rep_time
                    lockout_angle = 170
                    if angle < bottom_degree and stage != "bottom":
                        stage = "bottom"                       
                    if bottom_degree < angle < lockout_angle and stage == "bottom":
                        stage = "ongoing"
                        current_rep_time = time.time()
                    if angle > lockout_angle and stage == "ongoing":
                        stage = "lockout"
                        counter += 1
                        last_rep_time = time.time() 
                        rep_time = (last_rep_time - current_rep_time)
                        rep_speeds.append(rep_time)       

                if deadlift_type == "conventional":
                    process_deadlift(40)

                elif deadlift_type == "sumo":
                    process_deadlift(130)

            except:
                pass

            cv.rectangle(image, (0, 0), (int(width), 60), (0,0,0), -1)
            cv.putText(image, "REPS", (5, 25), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(image, str(counter), (5, 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

            cv.putText(image, 'STAGE', (60, 25), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(image, stage, (60, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)
            
            cv.putText(image, 'AVG SPEED', (170, 25), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)
            if len(rep_speeds) > 0:
                average_speed = sum(rep_speeds) / len(rep_speeds)            
                cv.putText(image, f'{average_speed:.2f} sec/rep', (170, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)
            


            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
            cv.imshow("Live Feed", image)
            if cv.waitKey(waitKey_delay) & 0xFF == ord('x'):
                break

    capture.release()
    cv.destroyAllWindows()

def calculate_angle(start, mid, end):
    start = np.array(start)
    mid = np.array(mid)
    end = np.array(end)

    radians = np.arctan2(end[1] - mid[1], end[0] - mid[0]) - np.arctan2(start[1] - mid[1], start[0] - mid[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

if __name__ == '__main__':
    main()
