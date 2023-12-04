import threading
from deadlift_tracker import DeadliftTracker
from squat_tracker import SquatTracker

def start_squat_tracking():
    tracker = SquatTracker()
    tracker.start_tracking()

def start_deadlift_tracking():
    tracker = DeadliftTracker()
    tracker.start_tracking()

if __name__ == '__main__':
    while True:
        print("""Choose which exercise to analyze:x
1 - Squat
2 - Bench Press
3 - Deadlift""")
        choice = input("Enter choice: ")
        if choice not in ("1","2","3"):
            continue
        elif choice == "1":
            tracking_thread = threading.Thread(target=start_squat_tracking)
            tracking_thread.start()
            tracking_thread.join()
        elif choice == "2":
            pass
        elif choice == "3":
            tracking_thread = threading.Thread(target=start_deadlift_tracking)
            tracking_thread.start()
            tracking_thread.join()