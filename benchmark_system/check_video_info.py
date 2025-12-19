import cv2
import os

video_path = "football_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open {video_path}")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Total Frames: {frame_count}")
    print(f"Duration: {frame_count/fps:.2f} seconds")

cap.release()
