# import cv2
# import numpy as np
# import subprocess
# import time
# from datetime import datetime
# from multiprocessing import Process, Queue

# def add_timestamp(frame, font_scale=1.0, position=(10, 30)):
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     cv2.putText(frame, timestamp, position, cv2.FONT_HERSHEY_SIMPLEX, 
#                 font_scale, (0, 255, 255), 2)
#     return frame

# def process_video_with_ffmpeg(
#     video_path, 
#     queue_for_display, 
#     camera_id, 
#     display_scale=0.125,  
#     display_fps=5,        
#     enforce_real_time=True
# ):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Cannot open video source for camera {camera_id}.")
#         return

#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     source_fps = cap.get(cv2.CAP_PROP_FPS)
#     if source_fps <= 0:
#         source_fps = 25.0

#     ffmpeg_command = [
#         "ffmpeg",
#         "-y",
#         "-f", "rawvideo",
#         "-vcodec", "rawvideo",
#         "-pix_fmt", "bgr24",
#         "-s", f"{frame_width}x{frame_height}",
#         "-i", "-",
#         "-vsync", "vfr",
#         "-c:v", "h264_videotoolbox",
#         "-preset", "veryfast",
#         "-pix_fmt", "yuv420p",
#         f"output_camera_{camera_id}.mp4"
#     ]
#     ffmpeg_process = subprocess.Popen(ffmpeg_command, 
#                                       stdin=subprocess.PIPE, 
#                                       stderr=subprocess.PIPE)

#     mosaic_interval = 1.0 / display_fps
#     last_mosaic_push_time = 0

#     frame_interval = 1.0 / source_fps
#     last_frame_time = time.time()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             queue_for_display.put(None)
#             break

#         # Add the *video's timeline* timestamp (not system time)
#         # We'll get the position in the video in ms:
#         pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
#         # Convert to H:M:S
#         seconds = pos_msec / 1000.0
#         hh = int(seconds // 3600)
#         mm = int((seconds % 3600) // 60)
#         ss = int(seconds % 60)
#         text = f"{hh:02d}:{mm:02d}:{ss:02d}"
#         cv2.putText(frame, text, (30, 80), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
#         full_res_frame = frame

#         # Write to FFmpeg
#         try:
#             ffmpeg_process.stdin.write(full_res_frame.tobytes())
#         except BrokenPipeError:
#             print("FFmpeg pipe closed unexpectedly.")
#             break

#         # Scale for mosaic
#         now = time.time()
#         if (now - last_mosaic_push_time) >= mosaic_interval:
#             last_mosaic_push_time = now

#             display_w = int(frame_width * display_scale)
#             display_h = int(frame_height * display_scale)
#             display_frame = cv2.resize(frame, (display_w, display_h))

#             # You could also overlay the same text on display_frame if desired
#             # cv2.putText(display_frame, text, (10, 20), ...)

#             queue_for_display.put(display_frame)

#         # Optionally enforce real-time pacing
#         if enforce_real_time:
#             elapsed = time.time() - last_frame_time
#             sleep_time = frame_interval - elapsed
#             if sleep_time > 0:
#                 time.sleep(sleep_time)
#             last_frame_time = time.time()

#     cap.release()
#     if ffmpeg_process.stdin:
#         ffmpeg_process.stdin.close()
#     if ffmpeg_process.stderr:
#         ffmpeg_process.stderr.close()
#     ffmpeg_process.wait()



# def consumer_mosaic(queues, consumer_fps=5):
#     """
#     Consumer (main process):
#       - Periodically reads from each cameras display queue.
#       - Builds a horizontal mosaic of scaled frames.
#       - Displays at consumer_fps (like 5 FPS).
#     """
#     import cv2
#     interval = 1.0 / consumer_fps

#     # We'll detect each camera's display frame size from the first frames we see.
#     frame_sizes = [None] * len(queues)

#     while True:
#         # Grab one frame from each queue
#         frames = []
#         for i, q in enumerate(queues):
#             try:
#                 frame = q.get(timeout=2.0)
#             except:
#                 frame = None
#             frames.append(frame)

#         # If all frames are None => all producers ended
#         if all(f is None for f in frames):
#             print("All camera streams ended.")
#             break

#         mosaic_parts = []
#         for i, f in enumerate(frames):
#             if f is not None:
#                 if frame_sizes[i] is None:
#                     # Store the known width/height
#                     h, w, _ = f.shape
#                     frame_sizes[i] = (w, h)
#                 mosaic_parts.append(f)
#             else:
#                 # This camera doesn't have a new frame this cycle
#                 # or has ended. If we know the size, put a blank.
#                 if frame_sizes[i] is not None:
#                     w, h = frame_sizes[i]
#                     mosaic_parts.append(np.zeros((h, w, 3), dtype=np.uint8))
#                 # else we just skip if we never had a frame yet.

#         if mosaic_parts:
#             mosaic = np.hstack(mosaic_parts)
#             mosaic = add_timestamp(mosaic, font_scale=0.8, position=(10, 40))

#             cv2.imshow('Mosaic View', mosaic)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         time.sleep(interval)

#     cv2.destroyAllWindows()


# def simulate_cameras(video_paths):
#     """
#     Main entry:
#       1) Create a queue per camera.
#       2) Start a producer (process_video_with_ffmpeg) for each camera.
#       3) Run the consumer mosaic in the main process.
#     """
#     from multiprocessing import Process, Queue

#     num_cameras = len(video_paths)
#     queues = [Queue(maxsize=20) for _ in range(num_cameras)]
#     producers = []

#     for i, video_path in enumerate(video_paths):
#         p = Process(
#             target=process_video_with_ffmpeg,
#             kwargs=dict(
#                 video_path=video_path,
#                 queue_for_display=queues[i],
#                 camera_id=i,
#                 display_scale=0.125,  # 1/8 resolution for mosaic
#                 display_fps=20,
#                 enforce_real_time=True
#             )
#         )
#         producers.append(p)
#         p.start()

#     consumer_mosaic(queues, consumer_fps=20)

#     # Wait for all producers to end
#     for p in producers:
#         p.join()


# if __name__ == "__main__":
#     simulate_cameras([
#         'Ipynb_Experiment/Test_Video_4096_2160_25fps.mp4',
#         'Ipynb_Experiment/Test_Video_4096_2160_25fps.mp4',
#         'Ipynb_Experiment/Test_Video_4096_2160_25fps.mp4',
#         'Ipynb_Experiment/Test_Video_4096_2160_25fps.mp4'
#     ])


import cv2
import numpy as np
import subprocess
import time
from datetime import datetime
from multiprocessing import Process, Queue

def add_timestamp(frame, font_scale=1.0, position=(10, 30)):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, timestamp, position, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (0, 255, 255), 2)
    return frame

def process_video_with_ffmpeg(
    video_path, 
    queue_for_display, 
    camera_id, 
    display_scale=0.125,  
    display_fps=5,        
    enforce_real_time=True
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video source for camera {camera_id}.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 25.0  # Default assumption if FPS not available

    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{frame_width}x{frame_height}",
        "-i", "-",
        "-vsync", "vfr",
        "-c:v", "h264_videotoolbox",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        f"output_camera_{camera_id}.mp4"
    ]
    ffmpeg_process = subprocess.Popen(ffmpeg_command, 
                                      stdin=subprocess.PIPE, 
                                      stderr=subprocess.PIPE)

    mosaic_interval = 1.0 / display_fps
    last_mosaic_push_time = 0

    frame_interval = 1.0 / source_fps
    last_frame_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            queue_for_display.put(None)
            break

        # Calculate timestamp based on frame count and FPS
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        pos_msec = (current_frame / source_fps) * 1000
        seconds = pos_msec / 1000.0
        hh = int(seconds // 3600)
        mm = int((seconds % 3600) // 60)
        ss = int(seconds % 60)
        text = f"{hh:02d}:{mm:02d}:{ss:02d}"

        # Add timestamp to the original frame (for output video)
        cv2.putText(frame, text, (30, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        add_timestamp(frame, position=(100, 30))
        full_res_frame = frame

        # Write to FFmpeg
        try:
            ffmpeg_process.stdin.write(full_res_frame.tobytes())
        except BrokenPipeError:
            print("FFmpeg pipe closed unexpectedly.")
            break

        # Scale for mosaic and add timestamp with appropriate size
        now = time.time()
        if (now - last_mosaic_push_time) >= mosaic_interval:
            last_mosaic_push_time = now

            display_w = int(frame_width * display_scale)
            display_h = int(frame_height * display_scale)
            display_frame = cv2.resize(frame, (display_w, display_h))

            # Add timestamp to display frame (adjust font scale and position)
            font_scale_display = 0.4
            text_x = 10  # Adjust based on display scale
            text_y = 20
            cv2.putText(display_frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_display,
                        (0, 255, 255), 1)

            queue_for_display.put(display_frame)

        # Enforce real-time pacing
        if enforce_real_time:
            elapsed = time.time() - last_frame_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_frame_time = time.time()

    cap.release()
    if ffmpeg_process.stdin:
        ffmpeg_process.stdin.close()
    if ffmpeg_process.stderr:
        ffmpeg_process.stderr.close()
    ffmpeg_process.wait()

def consumer_mosaic(queues, consumer_fps=5):
    interval = 1.0 / consumer_fps
    frame_sizes = [None] * len(queues)

    while True:
        frames = []
        for i, q in enumerate(queues):
            try:
                frame = q.get(timeout=2.0)
            except:
                frame = None
            frames.append(frame)

        if all(f is None for f in frames):
            print("All camera streams ended.")
            break

        mosaic_parts = []
        for i, f in enumerate(frames):
            if f is not None:
                if frame_sizes[i] is None:
                    h, w, _ = f.shape
                    frame_sizes[i] = (w, h)
                mosaic_parts.append(f)
            else:
                if frame_sizes[i] is not None:
                    w, h = frame_sizes[i]
                    mosaic_parts.append(np.zeros((h, w, 3), dtype=np.uint8))

        if mosaic_parts:
            mosaic = np.hstack(mosaic_parts)
            # Removed add_timestamp to avoid system time overlay
            cv2.imshow('Mosaic View', mosaic)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        time.sleep(interval)

    cv2.destroyAllWindows()

def simulate_cameras(video_paths):
    num_cameras = len(video_paths)
    queues = [Queue(maxsize=20) for _ in range(num_cameras)]
    producers = []

    for i, video_path in enumerate(video_paths):
        p = Process(
            target=process_video_with_ffmpeg,
            kwargs=dict(
                video_path=video_path,
                queue_for_display=queues[i],
                camera_id=i,
                display_scale=0.125,
                display_fps=20,
                enforce_real_time=True
            )
        )
        producers.append(p)
        p.start()

    consumer_mosaic(queues, consumer_fps=20)

    for p in producers:
        p.join()

if __name__ == "__main__":
    simulate_cameras([
        'Ipynb_Experiment/hd_1920_1080_30fps.mp4',
        'Ipynb_Experiment/hd_1920_1080_30fps.mp4',
        'Ipynb_Experiment/hd_1920_1080_30fps.mp4',
        'Ipynb_Experiment/hd_1920_1080_30fps.mp4'
        # 'Ipynb_Experiment/Test_Video_4096_2160_25fps.mp4',
    ])