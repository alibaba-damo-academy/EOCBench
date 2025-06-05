import numpy as np
from decord import VideoReader, cpu
from PIL import Image
import cv2
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32,fps_segments=None):
    frame_ids = []
    max_frame = max_frame + 1
    if num_segments ==1:
        return max_frame
    if not fps_segments:
        n_frames = max_frame // (num_segments - 1)
    else:
        n_frames = fps // fps_segments
    
    n_frames = min(n_frames,150)
    if n_frames == 0:
        return list(range(max_frame))
    for frame_count in range(max_frame):
        if (frame_count % n_frames == 0 or frame_count == max_frame) and (num_segments is None or len(frame_ids) < num_segments):
            frame_ids.append(frame_count)
    return frame_ids

def load_video(video_path,fps = None, bound=None, num_segments=32 , fps_segments = None, prompt='box', add_last_frame=True):
    imgs = []

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1

    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments,fps_segments=fps_segments)

    frame_times = []
    for frame_index in frame_indices:
        frame_time = float(frame_index) / fps
        frame_times.append(frame_time)
    frame_times = np.array(frame_times)

    last_frame_index = max_frame

    if add_last_frame:
        if last_frame_index not in frame_indices:
            frame_indices = np.delete(frame_indices, -1)
            frame_indices = np.append(frame_indices, last_frame_index)
            frame_times = np.delete(frame_times, -1)
            frame_times = np.append(frame_times, round(float(last_frame_index) / fps,1))
    else:
        if last_frame_index-1 not in frame_indices:
            frame_indices = np.delete(frame_indices, -1)
            frame_indices = np.append(last_frame_index-1, last_frame_index)
            frame_times = np.delete(frame_times, -1)
            frame_times = np.append(frame_times, round(float(last_frame_index-1) / fps,1))

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        imgs.append(img)
    return imgs,frame_times

def load_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    last_frame = None
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        last_frame = frame
    cap.release()
    return last_frame


def get_second_last_frame_from_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    last_frame = None
    second_last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        second_last_frame = last_frame
        last_frame = frame

    cap.release()

    return second_last_frame

