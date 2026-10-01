import cv2
import os

def extract_frames(video_path, interval, output_folder=None, save=False):
    '''extracts the frames from video at video_path and outputs them in output_folder. 
    interval defines how long between each frame will be (5 means one frame every 5 seconds)
    '''
    if save:
        if output_folder is None:
            raise ValueError("output_folder is required when save=True")
        elif not os.path.exists(output_folder):
            os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not video.isOpened():
        print("Error: Could not open the video.")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(interval * fps)

    frame_count = 0
    frame_number = 0
    
    while True:
        #set video to correct frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        # Read the next frame from the video
        success, frame = video.read()
        
        # If success is False, the video has ended
        if not success:
            break
            
        # Format filename with zero-padding (e.g., frame_0001.jpg) for easy sorting
        if save:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        
        # Save the current frame as an image
        if save:
            cv2.imwrite(frame_filename, frame)
        else:
            yield frame

        frame_number += frame_interval
        frame_count += 1

    # Clean up and release the video file
    video.release()
    print(f"Extraction complete!")

def extract_frame_batch(video_path, interval, output_folder=None, save=False, batch_size=9):
    '''extracts a batch of frames from video at video_path and outputs them in output_folder. 
    interval defines how long between each frame will be (5 means one frame every 5 seconds)
    '''
    if save:
        if output_folder is None:
            raise ValueError("output_folder is required when save=True")
        elif not os.path.exists(output_folder):
            os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not video.isOpened():
        print("Error: Could not open the video.")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(interval * fps)

    frame_count = 0
    frame_number = 0

    batch = []

    print("FPS:", fps)
    print("Frame interval:", frame_interval)
    print("Total frames:", video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        #set video to correct frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        # Read the next frame from the video
        success, frame = video.read()
        
        # If success is False, the video has ended
        if not success:
            break
            
        # Format filename with zero-padding (e.g., frame_0001.jpg) for easy sorting
        if save:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        
        # Save the current frame as an image
        if save:
            cv2.imwrite(frame_filename, frame)
        else:
            batch.append(frame)

        if len(batch) == batch_size:
            yield batch
            batch = []


        frame_number += frame_interval
        frame_count += 1

    #if there are lingering images
    if batch:
        yield batch
    
    # Clean up and release the video file
    video.release()
    print(f"Extraction complete!")