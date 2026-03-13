import os
import cv2


def save_frames(keyframe_indexes, video_path, save_path, folder_name, prefix=None):
    """Save keyframes to <save_path>/<folder_name>/.

    Args:
        keyframe_indexes: list of frame indices to save.
        video_path:       path to the source video file.
        save_path:        root directory for output.
        folder_name:      sub-folder name inside save_path.
        prefix:           optional filename prefix. When given, files are named
                          <prefix>_<frame_index>.jpg (e.g., prefix_00001.jpg).
                          When None, files are named <frame_index>.jpg (original behaviour).
    """
    cap = cv2.VideoCapture(video_path)

    folder_path = os.path.join(save_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Build a set for O(1) lookup
    keyframe_set = set(keyframe_indexes)

    current_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_index in keyframe_set:
            if prefix is not None:
                file_name = f"{prefix}_{current_index:05d}.jpg"
            else:
                file_name = f"{current_index}.jpg"

            cv2.imwrite(os.path.join(folder_path, file_name), frame)

        current_index += 1

    cap.release()
