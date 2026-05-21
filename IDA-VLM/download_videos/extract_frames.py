"""
Extract full frames from videos based on the cross_clothes.json specification.

The script:
1. Reads the EVENT_ID -> FILE_PATH mapping from CSV
2. Reads image paths from cross_clothes.json
3. Extracts frame numbers and event IDs from filenames
4. Groups frames by video (event ID) for efficiency
5. Extracts frames from videos using OpenCV
6. Saves extracted frames to the output folder
"""

import os
import json
import csv
import cv2
from collections import defaultdict
from pathlib import Path
import re
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def read_event_id_mapping(csv_path):
    """
    Read EVENT_ID -> FILE_PATH mapping from CSV.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Dictionary mapping EVENT_ID to FILE_PATH
    """
    mapping = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                event_id = row['EVENT_ID'].strip()
                file_path = row['FILE_PATH'].strip().split('/')[-1]  # Get the filename from the path
                mapping[event_id] = file_path
        logger.info(f"Loaded {len(mapping)} EVENT_ID mappings from {csv_path}")
        return mapping
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}")
        return {}


def read_json_image_list(json_path):
    """
    Read image paths from cross_clothes.json.

    Args:
        json_path: Path to the JSON file

    Returns:
        Dictionary with 'gallery' and 'query' keys containing lists of image paths
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {json_path}")
        return data
    except Exception as e:
        logger.error(f"Error reading JSON file {json_path}: {e}")
        return {}


def parse_image_filename(filename):
    """
    Parse an image filename to extract event ID and frame number.

    Expected format: gallery/10026312_0_D03F278DF935_D03F278DF935131697549241_000000_002.jpg
    Returns: (event_id, frame_number)

    The event ID is the 24-digit number (e.g., D03F278DF935131697549241)
    The frame number is the 6-digit number (e.g., 000000)
    """

    try:
        event_id = filename.split('_')[-3]
        frame_id = filename.split('_')[-2]
        logger.debug(f"Parsed {filename}: event_id={event_id}, frame_id={frame_id}")
        return event_id, frame_id
    except Exception as e:
        logger.error(f"Error parsing filename {filename}: {e}")
        return None, None


def group_frames_by_event(image_list, event_mapping):
    """
    Group images by event ID.

    Args:
        image_list: List of image paths
        event_mapping: Dictionary mapping EVENT_ID to FILE_PATH

    Returns:
        Dictionary mapping EVENT_ID to list of (full_path, frame_number) tuples
    """
    grouped = defaultdict(list)

    for image_path in image_list:
        event_id, frame_number = parse_image_filename(image_path)
        if event_id and frame_number is not None:
            grouped[event_id].append((image_path, frame_number))

    logger.info(f"Grouped {len(image_list)} images into {len(grouped)} event groups")
    return grouped


def extract_frames_from_video(video_path, frame_numbers):
    """
    Extract specific frames from a video file.

    Args:
        video_path: Path to the video file
        frame_numbers: List of frame numbers to extract (0-indexed)

    Returns:
        Dictionary mapping frame_number to frame (numpy array),
        or empty dict if video cannot be opened
    """
    frames = {}

    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return frames

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return frames

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # logger.debug(f"Video: {os.path.basename(video_path)}, Total frames: {total_frames}, FPS: {fps}")

        # Sort frame numbers for efficient extraction
        sorted_frames = sorted(frame_numbers)
        current_frame = 0

        for target_frame in sorted_frames:
            if int(target_frame) >= total_frames:
                logger.warning(f"Frame {target_frame} exceeds total frames ({total_frames})")
                continue

            # Seek to the target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_frame))
            ret, frame = cap.read()

            if ret:
                frames[target_frame] = frame
            else:
                logger.warning(f"Failed to read frame {target_frame} from {os.path.basename(video_path)}")

        cap.release()
        # logger.info(f"Extracted {len(frames)} frames from {os.path.basename(video_path)}")

    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")

    return frames


def save_frame(frame, output_path):
    """
    Save a frame (numpy array) to disk.

    Args:
        frame: Numpy array representing the frame
        output_path: Path where to save the frame

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        success = cv2.imwrite(output_path, frame)
        if not success:
            logger.error(f"Failed to save frame to {output_path}")
        return success
    except Exception as e:
        logger.error(f"Error saving frame to {output_path}: {e}")
        return False


def get_image_filenames(json_data):
    """
    Extract image filenames from JSON data, handling both list and dict formats.

    Args:
        json_data: Dictionary from JSON with 'gallery' (list) and 'queries' (dict)

    Returns:
        List of all image filenames
    """
    filenames = []

    # Gallery is a list of filenames
    if 'gallery' in json_data:
        filenames.extend(json_data['gallery'])

    # Queries is a dict where keys are filenames, values are indices
    if 'queries' in json_data:
        filenames.extend(json_data['queries'].keys())

    return filenames


def process_frames(json_path, csv_path, output_dir, video_base_dir=None):
    """
    Main function to process and extract all frames.

    Args:
        json_path: Path to cross_clothes.json
        csv_path: Path to the EVENT_ID CSV mapping
        output_dir: Directory to save extracted frames
        video_base_dir: Base directory for video files (optional, used if video paths are relative)
    """
    # Load mappings
    event_mapping = read_event_id_mapping(csv_path)
    json_data = read_json_image_list(json_path)

    if not event_mapping:
        logger.error("No event mappings loaded. Exiting.")
        return

    if not json_data:
        logger.error("No JSON data loaded. Exiting.")
        return

    # Get all image filenames (handles both gallery list and queries dict)
    image_list = get_image_filenames(json_data)
    logger.info(f"Processing {len(image_list)} total images")

    # Group frames by event ID
    grouped = group_frames_by_event(image_list, event_mapping)

    # Process each video
    for event_id, images in tqdm(grouped.items(), desc="Processing events"):
        # Find the video path from the mapping
        if event_id not in event_mapping:
            logger.warning(f"Event ID {event_id} not found in mapping")
            continue

        # logger.info(f"Processing event {event_id}: {len(images)} frames to extract")

        video_relative_path = event_mapping[event_id]

        # Construct full video path
        if video_base_dir:
            video_path = os.path.join(video_base_dir, video_relative_path)
        else:
            video_path = video_relative_path

        # Extract frame numbers
        frame_numbers = [frame_num for _, frame_num in images]

        # Extract frames from video
        extracted_frames = extract_frames_from_video(video_path, frame_numbers)

        # Save extracted frames
        saved_count = 0
        for image_path, frame_number in images:
            if frame_number in extracted_frames:
                frame = extracted_frames[frame_number]
                output_path = os.path.join(output_dir, image_path)
                if save_frame(frame, output_path):
                    saved_count += 1

        # logger.info(f"Saved {saved_count}/{len(images)} frames for event {event_id}")


def main():
    """Main entry point."""
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Parent of download_videos

    # Paths relative to project root
    json_path = os.path.join(
        project_root,
        '..',
        'data',
        'wyze_person_v2_cross_clothes_full_frame',
        'cross_clothes.json'
    )

    csv_path = os.path.join(script_dir, 'wyze_person_v2_cross_clothes.csv')

    output_dir = os.path.join(
        project_root,
        '..',
        'data',
        'wyze_person_v2_cross_clothes_full_frame'
    )

    # Base directory for videos (adjust if videos are stored elsewhere)
    video_base_dir = os.path.join(
        project_root,
        '..',
        'data',
        'wyze_person_v2_cross_clothes_full_frame',
        'videos')

    # Verify input files exist
    if not os.path.exists(json_path):
        logger.error(f"JSON file not found: {json_path}")
        return

    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return

    logger.info(f"JSON path: {json_path}")
    logger.info(f"CSV path: {csv_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Video base directory: {video_base_dir}")

    # Process frames
    process_frames(json_path, csv_path, output_dir, video_base_dir)


if __name__ == '__main__':
    main()
