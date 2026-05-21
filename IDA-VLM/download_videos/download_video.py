import os
from concurrent.futures import ThreadPoolExecutor
import boto3
import pandas as pd

# Use your specific profile here
PROFILE_NAME = 'AWSPowerUserAccess-447056034859'
TARGET_ROLE_ARN = 'arn:aws:iam::596358926690:role/ai-dev-feedbackvideo-read-only-role'
BUCKET_NAME = 'wyze-feedback-video-service-596358926690-us-west-2'

def get_assumed_credentials():
    """Gets temporary credentials from the target role."""
    session = boto3.Session(profile_name=PROFILE_NAME)
    sts_client = session.client('sts')
    response = sts_client.assume_role(
        RoleArn=TARGET_ROLE_ARN,
        RoleSessionName='ai-dev-download-session'
    )
    return response['Credentials']

def download_single_video(video_path, location, credentials):
    """
    Thread-safe download: Creates a fresh session and resource
    INSIDE the thread.
    """
    try:
        # Each thread gets its own session and resource
        session = boto3.Session(
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken'],
            region_name='us-west-2'
        )
        s3 = session.resource('s3')
        bucket = s3.Bucket(BUCKET_NAME)

        video_name = video_path.split('/')[-1]
        save_path = os.path.join(location, video_name)

        bucket.download_file(video_path, save_path)
    except Exception as e:
        print(f"Failed to download {video_path}: {e}")

def download_feedback_video_multi_thread(video_list, location, credentials):
    # Using 30 workers as in your original script
    with ThreadPoolExecutor(max_workers=30) as executor:
        # We pass the credentials to every execution
        executor.map(lambda path: download_single_video(path, location, credentials), video_list)

def get_video_names_from_csv(csv_dir):
    video_names = []
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    for csv_file in csv_files:
        print(f"Reading: {csv_file}")
        df = pd.read_csv(os.path.join(csv_dir, csv_file))
        if 'FILE_PATH' in df.columns:
            video_names.extend(df['FILE_PATH'].tolist())
    print('Total videos to download:', len(video_names))
    return video_names

if __name__ == "__main__":
    # 1. Get the list of videos
    video_list = get_video_names_from_csv('./')

    # 2. Get credentials ONCE (they usually last 1 hour)
    creds = get_assumed_credentials()

    # 3. Setup save folder
    folder_to_save = '../dataset/videos_wyze_person_v2_cross_clothes'
    os.makedirs(folder_to_save, exist_ok=True)

    # 4. Start multi-threaded download
    download_feedback_video_multi_thread(video_list, folder_to_save, creds)