# IDA-VLM
Building Identity-Aware VLM.


## Updates

* Mar 19, 2026: tested InternVL-3.5-8B
* Mar 17, 2026: created WYZE in-house Person ReID VLM benchmarks.


## Env



## Benchmarks

WYZE in-house data

| Dataset | Identities (total) | Query Images | Gallery Images | Train Split (identity/imgs) | Test Split (identity/imgs)
|---------|-----------|--------|-----------------|--------------|-----------------|
| wyze_person_v1 (cross-camera, same clothes)| 382 | 3,745 | 5,164 | | | |
| wyze_person_v2 (cross-camera, same clothes) | | | | | | |
| wyze_person_v2 (cross-camera, cross clothes) | | | | | | |





## Test VLM

```bash

```

1. Access data in the google storage bucket

```bash
# verify you have access
gcloud auth list

# locate the data
gsutil ls
gsutil ls gs://wyze-ai-team-data/

# check local VM disk
df -h

# check the size of the data folder to be downloaded
gsutil du -sh gs://wyze-ai-team-data/wyze_person_v1/
gsutil du -sh gs://wyze-ai-team-data/wyze_person_v2/
gsutil du -sh gs://wyze-ai-team-data/wyze_person_v2/annotated_identities/

# donwload the data folder to local VM
cd /home/tian.liu/tian_data/
gsutil -m cp -r gs://wyze-ai-team-data/wyze_person_v1/ .
gsutil -m cp -r gs://wyze-ai-team-data/wyze_person_v2/ .

gcloud storage cp -r gs://wyze-ai-team-data/wyze_person_v2/annotated_identities .

# to expand VM disk space, first update the disk space on GCP portal
# then grow the partition and stretch the file system
sudo rm -f /etc/apt/sources.list.d/backports.list
sudo apt-get update --allow-releaseinfo-change
sudo apt-get install fdisk -y

sudo growpart /dev/sda 1
sudo resize2fs /dev/sda1
```

Upload data to the GCP bucket

```bash
gcloud storage cp --recursive /home/tian.liu/IDA-VLM/training/runs/20260427_212918_annotated-distractor-sft-qwen7b-WYZEv03_23_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0323token_expert_and_image_attn_lr_0.0002_bs_4_captions_False gs://wyze-ai-team-data/tian_vlm/model_ckpts

gcloud storage cp --recursive /home/tian.liu/IDA-VLM/training/runs/20260430_040726_annotated-distractor-sft-qwen7b-WYZEv04_15_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0415token_expert_and_image_attn_lr_0.0002_bs_4_captions_False gs://wyze-ai-team-data/tian_vlm/model_ckpts

gcloud storage cp --recursive /home/tian.liu/IDA-VLM/training/runs/20260502_014222_annotated-distractor-sft-qwen7b-WYZEv04_15_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0415token_expert_and_image_attn_lr_0.0002_bs_4_captions_False gs://wyze-ai-team-data/tian_vlm/model_ckpts

```


To download videos from AWS S3 bucketsL

```bash
# install awscli
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o  "awscliv2.zip" && unzip awscliv2.zip && sudo ./aws/install --update

# refresh terminal command cache
hash -r

# autheticate in web
aws configure sso

# set the env variable
export AWS_PROFILE=AWSPowerUserAccess-447056034859

# run the downloading script
cd /home/tian.liu/IDA-VLM/download_videos
python download_video.py

# extract full frames from the videos
python extract_frames.py

ssh-keygen -t ed25519 -C "tian.liu@wyze.com"
cat /home/tian.liu/.ssh/id_ed25519.pub
# add to the github settings SSH keys, and then click "configure SSO" to authorize the access to wyze repo
git remote set-url origin git@github.com:tian-wyze/IDA-VLM.git
```