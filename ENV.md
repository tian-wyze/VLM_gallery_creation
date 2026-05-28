# How to install environments?

1. The environment below is used to prepare the benchmarks.

```bash
conda create -n ida-vlm python=3.10 -y
conda clean --all

pip install pandas tqdm boto3 matplotlib awscli opencv-python

# in case nvidia driver was swept due to GCP kernel update, reinstall drivers
sudo /opt/deeplearning/install-driver.sh

# install torch and xformers of compatible versions
pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# this is for running DINOv2 model
pip install xformers==0.0.28.post3
```

2. The environment below is used to test InternVL-3 models

```bash
git clone https://github.com/OpenGVLab/InternVL.git
cd InternVL

conda create -n internvl python=3.9 -y
conda activate internvl

pip install -r requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation

# to use InternVL3.0+, must usews python3.10 above
conda create -n internvl_py310 python=3.10 -y
conda activate internvl_py310
pip install torch torchvision torchaudio decord einops timm psutil accelerate

pip install transformers==4.55.0
# this may take quite a while
pip install flash-attn --no-build-isolation
```