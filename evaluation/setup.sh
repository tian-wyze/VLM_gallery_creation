sudo apt-get update && sudo apt-get install -y python3-venv python3-pip


python3 -m venv ~/envs/ida-vlm --system-site-packages
source ~/envs/ida-vlm/bin/activate

cat >> ~/.bashrc <<'EOF'

# Force torch to load its bundled cuBLAS (pip nvidia-cublas-cu12) instead of
# the apt /usr/local/cuda libcublasLt, which is ABI-mismatched and causes
# "Invalid handle. Cannot load symbol cublasLtCreate".
NV_LIB_BASE=/usr/local/lib/python3.10/dist-packages/nvidia
if [[ -d "$NV_LIB_BASE/cublas/lib" ]]; then
    export LD_LIBRARY_PATH="$NV_LIB_BASE/cublas/lib:$NV_LIB_BASE/cuda_runtime/lib:$NV_LIB_BASE/cudnn/lib:$NV_LIB_BASE/cufft/lib:$NV_LIB_BASE/curand/lib:$NV_LIB_BASE/cusolver/lib:$NV_LIB_BASE/cusparse/lib:$NV_LIB_BASE/nvjitlink/lib:${LD_LIBRARY_PATH}"
fi
EOF
source ~/.bashrc

#pip install -r requirements.txt
#pip uninstall -y google                 # remove the wrong one
pip install google-genai                # the SDK you actually need
pip install ultralytics hdbscan
pip install qwen-vl-utils
sudo apt-get install -y libgl1 libglib2.0-0
pip install --force-reinstall --no-deps opencv-python
pip install transformers
pip install accelerate


wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y --reinstall libcublas-12-9
# xformers is OPTIONAL (only speeds up DINOv2). Versions must match torch
# exactly or DINOv2's forward pass aborts with "cublasLtCreate" at runtime.
# Skip this line unless you know the matching wheel exists for your torch.
# pip install --no-deps --index-url https://download.pytorch.org/whl/cu129 'xformers==0.0.32.*'
# for setting up gcsfuse, which is needed to read the data stored in GCS buckets
# Add the gcsfuse distribution URL as a package source
export DISTRO=$(lsb_release -c -s)
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/cloud.google.gpg
echo "deb [signed-by=/etc/apt/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt gcsfuse-$DISTRO main" \
    | sudo tee /etc/apt/sources.list.d/gcsfuse.list

# Update package lists and install
sudo apt-get update
sudo apt-get install gcsfuse -y

mkdir -p ~/bucket_data/wyze_person_v2
gcsfuse --implicit-dirs --only-dir wyze_person_v2 xin_data ~/bucket_data/wyze_person_v2


# for gemini api
export GOOGLE_CLOUD_PROJECT=fluted-bit-436622-f3

