# SAR-YOLO
SAR-YOLO: Parallel Detection and Posture Recognition of Ultra-small Person for UAV-based Search and Rescue

# Dataset
Click [here](https://ieee-dataport.org/documents/search-and-rescue-image-dataset-person-detection-sard) to download the SARD dataset.

Click[here](https://surbhi-31.github.io/Aeriform-in-action/) to download the Aeriform in-action dataset. 

# Install Dependencies
```
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
conda create -n SAR-YOLO python=3.11
conda activate SAR-YOLO
pip install -r requirements.txt
pip install -e .
```
