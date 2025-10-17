# Data Preprocessing
We preprocess **ScanNet**, **ScanNet++** and **ARKitScenes** following the data processing pipeline provided by [VLM-3R](https://github.com/VITA-Group/VLM-3R/tree/main/vlm_3r_data_process). Specifically, for each scene dataset, we extract and store the following components: **RGB frames**, **depth frames**, **camera parameters (both intrinsics and extrinsics)**, and **axis alignment matrix**. Since we only adopt part of the original pipeline and make several modifications to the code, we provide a simplified preprocessing version here for convenience and reproducibility. 

After preprocessing, the directory structure should look like this:
```
GS-Reasoner
|—— data
    |—— processed_data
        |—— scannet
            |—— axis_align_matrix
            |—— color
            |—— depth
            |—— intrinsic
            |—— pose
        |—— scannetpp
        |—— arkitscenes
```


## 1. Download Raw Dataset
* **ScanNet**: Download data to `data/raw_data/scannet` . Follow instructions at https://github.com/ScanNet/ScanNet.
* **ScanNet++**: Download data to `data/raw_data/scannetpp` and clone repository to `data/datasets/scannetpp`. Follow instructions at https://github.com/scannetpp/scannetpp.
* **ARKitScenes**: Download data to `data/raw_data/arkitscenes` and clone repository to `data/datasets/ARKitScenes`. Follow instructions at https://github.com/apple/ARKitScenes.


## 2. Extract Data
Extract **RGB frames**, **depth frames**, **camera parameters (both intrinsics and extrinsics)**, and **axis alignment matrix** using following script:

### ScanNet
```bash
# extract frame info
python scripts/3d/preprocessing/scannet/export_sampled_frames.py \
    --scans_dir data/raw_data/scannet/scans \
    --output_dir data/processed_data/scannet \
    --train_val_splits_path scripts/3d/preprocessing/scannet \
    --num_frames 10000 \
    --max_workers 32 \
    --image_size 480 640
```


### ScanNet++
```bash
# install dependency
pip install munch

# extract frame info
# train split
export PYTHONPATH="$PYTHONPATH:./data/datasets/scannetpp"
python scripts/3d/preprocessing/scannetpp/export_iphone_frames.py \
    --data_root data/raw_data/scannetpp/data \
    --scene_list_file data/raw_data/scannetpp/splits/nvs_sem_train.txt \
    --output_dir data/processed_data/scannetpp \
    --num_frames 10000 \
    --num_processes 32 \
    --image_size 480 640 \
    --split_name train

# val split
export PYTHONPATH="$PYTHONPATH:./data/datasets/scannetpp"
python scripts/3d/preprocessing/scannetpp/export_iphone_frames.py \
    --data_root data/raw_data/scannetpp/data \
    --scene_list_file data/raw_data/scannetpp/splits/nvs_sem_val.txt \
    --output_dir data/processed_data/scannetpp \
    --num_frames 10000 \
    --num_processes 32 \
    --image_size 480 640 \
    --split_name val

# generate axis align matrix
# train spilt
python scripts/3d/preprocessing/axis_align_pcd.py \
    --scene_list data/raw_data/scannetpp/splits/nvs_sem_train.txt \
    --split train \
    --output_dir data/processed_data/scannetpp/ \
    --data_type scannetpp

# val split
python scripts/3d/preprocessing/axis_align_pcd.py \
    --scene_list data/raw_data/scannetpp/splits/nvs_sem_val.txt \
    --split val \
    --output_dir data/processed_data/scannetpp/ \
    --data_type scannetpp
```

### ARKitScenes
```bash
# extract frame info
# train split
export PYTHONPATH="$PYTHONPATH:./data/datasets/ARKitScenes/threedod/benchmark_scripts"
python scripts/3d/preprocessing/arkitscenes/export_sampled_frames.py \
    --arkit_data_root data/raw_data/arkitscenes/3dod \
    --scene_list_file scripts/3d/preprocessing/arkitscenes/scenes_train.txt \
    --split Training \
    --output_dir data/processed_data/arkitscenes \
    --num_frames 10000 \
    --image_size 480 640 \
    --max_workers 32

# val split
export PYTHONPATH="$PYTHONPATH:./data/datasets/ARKitScenes/threedod/benchmark_scripts"
python scripts/3d/preprocessing/arkitscenes/export_sampled_frames.py \
    --arkit_data_root data/raw_data/arkitscenes/3dod \
    --scene_list_file scripts/3d/preprocessing/arkitscenes/scenes_val.txt \
    --split Validation \
    --output_dir data/processed_data/arkitscenes \
    --num_frames 10000 \
    --image_size 480 640 \
    --max_workers 32

# generate axis align matrix
# train split
python scripts/3d/preprocessing/axis_align_pcd.py \
    --scene_list scripts/3d/preprocessing/arkitscenes/scenes_train.txt \
    --split train \
    --output_dir data/processed_data/arkitscenes/ \
    --data_type arkitscenes

# val split
python scripts/3d/preprocessing/axis_align_pcd.py \
    --scene_list scripts/3d/preprocessing/arkitscenes/scenes_val.txt \
    --split val \
    --output_dir data/processed_data/arkitscenes/ \
    --data_type arkitscenes
```
