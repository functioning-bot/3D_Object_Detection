# 3D Object Detection:

## 1. Setup & Environment
The project was executed on a high-performance deep learning setup. Below are the exact commands used to reproduce the environment.

* **Hardware:** NVIDIA GeForce RTX 4090 (24GB VRAM)
* **Environment:** Python 3.10, PyTorch 2.5.1, MMDetection3D v1.4.0

### Installation & Verification
```bash
conda activate functioning_bot
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
python -m pip install -U pip setuptools wheel
python -m pip install -U openmim

mim install mmengine
pip uninstall mmcv -y
MMCV_WITH_OPS=1 pip install -e . -v --force-reinstall --no-deps
mim install "mmdet>=3.0.0,<3.3.0"
mim install "mmdet3d>=1.1.0"

pip install open3d opencv-python matplotlib tqdm

pip uninstall numpy -y
pip install 'numpy<2'
pip uninstall matplotlib -y
pip install matplotlib

# Verification Script
python - <<'PY'
import sys, numpy, matplotlib
print("Python     :", sys.version.split()[0])
print("NumPy      :", numpy.__version__)
print("Matplotlib :", matplotlib.__version__)
from matplotlib import pyplot as plt
print("pyplot OK")
import torch, mmengine, mmcv
print("Torch      :", torch.__version__, "| CUDA:", torch.version.cuda, "| is_available:", torch.cuda.is_available())
print("MMEngine   :", mmengine.__version__)
import pkgutil
try:
    import mmcv
    print("MMCV   :", mmcv.__version__, "at", mmcv.__file__)
    print("has mmcv._ext ? ", pkgutil.find_loader("mmcv._ext") is not None)
except Exception as e:
    print("MMCV import error:", repr(e))
PY
```

## 2\. Models & Datasets Used

Evaluated four distinct configurations to compare the trade-offs between Single-Stage vs. Two-Stage and Anchor-Based vs. Anchor-Free architectures.

### **Datasets**

1.  **KITTI:** A foundational autonomous driving dataset. We used the training split (\~7.5k samples) focusing on 3 classes: **Car, Pedestrian, Cyclist**.
2.  **NuScenes:** A modern, complex dataset with 360° LiDAR and camera coverage. We used the `v1.0-Mini` split (10 scenes, \~400 samples) with 10 object classes.

### **Models Evaluated**

| Model | Type | Features | Used On |
| :--- | :--- | :--- | :--- |
| **PointPillars** | Single-Stage, Anchor-Based | Converts point clouds into vertical columns (pillars). Optimized for high-speed inference. | KITTI & NuScenes |
| **PV-RCNN** | Two-Stage, Point-Voxel | Combines voxel efficiency with raw point precision. Optimized for maximum accuracy. | KITTI |
| **CenterPoint** | Two-Stage, Anchor-Free | Detects objects as center points rather than using predefined anchors. Optimized for rotation robustness. | NuScenes |

## 3\. Methodology:

For this project, used two inference logic scripts: `mmdet3d_inference2.py` for KITTI and `simple_infer_main.py` for NuScenes. Did try to use NuScenes using `mmdet3d_inference2.py` but it produced many issues.

For KITTI (`mmdet3d_inference2.py`): KITTI is an older dataset with a simpler structure, but it has tricky coordinate systems (Camera vs. LiDAR). Used this script because it allowed to manually control the Open3D visualization loop. This fixed the invisible box issue by ensuring the bounding boxes matched the LiDAR frame before rendering.
For NuScenes (`simple_infer_main.py`): NuScenes is much more complex (it has a database, SDK, and 360° views). Writing a manual viewer for it is difficult and error-prone. Instead, patched an existing tool (`simple_infer_utils.py`) to handle the heavy lifting. This script needed specific "hacks" to work with setup—specifically, had to force it to accept the `v1.0-mini` split (which the library usually rejects) and strip the timestamp dimension that was crashing the PointPillars model.

Using two specialized tools allowed to debug each dataset's unique problems without breaking the other.

## 4\. Results & Metrics

| Dataset | Model | Accuracy (mAP / AP) | Latency | Speed | Model Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **KITTI** | **PointPillars** | **80.7%** (Car 2D) | \~17 ms | **\~59 FPS** | **18.4 MB** |
| **KITTI** | **PV-RCNN** | **83.5%** (Car 2D) | 100 ms | 10 FPS | 155.1 MB |
| **NuScenes** | **PointPillars** | 0.0% (Failed) | 109.7 ms | 9.1 FPS | 18.9 MB |
| **NuScenes** | **CenterPoint** | **47.9%** (mAP) | 139.0 ms | 7.2 FPS | 34.4 MB |

### Visual Evidence

**[Click Here to View Full Results and Images (Google Drive)](https://drive.google.com/drive/folders/1FRQqoTZpUniqMNSKDW-30dsVqt_VvhZD?usp=drive_link)**

## 5\. Issues Faced

One significant technical hurdle faced was the "Invisible Box" problem, where 3D bounding boxes failed to render during visualization. This occurred because the older model weights outputted coordinates in a Camera-based system (Z-forward), whereas modern visualizer anticipated a LiDAR-based system (Z-up), causing the boxes to be rendered underground. Resolved this by identifying the specific coordinate mismatch and upgrading to "v1.0" model weights that utilized the correct LiDAR coordinate standard.

Also encountered a "Missing Data" crash where the evaluation script failed instantly due to a `KeyError: 'lidar_points'`. The root cause was that the standard KITTI dataset files lacked specific metadata fields required by the newer evaluation codebase. To fix this, developed a custom Python script, `fix_kitti_deep.py`, which injected dummy 3D data into the files, effectively tricking the evaluator into processing the results without errors.

Finally, the NuScenes model initially failed to run on the "Mini" version of the dataset. The library was hardcoded to validate against the full `val` split and rejected our smaller `v1.0-mini` version. Addressed this by patching the inference code in `simple_infer_main.py` to accept the `mini_val` split explicitly and removing a timestamp dimension that was incompatible with the older PointPillars architecture.

## 6\. Comparisons & Takeaways

Comparison on the KITTI dataset highlighted a distinct "Accuracy Tax" where speed is traded for precision. The PV-RCNN model demonstrated superior performance with 83.5% accuracy, successfully detecting difficult classes like cyclists compared to PointPillars' 80.7%. However, this precision came at a steep cost, as PV-RCNN was approximately 6x slower and 8x heavier than its counterpart. Consequently, PointPillars emerges as the optimal choice for embedded systems requiring real-time response, whereas PV-RCNN is better suited for offline processing where latency is less critical.

On the NuScenes dataset, observed the fragility of anchor-based methods in complex environments. The PointPillars model failed completely with a 0.0% mAP, likely because its rigid, pre-defined anchor boxes could not adapt to the diverse 360-degree orientations of objects. In contrast, the anchor-free CenterPoint model achieved a robust 47.9% mAP by detecting object centers rather than fitting boxes. This suggests that anchor-free architectures are significantly more robust for complex, multi-class environments involving rotation and unusual object shapes.

## 7\. Reproduction Steps

**Step 1: Environment Setup**
Run the installation commands listed in **Section 1** to create the environment and install dependencies.

**Step 2: Prepare Datasets**
Download the datasets and organize them as follows:

  * **KITTI:** Place training data in `mmdetection3d/data/kitti/`
  * **NuScenes:** Place v1.0-Mini split in `mmdetection3d/data/nuscenes/`

**Step 3: Download Checkpoints**
Run these commands to verify or re-download the exact model weights used in this report.

```bash
mkdir -p checkpoints
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest mmdetection3d/checkpoints
mim download mmdet3d --config pv_rcnn_8xb2-80e_kitti-3d-3class --dest mmdetection3d/checkpoints
mim download mmdet3d --config pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d --dest mmdetection3d/checkpoints
mim download mmdet3d --config centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d --dest mmdetection3d/checkpoints
```

**Step 4: Execute Experiments**
Run the inference commands below to reproduce results.

**Exp 1: KITTI PointPillars**

```bash
python mmdetection3d/mmdet3d_inference2.py --dataset kitti --input-path mmdetection3d/data/kitti/training --model mmdetection3d/checkpoints/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py --checkpoint mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --out-dir mmdetection3d/results/kitti_pp --score-thr 0.3 --device cuda:0 --headless --frame-number -1
```

**Exp 2: KITTI PV-RCNN**

```bash
python mmdetection3d/mmdet3d_inference2.py --dataset kitti --input-path mmdetection3d/data/kitti/training --model mmdetection3d/checkpoints/pv_rcnn_8xb2-80e_kitti-3d-3class.py --checkpoint mmdetection3d/checkpoints/pv_rcnn_8xb2-80e_kitti-3d-3class_20210831_022655-14a92953.pth --out-dir mmdetection3d/results/kitti_pvrcnn --score-thr 0.3 --device cuda:0 --headless --frame-number -1
```

**Exp 3: NuScenes PointPillars**

```bash
env -u DISPLAY python mmdetection3d/simple_infer_main.py --dataset nuscenes --dataroot mmdetection3d/data/nuscenes --ann-file nuscenes_infos_val.pkl --config mmdetection3d/checkpoints/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py --checkpoint mmdetection3d/checkpoints/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth --out-dir mmdetection3d/results/nuscenes_pp --data-source custom --nus-version v1.0-mini --no-open3d --max-samples -1 --eval
```

**Exp 4: NuScenes CenterPoint**

```bash
env -u DISPLAY python mmdetection3d/simple_infer_main.py --dataset nuscenes --dataroot mmdetection3d/data/nuscenes --ann-file nuscenes_infos_val.pkl --config mmdetection3d/checkpoints/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py --checkpoint mmdetection3d/checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth --out-dir mmdetection3d/results/nuscenes_cp --data-source custom --nus-version v1.0-mini --no-open3d --max-samples -1 --eval
```
