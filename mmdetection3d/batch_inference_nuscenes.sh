#!/bin/bash

# Activate environment if needed
# conda activate functioning_bot

# Get first 10 samples from NuScenes mini
SAMPLES=$(ls ~/data/nuscenes/samples/LIDAR_TOP/*.pcd.bin | head -10)

# Counter
i=1

# Run inference on each sample
for SAMPLE in $SAMPLES; do
    echo "Processing sample $i: $SAMPLE"
   
    python mmdet3d_inference2.py \
      --dataset any \
      --input-path $(dirname $SAMPLE) \
      --model checkpoints/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
      --checkpoint checkpoints/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_*.pth \
      --modality lidar \
      --out-dir results_nuscenes_batch/sample_$i \
      --headless \
      --device cuda:0
   
    ((i++))
done

echo "Batch inference complete!"
