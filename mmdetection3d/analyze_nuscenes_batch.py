#!/usr/bin/env python3
import json
import glob
import numpy as np
from pathlib import Path

# Find all prediction JSON files
pred_files = sorted(glob.glob('results_nuscenes_batch/sample_*/*_predictions.json'))

print(f"Found {len(pred_files)} prediction files\n")

all_detections = []
all_scores = []
all_classes = []

for pred_file in pred_files:
    with open(pred_file, 'r') as f:
        data = json.load(f)
   
    num_det = len(data['bboxes_3d'])
    scores = data['scores_3d']
    labels = data['labels_3d']
   
    all_detections.append(num_det)
    all_scores.extend(scores)
    all_classes.extend(labels)
   
    print(f"{Path(pred_file).parent.name}: {num_det} detections, avg score: {np.mean(scores):.3f}")

print("\n" + "="*60)
print("OVERALL STATISTICS")
print("="*60)
print(f"Total samples processed: {len(pred_files)}")
print(f"Average detections per sample: {np.mean(all_detections):.1f}")
print(f"Total detections: {sum(all_detections)}")
print(f"Average confidence score: {np.mean(all_scores):.3f}")
print(f"Max confidence score: {np.max(all_scores):.3f}")
print(f"Min confidence score: {np.min(all_scores):.3f}")

# Class distribution
from collections import Counter
class_counts = Counter(all_classes)
print("\nClass Distribution:")
for class_id, count in sorted(class_counts.items()):
    print(f"  Class {class_id}: {count} detections")
