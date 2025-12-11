import json
import glob
import os
import numpy as np
def analyze(dir_path, dataset_name): 
    json_files = glob.glob(os.path.join(dir_path, "*_predictions.json")) 
    print(f"--- {dataset_name} Analysis ---") 
    if not json_files: 
        print("No results found.") 
        return
    for f in json_files: 
        with open(f, 'r') as fp: 
            data = json.load(fp)
            bboxes = data.get('bboxes_3d', []) 
            scores = data.get('scores_3d', [])
            labels = data.get('labels_3d', []) 
            print(f"File: {os.path.basename(f)}") 
            print(f" Detections: {len(bboxes)}") 
            if len(scores) > 0: 
                print(f" Avg Score: {np.mean(scores):.4f}") 
                print(f" Max Score: {np.max(scores):.4f}") 
                # Count classes 
                from collections import Counter 
                c = Counter(labels) 
                print(f" Class Counts: {dict(c)}") 
                print("")
if __name__ == "__main__": 
    analyze("results_kitti_full", "KITTI") 
    analyze("results_nuscenes_full", "NuScenes")
