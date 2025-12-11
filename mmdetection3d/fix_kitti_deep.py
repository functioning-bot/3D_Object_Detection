import pickle
import os
import numpy as np

# Files to fix
files = ['data/kitti/kitti_infos_val.pkl', 'data/kitti/kitti_infos_train.pkl']

# Class mapping
CLASS_MAP = {
    'Car': 0, 'Pedestrian': 1, 'Cyclist': 2,
    'Van': 0, 'Person_sitting': 1, 'Truck': -1,
    'Tram': -1, 'Misc': -1, 'DontCare': -1
}

def fix_info(info):
    # --- FIX 1: LIDAR PATH ---
    if 'lidar_points' not in info:
        v_path = None
        if 'velodyne_path' in info:
            v_path = info['velodyne_path']
        elif 'point_cloud' in info and 'velodyne_path' in info['point_cloud']:
            v_path = info['point_cloud']['velodyne_path']
        
        if v_path:
            info['lidar_points'] = {'lidar_path': v_path, 'num_pts_feats': 4}

    # --- FIX 2: CALIBRATION ---
    lidar2cam = np.eye(4)
    cam2img = np.eye(4)
    
    if 'calib' in info:
        calib = info['calib']
        if 'R0_rect' in calib and 'Tr_velo_to_cam' in calib:
            R0_rect = np.array(calib['R0_rect'])
            Tr_velo_to_cam = np.array(calib['Tr_velo_to_cam'])
            lidar2cam = R0_rect @ Tr_velo_to_cam
            
        if 'P2' in calib:
            cam2img = np.array(calib['P2'])

    # --- FIX 3: IMAGES (Added Height/Width) ---
    img_path = ""
    # Try to find existing path
    if 'images' in info and 'CAM2' in info['images']:
        img_path = info['images']['CAM2'].get('img_path', "")
    elif 'image' in info and 'image_path' in info['image']:
        img_path = info['image']['image_path']
        
    # Set standard KITTI dimensions (approximate is fine for 3D eval)
    info['images'] = {
        'CAM2': {
            'img_path': img_path,
            'height': 375,  # REQUIRED
            'width': 1242,  # REQUIRED
            'lidar2cam': lidar2cam.tolist(),
            'cam2img': cam2img.tolist()
        }
    }

    if 'image' in info and 'image_idx' in info['image']:
        info['sample_idx'] = info['image']['image_idx']

    # --- FIX 4: INSTANCES ---
    annos = info.get('annos', None)
    if annos is None and 'annos' in info.get('image', {}): 
            annos = info['image']['annos']
            
    instances = []
    if annos:
        num_objects = len(annos['name'])
        for i in range(num_objects):
            obj_name = annos['name'][i]
            label = CLASS_MAP.get(obj_name, -1)
            
            if label != -1:
                inst = {
                    'bbox_label': label,
                    'bbox_label_3d': label,
                    'bbox': annos['bbox'][i], 
                    'bbox_3d': [0.0]*7,
                    'depth': 0.0,
                    'center_2d': [0.0, 0.0],
                    'truncated': annos['truncated'][i] if 'truncated' in annos else 0.0,
                    'occluded': annos['occluded'][i] if 'occluded' in annos else 0,
                    'alpha': annos['alpha'][i] if 'alpha' in annos else 0.0,
                    'score': annos['score'][i] if 'score' in annos else 0.0,
                    'dimensions': annos['dimensions'][i] if 'dimensions' in annos else [0,0,0],
                    'location': annos['location'][i] if 'location' in annos else [0,0,0],
                    'rotation_y': annos['rotation_y'][i] if 'rotation_y' in annos else 0.0,
                    'index': i,
                    'group_id': -1
                }
                instances.append(inst)
    
    info['instances'] = instances
    return info

for file_path in files:
    if not os.path.exists(file_path):
        print(f"Skipping {file_path} (not found)")
        continue
        
    print(f"Fixing dimensions for {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Handle wrapper
    data_list = []
    if isinstance(data, dict) and 'data_list' in data:
        data_list = data['data_list']
    elif isinstance(data, list):
        data_list = data

    meta = {
        'classes': ['Car', 'Pedestrian', 'Cyclist'],
        'categories': {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
    }

    new_list = []
    for item in data_list:
        new_list.append(fix_info(item))

    final_data = {
        'metainfo': meta,
        'data_list': new_list
    }

    with open(file_path, 'wb') as f:
        pickle.dump(final_data, f)
    print("  -> Saved.")