import pickle

# Files to fix
files = ['data/kitti/kitti_infos_val.pkl', 'data/kitti/kitti_infos_train.pkl']

for file_path in files:
    try:
        print(f"Checking {file_path}...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # If it is a list, we need to convert it
        if isinstance(data, list):
            print(f"  -> Old format detected (List). Converting to Dict...")
            
            # Create the new structure
            new_data = {
                'metainfo': {'classes': ['Car', 'Pedestrian', 'Cyclist']},
                'data_list': data
            }
            
            # Save it back
            with open(file_path, 'wb') as f:
                pickle.dump(new_data, f)
            print("  -> Fixed and saved.")
        else:
            print("  -> Already in correct format (Dict). Skipping.")

    except FileNotFoundError:
        print(f"  -> File not found: {file_path}")