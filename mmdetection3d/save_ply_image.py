#!/usr/bin/env python3
import argparse
import os
import sys
import open3d as o3d

def load_if_exists(path: str, loader, name: str):
    if os.path.exists(path):
        try:
            obj = loader(path)
            print(f"[LOAD] {name}: {path}")
            return obj
        except Exception as e:
            print(f"[WARN] Failed to load {name} ({path}): {e}")
    else:
        print(f"[SKIP] {name} not found: {path}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Open3D viewer for saved PLY outputs")
    parser.add_argument("--dir", required=True, help="Folder containing PLY files")
    parser.add_argument("--basename", required=True, help="Base name, e.g. 000008")
    parser.add_argument("--out", help="Output image path (optional)")
    parser.add_argument("--video-out", help="Output video path (optional)")
    parser.add_argument("--width", type=int, default=1440, help="Viewer window width")
    parser.add_argument("--height", type=int, default=900, help="Viewer window height")
    args = parser.parse_args()

    base_dir = os.path.expanduser(args.dir)
    base = args.basename

    points_path = os.path.join(base_dir, f"{base}_points.ply")
    axes_path = os.path.join(base_dir, f"{base}_axes.ply")
    pred_bbox_path = os.path.join(base_dir, f"{base}_pred_bboxes.ply")
    pred_label_path = os.path.join(base_dir, f"{base}_pred_labels.ply")
    gt_bbox_path = os.path.join(base_dir, f"{base}_gt_bboxes.ply")

    geoms = []
    pcd = load_if_exists(points_path, o3d.io.read_point_cloud, "Point cloud")
    if pcd: geoms.append(pcd)
    axes = load_if_exists(axes_path, o3d.io.read_triangle_mesh, "Coordinate axes")
    if axes: geoms.append(axes)
    pred_bboxes = load_if_exists(pred_bbox_path, o3d.io.read_line_set, "Predicted bboxes")
    if pred_bboxes: geoms.append(pred_bboxes)
    pred_labels = load_if_exists(pred_label_path, o3d.io.read_line_set, "Predicted labels")
    if pred_labels: geoms.append(pred_labels)
    gt_bboxes = load_if_exists(gt_bbox_path, o3d.io.read_line_set, "Ground truth bboxes")
    if gt_bboxes: geoms.append(gt_bboxes)

    if not geoms:
        print("No geometries loaded.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=args.width, height=args.height)
    for geom in geoms:
        vis.add_geometry(geom)
    
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    if args.out:
        ctr.rotate(0, -200)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(args.out)
        print(f"Saved visualization to {args.out}")

    if args.video_out:
        import cv2
        import numpy as np
        
        print(f"Rendering video to {args.video_out}...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(args.video_out, fourcc, 30, (args.width, args.height))
        
        # Rotate 360 degrees
        steps = 90
        for i in range(steps):
            ctr.rotate(10.0, 0.0) # Rotate horizontally
            vis.poll_events()
            vis.update_renderer()
            
            # Capture image
            # Open3D headless capture to buffer is tricky, usually saves to file.
            # We'll save to temp file and read back, or use capture_screen_float_buffer
            
            tmp_img = f"tmp_{i}.png"
            vis.capture_screen_image(tmp_img, do_render=False)
            frame = cv2.imread(tmp_img)
            if frame is not None:
                video.write(frame)
            if os.path.exists(tmp_img):
                os.remove(tmp_img)
                
        video.release()
        print(f"Saved video to {args.video_out}")

    vis.destroy_window()

if __name__ == "__main__":
    main()

