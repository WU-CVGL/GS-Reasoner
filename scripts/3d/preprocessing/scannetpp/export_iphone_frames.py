# -----------------------------------------------------------------------------
# This file is modified from:
# https://github.com/VITA-Group/VLM-3R/blob/main/vlm_3r_data_process/src/metadata_generation/ScanNetpp/preprocess/export_iphone_frames.py
#
# Original license: MIT License
# Copyright (c) 2025 VLM-3R
# -----------------------------------------------------------------------------



import argparse
import json
from pathlib import Path
import numpy as np
import shutil
from tqdm import tqdm
import os
import cv2 # Added for image processing
import zlib # Added for depth processing
import lz4.block # Added for depth processing
import tempfile # Added for temporary directory management
import multiprocessing # Added for parallel processing
# import imageio as iio # Using cv2 for consistency, but iio is an option if cv2 fails for uint16 PNGs
# from export_sampled_frames import verify_scene_reprojection

# 确保 common 模块在 PYTHONPATH 中，或者将此脚本放在 scannetpp 代码库的合适位置
try:
    from common.scene_release import ScannetppScene_Release
    from common.utils.colmap import read_model, qvec2rotmat
    from common.utils.utils import read_txt_list, run_command
except ImportError as e:
    print("Error: Ensure that the common ScanNet++ modules are in your PYTHONPATH.")
    print("You might need to run this script from within the scannetpp repository structure,")
    print("or add the repository root to your PYTHONPATH environment variable.")
    # print(f"current PYTHONPATH={sys.path}")
    print(f"ImportError details: {e}")
    exit(1)


# ---- START: New helper function for specific depth frame extraction ----
def _extract_and_save_specific_depth_frames(
    scene_iphone_depth_path: Path,
    target_frame_indices: list[int],
    depth_out_dir: Path,
    original_depth_height: int,
    original_depth_width: int,
    target_image_size: tuple[int, int] | None,
    scene_id: str
):
    """
    Extracts, decodes, (optionally) resizes, and saves specific depth frames.
    """
    if not scene_iphone_depth_path.exists():
        print(f"Warning [Scene: {scene_id}]: iPhone raw depth path {scene_iphone_depth_path} not found. Skipping depth extraction.")
        return

    print(f"Info [Scene: {scene_id}]: Starting extraction of {len(target_frame_indices)} specific depth frames from {scene_iphone_depth_path}")
    
    # Sort target_frame_indices to process in order, which might be slightly more efficient for seeking.
    # Also remove duplicates.
    processed_target_indices = sorted(list(set(target_frame_indices)))
    
    # Attempt global decompression first (zlib)
    try:
        with open(scene_iphone_depth_path, 'rb') as infile:
            compressed_data = infile.read()
            decompressed_data = zlib.decompress(compressed_data, wbits=-zlib.MAX_WBITS)
            all_depth_frames = np.frombuffer(decompressed_data, dtype=np.float32).reshape(-1, original_depth_height, original_depth_width)

        for frame_idx in tqdm(processed_target_indices, desc=f"Saving globally decompressed depth frames for {scene_id}"):
            if 0 <= frame_idx < all_depth_frames.shape[0]:
                depth_map_float32 = all_depth_frames[frame_idx]
                # Convert to uint16 (standard for depth maps, representing millimeters)
                depth_map_uint16 = (depth_map_float32 * 1000).astype(np.uint16)
                
                output_depth_file_path = depth_out_dir / f"frame_{frame_idx:06}.png"

                if target_image_size:
                    # target_image_size is (HEIGHT, WIDTH)
                    # cv2.resize expects (WIDTH, HEIGHT)
                    depth_map_resized = cv2.resize(depth_map_uint16, (target_image_size[1], target_image_size[0]), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(str(output_depth_file_path), depth_map_resized)
                else:
                    cv2.imwrite(str(output_depth_file_path), depth_map_uint16)
            else:
                print(f"Warning [Scene: {scene_id}]: Requested frame index {frame_idx} out of bounds for globally decompressed depth data (num frames: {all_depth_frames.shape[0]}).")
        print(f"Info [Scene: {scene_id}]: Successfully extracted depth frames using global zlib decompression.")
        return # Successfully processed with global decompression

    except Exception as e_global:
        print(f"Info [Scene: {scene_id}]: Global zlib decompression failed ({e_global}). Attempting per-frame decompression.")
        # Fallback to per-frame decompression
        current_frame_id_in_file = 0
        frames_saved_count = 0
        target_indices_set = set(processed_target_indices) # For quick lookups

        try:
            with open(scene_iphone_depth_path, 'rb') as infile:
                while True:
                    # Read size of the compressed frame
                    size_bytes = infile.read(4) # 32-bit integer for size
                    if not size_bytes: # End of file
                        break
                    
                    compressed_frame_size = int.from_bytes(size_bytes, byteorder='little')

                    if current_frame_id_in_file in target_indices_set:
                        data = infile.read(compressed_frame_size)
                        if len(data) < compressed_frame_size:
                            print(f"Warning [Scene: {scene_id}]: Unexpected EOF while reading frame {current_frame_id_in_file}.")
                            break
                        
                        try:
                            # Try LZ4 decompression
                            decompressed_frame = lz4.block.decompress(data, uncompressed_size=original_depth_height * original_depth_width * 2) # uint16 = 2 bytes
                            depth_map_uint16 = np.frombuffer(decompressed_frame, dtype=np.uint16).reshape(original_depth_height, original_depth_width)
                        except Exception as e_lz4:
                            # Try Zlib decompression (for float32, then convert)
                            try:
                                decompressed_frame = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                                depth_map_float32 = np.frombuffer(decompressed_frame, dtype=np.float32).reshape(original_depth_height, original_depth_width)
                                depth_map_uint16 = (depth_map_float32 * 1000).astype(np.uint16)
                            except Exception as e_zlib_frame:
                                print(f"Warning [Scene: {scene_id}]: Decompression failed for frame {current_frame_id_in_file} (LZ4: {e_lz4}, ZLIB: {e_zlib_frame}). Skipping.")
                                current_frame_id_in_file += 1
                                continue
                        
                        output_depth_file_path = depth_out_dir / f"frame_{current_frame_id_in_file:06}.png"
                        if target_image_size:
                            depth_map_resized = cv2.resize(depth_map_uint16, (target_image_size[1], target_image_size[0]), interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(str(output_depth_file_path), depth_map_resized)
                        else:
                            cv2.imwrite(str(output_depth_file_path), depth_map_uint16)
                        
                        frames_saved_count += 1
                        if frames_saved_count == len(target_indices_set):
                            print(f"Info [Scene: {scene_id}]: All {len(target_indices_set)} requested depth frames extracted via per-frame decompression.")
                            return # All requested frames found and processed

                    else: # Not a target frame, seek past it
                        infile.seek(compressed_frame_size, 1) # 1 means relative to current position
                    
                    current_frame_id_in_file += 1

            if frames_saved_count < len(target_indices_set):
                print(f"Warning [Scene: {scene_id}]: Processed per-frame data, but only saved {frames_saved_count} out of {len(target_indices_set)} requested depth frames. Some might be missing or out of bounds.")

        except Exception as e_per_frame:
            print(f"Error [Scene: {scene_id}]: Per-frame depth extraction failed: {e_per_frame}")

# ---- END: New helper function ----

# ---- START: New helper function for specific RGB frame extraction ----
def _extract_and_save_specific_rgb_frames(
    iphone_video_path: Path,
    target_frame_indices: list[int], # These are 0-indexed
    color_out_dir: Path,
    target_image_size: tuple[int, int] | None, # (Height, Width)
    scene_id: str
):
    """
    Extracts, (optionally) resizes, and saves specific RGB frames from a video using OpenCV.
    """
    if not iphone_video_path.exists():
        print(f"Warning [Scene: {scene_id}]: iPhone video path {iphone_video_path} not found. Skipping RGB extraction.")
        return

    color_out_dir.mkdir(parents=True, exist_ok=True)
    # Sort and unique ensures we process frames in order and avoid redundant work.
    unique_sorted_indices = sorted(list(set(target_frame_indices)))

    if not unique_sorted_indices:
        print(f"Info [Scene: {scene_id}]: No specific RGB frames requested for extraction.")
        return

    print(f"Info [Scene: {scene_id}]: Starting extraction of {len(unique_sorted_indices)} specific RGB frames from {iphone_video_path} using OpenCV.")

    cap = cv2.VideoCapture(str(iphone_video_path))
    if not cap.isOpened():
        print(f"Error [Scene: {scene_id}]: Could not open video file {iphone_video_path} with OpenCV. Skipping RGB extraction.")
        return

    frames_extracted_count = 0
    # Consider adding tqdm here if it's available and you want a progress bar
    # for frame_idx in tqdm(unique_sorted_indices, desc=f"Extracting RGB frames for {scene_id}"):
    for frame_idx in unique_sorted_indices: # Using basic loop if tqdm is not critical path
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Warning [Scene: {scene_id}]: Could not read frame {frame_idx} from video {iphone_video_path}. It might be out of bounds or an error occurred.")
            continue

        # Construct output path
        output_rgb_file_path = color_out_dir / f"frame_{frame_idx:06}.jpg"

        try:
            if target_image_size: # target_image_size is (Height, Width)
                # cv2.resize expects (Width, Height)
                resized_frame = cv2.resize(frame, (target_image_size[1], target_image_size[0]), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(output_rgb_file_path), resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 95]) # Save with good quality
            else:
                cv2.imwrite(str(output_rgb_file_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frames_extracted_count += 1
        except Exception as e:
            print(f"Error [Scene: {scene_id}, Frame: {frame_idx}]: Failed to process or save frame: {e}")

    cap.release()

    if frames_extracted_count == len(unique_sorted_indices):
        print(f"Info [Scene: {scene_id}]: Successfully extracted {frames_extracted_count} RGB frames into {color_out_dir}.")
    else:
        print(f"Warning [Scene: {scene_id}]: Extracted {frames_extracted_count} out of {len(unique_sorted_indices)} requested RGB frames. Some frames may have been missed due to errors or being out of bounds.")

    print(f"Info [Scene: {scene_id}]: Finished extracting specific RGB frames for scene {scene_id} into {color_out_dir}")

# ---- END: New helper function for specific RGB frame extraction ----


def save_intrinsics_to_txt(camera_data, output_path, target_new_width=None, target_new_height=None):
    """
    Saves camera intrinsics to a TXT file.
    Assumes a single camera model for the iPhone data as is typical.
    Scales intrinsics if target_new_width and target_new_height are provided and different from original.
    """
    if not camera_data:
        print("Warning: No camera data found.")
        return

    # Typically, for iPhone data from COLMAP, there's one camera entry
    # Find the camera with the smallest ID or just the first one if IDs are not sequential integers
    # This handles cases where camera_data keys might not be simple integers like 1
    cam_id = min(camera_data.keys()) if camera_data else None
    if cam_id is None:
        print("Warning: Could not determine camera ID from camera_data.")
        return
    cam = camera_data[cam_id]

    original_fx = cam.params[0] if len(cam.params) > 0 else None
    original_fy = cam.params[1] if len(cam.params) > 1 else None
    original_cx = cam.params[2] if len(cam.params) > 2 else None
    original_cy = cam.params[3] if len(cam.params) > 3 else None
    original_width = cam.width
    original_height = cam.height

    current_fx, current_fy, current_cx, current_cy = original_fx, original_fy, original_cx, original_cy
    current_width, current_height = original_width, original_height

    if target_new_width is not None and target_new_height is not None and \
       (original_width != target_new_width or original_height != target_new_height):
        
        print(f"Scaling intrinsics from {original_width}x{original_height} to {target_new_width}x{target_new_height}")
        
        if original_width > 0 and original_height > 0:
            scale_x = target_new_width / original_width
            scale_y = target_new_height / original_height

            current_fx = original_fx * scale_x if original_fx is not None else None
            current_fy = original_fy * scale_y if original_fy is not None else None
            current_cx = original_cx * scale_x if original_cx is not None else None
            current_cy = original_cy * scale_y if original_cy is not None else None
            
            current_width = int(target_new_width)
            current_height = int(target_new_height)
        else:
            print(f"Warning: Original camera dimensions ({original_width}x{original_height}) are invalid for scaling. Intrinsics not scaled, but width/height will be updated.")
            current_width = int(target_new_width)
            current_height = int(target_new_height)
    else:
        # Ensure current_width and current_height are integers even if no scaling
        current_width = int(original_width)
        current_height = int(original_height)

    # 创建3x3内参矩阵
    intrinsics_matrix = np.array([
        [current_fx, 0, current_cx],
        [0, current_fy, current_cy],
        [0, 0, 1]
    ])
    
    # 保存为txt文件
    np.savetxt(output_path, intrinsics_matrix)
    print(f"保存内参矩阵到 {output_path}")


def save_pose_to_txt(pose_matrix, output_path):
    """Saves a 4x4 pose matrix to a text file."""
    np.savetxt(output_path, pose_matrix)


def process_scene(scene_id, data_root_path, output_base_path, device_type, num_frames_to_sample, target_image_size, split_name):
    """
    Processes a single scene: reads COLMAP intrinsics, samples frames,
    processes and saves per-frame pose, RGB, and (for iPhone) extracts specific depth images.
    """
    print(f"\nProcessing scene: {scene_id} ({device_type})")
    scene = ScannetppScene_Release(scene_id, data_root=data_root_path)

    # Fixed split name as per user's previous edits context
    # split_name = "train"

    # Define COLMAP directory and source paths based on device type
    if device_type == "iphone":
        colmap_dir = scene.iphone_colmap_dir
        scene_iphone_video_path = scene.iphone_video_path # Path to the video file
        scene_raw_depth_path = scene.iphone_depth_path

        if not scene_iphone_video_path or not scene_iphone_video_path.exists():
            print(f"Error [Scene: {scene_id}]: iPhone video path '{scene_iphone_video_path}' not found. Cannot extract RGB. Skipping scene.")
            return
        # Raw depth path existence is checked before its specific extraction logic later

    elif device_type == "dslr":
        colmap_dir = scene.dslr_colmap_dir
        # For DSLR, rgb_src_dir is where pre-processed (e.g. resized/undistorted) images are expected
        rgb_src_dir = scene.dslr_resized_dir # Used later in the loop for DSLR
        scene_raw_depth_path = None # No raw depth for DSLR in this flow
        scene_iphone_video_path = None # Not applicable for DSLR

        if not rgb_src_dir or not rgb_src_dir.exists():
             print(f"Error [Scene: {scene_id}]: DSLR RGB source directory ('{rgb_src_dir}') not found. Skipping scene.")
             return
        # print(f"Warning: For DSLR, depth data is typically rendered. This script expects pre-existing depth frames if 'depth_src_dir' is set.")
    else:
        print(f"Error: Unsupported device type '{device_type}'")
        return

    if not colmap_dir.exists():
        print(f"COLMAP directory not found for scene {scene_id} at {colmap_dir}. Skipping.")
        return
    # Removed generic rgb_src_dir.exists() check here, handled device-specifically above

    # if device_type == "iphone" and (not scene_raw_depth_path or not scene_raw_depth_path.exists()):
    #     print(f"Warning: iPhone raw depth file {scene_raw_depth_path} not found for scene {scene_id}. Depth frames will be skipped.")
    # This check is better placed right before depth extraction call


    # Create output directories for the current scene, organized by data type and split
    color_out_dir = output_base_path / "color" / split_name / scene_id
    depth_out_dir = output_base_path / "depth" / split_name / scene_id
    pose_out_dir = output_base_path / "pose" / split_name / scene_id
    intrinsic_split_dir = output_base_path / "intrinsic" / split_name

    color_out_dir.mkdir(parents=True, exist_ok=True)
    if device_type == "iphone":
        depth_out_dir.mkdir(parents=True, exist_ok=True)
    pose_out_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_split_dir.mkdir(parents=True, exist_ok=True)

    # Read COLMAP data
    try:
        cameras_colmap, images_colmap, _ = read_model(colmap_dir, ".txt")
    except FileNotFoundError:
        print(f"Could not read COLMAP model from {colmap_dir}. Ensure cameras.txt, images.txt, points3D.txt exist.")
        return
    except Exception as e:
        print(f"Error reading COLMAP model for {scene_id}: {e}")
        return


    # Save intrinsics
    intrinsic_file_path = intrinsic_split_dir / f"intrinsics_{scene_id}.txt"

    if target_image_size: # target_image_size is (HEIGHT, WIDTH)
        t_height, t_width = target_image_size
        save_intrinsics_to_txt(cameras_colmap, intrinsic_file_path, 
                                target_new_width=t_width, 
                                target_new_height=t_height)
    else:
        save_intrinsics_to_txt(cameras_colmap, intrinsic_file_path) # No scaling if target_image_size is None

    # Process each frame
    print(f"Found {len(images_colmap)} images in COLMAP for scene {scene_id}")

    # Sort image keys (IDs) to ensure deterministic sampling if order matters
    # COLMAP image IDs are typically integers.
    image_ids_sorted = sorted(images_colmap.keys())

    if not image_ids_sorted:
        print(f"Warning: No image IDs found in COLMAP for scene {scene_id}. Skipping frame processing.")
        return

    if num_frames_to_sample is not None and len(image_ids_sorted) > num_frames_to_sample:
        print(f"Sampling {num_frames_to_sample} frames from {len(image_ids_sorted)} available frames.")
        indices = np.linspace(0, len(image_ids_sorted) - 1, num_frames_to_sample, dtype=int)
        sampled_image_ids = [image_ids_sorted[i] for i in indices]
    else:
        print(f"Processing all {len(image_ids_sorted)} available frames (num_frames_to_sample={num_frames_to_sample}).")
        sampled_image_ids = image_ids_sorted
    
    print(f"Selected {len(sampled_image_ids)} frames for processing.")

    # ---- START: Parse target frame indices from COLMAP image names ----
    common_target_frame_indices = []
    if sampled_image_ids: # Ensure there are images from COLMAP
        for img_id_colmap in sampled_image_ids:
            img_data_colmap = images_colmap[img_id_colmap]
            try:
                # Assuming frame_name_stem is like "frame_000000"
                frame_idx_str = Path(img_data_colmap.name).stem.split('_')[-1]
                common_target_frame_indices.append(int(frame_idx_str))
            except ValueError:
                print(f"Warning [Scene: {scene_id}]: Could not parse frame index from {img_data_colmap.name}. Skipping this frame index.")
            except IndexError:
                print(f"Warning [Scene: {scene_id}]: Could not parse frame index from {img_data_colmap.name} (unexpected format). Skipping this frame index.")
        
        common_target_frame_indices = sorted(list(set(common_target_frame_indices)))
        print(f"Info [Scene: {scene_id}]: Parsed {len(common_target_frame_indices)} unique target frame indices from COLMAP data.")
    else:
        print(f"Warning [Scene: {scene_id}]: No COLMAP images sampled, cannot determine target frame indices.")
    # ---- END: Parse target frame indices ----

    # ---- START: New logic for iPhone RGB extraction ----
    if device_type == "iphone":
        if not scene_iphone_video_path: # Should have been caught earlier, but double check
            print(f"Critical Warning [Scene: {scene_id}]: iPhone video path is not set. Cannot extract RGB.")
        elif not common_target_frame_indices:
            print(f"Warning [Scene: {scene_id}]: No target frame indices from COLMAP. Skipping RGB frame extraction.")
        else:
            _extract_and_save_specific_rgb_frames(
                iphone_video_path=scene_iphone_video_path,
                target_frame_indices=common_target_frame_indices,
                color_out_dir=color_out_dir,
                target_image_size=target_image_size,
                scene_id=scene_id
            )
    # ---- END: New logic for iPhone RGB extraction ----

    # ---- START: Modified logic for iPhone depth extraction ----
    if device_type == "iphone":
        if not scene_raw_depth_path or not scene_raw_depth_path.exists():
            print(f"Warning [Scene: {scene_id}]: iPhone raw depth file '{scene_raw_depth_path}' not found. Depth frames will be skipped.")
        elif not common_target_frame_indices:
             print(f"Warning [Scene: {scene_id}]: No target frame indices from COLMAP. Skipping depth frame extraction.")
        else:
            original_iphone_depth_height = 192
            original_iphone_depth_width = 256
            _extract_and_save_specific_depth_frames(
                scene_iphone_depth_path=scene_raw_depth_path,
                target_frame_indices=common_target_frame_indices,
                depth_out_dir=depth_out_dir,
                original_depth_height=original_iphone_depth_height,
                original_depth_width=original_iphone_depth_width,
                target_image_size=target_image_size,
                scene_id=scene_id
            )
    # ---- END: Modified logic for iPhone depth extraction ----


    for img_id in tqdm(sampled_image_ids, desc="Processing sampled frames"):
        img_data = images_colmap[img_id]
        frame_name_stem = Path(img_data.name).stem # e.g., "frame_000000"

        # 1. Pose
        # COLMAP provides world-to-camera (w2c)
        # R = qvec2rotmat(img_data.qvec)
        # t = img_data.tvec
        # w2c_matrix = np.eye(4)
        # w2c_matrix[:3, :3] = R
        # w2c_matrix[:3, 3] = t
        # We usually want camera-to-world (c2w) for pose
        # c2w_matrix = np.linalg.inv(w2c_matrix)

        # The Image object from common.utils.colmap has a world_to_camera property
        # This is the one from the Colmap authors' original script.
        # The ScannetPP codebase also includes semantic.utils.colmap_utils.py
        # which might have slightly different Image class. Let's ensure we use
        # the one intended for read_model here. The `world_to_camera` property
        # in `common.utils.colmap.Image` should be correct.
        if hasattr(img_data, 'world_to_camera'): # Check if it's the full Image object
            w2c_matrix = img_data.world_to_camera
            c2w_matrix = np.linalg.inv(w2c_matrix)
        else: # Fallback if it's the BaseImage namedtuple (less likely with read_model)
            R = qvec2rotmat(img_data.qvec)
            t = img_data.tvec
            w2c_matrix_fallback = np.eye(4)
            w2c_matrix_fallback[:3, :3] = R
            w2c_matrix_fallback[:3, 3] = t
            c2w_matrix = np.linalg.inv(w2c_matrix_fallback)


        pose_file_path = pose_out_dir / f"{frame_name_stem}.txt"
        save_pose_to_txt(c2w_matrix, pose_file_path)

        # 2. RGB image (now 'color')
        frame_name_stem = Path(img_data.name).stem # e.g., "frame_000000"
        output_rgb_file_path = color_out_dir / img_data.name # e.g., color_out_dir/frame_000000.jpg

        if device_type == "iphone":
            # For iPhone, RGB frames are extracted by _extract_and_save_specific_rgb_frames.
            # We just check if the file was created successfully.
            if not output_rgb_file_path.exists():
                print(f"Warning [Scene: {scene_id}, Frame: {frame_name_stem}]: Expected RGB file {output_rgb_file_path} not found after extraction attempt.")
        elif device_type == "dslr":
            # For DSLR, continue with the existing logic of copying and resizing from rgb_src_dir.
            # rgb_src_dir for DSLR was defined as scene.dslr_resized_dir earlier.
            dslr_rgb_src_file = rgb_src_dir / img_data.name 
            if dslr_rgb_src_file.exists():
                try:
                    color_image = cv2.imread(str(dslr_rgb_src_file))
                    if color_image is None:
                        raise ValueError(f"cv2.imread failed to load {dslr_rgb_src_file}")
                    
                    # output_rgb_file_path is already defined correctly for the destination
                    if target_image_size:
                        color_image_resized = cv2.resize(color_image, (target_image_size[1], target_image_size[0]), interpolation=cv2.INTER_LINEAR_EXACT)
                        cv2.imwrite(str(output_rgb_file_path), color_image_resized)
                    else:
                        cv2.imwrite(str(output_rgb_file_path), color_image) # Save original if no resize, effectively a copy if paths differ
                except Exception as e:
                    print(f"Warning [Scene: {scene_id}, Frame: {frame_name_stem}]: Error processing/saving DSLR color image {dslr_rgb_src_file}: {e}. Skipping color.")
            else:
                print(f"Warning [Scene: {scene_id}, Frame: {frame_name_stem}]: DSLR Color file {dslr_rgb_src_file} not found.")

        # 3. Depth image
        if device_type == "iphone":
            # Depth frames should have been extracted by _extract_and_save_specific_depth_frames already
            # We just verify if the file exists in the output path.
            # The processing (like resizing) is handled within that function.
            depth_file_name = f"{frame_name_stem}.png"
            output_depth_file_path = depth_out_dir / depth_file_name
            if not output_depth_file_path.exists():
                print(f"Warning: Expected depth file {output_depth_file_path} was not found after extraction attempt for frame {frame_name_stem} in scene {scene_id}.")
        
        elif device_type == "dslr":
            # DSLR depth handling remains as placeholder or relies on pre-existing rendered depth
            # If you had a depth_src_dir for DSLR, the old logic would apply here.
            # For now, we assume no direct depth processing for DSLR in this part of the loop.
            pass 

    log_message_parts = [f"Finished processing scene {scene_id}."]
    log_message_parts.append(f"Color data in: {color_out_dir.relative_to(output_base_path) if output_base_path in color_out_dir.parents else color_out_dir}")
    if device_type == "iphone" and depth_out_dir.exists():
        log_message_parts.append(f"Depth data in: {depth_out_dir.relative_to(output_base_path) if output_base_path in depth_out_dir.parents else depth_out_dir}")
    log_message_parts.append(f"Pose data in: {pose_out_dir.relative_to(output_base_path) if output_base_path in pose_out_dir.parents else pose_out_dir}")
    log_message_parts.append(f"Intrinsic data in: {intrinsic_split_dir.relative_to(output_base_path) if output_base_path in intrinsic_split_dir.parents else intrinsic_split_dir}")
    log_message_parts.append(f"Using split '{split_name}' under base {output_base_path}")
    print(" ".join(log_message_parts))


def main():
    parser = argparse.ArgumentParser(
        description="Extracts COLMAP intrinsics, per-frame poses, RGB, and Depth data for ScanNet++ scenes."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of the ScanNet++ dataset (e.g., '/path/to/scannetpp/data')."
    )
    parser.add_argument(
        "--scene_list_file",
        type=str,
        required=True,
        help="Path to a text file containing a list of scene IDs to process (one ID per line)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base directory where the processed data will be saved."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="iphone",
        choices=["iphone", "dslr"],
        help="Device type to process (iphone or dslr). Default is iphone."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=32,
        help="Number of frames to uniformly sample per scene. Processes all if None or if total frames < num_frames. (default: 32)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        metavar=('HEIGHT', 'WIDTH'),
        default=[480, 640], # Default to 480x640 (Height, Width)
        help="Target image size (height width) to resize color and depth images to. (default: 480 640)"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1), # Default to num_cpu - 1
        help="Number of parallel processes to use for scene processing. (default: CPU count - 1)"
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="train",
        help="Split name to process (train or val). (default: train)"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        print(f"Error: Data root directory '{data_root}' does not exist.")
        return

    try:
        scene_ids = read_txt_list(args.scene_list_file)
    except FileNotFoundError:
        print(f"Error: Scene list file '{args.scene_list_file}' not found.")
        return

    print(f"Found {len(scene_ids)} scenes to process from {args.scene_list_file}")
    print(f"Using {args.num_processes} parallel processes.")

    # Prepare arguments for each process_scene call
    tasks = []
    for scene_id in scene_ids:
        # Ensure image_size is a tuple if provided, or None
        img_size_tuple = tuple(args.image_size) if args.image_size else None
        tasks.append((scene_id.strip(), data_root, output_dir, args.device, args.num_frames, img_size_tuple, args.split_name))

    # Use a process pool to parallelize scene processing
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        # Using starmap to pass multiple arguments to process_scene
        # Wrap with tqdm for a progress bar over the tasks
        list(tqdm(pool.starmap(process_scene, tasks), total=len(tasks), desc="Processing scenes"))

    print("\nAll scenes processed.")
    print(f"Output data saved in: {output_dir}")

if __name__ == "__main__":
    main()
    # from .export_sampled_frames import verify_scene_reprojection
    # verify_scene_reprojection(
    #     scene_id="fe1733741f",
    #     output_dir='data/processed_data/vsibench_points/ScanNetpp',
    #     split_name="train",
    #     verification_output_dir="verification_output"
    # )
    