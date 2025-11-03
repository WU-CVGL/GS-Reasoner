# -----------------------------------------------------------------------------
# This file is modified from:
# https://github.com/VITA-Group/VLM-3R/blob/main/vlm_3r_data_process/src/metadata_generation/ScanNet/preprocess/export_sampled_frames.py
#
# Original license: MIT License
# Copyright (c) 2025 VLM-3R
# -----------------------------------------------------------------------------


import argparse
import os
import sys
import cv2  # Requires opencv-python
import numpy as np
from tqdm import tqdm  # Requires tqdm
import glob
import concurrent.futures
import math
import png # Requires pypng
import zipfile # Added for zip file handling
import shutil # Added for file copying

# Assuming SensorData is in the same directory or PYTHONPATH is set
try:
    from .SensorData import SensorData
except ImportError as e:
    print(f"Error: Could not import SensorData class: {e}")
    # Attempt import from parent directory if running from preprocess/scannet
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from SensorData import SensorData
        print("Successfully imported SensorData from parent directory.")
    except ImportError:
        # If SensorData.py is in the preprocess/scannet directory itself
        try:
            from SensorData import SensorData
            print("Successfully imported SensorData from current directory.")
        except ImportError:
            print("Error: SensorData class not found even after checking parent/current directories.")
            sys.exit(1)

# Helper function to parse scene_id.txt
def parse_scene_meta_file(filename):
    """Parses the scene_id.txt file to extract axis alignment and depth intrinsics."""
    metadata = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split(' = ')
                if len(parts) == 2:
                    key, value = parts
                    metadata[key.strip()] = value.strip()

        # Extract and reshape axis alignment matrix
        axis_align_str = metadata.get('axisAlignment')
        if axis_align_str:
            axis_align_vals = list(map(float, axis_align_str.split()))
            if len(axis_align_vals) == 16:
                axis_align_matrix = np.array(axis_align_vals).reshape(4, 4)
            else:
                print(f"Warning: Invalid number of values for axisAlignment in {filename}. Expected 16, got {len(axis_align_vals)}.")
                axis_align_matrix = None
        else:
            print(f"Warning: axisAlignment key not found in {filename}.")
            axis_align_matrix = None

        # Construct depth intrinsics matrix
        try:
            fx = float(metadata['fx_depth'])
            fy = float(metadata['fy_depth'])
            mx = float(metadata['mx_depth'])
            my = float(metadata['my_depth'])
            depth_intrinsics = np.array([
                [fx, 0, mx, 0],
                [0, fy, my, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        except KeyError as e:
            print(f"Warning: Missing depth intrinsic key {e} in {filename}.")
            depth_intrinsics = None
        except ValueError as e:
            print(f"Warning: Invalid value for depth intrinsic key in {filename}: {e}.")
            depth_intrinsics = None

        return axis_align_matrix, depth_intrinsics

    except FileNotFoundError:
        print(f"Warning: Scene metadata file not found: {filename}")
        return None, None
    except Exception as e:
        print(f"Warning: Error parsing scene metadata file {filename}: {e}")
        return None, None

def save_matrix_to_file(matrix, filename):
    """Saves a numpy matrix to a text file."""
    with open(filename, 'w') as f:
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')

def export_scene_sampled_frames(sens_file_path, output_base_dir, num_frames_to_sample, split, image_size=None, min_valid_components_per_frame_initial=0, min_valid_frames_per_scene=1):
    """Exports uniformly sampled pose, depth, color, instance mask, and intrinsics for a single scene, applying axis alignment to poses."""
    scene_id = os.path.basename(os.path.dirname(sens_file_path))
    scene_dir_path = os.path.dirname(sens_file_path)
    print(f"Processing scene: {scene_id} (split: {split})")
    print(f'Loading {sens_file_path}...')
    try:
        sd = SensorData(sens_file_path)
    except NameError:
        print("Error: SensorData class failed to be imported.")
        return None 
    except Exception as e:
        print(f"Error loading SensorData from {sens_file_path}: {e}")
        return None 
    print(f'Loaded {len(sd.frames)} frames.')

    if not hasattr(sd, 'frames') or not sd.frames:
        print("Error: SensorData object does not contain frames or is empty.")
        return None
    
    total_raw_frames = len(sd.frames)
    if total_raw_frames == 0:
        print(f"Scene {scene_id} has 0 frames. Skipping.")
        return scene_id 

    # --- Locate additional input files ---
    scene_meta_path = os.path.join(scene_dir_path, f"{scene_id}.txt")
    instance_zip_path = os.path.join(scene_dir_path, f"{scene_id}_2d-instance-filt.zip")
    if not os.path.exists(instance_zip_path):
         instance_zip_path = os.path.join(scene_dir_path, f"{scene_id}_2d-instance.zip")

    instance_zip = None
    if os.path.exists(instance_zip_path):
        try:
            instance_zip = zipfile.ZipFile(instance_zip_path, 'r')
            print(f"Opened instance mask zip: {instance_zip_path}")
        except zipfile.BadZipFile:
            print(f"Warning: Bad zip file: {instance_zip_path}. Instance masks will be unavailable.")
            instance_zip = None
        except Exception as e:
            print(f"Warning: Error opening zip file {instance_zip_path}: {e}. Instance masks will be unavailable.")
            instance_zip = None
    else:
        print(f"Warning: Instance mask zip file not found at {instance_zip_path} or fallback. Instance masks will be unavailable.")

    # --- Phase 1: Validate all available frames to create a list of candidates ---
    print(f"Validating all {total_raw_frames} available frames for scene {scene_id}...")
    valid_frame_candidates_info = [] 
    
    for original_idx in tqdm(range(total_raw_frames), desc=f"Validating raw frames for {scene_id}", leave=False):
        try:
            frame = sd.frames[original_idx]
            num_available_components = 0

            if hasattr(frame, 'camera_to_world') and frame.camera_to_world is not None:
                num_available_components += 1
            
            if hasattr(frame, 'decompress_depth') and hasattr(sd, 'depth_compression_type') and sd.depth_compression_type.lower() != 'unknown':
                num_available_components += 1

            if hasattr(frame, 'decompress_color') and hasattr(sd, 'color_compression_type') and sd.color_compression_type.lower() == 'jpeg':
                num_available_components += 1
            
            instance_mask_potentially_available = False
            if instance_zip:
                mask_filename_in_zip = f'instance-filt/{original_idx}.png' 
                try:
                    instance_zip.getinfo(mask_filename_in_zip) 
                    instance_mask_potentially_available = True
                except KeyError:
                    instance_mask_potentially_available = False 
                except Exception: 
                    instance_mask_potentially_available = False
            
            if instance_mask_potentially_available:
                num_available_components +=1

            if num_available_components >= min_valid_components_per_frame_initial:
                 valid_frame_candidates_info.append({"original_idx": original_idx})

        except IndexError:
            print(f"Warning: Raw frame index {original_idx} out of bounds during validation for scene {scene_id}.")
            continue 
        except Exception as e_val:
            print(f"Warning: Error validating raw frame {original_idx} for scene {scene_id}: {e_val}. Skipping candidate.")
            continue
            
    if not valid_frame_candidates_info:
        print(f"No valid frame candidates found for scene {scene_id} after initial validation. Skipping scene processing.")
        if instance_zip: instance_zip.close()
        return None 

    num_candidates = len(valid_frame_candidates_info)
    print(f"Found {num_candidates} valid frame candidates for scene {scene_id}.")

    # --- Sampling Logic based on validated candidates ---
    if num_candidates <= num_frames_to_sample:
        selected_candidate_indices = np.arange(num_candidates)
    else:
        selected_candidate_indices = np.linspace(0, num_candidates - 1, num_frames_to_sample, dtype=int)
        selected_candidate_indices = np.unique(selected_candidate_indices)

    actual_indices_to_process = [valid_frame_candidates_info[i]["original_idx"] for i in selected_candidate_indices]
    if not actual_indices_to_process: # Should not happen if valid_frame_candidates_info is not empty
        print(f"No frames selected for processing for scene {scene_id} after sampling (num_candidates: {num_candidates}, num_frames_to_sample: {num_frames_to_sample}). Skipping.")
        if instance_zip: instance_zip.close()
        return None
    print(f"Selected {len(actual_indices_to_process)} frames for export. First few original indices: {actual_indices_to_process[:10]}...")


    # --- Create output directories ---
    pose_output_dir = os.path.join(output_base_dir, 'pose', split, scene_id)
    depth_output_dir = os.path.join(output_base_dir, 'depth', split, scene_id)
    color_output_dir = os.path.join(output_base_dir, 'color', split, scene_id)
    instance_output_dir = os.path.join(output_base_dir, 'instance', split, scene_id)
    intrinsic_base_output_dir = os.path.join(output_base_dir, 'intrinsic', split)
    
    os.makedirs(pose_output_dir, exist_ok=True)
    os.makedirs(depth_output_dir, exist_ok=True)
    os.makedirs(color_output_dir, exist_ok=True)
    os.makedirs(instance_output_dir, exist_ok=True)
    os.makedirs(intrinsic_base_output_dir, exist_ok=True)

    # --- Process scene-level data ---
    axis_align_matrix, intrinsics_matrix = parse_scene_meta_file(scene_meta_path)
    if intrinsics_matrix is not None:
        intrinsic_out_filename = os.path.join(intrinsic_base_output_dir, f'intrinsic_depth_{scene_id}.txt')
        save_matrix_to_file(intrinsics_matrix, intrinsic_out_filename)
        print(f"Saved intrinsics to {intrinsic_out_filename}")
    else:
        print(f"Warning: Could not read/parse or save intrinsics for scene {scene_id} from {scene_meta_path}.")

    if axis_align_matrix is None:
        print(f"Warning: Could not read/parse axis alignment matrix for scene {scene_id} from {scene_meta_path}. Poses will NOT be aligned.")

    # --- Phase 2: Process and Save Selected Frames ---
    processed_attempt_count = 0 
    final_valid_frames_count = 0 
    total_skipped_components_in_export = 0 
    min_successful_components_per_frame_export = min_valid_components_per_frame_initial
    
    try:
        for original_idx in tqdm(actual_indices_to_process, desc=f"Exporting selected frames for {scene_id}", leave=False):
            try:
                frame = sd.frames[original_idx] 
                components_skipped_this_frame_export = 0
                
                pose_saved_path = None
                depth_saved_path = None
                color_saved_path = None
                instance_saved_path = None
                color_export_failed_critically = False

                # Export Pose
                if hasattr(frame, 'camera_to_world') and frame.camera_to_world is not None and np.isnan(frame.camera_to_world).sum() == 0 and np.isinf(frame.camera_to_world).sum() == 0:
                    pose_matrix_original = frame.camera_to_world
                    pose_filename = os.path.join(pose_output_dir, f'{original_idx:06d}.txt')
                    if axis_align_matrix is not None:
                        try:
                            aligned_pose = np.dot(axis_align_matrix, pose_matrix_original)
                            save_matrix_to_file(aligned_pose, pose_filename)
                            pose_saved_path = pose_filename
                        except ValueError as e:
                            print(f"Warning: Error applying axis alignment for frame {original_idx}: {e}. Saving original.")
                            save_matrix_to_file(pose_matrix_original, pose_filename)
                            pose_saved_path = pose_filename 
                        except Exception as e:
                            print(f"Warning: Unexpected error during pose alignment/saving for frame {original_idx}: {e}. Skip pose comp.")
                            components_skipped_this_frame_export += 1
                    else:
                        save_matrix_to_file(pose_matrix_original, pose_filename)
                        pose_saved_path = pose_filename
                else:
                    print(f"Warning: Pose data not found for pre-validated frame {original_idx} during export. Skip pose comp.")
                    total_skipped_components_in_export += 4 
                    processed_attempt_count += 1
                    continue
                    # components_skipped_this_frame_export += 1

                # Export Depth
                if hasattr(frame, 'decompress_depth') and hasattr(sd, 'depth_compression_type') and sd.depth_compression_type.lower() != 'unknown':
                    depth_data = frame.decompress_depth(sd.depth_compression_type)
                    if depth_data is not None:
                        try:
                            depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape(sd.depth_height, sd.depth_width)
                            depth_filename = os.path.join(depth_output_dir, f'{original_idx:06d}.png')
                            depth_image_reshaped = depth_image.reshape(-1, depth_image.shape[1]).tolist()
                            with open(depth_filename, 'wb') as f_png:
                                writer = png.Writer(width=depth_image.shape[1], height=depth_image.shape[0], bitdepth=16, greyscale=True)
                                writer.write(f_png, depth_image_reshaped)
                            depth_saved_path = depth_filename
                        except (ValueError, AttributeError, Exception) as e: # Catch specific and general errors
                            print(f"Warning: Error processing/writing depth for frame {original_idx}: {e}. Skip depth comp.")
                            components_skipped_this_frame_export += 1
                    else:
                        print(f"Warning: Decompressed null depth data for frame {original_idx}. Skip depth comp.")
                        components_skipped_this_frame_export += 1
                else:
                    print(f"Warning: Depth data/decompression not available for pre-validated frame {original_idx}. Skip depth comp.")
                    components_skipped_this_frame_export += 1

                # Export Color
                if hasattr(frame, 'decompress_color') and hasattr(sd, 'color_compression_type') and sd.color_compression_type.lower() == 'jpeg':
                    color_image = frame.decompress_color(sd.color_compression_type)
                    if color_image is not None:
                        try:
                            color_filename = os.path.join(color_output_dir, f'{original_idx:06d}.jpg')
                            if image_size:
                                color_image_resized = cv2.resize(color_image, (image_size[1], image_size[0]))
                            else:
                                color_image_resized = color_image
                            color_image_bgr = cv2.cvtColor(color_image_resized, cv2.COLOR_RGB2BGR)
                            if not cv2.imwrite(color_filename, color_image_bgr):
                                raise ValueError(f"cv2.imwrite failed for color {color_filename}")
                            color_saved_path = color_filename
                        except Exception as e:
                            print(f"Warning: Error saving color image for frame {original_idx}: {e}. CRITICAL: Skip color & FRAME.")
                            components_skipped_this_frame_export += 1
                            color_export_failed_critically = True
                    else:
                        print(f"Warning: Decompressed null color data for frame {original_idx}. CRITICAL: Skip color & FRAME.")
                        components_skipped_this_frame_export += 1
                        color_export_failed_critically = True
                else:
                    print(f"Warning: Color data/decompression not JPEG for pre-validated frame {original_idx}. CRITICAL: Skip color & FRAME.")
                    components_skipped_this_frame_export += 1
                    color_export_failed_critically = True

                if color_export_failed_critically:
                    if pose_saved_path and os.path.exists(pose_saved_path): os.remove(pose_saved_path)
                    if depth_saved_path and os.path.exists(depth_saved_path): os.remove(depth_saved_path)
                    total_skipped_components_in_export += (4 - components_skipped_this_frame_export) # Add remaining potential skips
                    processed_attempt_count += 1
                    continue 

                # Export Instance Mask
                if instance_zip:
                    mask_filename_in_zip = f'instance-filt/{original_idx}.png'
                    instance_output_filename = os.path.join(instance_output_dir, f'{original_idx:06d}.png')
                    try:
                        with instance_zip.open(mask_filename_in_zip, 'r') as mask_file:
                            mask_data = mask_file.read()
                        instance_mask = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_UNCHANGED)
                        if instance_mask is None: raise ValueError(f"cv2.imdecode failed for {mask_filename_in_zip}")
                        if image_size:
                            target_h, target_w = image_size
                            instance_mask = cv2.resize(instance_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                        if not cv2.imwrite(instance_output_filename, instance_mask):
                            raise ValueError(f"cv2.imwrite failed for instance {instance_output_filename}")
                        instance_saved_path = instance_output_filename
                    except KeyError:
                        print(f"Warning: Instance mask {mask_filename_in_zip} not in zip for frame {original_idx}. Skip instance comp.")
                        components_skipped_this_frame_export += 1
                    except Exception as e:
                        print(f"Warning: Error processing instance mask {mask_filename_in_zip} for frame {original_idx}: {e}. Skip instance comp.")
                        components_skipped_this_frame_export += 1
                elif not instance_zip: # Instance masks were not available for the scene
                    pass # Not a skip for this frame if unavailable for scene, handled by num_successfully_exported

                processed_attempt_count += 1
                total_skipped_components_in_export += components_skipped_this_frame_export

                num_successfully_exported_components = 0
                if pose_saved_path: num_successfully_exported_components +=1
                if depth_saved_path: num_successfully_exported_components +=1
                if color_saved_path: num_successfully_exported_components +=1
                if instance_saved_path: num_successfully_exported_components +=1
                
                if num_successfully_exported_components >= min_successful_components_per_frame_export:
                    final_valid_frames_count += 1
                else:
                    print(f"Info: Frame {original_idx} did not meet min successful components ({num_successfully_exported_components}/{min_successful_components_per_frame_export}). Cleaning up.")
                    if pose_saved_path and os.path.exists(pose_saved_path): os.remove(pose_saved_path)
                    if depth_saved_path and os.path.exists(depth_saved_path): os.remove(depth_saved_path)
                    if color_saved_path and os.path.exists(color_saved_path): os.remove(color_saved_path)
                    if instance_saved_path and os.path.exists(instance_saved_path): os.remove(instance_saved_path)

            except IndexError: 
                print(f"Warning: Frame index {original_idx} out of bounds for scene {scene_id} during export. Critical error.")
                break 
            except Exception as e:
                print(f"Warning: Unhandled error processing selected frame {original_idx} for scene {scene_id}: {e}. Skipping frame.")
                total_skipped_components_in_export += 4 
                processed_attempt_count += 1
                continue
    finally:
        if instance_zip:
            instance_zip.close()
            print(f"Closed instance mask zip for {scene_id}")

    scene_export_successful = False
    if total_raw_frames == 0:
        scene_export_successful = True 
    elif processed_attempt_count > 0 and final_valid_frames_count >= min_valid_frames_per_scene:
        scene_export_successful = True
    elif not actual_indices_to_process and total_raw_frames > 0 : 
        scene_export_successful = False 
    else: 
        scene_export_successful = False
        
    if scene_export_successful:
        print(f'Finished exporting for scene {scene_id} (split: {split}).')
        print(f'Initial valid candidates: {num_candidates}. Selected for processing: {len(actual_indices_to_process)}.')
        print(f'Attempted export for {processed_attempt_count} selected frames.')
        print(f'Number of finally valid frames (>= {min_successful_components_per_frame_export} components exported): {final_valid_frames_count} (required >= {min_valid_frames_per_scene}).')
        if total_skipped_components_in_export > 0:
            print(f'Skipped {total_skipped_components_in_export} components during export phase.')
        return scene_id
    else:
        print(f"Failed to export scene {scene_id} (split: {split}).")
        print(f"  Initial valid candidates: {num_candidates}. Selected for processing: {len(actual_indices_to_process)}.")
        print(f"  Attempted export for {processed_attempt_count} selected frames.")
        print(f"  Number of finally valid frames (>= {min_successful_components_per_frame_export} components exported): {final_valid_frames_count} (required >= {min_valid_frames_per_scene}).")
        if not valid_frame_candidates_info and total_raw_frames > 0:
            print("  Reason: No frame candidates passed initial validation.")
        elif not actual_indices_to_process and valid_frame_candidates_info:
             print("  Reason: No frames were selected from candidates for processing.")
        elif final_valid_frames_count < min_valid_frames_per_scene and processed_attempt_count > 0 :
             print(f"  Reason: Number of finally valid frames ({final_valid_frames_count}) is less than the required minimum ({min_valid_frames_per_scene}).")
        return None

def main():
    parser = argparse.ArgumentParser(description="Export uniformly sampled pose (axis-aligned), depth, color, and instance masks from ScanNet .sens files, along with intrinsics, organised by train/val splits.")
    parser.add_argument('--scans_dir', required=True, help='Path to the directory containing scene subdirectories (e.g., data/scannet/scans)')
    parser.add_argument('--output_dir', required=True, help='Path to the base directory where output train/val folders (containing pose/depth/color/instance_mask/intrinsic) will be saved')
    parser.add_argument('--train_val_splits_path', type=str, default=None, help='Path to the directory containing scannetv2_train.txt and scannetv2_val.txt (optional). If provided, scenes will be sorted into train/val subfolders.')
    parser.add_argument('--num_frames', type=int, default=32, help='Number of frames to uniformly sample (default: 32)')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of processes to use for parallel processing (default: number of cores)')
    parser.add_argument('--image_size', type=int, nargs=2, metavar=('HEIGHT', 'WIDTH'), default=None, help='Target image size (height width) to resize color and instance images to (default: None)')
    parser.add_argument('--skip_existing', action='store_true', help='Skip processing scenes if their ID is found in the corresponding successful_scenes_<split>.txt in the output directory, or if output data directories exist.')
    parser.add_argument('--scene_list_file', type=str, default=None, help='Path to a text file containing a list of scene IDs to process, one per line. If provided, --scans_dir is still used to locate these scenes.')
    parser.add_argument('--min_valid_components_per_frame', type=int, default=0, help='Minimum number of components (pose, depth, color, instance) that must be available for a frame to be a candidate for sampling, AND successfully exported for a frame to be considered valid post-export. Default 0.')
    parser.add_argument('--min_valid_frames_per_scene', type=int, default=1, help='Minimum number of valid frames required for a scene to be considered successfully processed. Default 1.')


    opt = parser.parse_args()
    print("Script Options:")
    print(vars(opt)) 

    if opt.image_size and len(opt.image_size) != 2:
        print("Error: --image_size requires two arguments: HEIGHT WIDTH")
        sys.exit(1)
    if opt.image_size:
        print(f"Color images will be resized to Height={opt.image_size[0]}, Width={opt.image_size[1]}")

    train_scenes = set()
    val_scenes = set()
    if opt.train_val_splits_path:
        train_file_path = os.path.join(opt.train_val_splits_path, 'scannetv2_train.txt')
        val_file_path = os.path.join(opt.train_val_splits_path, 'scannetv2_val.txt')
        try:
            with open(train_file_path, 'r') as f:
                train_scenes = set(line.strip() for line in f if line.strip())
            print(f"Loaded {len(train_scenes)} train scene IDs from {train_file_path}")
        except FileNotFoundError:
            print(f"Warning: Train split file not found: {train_file_path}.")
        except Exception as e:
            print(f"Warning: Error reading train split file {train_file_path}: {e}")

        try:
            with open(val_file_path, 'r') as f:
                val_scenes = set(line.strip() for line in f if line.strip())
            print(f"Loaded {len(val_scenes)} val scene IDs from {val_file_path}")
        except FileNotFoundError:
            print(f"Warning: Validation split file not found: {val_file_path}.")
        except Exception as e:
            print(f"Warning: Error reading validation split file {val_file_path}: {e}")

        if not train_scenes and not val_scenes:
             print("Warning: Train/Val split path provided, but failed to load any scenes from the split files.")
    else:
        print("Train/Val split path not provided. Scenes will not be assigned to train/val splits and processing might be skipped depending on skip logic.")

    if not os.path.exists(opt.output_dir):
        print(f"Creating output directory: {opt.output_dir}")
        os.makedirs(opt.output_dir, exist_ok=True)

    successful_scenes_list_files = {
        'train': os.path.join(opt.output_dir, 'successful_scenes_train.txt'),
        'val': os.path.join(opt.output_dir, 'successful_scenes_val.txt')
    }
    existing_successful_scenes = {'train': set(), 'val': set()}

    if opt.skip_existing:
        for split_key, list_file in successful_scenes_list_files.items(): # Use split_key consistently
             if os.path.exists(list_file):
                try:
                    with open(list_file, 'r') as f:
                        existing_successful_scenes[split_key] = set(line.strip() for line in f if line.strip())
                    print(f"Loaded {len(existing_successful_scenes[split_key])} previously successful {split_key} scene IDs from {list_file}")
                except Exception as e:
                    print(f"Warning: Could not read {list_file}: {e}. Proceeding without skipping based on this list for {split_key} split.")

    scenes_to_consider = []
    if opt.scene_list_file:
        if not os.path.exists(opt.scene_list_file):
            print(f"Error: Scene list file not found: {opt.scene_list_file}")
            sys.exit(1)
        if not opt.scans_dir: # scans_dir is needed to locate the data
            print(f"Error: --scans_dir must be provided when using --scene_list_file.")
            sys.exit(1)
        try:
            with open(opt.scene_list_file, 'r') as f:
                target_scene_ids = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(target_scene_ids)} scene IDs from {opt.scene_list_file}")
            for scene_id_val in target_scene_ids: # Use scene_id_val to avoid conflict
                scene_dir_path = os.path.join(opt.scans_dir, scene_id_val)
                if os.path.isdir(scene_dir_path):
                    scenes_to_consider.append(scene_dir_path)
                else:
                    print(f"Warning: Scene directory for ID '{scene_id_val}' not found at {scene_dir_path}. Skipping.")
        except Exception as e:
            print(f"Error reading scene list file {opt.scene_list_file}: {e}")
            sys.exit(1)
    else:
        if not opt.scans_dir:
            print(f"Error: --scans_dir must be provided if --scene_list_file is not used.")
            sys.exit(1)
        try:
            scenes_to_consider = [d.path for d in os.scandir(opt.scans_dir) if d.is_dir()]
        except FileNotFoundError:
            print(f"Error: Scans directory not found: {opt.scans_dir}")
            sys.exit(1)
        except Exception as e:
            print(f"Error scanning directory {opt.scans_dir}: {e}")
            sys.exit(1)

    if not scenes_to_consider:
        print(f"No scene directories found or specified to process. Exiting.")
        sys.exit(0)

    print(f"Found {len(scenes_to_consider)} potential scene directories to evaluate.")

    total_scenes_found = len(scenes_to_consider) 
    successful_scenes_this_run = {'train': [], 'val': []} 
    scenes_to_process_args_list = [] 

    skipped_due_to_sens_or_skip_flag = 0
    skipped_due_to_no_split = 0
    for scene_dir_path in scenes_to_consider:
        current_scene_id = os.path.basename(scene_dir_path) # Renamed to avoid conflict
        sens_file_path = os.path.join(scene_dir_path, f"{current_scene_id}.sens")
        
        if not os.path.exists(sens_file_path):
            print(f"Warning: .sens file not found for scene {current_scene_id}. Skipping.")
            skipped_due_to_sens_or_skip_flag += 1
            continue

        current_split = None # Renamed to avoid conflict
        if opt.train_val_splits_path:
            if current_scene_id in train_scenes:
                current_split = 'train'
            elif current_scene_id in val_scenes:
                current_split = 'val'
            else:
                skipped_due_to_no_split += 1
                continue 
        else:
             print(f"Warning: --train_val_splits_path not provided. Skipping scene {current_scene_id} as split assignment is required.")
             skipped_due_to_no_split += 1
             continue

        if opt.skip_existing:
            if current_scene_id in existing_successful_scenes[current_split]:
                print(f"Scene {current_scene_id} (split: {current_split}) found in successful list. Skipping.")
                skipped_due_to_sens_or_skip_flag += 1
                continue

            # Simplified output path check - check for one key output directory for the scene in its split
            scene_output_check_path = os.path.join(opt.output_dir, current_split, 'pose', current_scene_id) # Check e.g. pose dir
            if os.path.exists(scene_output_check_path):
                 print(f"Output data likely exists for scene {current_scene_id} (split: {current_split}). Skipping.")
                 skipped_due_to_sens_or_skip_flag += 1
                 continue
        
        scenes_to_process_args_list.append((sens_file_path, opt.output_dir, opt.num_frames, current_split, opt.image_size, opt.min_valid_components_per_frame, opt.min_valid_frames_per_scene))

    num_scenes_to_actually_process = len(scenes_to_process_args_list)
    print(f"Skipped {skipped_due_to_sens_or_skip_flag} scenes based on missing .sens or --skip_existing flag.")
    if skipped_due_to_no_split > 0:
        print(f"Skipped {skipped_due_to_no_split} scenes because they were not found in train/val lists or --train_val_splits_path was not provided/required.")
    print(f"Attempting to process {num_scenes_to_actually_process} scenes.")

    if num_scenes_to_actually_process == 0:
        print("No scenes left to process. Exiting.")
        sys.exit(0)

    print(f"Starting parallel export using up to {opt.max_workers or os.cpu_count()} workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=opt.max_workers) as executor:
        futures_map = {}
        # Unpack arguments correctly for submit
        for sens_path_arg, output_base_arg, num_frames_arg, split_arg, image_size_arg, min_comp_arg, min_frames_arg in scenes_to_process_args_list:
            # Need scene_id for futures_map key association, extract from sens_path_arg
            proc_scene_id = os.path.basename(os.path.dirname(sens_path_arg))
            future = executor.submit(export_scene_sampled_frames, sens_path_arg, output_base_arg, num_frames_arg, split_arg, image_size_arg, min_comp_arg, min_frames_arg)
            futures_map[future] = (proc_scene_id, split_arg) # Use proc_scene_id and split_arg

        processed_counts_this_run = {'train': 0, 'val': 0} # Renamed for clarity
        failed_counts_this_run = {'train': 0, 'val': 0}    # Renamed for clarity
        for future in tqdm(concurrent.futures.as_completed(futures_map), total=len(futures_map), desc="Processing Scenes"):
            result_scene_id, result_split = futures_map[future] # Get associated scene_id and split
            try:
                returned_scene_id = future.result() 
                if returned_scene_id is not None: # export_scene_sampled_frames returns scene_id on success
                    successful_scenes_this_run[result_split].append(returned_scene_id)
                    processed_counts_this_run[result_split] += 1
                else:
                    failed_counts_this_run[result_split] += 1 
            except Exception as exc:
                print(f'\nScene {result_scene_id} (split: {result_split}) generated an exception during execution: {exc}')
                failed_counts_this_run[result_split] += 1

    all_successful_scenes_combined = {'train': set(), 'val': set()} # Renamed for clarity
    for split_key in ['train', 'val']: # Use split_key consistently
        if successful_scenes_this_run[split_key]:
            current_split_successful_combined = existing_successful_scenes[split_key].union(set(successful_scenes_this_run[split_key]))
            all_successful_scenes_combined[split_key] = current_split_successful_combined
            list_file = successful_scenes_list_files[split_key]
            try:
                with open(list_file, 'w') as f: 
                    for scene_id_val in sorted(list(current_split_successful_combined)): 
                        f.write(scene_id_val + '\n')
                print(f"Updated successful {split_key} scenes list at: {list_file}")
            except Exception as e:
                print(f"Error writing successful {split_key} scenes list to {list_file}: {e}")
        else:
             all_successful_scenes_combined[split_key] = existing_successful_scenes[split_key]


    total_processed_in_this_run = sum(processed_counts_this_run.values())
    total_failed_in_this_run = sum(failed_counts_this_run.values())
    total_successful_over_all_runs = sum(len(s) for s in all_successful_scenes_combined.values())


    print("\n" + "=" * 30)
    print("Processing Summary:")
    print(f"Total potential scenes found: {total_scenes_found}")
    print(f"Scenes skipped (missing .sens / --skip_existing): {skipped_due_to_sens_or_skip_flag}")
    print(f"Scenes skipped (no split assignment): {skipped_due_to_no_split}")
    print(f"Scenes submitted for processing in this run: {num_scenes_to_actually_process}")
    print(f"  - Train: {processed_counts_this_run['train']} successful, {failed_counts_this_run['train']} failed")
    print(f"  - Val:   {processed_counts_this_run['val']} successful, {failed_counts_this_run['val']} failed")
    print(f"Total successfully processed scenes (this run): {total_processed_in_this_run}")
    print(f"Total failed/incomplete scenes (this run): {total_failed_in_this_run}")
    print("-" * 30)
    print(f"Total successful scenes across all runs:")
    print(f"  - Train: {len(all_successful_scenes_combined['train'])} (list: {successful_scenes_list_files['train']})")
    print(f"  - Val:   {len(all_successful_scenes_combined['val'])} (list: {successful_scenes_list_files['val']})")
    print(f"  - Overall: {total_successful_over_all_runs}")
    print("Note: Success indicates the scene processing completed. Check logs for skipped components or frames.")
    print("=" * 30)

if __name__ == '__main__':
    main() 