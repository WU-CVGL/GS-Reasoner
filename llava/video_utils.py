import os
import json
import torch
import pickle
import cv2
import numpy as np
from PIL import Image
from transformers.image_utils import to_numpy_array
import json
from tqdm import tqdm
import random
import copy
from .utils_3d import transform_bbox, check_points_in_boxes

def convert_from_uvd(u, v, d, intr, pose):
    # extr = np.linalg.inv(pose)
    
    fx = intr[0, 0]
    fy = intr[1, 1]
    cx = intr[0, 2]
    cy = intr[1, 2]
    depth_scale = 1000
    
    z = d / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    world = (pose @ np.array([x, y, z, 1]))
    return world[:3] / world[3]
    
def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)


def unproject(intrinsics, poses, depths):
    """
        intrinsics: (V, 4, 4)
        poses: (V, 4, 4)
        depths: (V, H, W)
    """
    V, H, W = depths.shape
    y = torch.arange(0, H).to(depths.device)
    x = torch.arange(0, W).to(depths.device)
    y, x = torch.meshgrid(y, x)

    x = x.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)
    y = y.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)

    fx = intrinsics[:, 0, 0].unsqueeze(-1).repeat(1, H*W)
    fy = intrinsics[:, 1, 1].unsqueeze(-1).repeat(1, H*W)
    cx = intrinsics[:, 0, 2].unsqueeze(-1).repeat(1, H*W)
    cy = intrinsics[:, 1, 2].unsqueeze(-1).repeat(1, H*W)

    z = depths.view(V, H*W) / 1000       # (V, H*W)
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    cam_coords = torch.stack([
        x, y, z, torch.ones_like(x)
    ], -1)      # (V, H*W, 4)

    world_coords = (poses @ cam_coords.permute(0, 2, 1)).permute(0, 2, 1)       # (V, H*W, 4)
    world_coords = world_coords[..., :3] / world_coords[..., 3].unsqueeze(-1)   # (V, H*W, 3)
    world_coords = world_coords.view(V, H, W, 3)

    return world_coords


class VideoProcessor:
    def __init__(
        self, 
        data_folder="data/processed_data", 
        frame_sampling_strategy='uniform',
        img_data_augmentation=False,
        mode="train", # train/val
    ):
        self.data_folder = data_folder
        self.img_data_augmentation = (img_data_augmentation and mode=="train")
        self.frame_sampling_strategy = frame_sampling_strategy
        self.mode = mode
        print('============frame sampling strategy: {}============='.format(self.frame_sampling_strategy))

    def sample_frame_files(
        self,
        video_id: str,
        force_sample: bool = False,
        frames_upbound: int = 0,
    ):
        # 1. read all frame file paths 
        scene_type, scene_id = video_id.split("/")
        
        rgb_dir_path = os.path.join(self.data_folder, f"{scene_type}/color/{self.mode}/{scene_id}")
        rgb_files_path = [os.path.join(rgb_dir_path, f) for f in os.listdir(rgb_dir_path) if os.path.isfile(os.path.join(rgb_dir_path, f))]
        if scene_type == "scannet":
            rgb_files_path = sorted(rgb_files_path, key=lambda x:int(os.path.splitext(os.path.basename(x))[0]))
        elif scene_type == "scannetpp":
            rgb_files_path = sorted(rgb_files_path, key=lambda x:int(os.path.splitext(os.path.basename(x))[0].replace("frame_", "")))
        elif "arkitscenes" in scene_type:
            rgb_files_path = sorted(rgb_files_path, key=lambda x:float(os.path.splitext(os.path.basename(x))[0].split("_")[-1]))
        else:
            raise TypeError
        
        total_frames = len(rgb_files_path)

        # 2. sample files 
        # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
        if force_sample:
            num_frames_to_sample = frames_upbound if frames_upbound < total_frames else total_frames
        else:
            num_frames_to_sample = 10

        if self.img_data_augmentation:    
            # random sample frame number from 16~maxframe
            num_frames = random.randint(16, num_frames_to_sample)
            sampled_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

        return [rgb_files_path[i] for i in sampled_indices]

    def calculate_world_coords(
        self,
        video_id: str, 
        rgb_files_path,
    ):
        scene_type, scene_id = video_id.split("/")
        if scene_type == "scannet":
            depth_intrinsic_path = os.path.join(self.data_folder, f"{scene_type}/intrinsic/{self.mode}/intrinsic_depth_{scene_id}.txt")        
            depth_dir_name = "depth"
            img_ext = "jpg"
        elif scene_type == "scannetpp":
            depth_intrinsic_path = os.path.join(self.data_folder, f"{scene_type}/intrinsic/{self.mode}/intrinsics_{scene_id}.txt")
            # depth_dir_name = "render_depth"
            depth_dir_name = "depth"
            img_ext = "jpg"
        elif "arkitscenes" in scene_type:
            depth_intrinsic_path = os.path.join(self.data_folder, f"{scene_type}/intrinsic/{self.mode}/intrinsics_{scene_id}.txt")
            depth_dir_name = "depth"
            img_ext = "png"
        else:
            raise TypeError
        
        axis_align_matrix_path = os.path.join(self.data_folder, f"{scene_type}/axis_align_matrix/{self.mode}/{scene_id}_further_aligned_transform.txt")
        if os.path.exists(axis_align_matrix_path) is False:
            axis_align_matrix = torch.eye(4).float()
        else:
            axis_align_matrix = torch.from_numpy(np.loadtxt(axis_align_matrix_path))
            axis_align_matrix = axis_align_matrix.float()
        
        is_unified_intrinsic = os.path.exists(depth_intrinsic_path) 
        depth_intrinsic = torch.eye(4)
        if is_unified_intrinsic:
            _depth_intrinsic = torch.from_numpy(np.loadtxt(depth_intrinsic_path))
            depth_intrinsic[:3,:3] = _depth_intrinsic[:3,:3]

        depths = []
        poses = []
        depth_intrinsics = []
 
        # Read and store the sampled frames
        for file_path in rgb_files_path:

            # convert rgb file path to depth file path
            depth_path = file_path.replace("color", depth_dir_name).replace(img_ext, "png")
            # depth image
            with Image.open(depth_path) as depth_img:
                depth = np.array(depth_img).astype(np.int32)
                depths.append(torch.from_numpy(depth))

            # pose
            pose_file = file_path.replace("color", "pose").replace(img_ext, "txt")
            pose = np.loadtxt(pose_file)
            poses.append(torch.from_numpy(pose).float())

            # intrinsic (conditional)
            if is_unified_intrinsic is False:
                intrinsic_file = file_path.replace("color", "intrinsic").replace(img_ext, "txt")
                intrinsic = torch.from_numpy(np.loadtxt(intrinsic_file))
                depth_intrinsics.append(intrinsic.float())

        depth_intrinsics = depth_intrinsic.unsqueeze(0).repeat(len(rgb_files_path), 1, 1) if is_unified_intrinsic else torch.stack(depth_intrinsics)
        depths = torch.stack(depths)   # (V, H, W)
        poses = torch.stack([axis_align_matrix @ pose for pose in poses])     # (V, 4, 4)
        world_coords = unproject(depth_intrinsics.float(), poses.float(), depths.float())    # (V, H, W, 3)
        

        # global scale for estimated depth
        global_scale_path = os.path.join(self.data_folder, f"{scene_type}/global_scale/{self.mode}/{scene_id}.txt")
        if os.path.exists(global_scale_path):
            global_scale = torch.from_numpy(np.loadtxt(global_scale_path))
            world_coords = world_coords * global_scale
        
        return {
            "world_coords": world_coords,
            "poses": poses,
        }


    # def calculate_relative_coords(
    #     self,
    #     video_id: str, 
    #     frame_files,
    #     do_normalize=False,
    # ):
    #     """
    #     calculate coords relative to the first frame (not in canonical coordinate anymore)
    #     """
    #     meta_info = self.scene[video_id]
    #     scene_id = video_id.split('/')[-1]

    #     axis_align_matrix = torch.from_numpy(np.array(meta_info['axis_align_matrix']))
    #     depth_intrinsic = torch.from_numpy(np.array(meta_info["depth_cam2img"]))

    #     depths = []
    #     poses = []
 
    #     # Read and store the sampled frames
    #     for frame_path in frame_files:

    #         # depth image
    #         depth_path = frame_path.replace(".jpg", ".png")
    #         with Image.open(depth_path) as depth_img:
    #             depth = np.array(depth_img).astype(np.int32)
    #             depths.append(torch.from_numpy(depth))

    #         # pose
    #         pose_file = frame_path.replace("jpg", "txt")
    #         pose = np.loadtxt(pose_file)
    #         poses.append(torch.from_numpy(pose))


    #     depths = torch.stack(depths)   # (V, H, W)

    #     # caculate relative pose
    #     relative_pose = poses[0].inverse() # world to camera_0 
    #     # all the poses are from camera_n to camera_0 (no axis aligned)
    #     poses = torch.stack([relative_pose @ pose for pose in poses])     # (V, 4, 4)
    #     depth_intrinsic = depth_intrinsic.unsqueeze(0).repeat(len(frame_files), 1, 1)
        
    #     world_coords = unproject(depth_intrinsic.float(), poses.float(), depths.float())    # (V, H, W, 3)

    #     if do_normalize:
    #         world_coords = torch.maximum(world_coords, self.pc_min[scene_id].to(world_coords.device))
    #         world_coords = torch.minimum(world_coords, self.pc_max[scene_id].to(world_coords.device))
        
    #     return {
    #         "world_coords": world_coords,
    #         # used to transform bounding box from canonical coordinate to camera_0 
    #         "transformation": (relative_pose @ axis_align_matrix.inverse()).float(),
    #     }


    def preprocess(
        self,
        video_id: str, 
        image_processor,
        force_sample: bool = False,
        frames_upbound: int = 0,
        strategy: str = "center_crop",
        debug_frame_files = None,
    ):
        """
            read data info from file and preproess
        """
        # depth+pose -> pointmap (480, 640) -> resize to (384, 512) -> center crop to (384, 384)
        # image (968, 1296) -> resize to (384, 512) -> center crop to (384, 384)

        rgb_files_path = self.sample_frame_files(
            video_id,
            force_sample=force_sample,
            frames_upbound=frames_upbound,
        )

        if debug_frame_files is not None:
            # 从video_id中提取scene_type
            scene_type, scene_id = video_id.split("/")
            
            # 从rgb_files_path中提取frame索引
            frame_indices = []
            for rgb_path in rgb_files_path:
                filename = os.path.basename(rgb_path)
                # 提取frame_{idx:06}格式的索引
                if filename.startswith('frame_') and filename.endswith('.jpg'):
                    try:
                        idx = int(filename[6:12])  # 提取frame_后面的6位数字
                        frame_indices.append(idx)
                    except ValueError:
                        continue
            
            # 为每个debug_frame_file找到最近的帧
            new_rgb_files = []
            for debug_file in debug_frame_files:
                debug_filename = os.path.basename(debug_file)
                if debug_filename.startswith('frame_') and debug_filename.endswith('.jpg'):
                    try:
                        debug_idx = int(debug_filename[6:12])
                        # 找到最近的frame索引
                        if frame_indices:
                            nearest_idx = min(frame_indices, key=lambda x: abs(x - debug_idx))
                            # 找到对应的rgb文件路径
                            for rgb_path in rgb_files_path:
                                if f"frame_{nearest_idx:06d}.jpg" in rgb_path:
                                    new_rgb_files.append(rgb_path)
                                    break
                    except ValueError:
                        continue
                
            rgb_files_path = new_rgb_files

        video_dict = self.calculate_world_coords(
            video_id,
            rgb_files_path
        )
        
        world_coords = video_dict["world_coords"]
        transformation = video_dict.get("transformation", None)
        V, H, W, _ = world_coords.shape

        images = []
        for file_path in rgb_files_path:
            with Image.open(file_path) as img:
                frame = img.convert("RGB")
                images.append(frame)

        crop_size = image_processor.crop_size["width"]
        if strategy == "resize":
            images = [frame.resize((crop_size, crop_size)) for frame in images]
            resized_coords = [cv2.resize(coords.numpy(), (384, 384), interpolation=cv2.INTER_NEAREST) for coords in world_coords] 
        elif strategy == "center_crop":
            # Determine which dimension to use as the reference for scaling
            if H <= W:  # height <= width: scale based on height
                new_height = crop_size
                new_width = int(W * (crop_size / H))
            else:  # height > width: scale based on width
                new_width = crop_size
                new_height = int(H * (crop_size / W))
            
            images = [frame.resize((new_width, new_height)) for frame in images]
            resized_coords = [cv2.resize(coords.numpy(), (new_width, new_height), interpolation=cv2.INTER_NEAREST) for coords in world_coords]
            
            # Calculate the position and perform the center crop
            left = (new_width - crop_size) // 2
            right = left + crop_size
            top = (new_height - crop_size) // 2
            bottom = top + crop_size
            images = [frame.crop((left, top, right, bottom)) for frame in images]

            resized_coords = [coords[top:bottom, left:right, :] for coords in resized_coords]
        

        # objects = torch.tensor(self.scan2obj[video_id]) # [n, 6]
        # if transformation is not None:
        #     objects = transform_bbox(objects, transformation)

        resized_world_coords = torch.from_numpy(np.stack(resized_coords))
            
        world_coords = resized_world_coords
        
        # point cloud data augmentation
        # if self.pcd_data_augmentation:
        #     pcd = world_coords.flatten(0, 2)
        #     p_rot_z = 0.5 if data_aug_rotate_z else 0.
        #     aug_pcd, aug_obj = augment_point_cloud_torch(pcd, objects, p_rot_z=p_rot_z)
            
        #     V, H, W, _ = world_coords.shape
        #     world_coords = aug_pcd.reshape(V, H, W, -1)
        #     objects = aug_obj

            
        return {
            "images": images,
            "world_coords": world_coords,
            "video_size": len(images),
            # "boundry": boundry,
            # "objects": torch.tensor(self.scan2obj[video_id]),
            # "objects": objects,
            "transformation": transformation,
            
            "poses": video_dict["poses"]
        }
        
    
    def process_3d_video(
        self,
        video_id: str, 
        image_processor,
        force_sample: bool = False,
        frames_upbound: int = 0,
        strategy: str = "center_crop",
        debug_frame_files = None,
    ):
        video_dict = self.preprocess(
            video_id,
            image_processor,
            force_sample,
            frames_upbound,
            strategy,
            debug_frame_files=debug_frame_files,
        )
        video_dict["images"] = image_processor.preprocess(video_dict["images"], return_tensors="pt")["pixel_values"]
        return video_dict

    
    def discrete_point(self, xyz):
        xyz = torch.tensor(xyz)
        if self.min_xyz_range is not None:
            xyz = torch.maximum(xyz, self.min_xyz_range.to(xyz.device))
        if self.max_xyz_range is not None:
            xyz = torch.minimum(xyz, self.max_xyz_range.to(xyz.device))
        if self.min_xyz_range is not None:
            xyz = (xyz - self.min_xyz_range.to(xyz.device)) 
            
        xyz = xyz / self.voxel_size
        return xyz.round().int().tolist()
    

def merge_video_dict(video_dict_list):
    new_video_dict = {}
    new_video_dict['box_input'] = []
    for k in video_dict_list[0]:
        if k in ['world_coords', 'images', 'objects', 'transformation']:
            if video_dict_list[0][k] is not None:
                new_video_dict[k] = torch.stack([video_dict[k] for video_dict in video_dict_list])
            else:
                new_video_dict[k] = None
        elif k in ['box_input']:
            for video_dict in video_dict_list:
                if video_dict[k] is not None:
                    new_video_dict['box_input'].append(video_dict[k])

    new_video_dict['box_input'] = torch.Tensor(new_video_dict['box_input'])
    return new_video_dict
