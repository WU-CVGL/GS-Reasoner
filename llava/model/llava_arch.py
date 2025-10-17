#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random
import llava.submodules.sonata as sonata
from llava.submodules.sonata.model import PointTransformerV3
from llava.utils_3d import PointNet, CrossAttnFusion
import math


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
        
        if hasattr(self.config, 'world_position_embedding_type'):
            from llava.model.position_encoding import PositionEmbeddingSine3D, PositionEmbeddingMLP

            if "mlp" in self.config.world_position_embedding_type:
                # default MLP for 3d pointmap encoding
                self.world_position_embedding = PositionEmbeddingMLP(config.hidden_size)
            elif "sin3d" in self.config.world_position_embedding_type:
                self.world_position_embedding = PositionEmbeddingSine3D(config.hidden_size)
            elif "sonata" in self.config.world_position_embedding_type:
                sonata_hidden_size = 1232

                self.use_xyz = False
                self.pointnet_pooling = PointNet(input_dim=sonata_hidden_size, output_dim=2048, use_xyz=self.use_xyz)
                
                self.ada_crossattn = CrossAttnFusion(semantic_feat_dim=config.mm_hidden_size, spatial_feat_dim=sonata_hidden_size)
                
                self.point_transform = sonata.transform.nonorml_transform()
                
                self.mm_spatial_projector = nn.Sequential(
                    nn.Linear(2048+sonata_hidden_size, config.hidden_size),
                    nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size)
                ) # same archetecture with mm projector
                self.world_position_embedding = PositionEmbeddingSine3D(config.hidden_size)
                self.spatial_encoder = PointTransformerV3()
                
                self.n_points = int(self.config.world_position_embedding_type.split("_")[-1])
        
    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_spatial_encoder(self):
        spatial_encoder = getattr(self, "spatial_encoder", None)
        return spatial_encoder
    
    def set_spatial_encoder(self, spatial_encoder):
        self.spatial_encoder = spatial_encoder

    def initialize_spatial_encoder(self, dtype, device):
        print("initialize sonata spatial encoder!")
        spatial_encoder = sonata.load("data/models/sonata/sonata_wo_normal.pth", repo_id="facebook/sonata")
        # point transformer do not support bf16
        spatial_encoder.to(dtype=torch.float32 ,device=device)
        self.spatial_encoder = spatial_encoder

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)

        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_spatial_encoder(self):
        return self.get_model().get_spatial_encoder()
    
    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature


    def average_coordinate_in_patch(self, world_coords, patch_size=27):

        V, H, W, D = world_coords.size() # D = 3

        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :]    # [32, 378, 378, 3]
        world_coords = world_coords.permute(0, 3, 1, 2)   # [V, D, 378, 378]
        world_coords_avg = torch.nn.functional.avg_pool2d(world_coords, kernel_size=patch_size, stride=patch_size)  # [32, 3, 14,  14]
        patch_num = world_coords_avg.shape[-1]
        world_coords_avg = world_coords_avg.permute(0, 2, 3, 1)     # [32, 14, 14, 3]

        return world_coords_avg

    def minmax_coordinate_in_patch(self, world_coords, patch_size=27):

        V, H, W, D = world_coords.size() # D = 3

        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :]    # [32, 378, 378, 3]
        world_coords = world_coords.permute(0, 3, 1, 2)   # [V, D, 378, 378]

        world_coords_max = torch.nn.functional.max_pool2d(world_coords, kernel_size=patch_size, stride=patch_size)  # [32, 3, 14,  14]
        world_coords_max = world_coords_max.permute(0, 2, 3, 1)     # [32, 14, 14, 3]

        world_coords_min = - torch.nn.functional.max_pool2d(-world_coords, kernel_size=patch_size, stride=patch_size)  # [32, 3, 14,  14]
        world_coords_min = world_coords_min.permute(0, 2, 3, 1)     # [32, 14, 14, 3]
        world_coords = torch.stack([world_coords_min, world_coords_max], dim=3) # [32, 14, 14, 2, 3]

        return world_coords
    
    def patch_pooling(self, sample_coords_feat, sample_coords, patch_3d_position, pooling_strategy="coarse_to_fine"):
        """
        patch pooling geometry feature
        Args:
            sample_coords_feat: [B, V, num_patches, num_patches, N, feat_dim]
            sample_coords: [B, V, num_patches, num_patches, N, 3]
            patch_3d_position: [B, V, num_patches, num_patches, 3]
        Returns:
            pooling_feats: [B, V, num_patches, num_patches, feat_dim]
        """
        assert "sonata" in self.config.world_position_embedding_type,  "only sonata encoder compatibility!"
        B, V, num_patches, _, n_points, feat_dim = sample_coords_feat.shape
        if pooling_strategy in ["average", "max"]:
            pooling_feat = self.get_model().pointnet_pooling(sample_coords_feat, pooling_strategy=pooling_strategy)
        elif pooling_strategy == "interpolate":
            assert sample_coords is not None, "interpolate strategy need sample_coords input"
            assert patch_3d_position is not None, "interpolate pooling strategy need patch_3d_position input"
            
            pooling_feat = self.get_model().pointnet_pooling(sample_coords_feat, pooling_strategy=None) # [B, V, num_patches, num_patches, N, out_feat_dim]
            distances = torch.norm(sample_coords - patch_3d_position.unsqueeze(-2), dim=-1) # [B, V, num_patches, num_patches, N]
            eps = 1e-8
            weights = 1.0 / (distances + eps)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            pooling_feat = torch.sum(pooling_feat * weights.unsqueeze(-1), dim=-2)

        return pooling_feat


    def sample_n_points(self, world_coords, n_points=9):
        # fixed patch_size = 27

        V, H, W, D = world_coords.size() # D = 3
        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :] 
        world_coords = world_coords.view(-1, 14, 27, 14, 27, 3).permute(0, 1, 3, 2, 4, 5)
        if n_points == 1:
            world_coords_sample = world_coords[:, :, :, 4::9, 4::9, :].reshape(V, 14, 14, 9, 3)
            world_coords_sample = world_coords_sample[:, :, :, 4, :].reshape(V, 14, 14, 3)
        elif n_points == 0:
            world_coords_sample = world_coords.reshape(V, 14, 14, -1, 3)
        else:
            n_points_sqrt = int(math.sqrt(n_points))
            sample_index = torch.linspace(0, 26, steps=n_points_sqrt).to(device=world_coords.device)
            sample_index = torch.round(sample_index).to(dtype=torch.int64)
            ii, jj = torch.meshgrid(sample_index, sample_index, indexing='ij')
            world_coords_sample = world_coords[:, :, :, ii, jj, :].reshape(V, 14, 14, n_points, 3)
        
        return world_coords_sample

    def discrete_coords(self, world_coords, xyz_min):

        # V, H, W, D = world_coords.size() # D = 3
        # world_coords_discrete = (world_coords.view(-1, 3) - xyz_min.view(1, 3)) / self.config.voxel_size

        min_xyz_range = torch.tensor(self.config.min_xyz_range).to(world_coords.device)
        max_xyz_range = torch.tensor(self.config.max_xyz_range).to(world_coords.device)

        world_coords = torch.maximum(world_coords, min_xyz_range)
        world_coords = torch.minimum(world_coords, max_xyz_range)
        world_coords_discrete = (world_coords - min_xyz_range) / self.config.voxel_size
        world_coords_discrete = world_coords_discrete.round()

        return world_coords_discrete.detach()


    def encode_images(self, images, world_coords=None):
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        # image_features = self.get_model().mm_projector(image_features)

        return image_features


    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat is not 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past_key_values, 
        labels, 
        images, 
        modalities=["image"], 
        image_sizes=None, 
        video_dict=None,
    ):
        #-------------------------------------------------------------------------------------------------------------------------------#
        """
            1. build position-aware video representations
        """
        use_geo_feat = False
        if hasattr(self.config, 'world_position_embedding_type') and past_key_values is None:
            use_geo_feat = True
            B = input_ids.shape[0]

            if type(images) is list:
                images_batch = torch.stack(images, dim=0) # [B, V, 3, H, W]
            else:
                images_batch = images
            assert images_batch.ndim == 5, "wrong images_batch dim"
            # compatible with siglip encoder
            images_batch = images_batch.permute(0, 1, 3, 4, 2) # [B, V, H, W, 3]
            world_coords = video_dict['world_coords'] # [B, V, H, W, 3]

            # sample point from patch for faster calculation
            if "sample" in self.config.world_position_embedding_type:                
                n_points = self.get_model().n_points

                sample_world_coords = torch.stack([self.sample_n_points(coords, n_points=n_points) for coords in world_coords], dim=0) 
                sample_world_coords_rgb = torch.stack([self.sample_n_points(img, n_points=n_points) for img in images_batch], dim=0) 
            
            B, V, num_patches, _, _, _ = sample_world_coords.shape # [B, V, num_patches, num_patches, n_points, 3]
            sample_world_coords = sample_world_coords.flatten(start_dim=1, end_dim=-2)
            sample_world_coords_rgb = (sample_world_coords_rgb.flatten(start_dim=1, end_dim=-2) + 1) / 2 * 255 # convert to 0~255

            # prepare sonata input
            point = {
                "coord": sample_world_coords.float().cpu().numpy().squeeze(0),
                "color": sample_world_coords_rgb.float().cpu().numpy().squeeze(0),
            }
            point = self.get_model().point_transform(point)
            for key in point.keys():
                if isinstance(point[key], torch.Tensor):
                    point[key] = point[key].to(device=world_coords.device)
            
            with torch.autocast(device_type="cuda", enabled=False):
                spatial_encoder_fp32 = self.get_model().spatial_encoder.float()
                point = spatial_encoder_fp32(point)

            # backpropagate pointcloud to the origin size
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            point_feat = point.feat[point.inverse].reshape(B, V, num_patches, num_patches, n_points, -1).to(dtype=world_coords.dtype) # [B, V, num_patches, num_patches, n_points, 1232]

            patch_3d_position = torch.stack([self.sample_n_points(coords, n_points=1) for coords in world_coords], dim=0) # [B, V, num_patches, num_patches, 3]
            sample_world_coords = sample_world_coords.reshape(B, V, num_patches, num_patches, n_points, -1)
            pos_pooling_feat = self.patch_pooling(point_feat, sample_world_coords, patch_3d_position, pooling_strategy="interpolate") # [B, V, num_patches, num_patches, out_dim]
            pos_pooling_feat = pos_pooling_feat.flatten(start_dim=2, end_dim=3) # [B, V, num_patches*num_patches, out_dim]


        """
            1.2 video frames preprocess, including siglip image encoding, encoded feature pooling(from 27*27 pool to 14*14) and object feature calculation (for visual grounding task)
        """
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            # siglip encoding (patch_size=14)
            encoded_image_features = self.encode_images(concat_images) # [V, 27*27, 3584]

            pooling_img_feat = self.get_2dPool(encoded_image_features.detach()) # [V, 14*14, 3584]
            ada_pooling_feat = self.get_model().ada_crossattn(pooling_img_feat.unsqueeze(-2), point_feat[0].flatten(1, 2)).unsqueeze(0) # [B, V, 14*14, sonata_hidden_size]
            pooling_feat = torch.cat([pos_pooling_feat, ada_pooling_feat], dim=-1)
            spatial_feat = self.get_model().mm_spatial_projector(pooling_feat) # [B, V, 14*14, hidden_size]
            position_feat = self.get_model().world_position_embedding(patch_3d_position.flatten(start_dim=1, end_dim=3)).reshape(B, V, num_patches*num_patches, -1) # [B, V, 14*14, hidden_size]

            encoded_image_features = self.get_model().mm_projector(encoded_image_features)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)

            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            """
                1.3 add image feature and position encoded pointmap as spaital video
            """
            if use_geo_feat:
                new_image_features = []
                for idx, image_feat in enumerate(image_features):
                    image_feat = image_feat + spatial_feat[idx] + position_feat[idx]
                    new_image_features.append(image_feat)
                image_features = new_image_features # [n_frame, n_patch, 3584] list
            
            """
                1.4 convert spatial representation with shape [n_frame, n_patch, 3584] to flat token embedding with shape [N, 3584]
                In the spatial patch merge method, a special token embedding will be added to each line end of the image feature. 
                Note that between different frames there is no any special process.
            """
            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)

                                # print("!!!!!!!!!!!!")
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)
        #-------------------------------------------------------------------------------------------------------------------------------#
        
        
        #-------------------------------------------------------------------------------------------------------------------------------#
        """
            2. convert token id to token embeddings, and substitute the <image> token with position-aware video representation 
        """
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        
        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
 
        """
            2.1 split tokens with <image>, and then convert all no image tokens to embeddings. 
            Then when concat token embeddings, substitute the <image> token place with position-aware video representation(already projected to the same dim as text embeddings)
            The labels are also updated, with the substituted <image> token all place with IGNORE_INDEX
            Note: may position encoding coordinate tokens if have.
        """
        new_input_embeds = []
        new_labels = []
        noimage_mask = []

        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                
                
            split_sizes = [x.shape[0] for x in cur_labels_noim]

            cat_cur_input_ids_noim = torch.cat(cur_input_ids_noim)
            # get token(wo <image>) embedding
            cur_input_embeds = self.get_model().embed_tokens(cat_cur_input_ids_noim)
            
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_noimg_mask = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                
                cur_noimg_mask.append(torch.ones_like(cur_labels_noim[i], dtype=torch.bool))
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_noimg_mask.append(torch.zeros((cur_image_features.shape[0],), device=cur_labels.device, dtype=torch.bool))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_noimg_mask = torch.cat(cur_noimg_mask)
            
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            noimage_mask.append(cur_noimg_mask)


        """
            2.2 Truncate sequences to max length, and padding tokens in same batch to same length(max length in this batch) for transformer input.  
        """            
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")
        # truncate tokens after max length
        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        noimage_mask = [x[:tokenizer_model_max_length] for x, modality in zip(noimage_mask, modalities)]


        # Combine them. Specifically, pad tokens in same batch to same length for transformer input
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        noimage_mask_padded = torch.ones((batch_size, max_len), dtype=torch.bool, device=noimage_mask[0].device)
        
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        for i, (cur_new_embed, cur_new_labels, cur_noimg_mask) in enumerate(zip(new_input_embeds, new_labels, noimage_mask)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    noimage_mask_padded[i, -cur_len:] = cur_noimg_mask
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

                    noimage_mask_padded[i, :cur_len] = cur_noimg_mask
                
                
        # mrope_position_ids = mrope_position_ids.permute(2, 0, 1)
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0) # [B, padded_token_num, token_embedding_dim]
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        noimage_mask = noimage_mask_padded
        
        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)        
        
        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        
        #-------------------------------------------------------------------------------------------------------------------------------#

        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, None, None, noimage_mask

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
