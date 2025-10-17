import numpy as np
import torch
import math
import scipy
import random
import re
from llava.constants import BBOX_START_TOKEN, BBOX_END_TOKEN
import torch.nn as nn


def convert_pc_to_box(obj_pc):
    # converting point clouds into bounding box
    xmin = np.min(obj_pc[:,0])
    ymin = np.min(obj_pc[:,1])
    zmin = np.min(obj_pc[:,2])
    xmax = np.max(obj_pc[:,0])
    ymax = np.max(obj_pc[:,1])
    zmax = np.max(obj_pc[:,2])
    center = [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]
    box_size = [xmax-xmin, ymax-ymin, zmax-zmin]
    return center, box_size


def transform_bbox(bboxes, transformation):
    bboxes_center = torch.cat([bboxes[..., :3], torch.ones_like(bboxes[..., :1])], dim=-1) # [n, 4]
    transformed_bboxes_center = (transformation @ bboxes_center.permute(1, 0)).permute(1, 0) # [n, 4]
    transformed_bboxes = torch.cat([transformed_bboxes_center[..., :3], bboxes[..., 3:]], dim=-1) # [n, 6]
    return transformed_bboxes

def transform_point(points, transformation):
    homo_points = torch.cat([points[..., :3], torch.ones_like(points[..., :1])], dim=-1) # [n, 4]
    transformed_points = (transformation @ homo_points.permute(1, 0)).permute(1, 0) # [n, 4]
    transformed_points = transformed_points[..., :3]
    return transformed_points


def convert_bbox_to_str(bboxes):
    """
    bboxes: [N, 6] -> [x1, y1, z1, x2, y2, z2]
    """
    bbox_str_list = []
    
    for bbox in bboxes:    
        bbox_float = [i.item() for i in bbox]        
        # preserve 4 digit after point
        bbox_str = "({})".format(', '.join(f"{x:.4f}" for x in bbox_float))
        
        # preserve 2 digit after point
        # bbox_str = "({})".format(', '.join(f"{x:.2f}" for x in bbox_float))
        
        bbox_str_list.append(bbox_str)
    
    return bbox_str_list

# def convert_str_to_bbox(bbox_str: str):
#     """
#     bbox_str: (x1, y1, z1, x2, y2, z2)
#     """
#     cleaned_str = bbox_str.strip("()")  # Remove surrounding parentheses
#     bbox_list = [float(x.strip()) for x in cleaned_str.split(",")]  # Split, clean, and convert
#     return bbox_list

# def decode_str_to_bbox(outputs_str):
#     # 正则匹配每个object及其后续所有的<bbox_start>(...6个float)...<bbox_end>
#     object_pattern = re.finditer(rf"([\w\s]+?)\s+((?:{BBOX_START_TOKEN}\([^)]*\){BBOX_END_TOKEN})+)", outputs_str)
#     result = []

#     for match in object_pattern:
#         name = match.group(1)
#         bbox_block = match.group(2)
        
#         # 匹配该物体所有的bbox
#         bbox_matches = re.findall(rf"{BBOX_START_TOKEN}\(([^)]*)\){BBOX_END_TOKEN}", bbox_block)
#         bboxes = [[float(x) for x in bbox.split(",")] for bbox in bbox_matches]
        
#         result.append({
#             "object_name": name,
#             "object_bbox": bboxes
#         })
#     return result



def decode_str_to_bbox(outputs_str: str):
    """
    从字符串中解码出一个或多个边界框(bbox)信息。

    该函数可以解析以下两种格式:
    1. 单个bbox: "bbox_name bbox_num <bbox>(x1, y1, z1, x2, y2, z2)</bbox>"
    2. 多个bbox: "bbox_name bbox_num <bbox>(...)</bbox><bbox>(...)</bbox>"

    Args:
        outputs_str: 包含bbox信息的输入字符串。

    Returns:
        一个列表，每个元素是一个字典，代表一个被识别的对象及其所有bbox。
        e.g., [
            {
                "object_name": "some_name",
                "bbox_num": 2,
                "object_bbox": [[x1, y1, z1, x2, y2, z2], [x1, y1, z1, x2, y2, z2]]
            }
        ]
    """
    # 步骤1: 修改主正则表达式，以捕获名称、数量以及其后跟随的【所有】连续的bbox块
    # ((?:...)+) 会匹配一个或多个连续的 bbox 块，并将其作为一个整体捕获
    pattern = rf"(\w+)\s+(\d+)\s+((?:{BBOX_START_TOKEN}\([^)]*\){BBOX_END_TOKEN})+)"
    
    # re.finditer 更适合处理复杂的匹配和后续操作，也更节省内存
    matches = re.finditer(pattern, outputs_str)
    
    result = []
    for match in matches:
        bbox_name = match.group(1)
        bbox_num = int(match.group(2))
        # all_bboxes_str 包含了这个对象所有连续的 "<bbox>(...)</bbox><bbox>(...)</bbox>..." 字符串
        all_bboxes_str = match.group(3)
        
        # 步骤2: 在捕获到的 all_bboxes_str 上，使用简单的表达式提取【每个】bbox的坐标
        bbox_pattern = rf"{BBOX_START_TOKEN}\(([^)]*)\){BBOX_END_TOKEN}"
        # re.findall 会返回所有匹配到的坐标字符串列表, e.g., ['x1,y1,...', 'x1,y1,...']
        coord_matches = re.findall(bbox_pattern, all_bboxes_str)
        
        object_bboxes = []
        for coords_str in coord_matches:
            # 分割坐标字符串并转换为浮点数
            bbox_coords = [float(x.strip()) for x in coords_str.split(",")]
            object_bboxes.append(bbox_coords)
            
        result.append({
            "object_name": bbox_name,
            "bbox_num": bbox_num,
            "object_bbox": object_bboxes
        })
    
    return result


def aabb_to_xyzwlh(bbox):
    """
    Convert AABB bbox [x1, y1, z1, x2, y2, z2] to [x, y, z, w, h, l] format.
    Supports list, numpy array, and torch.Tensor inputs, and returns the same type.
    """
    input_type = type(bbox)
    
    if isinstance(bbox, (list, tuple)):
        bbox = np.array(bbox)
    elif isinstance(bbox, torch.Tensor):
        bbox_np = bbox.detach().cpu().numpy()
    else:
        bbox_np = bbox  # Assume numpy array if not list/tuple/torch.Tensor
    
    # Compute center and dimensions
    x1, y1, z1, x2, y2, z2 = bbox_np if 'bbox_np' in locals() else bbox
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    z = (z1 + z2) / 2
    w = x2 - x1  # width (x-axis dimension)
    h = y2 - y1  # height (y-axis dimension)
    l = z2 - z1  # length (z-axis dimension)
    
    xyzwlh = np.array([x, y, z, w, h, l])
    
    # Convert back to input type
    if input_type in (list, tuple):
        return xyzwlh.tolist()
    elif input_type is torch.Tensor:
        return torch.from_numpy(xyzwlh).to(bbox.device).type(bbox.dtype)
    else:
        return xyzwlh


def compute_3d_iou(bbox1, bbox2):
    """
    Compute 3D IoU between two bounding boxes in [x, y, z, w, h, l] format.
    
    Args:
        bbox1: [x, y, z, w, h, l] - center coordinates and dimensions
        bbox2: [x, y, z, w, h, l] - center coordinates and dimensions
    
    Returns:
        float: IoU value between 0 and 1
    """
    # Convert to numpy arrays for easier computation
    if isinstance(bbox1, (list, tuple)):
        bbox1 = np.array(bbox1)
    if isinstance(bbox2, (list, tuple)):
        bbox2 = np.array(bbox2)
    
    # Extract center coordinates and dimensions
    x1, y1, z1, w1, h1, l1 = bbox1
    x2, y2, z2, w2, h2, l2 = bbox2
    
    # Calculate the coordinates of the intersection box
    x_min = max(x1 - w1/2, x2 - w2/2)
    y_min = max(y1 - h1/2, y2 - h2/2)
    z_min = max(z1 - l1/2, z2 - l2/2)
    x_max = min(x1 + w1/2, x2 + w2/2)
    y_max = min(y1 + h1/2, y2 + h2/2)
    z_max = min(z1 + l1/2, z2 + l2/2)
    
    # Check if there is an intersection
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        return 0.0
    
    # Calculate intersection volume
    intersection_volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    
    # Calculate union volume
    volume1 = w1 * h1 * l1
    volume2 = w2 * h2 * l2
    union_volume = volume1 + volume2 - intersection_volume
    
    # Calculate IoU
    iou = intersection_volume / union_volume if union_volume > 0 else 0.0
    
    return iou


def find_best_bbox_by_iou(all_pred_bbox, gt_response):
    """
    Find the bbox from all_pred_bbox that has the highest IoU with gt_response.
    
    Args:
        all_pred_bbox: List of predicted bboxes in [x, y, z, w, h, l] format
        gt_response: Ground truth bbox in [x, y, z, w, h, l] format
    
    Returns:
        tuple: (best_bbox, best_iou) - the bbox with highest IoU and its IoU value
    """
    if not all_pred_bbox:
        return None, 0.0
    
    best_bbox = None
    best_iou = -1.0
    
    for pred_bbox in all_pred_bbox:
        iou = compute_3d_iou(pred_bbox, gt_response)
        if iou > best_iou:
            best_iou = iou
            best_bbox = pred_bbox
    
    return best_bbox, best_iou


@torch.no_grad()
def check_points_in_boxes(boxes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    boxes: [N, 6] -> [cx, cy, cz, w, h, l]
    points: [K, 3] -> [x, y, z]
    returns: [N] bool tensor, each True means the box contains at least one point
    """
    N = boxes.shape[0]
    K = points.shape[0]

    # Expand boxes to [N, K, 3]
    centers = boxes[:, :3].unsqueeze(1)  # [N, 1, 3]
    sizes = boxes[:, 3:].unsqueeze(1)    # [N, 1, 3]

    # Expand points to [1, K, 3]
    points_exp = points.unsqueeze(0)     # [1, K, 3]

    # Calculate min and max corners of each box
    box_mins = centers - sizes / 2  # [N, 1, 3]
    box_maxs = centers + sizes / 2  # [N, 1, 3]

    # Check if points are inside each box
    inside_mask = (points_exp >= box_mins) & (points_exp <= box_maxs)  # [N, K, 3]
    inside_all_axes = inside_mask.all(dim=2)  # [N, K]

    # If any point in box, return True for that box
    box_has_points = inside_all_axes.any(dim=1)  # [N]

    return box_has_points


def _random_scales(device, dtype):
    """生成全局缩放因子 ∈ [0.25, 1.25]."""
    scale_limit = (-0.25, 0.25)
    scales = torch.empty(3, device=device, dtype=dtype).uniform_(*scale_limit)
    return 1.0 + scales

def _rotation_matrix(axis, angle, device, dtype):
    """返回围绕给定 axis 的 3×3 旋转矩阵."""
    x, y, z = axis / torch.norm(axis)
    c, s, C = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    R = torch.tensor([
        [x * x * C + c,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c    ]
    ], dtype=dtype, device=device)
    return R


def _random_jitter(points: torch.Tensor, sigma=0.01, clip=0.05, ratio=0.5, p=0.5):
    """
    Apply random jitter to a subset of points in the point cloud.

    Args:
        points (torch.Tensor): Input point cloud of shape [N, 3].
        sigma (float): Standard deviation of the jitter.
        clip (float): Maximum absolute value of the jitter.
        ratio (float): Ratio of points to apply jitter to.
        p (float): Probability of applying jitter.

    Returns:
        torch.Tensor: Jittered point cloud of shape [N, 3].
    """
    assert points.ndim == 2 and points.shape[1] == 3, "Input must be of shape [N, 3]"
    if torch.rand(1).item() < p:
        N = points.shape[0]
        jitter = torch.clamp(
            sigma * torch.randn(N, 3, device=points.device),
            -clip,
            clip
        )
        mask = torch.rand(N, device=points.device) < ratio
        points[mask] += jitter[mask]
    return points


def _elastic_distortion(
    coords: torch.Tensor,
    distortion_params=[[0.2, 0.4], [0.8, 1.6]],
    p=[0.8, 0.5]
) -> torch.Tensor:
    """
    Apply elastic distortion on point cloud coordinates (torch version).

    Args:
        coords (torch.Tensor): Input point cloud of shape [N, 3].
        distortion_params (List[List[float]]): List of [granularity, magnitude] pairs.
        p (List[float]): Probability of applying each corresponding distortion.

    Returns:
        torch.Tensor: Distorted point cloud of shape [N, 3].
    """
    assert coords.ndim == 2 and coords.shape[1] == 3, "Input must be of shape [N, 3]"
    device = coords.device
    coords_np = coords.cpu().numpy().copy()

    def _apply_single_distortion(coords_np, granularity, magnitude):
        blurx = np.ones((3, 1, 1, 1), dtype=np.float32) / 3
        blury = np.ones((1, 3, 1, 1), dtype=np.float32) / 3
        blurz = np.ones((1, 1, 3, 1), dtype=np.float32) / 3

        coords_min = coords_np.min(0)
        noise_dim = ((coords_np - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        for _ in range(2):
            noise = scipy.ndimage.convolve(noise, blurx, mode="constant", cval=0)
            noise = scipy.ndimage.convolve(noise, blury, mode="constant", cval=0)
            noise = scipy.ndimage.convolve(noise, blurz, mode="constant", cval=0)

        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (np.array(noise_dim) - 2),
                noise_dim
            )
        ]

        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords_np += interp(coords_np) * magnitude
        return coords_np

    for i, (granularity, magnitude) in enumerate(distortion_params):
        if np.random.rand() > p[i]:
            continue
        coords_np = _apply_single_distortion(coords_np, granularity, magnitude)

    return torch.from_numpy(coords_np).to(device=device, dtype=coords.dtype)




def augment_point_cloud_torch(points: torch.Tensor,
                              bboxes: torch.Tensor | None = None,
                              data_aug_conf = None,
                              p_scale: float = 0.5,
                              p_rot_z: float = 0.5,
                              p_trans_x: float = 0.5,
                              p_trans_y: float = 0.5,
                              p_trans_z: float = 0.5,
                              ):
    """
    参数
    ----
    points : (N, 3 or D) float Tensor
    bboxes : (K, 6) float Tensor or None
             (x, y, z, w, l, h) — w∥x轴, l∥y轴, h∥z轴
    返回
    ----
    若 bboxes 为 None → 仅返回增强后的 points
    否则        → 返回 (points', bboxes')
    """
    device, dtype = points.device, points.dtype
    pts = points.clone()

    if bboxes is not None:
        boxes = bboxes.clone()
        ctrs = boxes[:, 0:3]                                # (x, y, z)
        dims_xyz = boxes[:, [3, 4, 5]]                      # (w, l, h) → (x轴, y轴, z轴) 尺寸
    else:
        ctrs = dims_xyz = None

    # ── 1. 缩放 ─────────────────────────────────────
    if data_aug_conf is None:
        apply_scale = torch.rand(1) < p_scale
        scales = _random_scales(device, dtype) if apply_scale else None
    else:
        apply_scale = data_aug_conf["rescale"]["flag"]
        scales = (torch.tensor(data_aug_conf["rescale"]["scale"], 
                            device=device, dtype=dtype) 
                if apply_scale else None)
    
    if apply_scale:
        pts[:, :3] *= scales
        if bboxes is not None:
            ctrs *= scales
            dims_xyz *= scales
    
    
    # ── 2. 绕 Z 轴旋转 ─────────────────────────────
    if data_aug_conf is None:
        apply_rot_z = torch.rand(1) < p_rot_z
        # Notice: only rotate 90 degrees multiple
        angle = np.random.choice(np.array([0, 0.5, 1.0, 1.5]) * np.pi).item() if apply_rot_z else None
    else:
        apply_rot_z = data_aug_conf["rot_z"]["flag"]
        angle = (np.array(data_aug_conf["rot_z"]["rot"]) * np.pi).item() if apply_rot_z else None
        
    if apply_rot_z:        
        Rz = _rotation_matrix(torch.tensor([0., 0., 1.], device=device), angle, device, dtype)
        pts[:, :3] = pts[:, :3] @ Rz.T
        if bboxes is not None:
            ctrs = ctrs @ Rz.T
            dims_xyz = dims_xyz @ torch.abs(Rz).T

    # ── 3. 沿X, Y, Z轴平移 ─────────────────────────────
    def apply_translation(axis_idx, p_trans, conf_key):
        if data_aug_conf is None:
            apply = torch.rand(1) < p_trans
            trans = random.uniform(-1, 1) if apply else None
        else:
            apply = data_aug_conf[conf_key]["flag"]
            trans = data_aug_conf[conf_key]["trans"] if apply else None

        if apply:
            pts[:, axis_idx] += trans
            if bboxes is not None:
                ctrs[:, axis_idx] += trans
    
    apply_translation(0, p_trans_x, "trans_x")
    apply_translation(1, p_trans_y, "trans_y")
    apply_translation(2, p_trans_z, "trans_z")

    if False:
        pts = _random_jitter(pts, sigma=0.025, clip=0.05, ratio=0.8, p=0.9)
        pts = _random_jitter(pts, sigma=0.2, clip=0.2, ratio=0.05, p=0.85)
        pts = _random_jitter(pts, sigma=0.4, clip=1.0, ratio=0.001, p=0.75)
        pts = _random_jitter(pts, sigma=0.5, clip=4.0, ratio=0.0005, p=0.7)
        pts = _elastic_distortion(pts, distortion_params=[[0.2, 0.4], [0.8, 1.6]], p=[0.85, 0.5])


    # Should not rotate X or Y
    # # ── 3. 绕 Y 轴旋转 ─────────────────────────────
    # if torch.rand(1) < p_rot_y:
    #     angle = torch.empty(1, device=device).uniform_(-0.1309, 0.1309).item()
    #     Ry = _rotation_matrix(torch.tensor([0., 1., 0.], device=device), angle, device, dtype)
    #     pts[:, :3] = pts[:, :3] @ Ry.T
    #     if bboxes is not None:
    #         ctrs = ctrs @ Ry.T
    #         dims_xyz = dims_xyz @ torch.abs(Ry).T

    # # ── 4. 绕 X 轴旋转 ─────────────────────────────
    # if torch.rand(1) < p_rot_x:
    #     angle = torch.empty(1, device=device).uniform_(-0.1309, 0.1309).item()
    #     Rx = _rotation_matrix(torch.tensor([1., 0., 0.], device=device), angle, device, dtype)
    #     pts[:, :3] = pts[:, :3] @ Rx.T
    #     if bboxes is not None:
    #         ctrs = ctrs @ Rx.T
    #         dims_xyz = dims_xyz @ torch.abs(Rx).T

    # ── 返回 ───────────────────────────────────────
    if bboxes is None:
        return pts, None
    else:
        # 尺寸重建回 (w, l, h) 顺序
        boxes_out = torch.cat([ctrs, dims_xyz], dim=-1)
        return pts, boxes_out



class PointNet(nn.Module):
    def __init__(self, input_dim, output_dim, use_xyz=False):
        super().__init__()
        self.use_xyz = use_xyz
        self.input_dim = input_dim + 3 if use_xyz else input_dim
        self.output_dim = output_dim
        self.mlp = nn.Linear(self.input_dim, self.output_dim)
        
        # add temperature here to prevent speed problem in deepspeed
        temperature = 0.1
        self.logit_scale = nn.Parameter(torch.tensor(1 / temperature).log())
        
    def forward(self, feature, xyz=None, pooling_strategy="max"):
        """
        Args:
            feature: (B, ..., K, feat_dim)
            xyz: (B, ..., K, 3)
        return: 
            torch.tensor: (B, N, out_dim)
        """
        
        if self.use_xyz and xyz is not None:
            norm_xyz = xyz - torch.mean(xyz, dim=-2, keepdim=True)
            feature = torch.cat([feature, norm_xyz], dim=-1)
        
        out_feat = self.mlp(feature)
        
        if pooling_strategy == "max":
            out_feat = torch.max(out_feat, dim=-2)[0]
        elif pooling_strategy == "average":
            out_feat = torch.mean(out_feat, dim=-2)[0]

        return out_feat
    

class CrossAttnFusion(nn.Module):
    def __init__(self, semantic_feat_dim, spatial_feat_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.semantic_feat_dim = semantic_feat_dim
        self.spatial_feat_dim = spatial_feat_dim
        self.num_heads = num_heads
        self.head_dim = spatial_feat_dim // num_heads
        
        assert spatial_feat_dim % num_heads == 0, "spatial_feat_dim must be divisible by num_heads"
        
        # Linear projections for query (semantic), key and value (spatial)
        self.q_proj = nn.Linear(semantic_feat_dim, spatial_feat_dim, bias=False)
        self.k_proj = nn.Linear(spatial_feat_dim, spatial_feat_dim, bias=False)
        self.v_proj = nn.Linear(spatial_feat_dim, spatial_feat_dim, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, semantic_feat, spatial_feat):
        """
        Args:
            semantic_feat: [..., 1, semantic_feat_dim] - query features
            spatial_feat: [..., n, spatial_feat_dim] - key and value features
        
        Returns:
            fused_feat: [..., spatial_feat_dim] - weighted spatial features
        """
        # Get input shapes
        *batch_dims, seq_len_semantic, semantic_dim = semantic_feat.shape
        *batch_dims, seq_len_spatial, spatial_dim = spatial_feat.shape
        
        # Project features
        q = self.q_proj(semantic_feat)  # [..., 1, spatial_feat_dim]
        k = self.k_proj(spatial_feat)   # [..., n, spatial_feat_dim]
        v = self.v_proj(spatial_feat)   # [..., n, spatial_feat_dim]
        
        # Reshape for multi-head attention
        q = q.view(*batch_dims, seq_len_semantic, self.num_heads, self.head_dim)
        k = k.view(*batch_dims, seq_len_spatial, self.num_heads, self.head_dim)
        v = v.view(*batch_dims, seq_len_spatial, self.num_heads, self.head_dim)
        
        # Transpose for attention computation: [..., num_heads, seq_len, head_dim]
        q = q.transpose(-3, -2)  # [..., num_heads, 1, head_dim]
        k = k.transpose(-3, -2)  # [..., num_heads, n, head_dim]
        v = v.transpose(-3, -2)  # [..., num_heads, n, head_dim]
        
        # Compute attention scores
        # q: [..., num_heads, 1, head_dim]
        # k: [..., num_heads, n, head_dim]
        # scores: [..., num_heads, 1, n]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # [..., num_heads, 1, n]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # attn_weights: [..., num_heads, 1, n]
        # v: [..., num_heads, n, head_dim]
        # attended: [..., num_heads, 1, head_dim]
        attended = torch.matmul(attn_weights, v)
        
        # Reshape back to original format
        attended = attended.transpose(-3, -2)  # [..., 1, num_heads, head_dim]
        attended = attended.contiguous().view(*batch_dims, spatial_dim)
                
        # Apply output projection
        # output = self.out_proj(attended)
        output = attended
        
        return output



# class PointNetpp(nn.Module):
#     def __init__(self, input_dim, output_dim, use_xyz=False):
#         super().__init__()
    
#     def forward(self, feature, xyz=None):
#         """
#         Args:
#             feature: (B, ..., K, feat_dim)
#             xyz: (B, ...,N, K, 3)
            
#         """