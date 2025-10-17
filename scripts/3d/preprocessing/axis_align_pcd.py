import open3d as o3d
import numpy as np
import os
import argparse
from sklearn.decomposition import PCA


def align_scene_to_z_up(pcd, save_pcd_path=None, save_transform_path=None, visualize=False, fit_ground=True):
    print("=== 开始对齐 ===")
    points = np.asarray(pcd.points)
    print(f"[DEBUG] 原始点云点数: {points.shape[0]}")
    centroid = np.mean(points, axis=0)
    print(f"[DEBUG] 点云质心: {centroid}")

    distance_threshold = 0.02
    R_ground = np.eye(3)

    if fit_ground:
        print("[DEBUG] 开始检测地面平面")
        plane_model, ground_inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=500
        )
        [a, b, c, d] = plane_model
        ground_normal = np.array([a, b, c])
        print(f"[DEBUG] 地面法向量: {ground_normal}, d={d}, 内点数={len(ground_inliers)}")
        
        z_axis = np.array([0, 0, 1])
        ground_normal = ground_normal / np.linalg.norm(ground_normal)
        rotation_axis = np.cross(ground_normal, z_axis)

        if np.linalg.norm(rotation_axis) < 1e-6:
            print("[DEBUG] 地面已接近Z轴，不需要旋转")
            rotation_axis = np.array([0, 1, 0])
        else:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        cos_theta = np.dot(ground_normal, z_axis)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        print(f"[DEBUG] 地面旋转角度: {np.degrees(angle):.4f}°, 旋转轴: {rotation_axis}")

        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        R_ground = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        centered_points = points - centroid
        ground_rotated = (R_ground @ centered_points.T).T
        all_indices = np.arange(len(points))
        non_ground_indices = np.setdiff1d(all_indices, ground_inliers)
        non_ground_points = ground_rotated[non_ground_indices]

        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(non_ground_points)
        remaining_pcd = temp_pcd
    else:
        remaining_pcd = pcd

    wall_directions = []
    max_points = 0
    best_direction = None
    R_walls = np.eye(3)

    print("[DEBUG] 开始检测墙面")
    for i in range(6):
        if len(remaining_pcd.points) < 100:
            break

        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=500
        )
        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])

        if abs(normal[2]) < 0.1 and np.linalg.norm(normal[:2]) > 0.5:
            horizontal_dir = normal[:2] / np.linalg.norm(normal[:2])
            if max_points < len(inliers):
                max_points = len(inliers)
                best_direction = horizontal_dir
                angle = np.arctan2(best_direction[1], best_direction[0])
                R_walls = np.array([
                    [np.cos(-angle), -np.sin(-angle), 0],
                    [np.sin(-angle), np.cos(-angle), 0],
                    [0, 0, 1]
                ])
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

    if fit_ground:
        combined_rotation = R_walls @ R_ground
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = combined_rotation
        transform_matrix[:3, 3] = -combined_rotation @ centroid
    else:
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R_walls
        transform_matrix[:3, 3] = -R_walls @ centroid

    aligned_points = (transform_matrix[:3, :3] @ points.T + transform_matrix[:3, 3:4]).T
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned_points)

    if pcd.colors:
        aligned_pcd.colors = pcd.colors

    if save_pcd_path:
        o3d.io.write_point_cloud(save_pcd_path, aligned_pcd)
        print(f"[DEBUG] 对齐后的点云已保存到 {save_pcd_path}")

    if save_transform_path:
        np.savetxt(save_transform_path, transform_matrix, fmt='%.8f')
        print(f"[DEBUG] 变换矩阵已保存到 {save_transform_path}")

    return aligned_pcd, transform_matrix


def process_point_cloud(input_path, output_path, scene_id, split, fit_ground=True):
    print(f"\n=== 处理文件: {input_path} ===")
    pcd = o3d.io.read_point_cloud(input_path)
    print(f"[DEBUG] 读取点云: {len(pcd.points)} points")

    if len(pcd.points) == 0:
        print(f"[ERROR] 点云为空: {input_path}")
        return

    if len(pcd.points) > 500000:
        print("[DEBUG] 点数过多，进行降采样...")
        pcd = pcd.voxel_down_sample(voxel_size=0.02)


    save_pcd_path = f"{output_path}/pointcloud/{split}/{scene_id}_further_aligned.ply"
    save_transform_path = f"{output_path}/axis_align_matrix/{split}/{scene_id}_further_aligned_transform.txt"

    os.makedirs(os.path.dirname(save_pcd_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_transform_path), exist_ok=True)

    transformed_pcd, transform = align_scene_to_z_up(
        pcd,
        save_transform_path=save_transform_path,
        save_pcd_path=save_pcd_path,
        visualize=False,
        fit_ground=fit_ground
    )

    aligned_points = np.asarray(transformed_pcd.points)
    min_z = np.min(aligned_points[:, 2])
    max_z = np.max(aligned_points[:, 2])
    height = max_z - min_z
    print(f"[DEBUG] 对齐后高度: {height:.4f}")

def load_scene_list(scene_list_path):
    """从文件中加载场景列表"""
    scene_list = []
    with open(scene_list_path, "r") as f:
        for line in f:
            scene_list.append(line.strip())
    return scene_list


def main():
    parser = argparse.ArgumentParser(description="处理ScanNetpp点云数据，进行轴对齐")
    parser.add_argument("--scene_list", type=str, 
                       default="data/raw_data/scannetpp/splits/nvs_sem_train.txt",
                       help="训练集场景列表文件路径")
    parser.add_argument("--split", type=str,
                       default="train",
                       help="数据集分割")
    parser.add_argument("--output_dir", type=str,
                       default="data/processed_data/ScanNetpp/",
                       help="输出目录路径")
    parser.add_argument("--data_type", type=str,
                       default="scannetpp",
                       help="数据集类型")

    parser.add_argument("--fit_ground", action="store_true",
                       help="是否进行地面拟合对齐")
    

    args = parser.parse_args()
    
    print(f"场景列表: {args.scene_list}")
    print(f"输出目录: {args.output_dir}")
    print(f"地面拟合: {args.fit_ground}")
    
    # 加载场景列表
    scene_list = load_scene_list(args.scene_list)
    
    print(f"场景数量: {len(scene_list)}")
    
    print("\n=== 开始处理 ===")
    for scene_id in scene_list:
        if args.data_type == "scannetpp":
            input_file = f"data/raw_data/scannetpp/data/{scene_id}/scans/mesh_aligned_0.05.ply"
        elif args.data_type == "arkitscenes":
            split_name = "Training" if args.split == "train" else "Validation"
            input_file = f"data/raw_data/arkitscenes/3dod/{split_name}/{scene_id}/{scene_id}_3dod_mesh.ply"
    
        process_point_cloud(input_file, args.output_dir, scene_id, args.split, fit_ground=args.fit_ground)


if __name__ == "__main__":
    main()

