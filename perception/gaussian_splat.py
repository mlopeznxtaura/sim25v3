"""
Gaussian Splatting scene reconstruction for robot perception.
Reconstructs real-world environments as 3D Gaussians for sim-to-real transfer.
Integrates with nerfstudio for training and Open3D for point cloud processing.
"""
import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
try:
    import torch
    import open3d as o3d
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GaussianSplatReconstructor:
    """
    Reconstruct real-world environments using 3D Gaussian Splatting.
    Pipeline: images/video -> COLMAP -> nerfstudio gsplat training -> point cloud export
    Used to create realistic visual backgrounds for sim training.
    """

    def __init__(
        self,
        workspace: str = "./splat_workspace",
        device: str = "cuda",
    ):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.device = device

    def prepare_data_from_video(self, video_path: str, fps: int = 2) -> str:
        """
        Extract frames from video for NeRF/GSplat training.
        Returns path to frames directory.
        """
        import subprocess
        frames_dir = self.workspace / "frames"
        frames_dir.mkdir(exist_ok=True)

        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", "2",
            str(frames_dir / "frame_%04d.jpg")
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

        n_frames = len(list(frames_dir.glob("*.jpg")))
        print(f"[GSplat] Extracted {n_frames} frames at {fps}fps")
        return str(frames_dir)

    def run_colmap(self, frames_dir: str) -> str:
        """Run COLMAP SfM to get camera poses."""
        import subprocess
        sparse_dir = self.workspace / "sparse"
        db_path = self.workspace / "colmap.db"

        # Feature extraction
        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", frames_dir,
            "--ImageReader.single_camera", "1",
        ], check=True)

        # Feature matching
        subprocess.run([
            "colmap", "exhaustive_matcher",
            "--database_path", str(db_path),
        ], check=True)

        # Sparse reconstruction
        sparse_dir.mkdir(exist_ok=True)
        subprocess.run([
            "colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", frames_dir,
            "--output_path", str(sparse_dir),
        ], check=True)

        print(f"[GSplat] COLMAP reconstruction at {sparse_dir}")
        return str(sparse_dir)

    def train_gaussian_splat(
        self,
        data_dir: str,
        method: str = "splatfacto",
        max_num_iterations: int = 30_000,
    ) -> str:
        """
        Train 3D Gaussian Splatting model using nerfstudio.
        method: 'splatfacto' (fast) or 'gaussian-splatting' (original 3DGS)
        Returns path to trained model.
        """
        output_dir = self.workspace / "model" / method
        cmd = [
            "ns-train", method,
            "--data", data_dir,
            "--output-dir", str(output_dir),
            "--max-num-iterations", str(max_num_iterations),
            "--pipeline.model.cull-alpha-thresh", "0.005",
            "--pipeline.model.densify-grad-thresh", "0.0002",
            "--viewer.quit-on-train-completion", "True",
        ]

        import subprocess
        print(f"[GSplat] Training {method} for {max_num_iterations} iterations...")
        subprocess.run(cmd, check=True)
        print(f"[GSplat] Model saved to {output_dir}")
        return str(output_dir)

    def export_point_cloud(self, model_dir: str, output_path: str = None) -> str:
        """
        Export trained Gaussian Splat as point cloud for sim integration.
        Returns path to .ply file.
        """
        output_path = output_path or str(self.workspace / "scene_pointcloud.ply")
        cmd = [
            "ns-export", "pointcloud",
            "--load-config", str(Path(model_dir) / "config.yml"),
            "--output-dir", str(Path(output_path).parent),
            "--num-points", "1000000",
            "--remove-outliers", "True",
        ]

        import subprocess
        subprocess.run(cmd, check=True)
        print(f"[GSplat] Point cloud exported to {output_path}")
        return output_path

    def load_point_cloud_as_open3d(self, ply_path: str) -> "o3d.geometry.PointCloud":
        """Load exported point cloud into Open3D for processing."""
        pcd = o3d.io.read_point_cloud(ply_path)
        print(f"[Open3D] Loaded {len(pcd.points):,} points")
        return pcd

    def crop_workspace_region(
        self,
        pcd: "o3d.geometry.PointCloud",
        bounds_min: List[float] = [-0.5, -0.5, 0.0],
        bounds_max: List[float] = [1.0, 0.5, 0.5],
    ) -> "o3d.geometry.PointCloud":
        """Crop point cloud to robot workspace volume."""
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=bounds_min,
            max_bound=bounds_max,
        )
        cropped = pcd.crop(bbox)
        print(f"[Open3D] Cropped to {len(cropped.points):,} points in workspace")
        return cropped

    def fit_object_bounding_boxes(
        self, pcd: "o3d.geometry.PointCloud", eps: float = 0.02, min_points: int = 50
    ) -> List[dict]:
        """
        Segment objects in scene using DBSCAN clustering, fit bounding boxes.
        Returns list of {center, extent, rotation, n_points} dicts.
        """
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
        n_clusters = labels.max() + 1
        print(f"[Open3D] Found {n_clusters} object clusters")

        objects = []
        for i in range(n_clusters):
            mask = labels == i
            cluster_pcd = pcd.select_by_index(np.where(mask)[0])
            obb = cluster_pcd.get_oriented_bounding_box()
            objects.append({
                "center": np.array(obb.center),
                "extent": np.array(obb.extent),
                "rotation": np.array(obb.R),
                "n_points": mask.sum(),
            })

        return objects
