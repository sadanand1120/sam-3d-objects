#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Reconstruct per-instance 3D assets from SAM3 masks using SAM-3D Objects.

This script is designed for offline pipelines:
1) Load one image + one SAM3 .npz mask file.
2) Reconstruct each mask instance with SAM-3D Objects.
3) Save per-instance assets and metadata for later retrieval.
4) Save full pointmap and intrinsics for later scale calibration.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from notebook.inference import Inference, make_scene, interactive_visualizer


@dataclass
class InstanceResult:
    instance_id: int
    mask_index: int
    gs_local_path: str
    pose_path: str


class SAM3DInstanceReconstructor:
    def __init__(
        self,
        config_path: str,
        output_root: str,
        compile_model: bool = False,
        seed: int = 42,
        save_combined_scene: bool = True,
        with_layout_postprocess: bool = False,
        gs_enable_occlusion_check: bool = False,
        gs_enable_manual_alignment: bool = False,
        gs_enable_shape_icp: bool = False,
        gs_enable_rendering_optimization: bool = True,
        gs_min_size: int = 518,
        gs_backend: str = "gsplat",
        gs_alignment_depth_edge_rtol: float = 0.03,
        gs_alignment_flip_xy: bool = False,
        gs_icp_threshold: float = 0.05,
        gs_icp_with_scaling: bool = False,
        gs_icp_max_iteration: int | None = None,
        gs_accept_icp_on_tie: bool = False,
        gs_accept_icp_if_rmse_improves: bool = False,
    ) -> None:
        self.config_path = str(config_path)
        self.output_root = Path(output_root)
        self.seed = int(seed)
        self.save_combined_scene = bool(save_combined_scene)
        self.with_layout_postprocess = bool(with_layout_postprocess)
        self.gs_enable_occlusion_check = bool(gs_enable_occlusion_check)
        self.gs_enable_manual_alignment = bool(gs_enable_manual_alignment)
        self.gs_enable_shape_icp = bool(gs_enable_shape_icp)
        self.gs_enable_rendering_optimization = bool(gs_enable_rendering_optimization)
        self.gs_min_size = int(gs_min_size)
        self.gs_backend = str(gs_backend)
        self.gs_alignment_depth_edge_rtol = float(gs_alignment_depth_edge_rtol)
        self.gs_alignment_flip_xy = bool(gs_alignment_flip_xy)
        self.gs_icp_threshold = float(gs_icp_threshold)
        self.gs_icp_with_scaling = bool(gs_icp_with_scaling)
        self.gs_icp_max_iteration = None if gs_icp_max_iteration is None else int(gs_icp_max_iteration)
        self.gs_accept_icp_on_tie = bool(gs_accept_icp_on_tie)
        self.gs_accept_icp_if_rmse_improves = bool(gs_accept_icp_if_rmse_improves)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.inference = Inference(self.config_path, compile=compile_model)

    @staticmethod
    def load_image_uint8(image_path: str) -> np.ndarray:
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        return image

    @staticmethod
    def load_masks_npz(npz_path: str) -> np.ndarray:
        data = np.load(npz_path)
        if "masks" not in data:
            raise KeyError(f"Expected key 'masks' in {npz_path}, got keys: {list(data.keys())}")
        masks = data["masks"]
        if masks.ndim == 3:
            masks = masks[:, None, ...]
        if masks.ndim != 4 or masks.shape[1] != 1:
            raise ValueError(f"Expected masks shape (N,1,H,W), got {masks.shape}")
        return masks.astype(bool)

    def compute_full_pointmap(self, image_rgb: np.ndarray) -> dict[str, np.ndarray]:
        h, w = image_rgb.shape[:2]
        alpha = np.ones((h, w, 1), dtype=np.uint8) * 255
        rgba = np.concatenate([image_rgb, alpha], axis=-1)

        pm_dict = self.inference._pipeline.compute_pointmap(rgba)
        pointmap_chw = pm_dict["pointmap"].detach().cpu().numpy().astype(np.float32)
        pointmap_hwc = np.transpose(pointmap_chw, (1, 2, 0))
        intrinsics = pm_dict["intrinsics"].detach().cpu().numpy().astype(np.float32)
        pointmap_colors_chw = pm_dict["pts_color"].detach().cpu().numpy().astype(np.float32)
        pointmap_colors_hwc = np.transpose(pointmap_colors_chw, (1, 2, 0))
        return {
            "pointmap_hwc": pointmap_hwc,
            "intrinsics": intrinsics,
            "pointmap_colors_hwc": pointmap_colors_hwc,
        }

    @staticmethod
    def _to_serializable(obj: Any) -> Any:
        if hasattr(obj, "detach"):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    @staticmethod
    def show_scene(scene_ply_path: str) -> None:
        scene_ply_path = str(scene_ply_path)
        if not Path(scene_ply_path).exists():
            raise FileNotFoundError(f"Scene file not found: {scene_ply_path}")
        interactive_visualizer(scene_ply_path)

    def reconstruct_single_instance(
        self,
        image_rgb: np.ndarray,
        mask_hw: np.ndarray,
        full_pointmap_hwc: np.ndarray,
    ) -> dict[str, Any]:
        pointmap = full_pointmap_hwc
        if isinstance(pointmap, np.ndarray):
            pointmap = torch.from_numpy(pointmap.astype(np.float32, copy=False))

        rgba = self.inference.merge_mask_to_rgba(image_rgb, mask_hw)
        output = self.inference._pipeline.run(
            rgba,
            None,
            self.seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=self.with_layout_postprocess,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=pointmap,
            gs_post_enable_occlusion_check=self.gs_enable_occlusion_check,
            gs_post_enable_manual_alignment=self.gs_enable_manual_alignment,
            gs_post_enable_shape_icp=self.gs_enable_shape_icp,
            gs_post_enable_rendering_optimization=self.gs_enable_rendering_optimization,
            gs_post_min_size=self.gs_min_size,
            gs_post_backend=self.gs_backend,
            gs_post_alignment_depth_edge_rtol=self.gs_alignment_depth_edge_rtol,
            gs_post_alignment_flip_xy=self.gs_alignment_flip_xy,
            gs_post_icp_threshold=self.gs_icp_threshold,
            gs_post_icp_with_scaling=self.gs_icp_with_scaling,
            gs_post_icp_max_iteration=self.gs_icp_max_iteration,
            gs_post_accept_icp_on_tie=self.gs_accept_icp_on_tie,
            gs_post_accept_icp_if_rmse_improves=self.gs_accept_icp_if_rmse_improves,
        )
        return output

    def save_instance(
        self,
        frame_dir: Path,
        instance_id: int,
        mask_index: int,
        output: dict[str, Any],
    ) -> InstanceResult:
        instance_dir = frame_dir / "instances" / f"instance_{instance_id:03d}"
        instance_dir.mkdir(parents=True, exist_ok=True)

        gs_local_path = instance_dir / "gs_local.ply"
        output["gs"].save_ply(str(gs_local_path))

        pose_meta = {
            "rotation_wxyz_l2c": self._to_serializable(output["rotation"]),
            "translation_l2c": self._to_serializable(output["translation"]),
            "scale_l2c": self._to_serializable(output["scale"]),
        }
        pose_path = instance_dir / "pose_l2c.json"
        pose_path.write_text(json.dumps(pose_meta, indent=2))

        return InstanceResult(
            instance_id=instance_id,
            mask_index=mask_index,
            gs_local_path=str(gs_local_path),
            pose_path=str(pose_path),
        )

    def run(
        self,
        image_path: str,
        sam3_npz_path: str,
        min_mask_area_pixels: int = 16,
    ) -> Path:
        image_path = str(image_path)
        sam3_npz_path = str(sam3_npz_path)

        image_rgb = self.load_image_uint8(image_path)
        masks = self.load_masks_npz(sam3_npz_path)  # (N,1,H,W)
        h, w = image_rgb.shape[:2]
        if masks.shape[-2:] != (h, w):
            raise RuntimeError(
                f"Mask/image size mismatch. masks={masks.shape[-2:]}, image={(h, w)}"
            )

        frame_tag = Path(image_path).stem
        frame_dir = self.output_root / frame_tag
        frame_dir.mkdir(parents=True, exist_ok=True)

        pointmap_bundle = self.compute_full_pointmap(image_rgb)
        pointmap_out = frame_dir / "pointmap_full.npz"
        np.savez_compressed(
            pointmap_out,
            pointmap_hwc=pointmap_bundle["pointmap_hwc"],
            intrinsics=pointmap_bundle["intrinsics"],
            pointmap_colors_hwc=pointmap_bundle["pointmap_colors_hwc"],
        )

        outputs_for_scene = []
        instance_records: list[InstanceResult] = []
        instance_id = 0
        for mask_index in range(masks.shape[0]):
            mask_hw = masks[mask_index, 0]
            area = int(mask_hw.sum())
            if area < min_mask_area_pixels:
                continue

            pred = self.reconstruct_single_instance(
                image_rgb=image_rgb,
                mask_hw=mask_hw,
                full_pointmap_hwc=pointmap_bundle["pointmap_hwc"],
            )
            outputs_for_scene.append(pred)
            result = self.save_instance(
                frame_dir=frame_dir,
                instance_id=instance_id,
                mask_index=mask_index,
                output=pred,
            )
            instance_records.append(result)
            instance_id += 1

        scene_path = None
        if self.save_combined_scene and outputs_for_scene:
            scene_dir = frame_dir / "scene"
            scene_dir.mkdir(parents=True, exist_ok=True)
            scene_gs = make_scene(*outputs_for_scene)
            scene_path = scene_dir / "scene_cam.ply"
            scene_gs.save_ply(str(scene_path))

        manifest = {
            "image_path": image_path,
            "sam3_npz_path": sam3_npz_path,
            "pointmap_file": str(pointmap_out),
            "scene_cam_ply": str(scene_path) if scene_path is not None else None,
            "instances": [r.__dict__ for r in instance_records],
        }
        manifest_path = frame_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return manifest_path


if __name__ == "__main__":
    config_path = "checkpoints/hf/pipeline.yaml"
    output_root = "testing_outputs/sam3d_instances"

    reconstructor = SAM3DInstanceReconstructor(
        config_path=config_path,
        output_root=output_root,
        compile_model=False,
        seed=42,
        save_combined_scene=True,
    )

    # Demo 1
    image_path_1 = "/robodata/smodak/repos/f3rm/datasets/f3rm/fresh/objaverse/car2/images/frame_00001.png"
    sam3_npz_path_1 = "/robodata/smodak/repos/f3rm/datasets/f3rm/fresh/objaverse/car2/features/sam3_/image_000000.npz"
    manifest_path_1 = reconstructor.run(
        image_path=image_path_1,
        sam3_npz_path=sam3_npz_path_1,
        min_mask_area_pixels=16,
    )
    print(f"Demo 1 done. Manifest written to: {manifest_path_1}")
    manifest_1 = json.loads(manifest_path_1.read_text())
    if manifest_1.get("scene_cam_ply"):
        # This launches a Gradio app and blocks until closed (ctrl+C)
        reconstructor.show_scene(manifest_1["scene_cam_ply"])

    # Demo 3 (tight-to-pointmap variant of Demo 1)
    reconstructor_tight = SAM3DInstanceReconstructor(
        config_path=config_path,
        output_root=f"{output_root}_tight",
        compile_model=False,
        seed=42,
        save_combined_scene=True,
        with_layout_postprocess=True,
        gs_enable_occlusion_check=False,
        gs_enable_manual_alignment=True,
        gs_enable_shape_icp=True,
        gs_enable_rendering_optimization=False,
        gs_min_size=518,
        gs_backend="gsplat",
        gs_alignment_depth_edge_rtol=0.08,
        gs_alignment_flip_xy=False,
        gs_icp_threshold=0.08,
        gs_icp_with_scaling=False,
        gs_icp_max_iteration=80,
        gs_accept_icp_on_tie=True,
        gs_accept_icp_if_rmse_improves=False,
    )
    manifest_path_3 = reconstructor_tight.run(
        image_path=image_path_1,
        sam3_npz_path=sam3_npz_path_1,
        min_mask_area_pixels=16,
    )
    print(f"Demo 3 (tight) done. Manifest written to: {manifest_path_3}")
    manifest_3 = json.loads(manifest_path_3.read_text())
    if manifest_3.get("scene_cam_ply"):
        reconstructor_tight.show_scene(manifest_3["scene_cam_ply"])

    # Demo 2 (same reconstructor object reused for a second image)
    image_path_2 = "/robodata/smodak/repos/f3rm/datasets/f3rm/fresh/objaverse/car2/images/frame_00002.png"
    sam3_npz_path_2 = "/robodata/smodak/repos/f3rm/datasets/f3rm/fresh/objaverse/car2/features/sam3_/image_000001.npz"
    manifest_path_2 = reconstructor.run(
        image_path=image_path_2,
        sam3_npz_path=sam3_npz_path_2,
        min_mask_area_pixels=16,
    )
    print(f"Demo 2 done. Manifest written to: {manifest_path_2}")
    manifest_2 = json.loads(manifest_path_2.read_text())
    if manifest_2.get("scene_cam_ply"):
        reconstructor.show_scene(manifest_2["scene_cam_ply"])
