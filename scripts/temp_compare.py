#!/usr/bin/env python3
import json
import time
from pathlib import Path
from pprint import pformat

import numpy as np
from PIL import Image
from plyfile import PlyData
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
import torch
import viser

from scripts.sam3d_instance_reconstructor import SAM3DInstanceReconstructor
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
from sam3d_objects.model.backbone.tdfy_dit.representations import Gaussian
from sam3d_objects.model.backbone.tdfy_dit.renderers.gaussian_render import GaussianRenderer
from sam3d_objects.pipeline.layout_post_optimization_utils import (
    flip_coords_pytorch3d_to_opencv,
    get_gs_transformed,
)


def downsample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    keep = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[keep]


def solve_rigid_transform_fixed_scale(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve Y ~= X @ R + t (row-vector form), fixed scale=1."""
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean

    cov = src_centered.T @ tgt_centered
    u, _, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = tgt_mean - src_mean @ r
    return r.astype(np.float32), t.astype(np.float32)


def solve_similarity_transform_uniform_scale(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve Y ~= s * X @ R + t (row-vector form), uniform scale."""
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean

    cov = src_centered.T @ tgt_centered
    u, _, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T

    src_rot = src_centered @ r
    denom = float(np.sum(src_rot * src_rot)) + 1e-12
    scale = float(np.sum(src_rot * tgt_centered) / denom)
    t = tgt_mean - scale * (src_mean @ r)
    return r.astype(np.float32), t.astype(np.float32), scale


def solve_camera_origin_scale(source: np.ndarray, target: np.ndarray) -> float:
    """Solve target ~= k * source with camera origin fixed."""
    denom = float(np.sum(source * source)) + 1e-12
    return float(np.sum(source * target) / denom)


def compute_alignment_metrics(source: np.ndarray, target: np.ndarray) -> dict:
    residual_vec = source - target
    residual = np.linalg.norm(residual_vec, ord=2, axis=1)
    depth_residual = np.abs(residual_vec[:, 2])
    return {
        "count": int(source.shape[0]),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "median": float(np.median(residual)),
        "p90": float(np.quantile(residual, 0.90)),
        "depth_rmse": float(np.sqrt(np.mean(depth_residual ** 2))),
    }


def save_side_by_side(left_rgb: np.ndarray, right_rgb: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.concatenate([left_rgb, right_rgb], axis=1)
    Image.fromarray(canvas).save(out_path)


def pretty_pose_block(name: str, pose_dict: dict) -> str:
    return f"{name}:\n{pformat(pose_dict, sort_dicts=False, width=100)}"


def build_match_overlay(image_rgb: np.ndarray, mask_hw: np.ndarray, matched_hw: np.ndarray) -> np.ndarray:
    """Green=matched in-mask pixels, red=in-mask unmatched."""
    overlay = image_rgb.copy()
    overlay[(mask_hw & ~matched_hw)] = np.array([255, 0, 0], dtype=np.uint8)
    overlay[matched_hw] = np.array([0, 255, 0], dtype=np.uint8)
    return overlay


def build_pointcloud_overlay(
    image_rgb: np.ndarray,
    black_valid_hw: np.ndarray,
    color_valid_hw: np.ndarray,
    color_rgb: np.ndarray,
) -> np.ndarray:
    overlay = image_rgb.copy()
    overlay[black_valid_hw] = np.array([0, 0, 0], dtype=np.uint8)
    overlay[color_valid_hw] = color_rgb.astype(np.uint8)
    return overlay


def load_gaussian_from_ply(ply_path: Path, device: str = "cuda") -> tuple[Gaussian, np.ndarray, np.ndarray]:
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    points_local = np.stack(
        [
            np.asarray(vertex["x"]),
            np.asarray(vertex["y"]),
            np.asarray(vertex["z"]),
        ],
        axis=1,
    ).astype(np.float32)
    point_rgb_u8 = extract_point_rgb_u8(vertex)

    xyz_min = points_local.min(axis=0)
    xyz_extent = np.maximum(points_local.max(axis=0) - xyz_min, 1e-3)
    gaussian_local = Gaussian(aabb=np.concatenate([xyz_min, xyz_extent]).tolist(), sh_degree=0, device=device)
    gaussian_local.load_ply(str(ply_path))
    return gaussian_local, points_local, point_rgb_u8


class ProjectiveICPFixedScale:
    def __init__(
        self,
        min_correspondences: int = 2000,
        residual_mad_sigma: float = 3.0,
        gs_alpha_threshold: float = 1e-3,
        gs_render_depth_mode: str = "ED",
        max_iters: int = 5,
    ) -> None:
        self.min_correspondences = int(min_correspondences)
        self.residual_mad_sigma = float(residual_mad_sigma)
        self.gs_alpha_threshold = float(gs_alpha_threshold)
        self.gs_render_depth_mode = str(gs_render_depth_mode)
        self.max_iters = int(max_iters)

    @staticmethod
    def _intrinsics_to_pixels(intrinsics: np.ndarray, h: int, w: int) -> tuple[float, float, float, float]:
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        if cx <= 2.0 and cy <= 2.0:
            fx *= w
            fy *= h
            cx *= w
            cy *= h
        return fx, fy, cx, cy

    @staticmethod
    def points_pytorch3d_to_opencv(points_cam: np.ndarray) -> np.ndarray:
        """SAM3D pointmaps live in PyTorch3D camera convention, but image projection uses OpenCV axes."""
        points_opencv = points_cam.copy()
        points_opencv[:, 0] *= -1.0
        points_opencv[:, 1] *= -1.0
        return points_opencv

    @staticmethod
    def normalize_intrinsics(intrinsics: np.ndarray, h: int, w: int) -> np.ndarray:
        intrinsics_norm = intrinsics.astype(np.float32).copy()
        if intrinsics_norm[0, 2] > 2.0 or intrinsics_norm[1, 2] > 2.0:
            intrinsics_norm[0, 0] /= w
            intrinsics_norm[0, 2] /= w
            intrinsics_norm[1, 1] /= h
            intrinsics_norm[1, 2] /= h
        return intrinsics_norm

    @classmethod
    def depth_to_pointmap_pytorch3d(cls, depth_hw: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        h, w = depth_hw.shape
        fx, fy, cx, cy = cls._intrinsics_to_pixels(intrinsics, h, w)
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        uu, vv = np.meshgrid(u, v, indexing="xy")
        z = depth_hw.astype(np.float32)
        x_opencv = (uu - cx) * z / fx
        y_opencv = (vv - cy) * z / fy
        pointmap_hwc = np.stack([-x_opencv, -y_opencv, z], axis=-1)
        pointmap_hwc[~np.isfinite(z)] = np.nan
        return pointmap_hwc

    @staticmethod
    def render_projected_buffers(
        points_cam: np.ndarray,
        point_rgb_u8: np.ndarray,
        intrinsics: np.ndarray,
        h: int,
        w: int,
    ) -> dict:
        """
        Project to image and z-buffer by closest depth per pixel.
        Returns:
          depth_hw: (H,W)
          valid_hw: (H,W) bool
          index_hw: (H,W) int (point index selected, -1 if invalid)
          image_rgb_hw3: (H,W,3) uint8 projected RGB
          pointmap_hwc: (H,W,3) float32 3D camera point at selected pixel, kept in PyTorch3D frame
        """
        fx, fy, cx, cy = ProjectiveICPFixedScale._intrinsics_to_pixels(intrinsics, h, w)
        points_cam_project = ProjectiveICPFixedScale.points_pytorch3d_to_opencv(points_cam)

        z = points_cam_project[:, 2]
        x = points_cam_project[:, 0]
        y = points_cam_project[:, 1]

        valid = z > 1e-6
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        valid &= (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)

        depth_hw = np.full((h, w), np.inf, dtype=np.float32)
        index_hw = np.full((h, w), -1, dtype=np.int32)

        if np.any(valid):
            idx = np.nonzero(valid)[0]
            z_valid = z[idx]
            pix = vi[idx] * w + ui[idx]

            order = np.argsort(z_valid)  # near->far
            pix_sorted = pix[order]
            idx_sorted = idx[order]
            z_sorted = z_valid[order]

            unique_pix, first = np.unique(pix_sorted, return_index=True)
            chosen_idx = idx_sorted[first]
            chosen_z = z_sorted[first]

            depth_hw.ravel()[unique_pix] = chosen_z
            index_hw.ravel()[unique_pix] = chosen_idx

        valid_hw = index_hw >= 0

        image_rgb_hw3 = np.zeros((h, w, 3), dtype=np.uint8)
        pointmap_hwc = np.full((h, w, 3), np.nan, dtype=np.float32)

        if np.any(valid_hw):
            chosen = index_hw[valid_hw]
            image_rgb_hw3[valid_hw] = point_rgb_u8[chosen]
            pointmap_hwc[valid_hw] = points_cam[chosen]

        return {
            "depth_hw": depth_hw,
            "valid_hw": valid_hw,
            "index_hw": index_hw,
            "image_rgb_hw3": image_rgb_hw3,
            "pointmap_hwc": pointmap_hwc,
        }

    def render_gaussian_buffers(
        self,
        gaussian_local: Gaussian,
        scale_l2c: np.ndarray,
        r_l2c: np.ndarray,
        t_l2c: np.ndarray,
        intrinsics: np.ndarray,
        h: int,
        w: int,
    ) -> dict:
        device = gaussian_local.device
        scale_t = torch.from_numpy(scale_l2c.astype(np.float32)).to(device)[None, :]
        rotation_t = torch.from_numpy(r_l2c.astype(np.float32)).to(device)[None, :, :]
        translation_t = torch.from_numpy(t_l2c.astype(np.float32)).to(device)[None, :]
        tfm_ori = compose_transform(scale=scale_t, rotation=rotation_t, translation=translation_t)

        gaussian_cam, _ = get_gs_transformed(gaussian_local, tfm_ori, scale_t[0], device)
        points_cam = gaussian_cam.get_xyz.detach().cpu().numpy().astype(np.float32)
        flip_coords_pytorch3d_to_opencv(gaussian_cam)

        intrinsics_norm = torch.from_numpy(self.normalize_intrinsics(intrinsics, h, w)).to(device)
        renderer = GaussianRenderer(
            {
                "image_height": h,
                "image_width": w,
                "near": 0.01,
                "far": 100.0,
                "ssaa": 1,
                "bg_color": (1.0, 1.0, 1.0),
                "backend": "gsplat",
            }
        )

        with torch.no_grad():
            rendered = renderer.render(
                gaussian_cam,
                torch.eye(4, device=device, dtype=torch.float32),
                intrinsics_norm,
                render_mode=f"RGB+{self.gs_render_depth_mode}",
            )

        depth_hw = rendered["depth"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        alpha_hw = rendered["alpha"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        color_hw3 = rendered["color"].permute(1, 2, 0).detach().cpu().clamp(0.0, 1.0).numpy()
        image_rgb_hw3 = (
            (color_hw3 * alpha_hw[..., None] + (1.0 - alpha_hw[..., None])) * 255.0
        ).clip(0.0, 255.0).astype(np.uint8)
        valid_hw = np.isfinite(depth_hw) & (depth_hw > 1e-6) & (alpha_hw > self.gs_alpha_threshold)
        pointmap_hwc = self.depth_to_pointmap_pytorch3d(depth_hw, intrinsics)
        pointmap_hwc[~valid_hw] = np.nan

        return {
            "depth_hw": depth_hw,
            "alpha_hw": alpha_hw,
            "valid_hw": valid_hw,
            "image_rgb_hw3": image_rgb_hw3,
            "pointmap_hwc": pointmap_hwc.astype(np.float32),
            "points_cam": points_cam,
        }

    def run(
        self,
        gaussian_local: Gaussian,
        scale_l2c: np.ndarray,
        r_l2c: np.ndarray,
        t_l2c: np.ndarray,
        pointmap_hwc_target: np.ndarray,
        mask_hw: np.ndarray,
        intrinsics: np.ndarray,
    ) -> dict:
        scale_l2c = scale_l2c.astype(np.float32)
        r = r_l2c.astype(np.float32).copy()
        t = t_l2c.astype(np.float32).copy()

        h, w = pointmap_hwc_target.shape[:2]
        target_valid = np.isfinite(pointmap_hwc_target).all(axis=-1)
        history = []
        rendered_initial = None
        source_initial = None
        target_initial = None
        source_optimized = None
        metrics_initial = None

        for iter_idx in range(self.max_iters):
            rendered_current = self.render_gaussian_buffers(
                gaussian_local=gaussian_local,
                scale_l2c=scale_l2c,
                r_l2c=r,
                t_l2c=t,
                intrinsics=intrinsics,
                h=h,
                w=w,
            )

            matched_pixels = mask_hw & target_valid & rendered_current["valid_hw"]
            n_corr = int(matched_pixels.sum())
            if n_corr < self.min_correspondences:
                rendered_full = rendered_current
                break

            source_all = rendered_current["pointmap_hwc"][matched_pixels]
            target_all = pointmap_hwc_target[matched_pixels]
            if rendered_initial is None:
                rendered_initial = rendered_current
                source_initial = source_all.copy()
                target_initial = target_all.copy()
                metrics_initial = compute_alignment_metrics(source_all, target_all)

            residual = np.linalg.norm(source_all - target_all, ord=2, axis=1)
            med = float(np.median(residual))
            mad = float(np.median(np.abs(residual - med))) + 1e-6
            thr = med + self.residual_mad_sigma * 1.4826 * mad
            keep = residual <= thr
            n_keep = int(keep.sum())
            if n_keep < self.min_correspondences:
                keep = np.ones_like(residual, dtype=bool)
                n_keep = int(keep.sum())

            src_keep = source_all[keep]
            tgt_keep = target_all[keep]
            rmse_before = float(np.sqrt(np.mean(np.sum((src_keep - tgt_keep) ** 2, axis=1))))

            camera_scale = solve_camera_origin_scale(src_keep, tgt_keep)
            src_after_keep_scale = camera_scale * src_keep
            rmse_after = float(np.sqrt(np.mean(np.sum((src_after_keep_scale - tgt_keep) ** 2, axis=1))))
            accepted = rmse_after + 1e-6 < rmse_before

            applied_scale = 1.0
            optimization_mode = "none"
            source_optimized = source_all.copy()
            if accepted:
                optimization_mode = "camera_origin_scale"
                applied_scale = camera_scale
                scale_l2c = scale_l2c * camera_scale
                t = t * camera_scale
                source_optimized = camera_scale * source_all

            history.append({
                "iter": iter_idx,
                "n_corr": n_corr,
                "n_keep": n_keep,
                "rmse_before": rmse_before,
                "rmse_after": rmse_after,
                "accepted": accepted,
                "mode": optimization_mode,
                "scale": float(applied_scale),
                "residual_median": med,
                "residual_threshold": thr,
            })

            rendered_full = rendered_current
            if not accepted:
                break
        else:
            rendered_full = rendered_current

        if rendered_initial is None:
            return {
                "r_l2c": r,
                "t_l2c": t,
                "scale_l2c": scale_l2c,
                "points_cam": rendered_full["points_cam"],
                "rendered": rendered_full,
                "matched_pixels": mask_hw & target_valid & rendered_full["valid_hw"],
                "history": [{"iter": 0, "n_corr": 0, "n_keep": 0, "rmse": np.nan, "accepted": False}],
                "source_initial": np.empty((0, 3), dtype=np.float32),
                "target_initial": np.empty((0, 3), dtype=np.float32),
                "source_optimized": np.empty((0, 3), dtype=np.float32),
                "source_final": np.empty((0, 3), dtype=np.float32),
                "target_final": np.empty((0, 3), dtype=np.float32),
                "metrics_initial": {"count": 0, "rmse": float("nan"), "median": float("nan"), "p90": float("nan"), "depth_rmse": float("nan")},
                "metrics_final": {"count": 0, "rmse": float("nan"), "median": float("nan"), "p90": float("nan"), "depth_rmse": float("nan")},
            }

        rendered_full = self.render_gaussian_buffers(
            gaussian_local=gaussian_local,
            scale_l2c=scale_l2c,
            r_l2c=r,
            t_l2c=t,
            intrinsics=intrinsics,
            h=h,
            w=w,
        )
        final_matched_pixels = mask_hw & target_valid & rendered_full["valid_hw"]
        source_final = rendered_full["pointmap_hwc"][final_matched_pixels]
        target_final = pointmap_hwc_target[final_matched_pixels]
        metrics_final = compute_alignment_metrics(source_final, target_final) if source_final.shape[0] else {
            "count": 0,
            "rmse": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "depth_rmse": float("nan"),
        }

        return {
            "r_l2c": r,
            "t_l2c": t,
            "scale_l2c": scale_l2c,
            "points_cam": rendered_full["points_cam"],
            "rendered": rendered_full,
            "matched_pixels": final_matched_pixels,
            "history": history,
            "source_initial": source_all,
            "target_initial": target_all,
            "source_optimized": source_optimized,
            "source_final": source_final,
            "target_final": target_final,
            "metrics_initial": metrics_initial,
            "metrics_final": metrics_final,
        }


def extract_point_rgb_u8(vertex) -> np.ndarray:
    names = set(vertex.data.dtype.names)

    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack([
            np.asarray(vertex["red"]),
            np.asarray(vertex["green"]),
            np.asarray(vertex["blue"]),
        ], axis=1)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    # 3DGS-style SH DC colors.
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        dc = np.stack([
            np.asarray(vertex["f_dc_0"]),
            np.asarray(vertex["f_dc_1"]),
            np.asarray(vertex["f_dc_2"]),
        ], axis=1).astype(np.float32)
        rgb = np.clip(0.5 + 0.28209479177387814 * dc, 0.0, 1.0)
        return (255.0 * rgb).astype(np.uint8)

    # fallback: white
    return np.full((len(vertex), 3), 255, dtype=np.uint8)


def load_manifest_bundle(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    pointmap_npz = np.load(Path(manifest["pointmap_file"]))
    pointmap_hwc = pointmap_npz["pointmap_hwc"].astype(np.float32)
    intrinsics = pointmap_npz["intrinsics"].astype(np.float32)

    image_rgb = np.asarray(Image.open(Path(manifest["image_path"])).convert("RGB"), dtype=np.uint8)
    if image_rgb.shape[:2] != pointmap_hwc.shape[:2]:
        raise ValueError(
            f"Image/pointmap shape mismatch: image={image_rgb.shape[:2]} pointmap={pointmap_hwc.shape[:2]}"
        )

    instance = manifest["instances"][0]
    mask_index = int(instance["mask_index"])

    sam3 = np.load(Path(manifest["sam3_npz_path"]))
    masks = sam3["masks"]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    mask = masks[mask_index].astype(bool)

    gaussian_local, points_local, point_rgb_u8 = load_gaussian_from_ply(Path(instance["gs_local_path"]))

    pose = json.loads(Path(instance["pose_path"]).read_text())
    quat_wxyz = np.asarray(pose["rotation_wxyz_l2c"], dtype=np.float32).reshape(4)
    trans_l2c = np.asarray(pose["translation_l2c"], dtype=np.float32).reshape(3)
    scale_l2c = np.asarray(pose["scale_l2c"], dtype=np.float32).reshape(3)

    r_l2c = quaternion_to_matrix(torch.from_numpy(quat_wxyz).float().unsqueeze(0))[0].cpu().numpy().astype(np.float32)
    model_points = (points_local * scale_l2c[None, :]) @ r_l2c + trans_l2c[None, :]

    return {
        "image_rgb": image_rgb,
        "mask_index": mask_index,
        "mask": mask,
        "intrinsics": intrinsics,
        "pointmap_hwc": pointmap_hwc,
        "quat_wxyz_l2c": quat_wxyz,
        "gaussian_local": gaussian_local,
        "points_local": points_local,
        "point_rgb_u8": point_rgb_u8,
        "r_l2c": r_l2c,
        "t_l2c": trans_l2c,
        "scale_l2c": scale_l2c,
        "model_points": model_points,
    }


if __name__ == "__main__":
    CONFIG_PATH = "checkpoints/hf/pipeline.yaml"
    IMAGE_PATH = "/robodata/smodak/repos/f3rm/datasets/f3rm/fresh/objaverse/car2new/images/frame_00057.png"
    SAM3_NPZ_PATH = "/robodata/smodak/repos/f3rm/datasets/f3rm/fresh/objaverse/car2new/features/sam3_/image_000056.npz"
    OUTPUT_ROOT = Path("testing_outputs/temp_compare")

    DO_GREEN_MODEL = False
    VISER_ENABLE = True

    MIN_MASK_AREA_PIXELS = 16
    MAX_POINTS = 150_000
    GS_ALPHA_THRESHOLD = 0.3
    VISER_PORT = 8895
    ENABLE_PUBLIC_VISER = True

    recon_default = SAM3DInstanceReconstructor(
        config_path=CONFIG_PATH,
        output_root=str(OUTPUT_ROOT / "default"),
        compile_model=False,
        seed=42,
        save_combined_scene=True,
        with_layout_postprocess=False,
    )

    if DO_GREEN_MODEL:
        recon_tight = SAM3DInstanceReconstructor(
            config_path=CONFIG_PATH,
            output_root=str(OUTPUT_ROOT / "tight"),
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

    manifest_default = recon_default.run(
        image_path=IMAGE_PATH,
        sam3_npz_path=SAM3_NPZ_PATH,
        min_mask_area_pixels=MIN_MASK_AREA_PIXELS,
    )
    bundle_default = load_manifest_bundle(manifest_default)

    if DO_GREEN_MODEL:
        manifest_tight = recon_tight.run(
            image_path=IMAGE_PATH,
            sam3_npz_path=SAM3_NPZ_PATH,
            min_mask_area_pixels=MIN_MASK_AREA_PIXELS,
        )
        bundle_tight = load_manifest_bundle(manifest_tight)

    projective_icp = ProjectiveICPFixedScale(
        min_correspondences=2500,
        residual_mad_sigma=3.0,
        gs_alpha_threshold=GS_ALPHA_THRESHOLD,
    )

    h, w = bundle_default["pointmap_hwc"].shape[:2]

    # Step 1: actual GS render from the initial pose.
    rendered_default = projective_icp.render_gaussian_buffers(
        gaussian_local=bundle_default["gaussian_local"],
        scale_l2c=bundle_default["scale_l2c"],
        r_l2c=bundle_default["r_l2c"],
        t_l2c=bundle_default["t_l2c"],
        intrinsics=bundle_default["intrinsics"],
        h=h,
        w=w,
    )

    # Step 2: establish matched pixels (same (i,j), within mask).
    target_valid = np.isfinite(bundle_default["pointmap_hwc"]).all(axis=-1)
    matched_default = bundle_default["mask"] & target_valid & rendered_default["valid_hw"]

    # Debug save: original image vs projected RGB render.
    debug_dir = OUTPUT_ROOT / "debug"
    save_side_by_side(
        bundle_default["image_rgb"],
        rendered_default["image_rgb_hw3"],
        debug_dir / "image_vs_gsplat_render_rgb_default.png",
    )
    save_side_by_side(
        bundle_default["image_rgb"],
        build_match_overlay(bundle_default["image_rgb"], bundle_default["mask"], matched_default),
        debug_dir / "image_vs_match_overlay_default.png",
    )

    # Step 3/4: optimize pose (fixed scale) using source=gsplat_rendered_pointmap, target=pointmap at matched pixels.
    proj_result = projective_icp.run(
        gaussian_local=bundle_default["gaussian_local"],
        scale_l2c=bundle_default["scale_l2c"],
        r_l2c=bundle_default["r_l2c"],
        t_l2c=bundle_default["t_l2c"],
        pointmap_hwc_target=bundle_default["pointmap_hwc"],
        mask_hw=bundle_default["mask"],
        intrinsics=bundle_default["intrinsics"],
    )

    rendered_final = proj_result["rendered"]
    matched_final = proj_result["matched_pixels"]

    save_side_by_side(
        bundle_default["image_rgb"],
        rendered_final["image_rgb_hw3"],
        debug_dir / "image_vs_gsplat_render_rgb_final.png",
    )
    save_side_by_side(
        bundle_default["image_rgb"],
        build_match_overlay(bundle_default["image_rgb"], bundle_default["mask"], matched_final),
        debug_dir / "image_vs_match_overlay_final.png",
    )

    # Clouds used in correspondence for visual check.
    black_target = proj_result["target_initial"]
    yellow_source = proj_result["source_initial"]
    black_target_final = proj_result["target_final"]
    green_source = proj_result["source_final"]

    white_canvas = np.full_like(bundle_default["image_rgb"], 255, dtype=np.uint8)

    black_render_default = projective_icp.render_projected_buffers(
        points_cam=black_target,
        point_rgb_u8=np.zeros((black_target.shape[0], 3), dtype=np.uint8),
        intrinsics=bundle_default["intrinsics"],
        h=h,
        w=w,
    )
    yellow_render = projective_icp.render_projected_buffers(
        points_cam=yellow_source,
        point_rgb_u8=np.full((yellow_source.shape[0], 3), (255, 255, 0), dtype=np.uint8),
        intrinsics=bundle_default["intrinsics"],
        h=h,
        w=w,
    )
    black_render_final = projective_icp.render_projected_buffers(
        points_cam=black_target_final,
        point_rgb_u8=np.zeros((black_target_final.shape[0], 3), dtype=np.uint8),
        intrinsics=bundle_default["intrinsics"],
        h=h,
        w=w,
    )
    green_render = projective_icp.render_projected_buffers(
        points_cam=green_source,
        point_rgb_u8=np.full((green_source.shape[0], 3), (0, 255, 0), dtype=np.uint8),
        intrinsics=bundle_default["intrinsics"],
        h=h,
        w=w,
    )

    save_side_by_side(
        white_canvas,
        build_pointcloud_overlay(
            white_canvas,
            black_render_default["valid_hw"],
            yellow_render["valid_hw"],
            np.array([255, 255, 0], dtype=np.uint8),
        ),
        debug_dir / "image_vs_black_yellow_projection.png",
    )
    save_side_by_side(
        white_canvas,
        build_pointcloud_overlay(
            white_canvas,
            black_render_final["valid_hw"],
            green_render["valid_hw"],
            np.array([0, 255, 0], dtype=np.uint8),
        ),
        debug_dir / "image_vs_black_green_projection.png",
    )

    black_full = black_target_final
    yellow_full = yellow_source
    green_full = green_source

    pm_points = downsample_points(black_full, MAX_POINTS, seed=1)
    pm_points_matched = downsample_points(black_target, MAX_POINTS, seed=11)
    yellow_points = downsample_points(yellow_full, MAX_POINTS, seed=5)
    yellow_points_matched = downsample_points(yellow_source, MAX_POINTS, seed=15)
    green_points = downsample_points(green_full, MAX_POINTS, seed=6)
    model_default_points = downsample_points(bundle_default["model_points"], MAX_POINTS, seed=21)
    model_optimized_points = downsample_points(proj_result["points_cam"], MAX_POINTS, seed=22)
    print(f"mask_index={bundle_default['mask_index']}")
    print(f"matched_default={int(matched_default.sum())} / mask={int(bundle_default['mask'].sum())}")
    print(f"matched_final={int(matched_final.sum())} / mask={int(bundle_default['mask'].sum())}")
    print(f"gs_alpha_threshold={GS_ALPHA_THRESHOLD}")
    print(f"metrics_initial={proj_result['metrics_initial']}")
    print(f"metrics_final={proj_result['metrics_final']}")
    for row in proj_result["history"]:
        print(row)

    print(f"saved: {debug_dir / 'image_vs_gsplat_render_rgb_default.png'}")
    print(f"saved: {debug_dir / 'image_vs_match_overlay_default.png'}")
    print(f"saved: {debug_dir / 'image_vs_gsplat_render_rgb_final.png'}")
    print(f"saved: {debug_dir / 'image_vs_match_overlay_final.png'}")
    print(f"saved: {debug_dir / 'image_vs_black_yellow_projection.png'}")
    print(f"saved: {debug_dir / 'image_vs_black_green_projection.png'}")

    if VISER_ENABLE:
        server = viser.ViserServer(port=VISER_PORT)
        server.scene.add_point_cloud(
            "/sam3d_pointmap_instance_all",
            points=pm_points,
            colors=(0.0, 0.0, 0.0),
            point_size=0.003,
        )
        server.scene.add_point_cloud(
            "/sam3d_pointmap_instance_matched",
            points=pm_points_matched,
            colors=(0.6, 0.0, 0.8),
            point_size=0.0035,
        )
        server.scene.add_point_cloud(
            "/gsplat_rendered_pointmap_all",
            points=yellow_points,
            colors=(1.0, 0.95, 0.0),
            point_size=0.0025,
        )
        server.scene.add_point_cloud(
            "/gsplat_rendered_pointmap_matched",
            points=yellow_points_matched,
            colors=(0.6, 0.0, 0.8),
            point_size=0.003,
        )
        server.scene.add_point_cloud(
            "/gsplat_rendered_pointmap_optimized",
            points=green_points,
            colors=(0.1, 0.95, 0.1),
            point_size=0.0025,
        )
        server.scene.add_point_cloud(
            "/model_default",
            points=model_default_points,
            colors=(0.95, 0.1, 0.1),
            point_size=0.0015,
        )
        server.scene.add_point_cloud(
            "/model_optimized",
            points=model_optimized_points,
            colors=(0.1, 0.3, 0.95),
            point_size=0.0015,
        )
        optimized_quat_wxyz = (
            matrix_to_quaternion(torch.from_numpy(proj_result["r_l2c"]).float().unsqueeze(0))[0]
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        print(pretty_pose_block("Loaded pose from saved file", {
            "rotation_wxyz_l2c": bundle_default["quat_wxyz_l2c"].tolist(),
            "translation_l2c": bundle_default["t_l2c"].tolist(),
            "scale_l2c": bundle_default["scale_l2c"].tolist(),
        }))
        print(pretty_pose_block("Model default pose used in viser", {
            "rotation_wxyz_l2c": bundle_default["quat_wxyz_l2c"].tolist(),
            "translation_l2c": bundle_default["t_l2c"].tolist(),
            "scale_l2c": bundle_default["scale_l2c"].tolist(),
        }))
        print(pretty_pose_block("Model optimized pose used in viser", {
            "rotation_wxyz_l2c": optimized_quat_wxyz.tolist(),
            "translation_l2c": proj_result["t_l2c"].tolist(),
            "scale_l2c": proj_result["scale_l2c"].tolist(),
        }))
        actual_port = server.get_port()
        print(f"Viser: http://localhost:{actual_port}")
        print("Black=target pointmap, Yellow=rendered pointmap before optimization, Purple=matched correspondences used for optimization, Green=rendered pointmap after optimization, Red=default model points, Blue=optimized model points")

        if ENABLE_PUBLIC_VISER:
            print(f"Public URL: {server.request_share_url(verbose=True)}")

        print("Press Ctrl+C to stop")
        while True:
            time.sleep(1)
