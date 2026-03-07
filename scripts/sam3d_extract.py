#!/usr/bin/env python3
import argparse
import concurrent.futures
import gc
import json
import multiprocessing as mp
import random
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from tqdm.auto import tqdm

try:
    from sam3d_instance_reconstructor import SAM3DInstanceReconstructor
except ModuleNotFoundError:
    from scripts.sam3d_instance_reconstructor import SAM3DInstanceReconstructor


class SAM3DArgs:
    config_path: str = "checkpoints/hf/pipeline.yaml"
    compile: bool = False
    seed: int = 42
    min_mask_area_pixels: int = 16
    save_combined_scene: bool = True
    batch_size_per_gpu: int = 1

    @classmethod
    def id_dict(cls) -> dict[str, Any]:
        return {
            "config_path": str(cls.config_path),
            "compile": bool(cls.compile),
            "seed": int(cls.seed),
            "min_mask_area_pixels": int(cls.min_mask_area_pixels),
            "save_combined_scene": bool(cls.save_combined_scene),
        }


def parse_text_prompts(text_prompts_arg: str) -> list[str]:
    raw = (text_prompts_arg or "").strip()
    if raw == "":
        return []
    return [word.lower() for word in raw.split("_") if word.strip()]


def get_feature_names(text_prompts: list[str]) -> tuple[str, str]:
    if len(text_prompts) == 0:
        return "sam3_", "sam3d_"
    suffix = "_".join(text_prompts)
    return f"sam3_{suffix}", f"sam3d_{suffix}"


def resolve_worker_devices(
    device: torch.device, batch_size_per_gpu: int
) -> list[torch.device]:
    workers_per_gpu = max(1, int(batch_size_per_gpu))
    if device.type == "cuda":
        if device.index is None:
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if n_gpus == 0:
                return [torch.device("cpu")]
            return [
                torch.device(f"cuda:{gpu_idx}")
                for gpu_idx in range(n_gpus)
                for _ in range(workers_per_gpu)
            ]
        return [torch.device(f"cuda:{device.index}") for _ in range(workers_per_gpu)]
    return [torch.device("cpu")]


def _list_images(image_dir: Path) -> list[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    image_paths: list[Path] = []
    for pattern in patterns:
        image_paths.extend(image_dir.glob(pattern))
    image_paths = sorted(set(image_paths))
    return image_paths


def _list_sam3_masks(mask_dir: Path) -> list[Path]:
    return sorted(mask_dir.glob("image_*.npz"))


_MP_RECONSTRUCTOR: SAM3DInstanceReconstructor | None = None


def _mp_worker_init(
    worker_rank: int,
    worker_total: int,
    device_str: str,
    config_path: str,
    reconstructor_output_root: str,
    compile_model: bool,
    seed: int,
    save_combined_scene: bool,
) -> None:
    global _MP_RECONSTRUCTOR
    logger.disable("sam3d_objects")
    device = torch.device(device_str)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    _MP_RECONSTRUCTOR = SAM3DInstanceReconstructor(
        config_path=config_path,
        output_root=reconstructor_output_root,
        compile_model=compile_model,
        seed=seed,
        save_combined_scene=save_combined_scene,
    )

    if device.type == "cuda" and device.index is not None:
        pipeline = _MP_RECONSTRUCTOR.inference._pipeline
        pipeline.device = device
        if hasattr(pipeline, "depth_model"):
            pipeline.depth_model.device = device
            if hasattr(pipeline.depth_model, "model"):
                pipeline.depth_model.model.to(device)

    print(f"[init] SAM3D worker {worker_rank}/{worker_total} ready on {device}")


def _mp_worker_run(
    task_idx: int,
    image_path: str,
    sam3_npz_path: str,
    min_mask_area_pixels: int,
) -> tuple[int, dict[str, Any]]:
    global _MP_RECONSTRUCTOR
    if _MP_RECONSTRUCTOR is None:
        raise RuntimeError("SAM3D multiprocessing worker is not initialized.")

    manifest_path = _MP_RECONSTRUCTOR.run(
        image_path=image_path,
        sam3_npz_path=sam3_npz_path,
        min_mask_area_pixels=min_mask_area_pixels,
    )
    manifest = json.loads(Path(manifest_path).read_text())
    result = {
        "image_path": str(image_path),
        "sam3_npz_path": str(sam3_npz_path),
        "manifest_path": str(manifest_path),
        "scene_cam_ply": manifest.get("scene_cam_ply"),
        "num_instances": len(manifest.get("instances", [])),
    }
    return task_idx, result


class SAM3DExtractor:
    def __init__(
        self,
        device: torch.device,
        data_dir: Path,
        sam3_feature_name: str,
        sam3d_feature_name: str,
        verbose: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.data_dir = Path(data_dir)
        self.sam3_feature_name = str(sam3_feature_name)
        self.sam3d_feature_name = str(sam3d_feature_name)
        self.verbose = bool(verbose)
        self._executors: list[concurrent.futures.ProcessPoolExecutor] = []

        self.feature_root = self.data_dir / "features" / self.sam3d_feature_name
        self.recon_root = self.feature_root / "reconstructions"
        self.feature_root.mkdir(parents=True, exist_ok=True)
        self.recon_root.mkdir(parents=True, exist_ok=True)

        self.worker_devices = resolve_worker_devices(self.device, SAM3DArgs.batch_size_per_gpu)
        num_workers = len(self.worker_devices)
        if self.verbose:
            print(
                f"Initializing SAM3D workers: num_workers={num_workers}, devices={self.worker_devices}"
            )
        self.num_workers = num_workers

    def _shutdown_executors(self, force: bool) -> None:
        if not self._executors:
            return
        executors = self._executors
        self._executors = []

        for executor in executors:
            if force:
                # Best-effort hard stop for running children on interrupt/error.
                processes = getattr(executor, "_processes", None)
                if processes:
                    for proc in list(processes.values()):
                        if proc.is_alive():
                            proc.terminate()
                executor.shutdown(wait=False, cancel_futures=True)
                if processes:
                    for proc in list(processes.values()):
                        proc.join(timeout=1.0)
                        if proc.is_alive():
                            proc.kill()
            else:
                executor.shutdown(wait=True, cancel_futures=False)

    def discover_pairs(self, max_images: int | None = None) -> list[tuple[Path, Path]]:
        image_dir = self.data_dir / "images"
        sam3_dir = self.data_dir / "features" / self.sam3_feature_name

        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image dir: {image_dir}")
        if not sam3_dir.exists():
            raise FileNotFoundError(f"Missing SAM3 mask dir: {sam3_dir}")

        image_paths = _list_images(image_dir)
        mask_paths = _list_sam3_masks(sam3_dir)

        if len(image_paths) == 0:
            raise RuntimeError(f"No images found under {image_dir}")
        if len(mask_paths) == 0:
            raise RuntimeError(f"No SAM3 mask files found under {sam3_dir}")
        if len(image_paths) != len(mask_paths):
            raise RuntimeError(
                f"Image/mask count mismatch: images={len(image_paths)} masks={len(mask_paths)}"
            )

        pairs = list(zip(image_paths, mask_paths))
        if max_images is not None:
            pairs = pairs[: int(max_images)]
        return pairs

    def extract_batch(self, pairs: list[tuple[Path, Path]]) -> list[dict[str, Any]]:
        if len(pairs) == 0:
            return []

        ctx = mp.get_context("spawn")
        futures: list[concurrent.futures.Future] = []
        try:
            for worker_rank, worker_device in enumerate(self.worker_devices, start=1):
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=ctx,
                    initializer=_mp_worker_init,
                    initargs=(
                        worker_rank,
                        self.num_workers,
                        str(worker_device),
                        SAM3DArgs.config_path,
                        str(self.recon_root),
                        SAM3DArgs.compile,
                        SAM3DArgs.seed,
                        SAM3DArgs.save_combined_scene,
                    ),
                )
                self._executors.append(executor)

            for task_idx, (image_path, mask_path) in enumerate(pairs):
                executor = self._executors[task_idx % len(self._executors)]
                futures.append(
                    executor.submit(
                        _mp_worker_run,
                        task_idx,
                        str(image_path),
                        str(mask_path),
                        SAM3DArgs.min_mask_area_pixels,
                    )
                )

            ordered_results: list[dict[str, Any] | None] = [None] * len(pairs)
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Extracting SAM3D",
                leave=False,
            ):
                task_idx, result = future.result()
                ordered_results[task_idx] = result

            return [r for r in ordered_results if r is not None]
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            self._shutdown_executors(force=True)
            raise
        except Exception:
            self._shutdown_executors(force=True)
            raise
        finally:
            self._shutdown_executors(force=False)
            gc.collect()

    def save_results(
        self,
        pairs: list[tuple[Path, Path]],
        results: list[dict[str, Any]],
    ) -> Path:
        if len(pairs) != len(results):
            raise RuntimeError(f"pairs/results mismatch: {len(pairs)} vs {len(results)}")

        for idx, ((image_path, mask_path), result) in enumerate(zip(pairs, results)):
            payload = {
                "image_path": str(image_path),
                "sam3_npz_path": str(mask_path),
                "manifest_path": str(result["manifest_path"]),
                "scene_cam_ply": result.get("scene_cam_ply"),
                "num_instances": int(result.get("num_instances", 0)),
            }
            (self.feature_root / f"image_{idx:06d}.json").write_text(
                json.dumps(payload, indent=2)
            )

        meta = {
            "args": SAM3DArgs.id_dict(),
            "image_fnames": [str(p[0]) for p in pairs],
            "sam3_npz_fnames": [str(p[1]) for p in pairs],
        }
        torch.save(meta, self.feature_root / "meta.pt")
        return self.feature_root

    def load_saved_result(self, index: int) -> dict[str, Any]:
        entry_path = self.feature_root / f"image_{index:06d}.json"
        if not entry_path.exists():
            raise FileNotFoundError(f"Missing saved entry: {entry_path}")
        return json.loads(entry_path.read_text())

    @staticmethod
    def show_scene(scene_ply_path: str) -> None:
        SAM3DInstanceReconstructor.show_scene(scene_ply_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM3D extraction using image files + precomputed SAM3 masks."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Dataset root containing images/ and features/sam3_/",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Max image/mask pairs to process (0 means all)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="checkpoints/hf/pipeline.yaml",
        help="SAM3D pipeline config path",
    )
    parser.add_argument(
        "--batch-size-per-gpu",
        type=int,
        default=1,
        help="Workers per GPU for AsyncMultiWrapper",
    )
    parser.add_argument(
        "--text-prompts",
        dest="text_prompts",
        type=str,
        default="",
        help="Underscore-separated prompts, e.g. 'car_dog'. Loads sam3_<prompts> and writes sam3d_<prompts>.",
    )
    parser.add_argument(
        "--min-mask-area-pixels",
        type=int,
        default=16,
        help="Skip SAM3 masks below this pixel area",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch compile in SAM3D inference pipeline",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to SAM3D inference",
    )
    parser.add_argument(
        "--no-scene",
        action="store_true",
        help="Disable combined scene_cam.ply export",
    )
    args = parser.parse_args()

    logger.disable("sam3d_objects")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SAM3DArgs.config_path = str(args.config_path)
    SAM3DArgs.compile = bool(args.compile)
    SAM3DArgs.seed = int(args.seed)
    SAM3DArgs.min_mask_area_pixels = int(args.min_mask_area_pixels)
    SAM3DArgs.save_combined_scene = not bool(args.no_scene)
    SAM3DArgs.batch_size_per_gpu = int(args.batch_size_per_gpu)
    text_prompts = parse_text_prompts(str(args.text_prompts))
    sam3_feature_name, sam3d_feature_name = get_feature_names(text_prompts)

    extractor = SAM3DExtractor(
        device=device,
        data_dir=args.data,
        sam3_feature_name=sam3_feature_name,
        sam3d_feature_name=sam3d_feature_name,
        verbose=True,
    )
    max_images = None if int(args.max_images) <= 0 else int(args.max_images)
    pairs = extractor.discover_pairs(max_images=max_images)
    interrupted = False
    try:
        print(f"Processing {len(pairs)} image/mask pairs...")
        results = extractor.extract_batch(pairs)
        saved_dir = extractor.save_results(pairs, results)
        print(f"Saved SAM3D extraction outputs to: {saved_dir}")

        scene_paths = [
            str(r["scene_cam_ply"])
            for r in results
            if r.get("scene_cam_ply") and Path(r["scene_cam_ply"]).exists()
        ]
        if scene_paths:
            sampled_scene = random.choice(scene_paths)
            print(f"Showing one random extracted scene: {sampled_scene}")
            # extractor.show_scene(sampled_scene)
        else:
            print("No scene_cam_ply outputs available to visualize.")
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user (Ctrl+C). Exiting cleanly.")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if interrupted:
        raise SystemExit(130)
