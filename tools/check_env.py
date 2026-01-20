#!/usr/bin/env python3
"""
环境自检脚本：尽量在运行主程序/消融前发现问题。
- 检查关键依赖是否可导入
- 输出 CUDA/GPU 相关信息
- 检查数据集路径和语义配置文件路径
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


def _check_import(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


def _print_section(title: str) -> None:
    print("\n" + "=" * 10 + f" {title} " + "=" * 10)


def main() -> int:
    parser = argparse.ArgumentParser(description="VGGT-SLAM 环境自检")
    parser.add_argument("--data_dir", type=str, default="", help="数据集路径（可选）")
    parser.add_argument("--semantic_backend_cfg", type=str, default="", help="语义配置文件路径（可选）")
    args = parser.parse_args()

    _print_section("Python")
    print(f"Python: {sys.version}")

    _print_section("核心依赖导入")
    modules = [
        "torch",
        "cv2",
        "open3d",
        "gtsam",
        "numpy",
    ]
    for name in modules:
        ok, detail = _check_import(name)
        status = "OK" if ok else "FAIL"
        print(f"{name:10s} : {status} {detail if not ok else ''}")

    _print_section("CUDA/GPU")
    try:
        import torch

        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        print(f"torch.version.cuda: {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        print(f"CUDA check failed: {exc}")

    _print_section("路径检查")
    if args.data_dir:
        data_path = Path(args.data_dir)
        print(f"data_dir: {data_path} -> {'exists' if data_path.exists() else 'missing'}")
        if data_path.exists():
            # 简单统计 jpg/png 数量，避免空目录误用
            imgs = list(data_path.rglob("*.jpg")) + list(data_path.rglob("*.png"))
            print(f"image files: {len(imgs)}")
    else:
        print("data_dir: (未指定)")

    if args.semantic_backend_cfg:
        cfg_path = Path(args.semantic_backend_cfg)
        print(f"semantic_backend_cfg: {cfg_path} -> {'exists' if cfg_path.exists() else 'missing'}")
    else:
        print("semantic_backend_cfg: (未指定)")

    _print_section("提示")
    print("若导入失败，请先安装对应依赖；若 CUDA 不可用，请考虑使用 CPU 模式或 GPU 环境。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
