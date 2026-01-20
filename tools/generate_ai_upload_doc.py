#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a single Markdown document that contains:
- Project tree (with excluded heavy dirs)
- Concatenated source code and text files

Usage:
  python tools/generate_ai_upload_doc.py --root VGGT-SEM --out VGGT-SEM/AI_UPLOAD_DOC.md

By default, excludes common heavy directories and binary file types. Adjust via CLI.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

DEFAULT_EXCLUDE_DIRS = {
    # VCS / IDE / caches
    ".git", ".hg", ".svn", ".idea", ".vscode", "__pycache__", 
    ".mypy_cache", ".pytest_cache", "cache", "jedi", "matplotlib", "torch",
    # envs / builds
    "venv", ".venv", "env", ".env", "envs", "build", "dist", ".eggs",
    # project-specific heavy dirs
    "DATA", "RUNS", "LOG", "outputs", "output", "results", "logs",
    # third-party large components (can include just tree, not file contents)
    "salad", "vggt",
}

DEFAULT_EXCLUDE_FILE_EXTS = {
    # images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
    # videos
    ".mp4", ".mov", ".avi", ".mkv",
    # models / binaries
    ".pt", ".pth", ".ckpt", ".onnx", ".tflite", ".so", ".dll", ".dylib",
    # archives
    ".npz", ".npy", ".zip", ".tar", ".gz", ".7z", ".rar",
    # docs that are not plain text
    ".pdf",
}

DEFAULT_INCLUDE_FILES = {
    # Always include these if present
    "pyproject.toml", "requirements.txt", "requirements_demo.txt", "environment.yml",
    "setup.py", "setup.cfg", "Pipfile", "Pipfile.lock",
    "README.md", "README.rst", "LICENSE", "LICENSE.txt",
}

DEFAULT_EXTS = {
    ".py", ".md", ".yaml", ".yml", ".json", ".sh", ".txt", ".toml", ".ini", ".cfg",
}


def build_tree(root: Path, exclude_dirs: set, max_entries: int = 20000) -> str:
    """Return a tree-like text representation, excluding certain directories."""
    root = root.resolve()
    lines = [str(root)]
    entries_count = 0

    def is_excluded_dir(p: Path) -> bool:
        return p.name in exclude_dirs

    def walk_dir(current: Path, prefix: str = ""):
        nonlocal entries_count
        if entries_count >= max_entries:
            lines.append(prefix + "└── [TRUNCATED: too many entries]")
            return

        try:
            children = sorted(list(current.iterdir()), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            lines.append(prefix + "└── [PermissionError]")
            return

        filtered = []
        for c in children:
            if c.is_dir() and is_excluded_dir(c):
                lines.append(prefix + "├── " + c.name + " [excluded]")
                continue
            filtered.append(c)

        for idx, c in enumerate(filtered):
            if entries_count >= max_entries:
                lines.append(prefix + "└── [TRUNCATED: too many entries]")
                return
            entries_count += 1

            is_last = (idx == len(filtered) - 1)
            connector = "└── " if is_last else "├── "
            lines.append(prefix + connector + c.name)

            if c.is_dir():
                extension = "    " if is_last else "│   "
                walk_dir(c, prefix + extension)

    walk_dir(root, "")
    return "\n".join(lines)


def should_include_file(path: Path, exts: set, exclude_file_exts: set) -> bool:
    if path.name in DEFAULT_INCLUDE_FILES:
        return True
    if path.suffix.lower() in exclude_file_exts:
        return False
    return path.suffix.lower() in exts


def iter_files(root: Path, exclude_dirs: set, exts: set, exclude_file_exts: set):
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        # prune dirs in-place
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if should_include_file(p, exts, exclude_file_exts):
                yield p


def is_binary_bytes(data: bytes) -> bool:
    # Heuristic: presence of null bytes or high non-text ratio
    if b"\x00" in data:
        return True
    # consider ASCII + common UTF-8 range
    text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(32, 127)))
    if not data:
        return False
    nontext = sum(1 for b in data if b not in text_chars)
    return (nontext / len(data)) > 0.30


def read_text_file(path: Path, max_bytes: int, max_lines: int):
    data = path.read_bytes()
    truncated = False

    if len(data) > max_bytes:
        data = data[:max_bytes]
        truncated = True

    if is_binary_bytes(data):
        return "[SKIPPED: detected binary content]", True

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")

    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True

    return "\n".join(lines), truncated


def infer_lang(ext: str) -> str:
    return {
        ".py": "python",
        ".sh": "bash",
        ".md": "markdown",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".txt": "text",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
    }.get(ext.lower(), "text")


def write_md_header(f, title: str, level: int = 1):
    f.write("{} {}\n\n".format("#" * level, title))


def generate_markdown(root: Path, out_path: Path, exts: set, exclude_dirs: set, exclude_file_exts: set,
                      max_bytes: int, max_lines: int, max_tree_entries: int):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        # Title & meta
        write_md_header(f, f"Project Dump: {root.name}", level=1)
        f.write(f"- Root: {root}\n")
        f.write(f"- Generated at: {now}\n")
        f.write(f"- Included extensions: {', '.join(sorted(exts))}\n")
        f.write(f"- Excluded dirs: {', '.join(sorted(exclude_dirs))}\n")
        f.write(f"- Per-file truncation: max_bytes={max_bytes}, max_lines={max_lines}\n\n")

        # Tree
        write_md_header(f, "Project Tree", level=2)
        f.write("```text\n")
        f.write(build_tree(root, exclude_dirs=exclude_dirs, max_entries=max_tree_entries))
        f.write("\n```\n\n")

        # Files
        write_md_header(f, "Files", level=2)
        file_count = 0
        total_bytes = 0
        for p in iter_files(root, exclude_dirs=exclude_dirs, exts=exts, exclude_file_exts=exclude_file_exts):
            try:
                size = p.stat().st_size
            except OSError:
                continue

            file_count += 1
            total_bytes += size

            rel = p.relative_to(root)
            write_md_header(f, str(rel), level=3)
            f.write(f"- Size: {size} bytes\n\n")

            content, truncated = read_text_file(p, max_bytes=max_bytes, max_lines=max_lines)
            lang = infer_lang(p.suffix)
            f.write(f"```{lang}\n")
            f.write(content)
            f.write("\n````\n\n" if lang == "markdown" else "\n```\n\n")

            if truncated:
                f.write("[TRUNCATED] content was cut by size/line limits.\n\n")

        # Summary
        write_md_header(f, "Summary", level=2)
        f.write(f"- Included files: {file_count}\n")
        f.write(f"- Total raw size (before truncation): {total_bytes} bytes\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="project root")
    ap.add_argument("--out", type=str, default="AI_UPLOAD_DOC.md", help="output markdown path")
    ap.add_argument("--ext", nargs="*", default=sorted(DEFAULT_EXTS), help="extensions to include")
    ap.add_argument("--exclude-dirs", nargs="*", default=sorted(DEFAULT_EXCLUDE_DIRS), help="dir names to exclude")
    ap.add_argument("--max-bytes", type=int, default=300_000, help="max bytes per file before truncation")
    ap.add_argument("--max-lines", type=int, default=5000, help="max lines per file before truncation")
    ap.add_argument("--max-tree-entries", type=int, default=20000, help="cap project tree entries")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out).resolve()

    exts = {e if e.startswith(".") else ("." + e) for e in args.ext}
    exclude_dirs = set(args.exclude_dirs)

    generate_markdown(
        root=root,
        out_path=out_path,
        exts=exts,
        exclude_dirs=exclude_dirs,
        exclude_file_exts=DEFAULT_EXCLUDE_FILE_EXTS,
        max_bytes=args.max_bytes,
        max_lines=args.max_lines,
        max_tree_entries=args.max_tree_entries,
    )

    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
