from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def newest_best_pt(search_root: Path) -> Path | None:
    candidates = list(search_root.rglob("best.pt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="自动查找并复制最新 best.pt 到 weights/best.pt")
    parser.add_argument("--search-root", type=Path, default=Path("runs"))
    parser.add_argument("--target", type=Path, default=Path("weights/best.pt"))
    args = parser.parse_args()

    search_root = args.search_root.resolve()
    target = args.target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if not search_root.exists():
        print(f"训练输出目录不存在: {search_root}")
        print("请先运行训练命令，再执行本脚本。")
        return

    src = newest_best_pt(search_root)
    if src is None:
        print(f"在目录 {search_root} 下未找到 best.pt。")
        print("请确认训练已完成且无报错。")
        return

    shutil.copy2(src, target)
    print(f"已复制 best.pt:\n  来源: {src}\n  目标: {target}")


if __name__ == "__main__":
    main()
