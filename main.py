# -*- coding: utf-8 -*-
"""執行 50 Startups 的 CRISP-DM 分析：python main.py。"""

import os
from pathlib import Path
import runpy
import sys
import warnings


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    analysis_file = project_dir / "analysis.py"
    data_file = project_dir / "data" / "50_Startups.csv"

    for required_file in (analysis_file, data_file):
        if not required_file.is_file():
            print(f"找不到必要檔案：{required_file}", file=sys.stderr)
            return 1

    # analysis.py 使用相對路徑，固定從專案資料夾讀取與儲存檔案。
    previous_dir = Path.cwd()
    try:
        os.chdir(project_dir)
        with warnings.catch_warnings():
            # 圖表已存成 PNG，命令列執行時不需要互動視窗。
            warnings.filterwarnings(
                "ignore",
                message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
                category=UserWarning,
            )
            runpy.run_path(str(analysis_file), run_name="__main__")
    except ModuleNotFoundError as exc:
        print(f"缺少 Python 套件：{exc.name}", file=sys.stderr)
        print("請使用執行程式的同一個 Python 安裝套件：", file=sys.stderr)
        print(
            f'"{sys.executable}" -m pip install -r "{project_dir / "requirements.txt"}"',
            file=sys.stderr,
        )
        return 1
    finally:
        os.chdir(previous_dir)

    from feature_selection import run
    run(project_dir)
    from build_dashboard import build
    build(project_dir)

    print(f"\n分析完成！圖表、指標與模型已儲存至：{project_dir / 'outputs'}")
    print(f"分析報告：{project_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


