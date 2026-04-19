# -*- coding: utf-8 -*-
"""
Slash normalizer v2: convert path slashes even when paths contain spaces.
- First pass: convert quoted Windows absolute paths (handles spaces).
- Second pass: convert unquoted Windows paths (no spaces) as fallback.
"""

import argparse, os, re, shutil
from pathlib import Path
from datetime import datetime

THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parent.parent
REPORT_FILE = PROJECT_ROOT / "slash_normalizer_report.txt"

IGNORE_DIRS = {".git", ".idea", "__pycache__", "venv", ".venv", "build", "dist"}
SCAN_EXTS = {".py", ".txt", ".md", ".json", ".yml", ".yaml", ".ini", ".cfg", ".bat", ".ps1", ".ipynb"}

def should_scan(p: Path) -> bool:
    if not p.is_file(): return False
    if p.suffix.lower() not in SCAN_EXTS: return False
    for part in p.parts:
        if part in IGNORE_DIRS: return False
    return True

# 1) 处理**加引号**的绝对路径（允许空格），例如：
#    "D:/pycharm file/Autoencoder/run_model/TF_quantization_test.py"
#    r'D:/pycharm file/Autoencoder/run_model/Autoencoder_revision'
RX_QUOTED_WIN_PATH = re.compile(
    r"""(?P<prefix>[rbfuRBFU]*?)          # 可选前缀 r/b/f/u
        (?P<q>['\"])                      # 引号
        (?P<body>                         # 路径主体（允许空格，直到配对引号前）
            [A-Za-z]:(?:\\|/)[^'"]*
        )
        (?P=q)                            # 配对引号
    """,
    re.VERBOSE
)

# 2) 兜底：未加引号、且不含空格的绝对路径
RX_WIN_BS = re.compile(r"[A-Za-z]:\\[^\s\"']+")
RX_WIN_FS = re.compile(r"[A-Za-z]:/[^\s\"']+")

def to_posix(s: str) -> str:
    return s.replace("\\", "/")

def to_windows(s: str) -> str:
    out = s.replace("/", "\\")
    out = re.sub(r"\\{2,}", r"\\", out)
    return out

def convert_body(body: str, target: str) -> str:
    return to_posix(body) if target == "posix" else to_windows(body)

def repl_quoted(m: re.Match, target: str) -> str:
    prefix, q, body = m.group("prefix"), m.group("q"), m.group("body")
    return f"{prefix}{q}{convert_body(body, target)}{q}"

def convert_text(text: str, target: str):
    changed = False

    # Pass 1: 处理**加引号**的路径（包含空格的主要场景）
    def _quoted_sub(m):
        nonlocal changed
        changed = True
        return repl_quoted(m, target)
    new_text = RX_QUOTED_WIN_PATH.sub(_quoted_sub, text)

    # Pass 2: 兜底处理未加引号的路径（不含空格）
    def _bs_sub(m):
        nonlocal changed
        changed = True
        return convert_body(m.group(0), target)
    def _fs_sub(m):
        nonlocal changed
        changed = True
        return convert_body(m.group(0), target)

    newer_text = RX_WIN_BS.sub(_bs_sub, new_text)
    newest_text = RX_WIN_FS.sub(_fs_sub, newer_text)

    return newest_text, changed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes with .bak backups")
    ap.add_argument("--slash", choices=["posix", "windows"], default="posix",
                    help="target slash style (default: posix '/')")
    args = ap.parse_args()

    lines = []
    total_files = suspicious = changed_files = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for fn in files:
            p = Path(root) / fn
            if not should_scan(p): continue
            total_files += 1
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                lines.append(f"[SKIP] {p} (read error: {e})")
                continue

            # 粗判：只要出现盘符就当可疑
            if re.search(r"[A-Za-z]:(?:\\|/)", text):
                suspicious += 1
                lines.append(f"\n[FILE] {p}")

            new_text, did_change = convert_text(text, args.slash)
            if did_change:
                lines.append(f"  * would convert -> {args.slash}")
                if args.apply:
                    bak = p.with_suffix(p.suffix + ".bak")
                    try:
                        shutil.copy2(p, bak)
                        p.write_text(new_text, encoding="utf-8")
                        changed_files += 1
                        lines.append(f"  * wrote changes (backup: {bak.name})")
                    except Exception as e:
                        lines.append(f"  ! write failed: {e}")

    header = [
        "=== Slash Normalizer Report ===",
        f"Project root : {PROJECT_ROOT}",
        f"Run at       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Target style : {args.slash}",
        f"Total files  : {total_files}",
        f"Suspicious   : {suspicious}",
        f"Changed      : {changed_files}",
        "-"*60,
    ]
    report = "\n".join(header + lines) + "\n"
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"[REPORT SAVED] {REPORT_FILE}")

if __name__ == "__main__":
    main()
