#!/usr/bin/env python3
"""Build a consolidated, self-contained report from final fresh confirmations."""

from __future__ import annotations

import base64
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "batch_results" / "final_iteration_summary"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

# (source JSON, case, strategy, display name, category)
SELECTIONS = [
    ("final_regression_standard_amputated/result.json", "standard_6", "geometry_wave_1.25", "标准六足", "标准"),
    ("final_regression_standard_amputated/result.json", "standard_missing_leg_1", "geometry_wave_1.25", "标准六足单腿缺失", "缺腿"),
    ("final_regression_double_missing_calibrated/result.json", "standard_missing_legs_1_4", "binary", "标准六足双腿缺失", "缺腿"),
    ("contact_latch_seed101/result.json", "seed_101", "hildebrand", "任意构型 seed 101", "任意构型"),
    ("fresh_screen_seed19/result.json", "seed_19", "hildebrand", "任意构型 seed 19", "任意构型"),
    ("contact_latch_binary_high/result.json", "seed_211", "binary", "任意构型 seed 211", "任意构型"),
    ("final_selected_binary_23/result.json", "seed_23", "binary", "任意构型 seed 23", "任意构型"),
    ("contact_latch_balanced/result.json", "seed_307", "balanced_wave", "任意构型 seed 307", "任意构型"),
    ("contact_latch_balanced/result.json", "seed_419", "balanced_wave", "任意构型 seed 419", "任意构型"),
    ("contact_latch_binary_high/result.json", "seed_42", "binary", "任意构型 seed 42", "任意构型"),
    ("contact_latch_seed7/result.json", "seed_7", "binary", "任意构型 seed 7", "任意构型"),
]


def _font(size: float) -> FontProperties:
    return FontProperties(fname=str(FONT_PATH), size=size)


def _record(source_name: str, case: str, strategy: str) -> tuple[dict, Path]:
    source = ROOT / "batch_results" / source_name
    document = json.loads(source.read_text(encoding="utf-8"))
    for record in document.get("records", []):
        if record.get("case") == case and record.get("strategy") == strategy:
            return record, source.parent
    raise KeyError(f"missing selection {case}/{strategy} in {source}")


def _copy_image(source_root: Path, relative: str, target: Path) -> Path:
    source = (source_root / relative).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def collect() -> list[dict]:
    rows = []
    for source_name, case, strategy, display_name, category in SELECTIONS:
        record, source_root = _record(source_name, case, strategy)
        target_root = OUTPUT / "assets" / case
        trajectory = _copy_image(
            source_root, record["artifacts"]["trajectory_plot"],
            target_root / "trajectory.png",
        )
        heatmap = _copy_image(
            source_root, record["artifacts"]["gait_heatmap"],
            target_root / "gait_heatmap.png",
        )
        row = dict(record)
        row["display_name"] = display_name
        row["category"] = category
        row["source_result"] = str((ROOT / "batch_results" / source_name).resolve())
        row["consolidated_artifacts"] = {
            "trajectory_plot": trajectory.relative_to(OUTPUT).as_posix(),
            "gait_heatmap": heatmap.relative_to(OUTPUT).as_posix(),
        }
        rows.append(row)
    return rows


def write_json(rows: list[dict]) -> Path:
    document = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "probe_and_confirmation": "separate fresh PhysX contexts",
            "probe_steps": 600,
            "confirmation_steps": 600,
            "trajectory_and_safety_are_reported_separately": True,
            "heatmap_basis": "planned swing-joint targets, split into left/right leg panels",
        },
        "counts": {
            "cases": len(rows),
            "trajectory_pass": sum(bool(row.get("trajectory_success")) for row in rows),
            "strict_full_pass": sum(bool(row.get("success")) for row in rows),
            "arbitrary_trajectory_pass": sum(
                bool(row.get("trajectory_success"))
                for row in rows if row["category"] == "任意构型"
            ),
            "arbitrary_cases": sum(row["category"] == "任意构型" for row in rows),
        },
        "records": rows,
    }
    output = OUTPUT / "summary.json"
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _metric_lines(row: dict) -> list[str]:
    metrics = row.get("trajectory_metrics", {})
    diagnostics = row.get("episode_diagnostics", {})
    return [
        f"构型：{row['display_name']}    腿数：{row.get('leg_count')}    策略：{row.get('strategy')}",
        f"轨迹通过：{'是' if row.get('trajectory_success') else '否'}    严格安全通过：{'是' if row.get('safety_event_free') else '否'}",
        f"前向速度：{float(metrics.get('forward_speed', 0.0)):.4f} m/s    横漂比：{float(metrics.get('drift_ratio', 0.0)):.4f}    速度 CV：{float(metrics.get('speed_cv', 0.0)):.4f}",
        f"横向速度：{float(metrics.get('lateral_speed', 0.0)):.4f} m/s    航向误差：{float(metrics.get('heading_error_deg', 0.0)):.2f}°",
        f"接触失配：{float(diagnostics.get('contact_mismatch_ratio') or 0.0):.4f}    打滑比：{float(diagnostics.get('foot_slip_ratio') or 0.0):.4f}    SSM P05：{float(diagnostics.get('ssm_p05') or 0.0):.4f}",
        f"失败原因：{diagnostics.get('failure_reason') or '无'}",
    ]


def write_pdf(rows: list[dict]) -> Path:
    output = OUTPUT / "final_iteration_report.pdf"
    full_pass_names = "、".join(
        row["display_name"] for row in rows if row.get("success")
    )
    with PdfPages(output) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.06, 0.90, "任意构型多足机器人步态优化——最终迭代报告", fontproperties=_font(24))
        fig.text(
            0.06, 0.81,
            f"结论：{sum(bool(row.get('trajectory_success')) for row in rows)}/{len(rows)} 构型通过匀速直线轨迹验收；"
            f"严格完整通过：{full_pass_names}。\n"
            "其余构型仍保留动态支撑多边形告警，不能表述为完整安全通过。",
            fontproperties=_font(14), va="top", linespacing=1.7,
        )
        columns = ["构型", "策略", "速度", "横漂", "CV", "轨迹", "安全"]
        cells = []
        for row in rows:
            metrics = row["trajectory_metrics"]
            cells.append([
                row["display_name"], row["strategy"],
                f"{metrics['forward_speed']:.3f}", f"{metrics['drift_ratio']:.3f}",
                f"{metrics['speed_cv']:.3f}",
                "通过" if row["trajectory_success"] else "失败",
                "通过" if row["safety_event_free"] else "待改进",
            ])
        axis = fig.add_axes([0.04, 0.08, 0.92, 0.60]); axis.axis("off")
        table = axis.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1.0, 1.45)
        for cell in table.get_celld().values():
            cell.get_text().set_fontproperties(_font(8.5))
        pdf.savefig(fig, dpi=170); plt.close(fig)

        for row in rows:
            fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
            grid = fig.add_gridspec(2, 2, height_ratios=[0.18, 0.82], width_ratios=[0.38, 0.62])
            title = fig.add_subplot(grid[0, :]); title.axis("off")
            title.text(0.0, 0.93, row["display_name"], fontproperties=_font(18), va="top")
            title.text(0.0, 0.62, "\n".join(_metric_lines(row)), fontproperties=_font(9.5), va="top", linespacing=1.45)
            trajectory_axis = fig.add_subplot(grid[1, 0]); trajectory_axis.axis("off")
            trajectory_axis.imshow(mpimg.imread(OUTPUT / row["consolidated_artifacts"]["trajectory_plot"]))
            trajectory_axis.set_title("路径图", fontproperties=_font(12))
            heatmap_axis = fig.add_subplot(grid[1, 1]); heatmap_axis.axis("off")
            heatmap_axis.imshow(mpimg.imread(OUTPUT / row["consolidated_artifacts"]["gait_heatmap"]))
            heatmap_axis.set_title("左右腿步态热力图", fontproperties=_font(12))
            pdf.savefig(fig, dpi=170); plt.close(fig)
    return output


def _data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def write_html(rows: list[dict]) -> Path:
    cards = []
    for row in rows:
        lines = "<br>".join(html.escape(line) for line in _metric_lines(row))
        trajectory = _data_uri(OUTPUT / row["consolidated_artifacts"]["trajectory_plot"])
        heatmap = _data_uri(OUTPUT / row["consolidated_artifacts"]["gait_heatmap"])
        cards.append(
            f'<section><h2>{html.escape(row["display_name"])}</h2><p>{lines}</p>'
            f'<div class="images"><img src="{trajectory}" alt="trajectory">'
            f'<img src="{heatmap}" alt="left right gait heatmap"></div></section>'
        )
    output = OUTPUT / "final_iteration_report.html"
    output.write_text(
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>最终迭代报告</title><style>'
        'body{font-family:sans-serif;margin:22px;background:#f3f5f7;color:#20252b}'
        'section{background:#fff;margin:18px 0;padding:16px;border-radius:10px}'
        '.images{display:grid;grid-template-columns:minmax(280px,.75fr) minmax(420px,1.25fr);gap:12px}'
        'img{width:100%;height:auto;border:1px solid #ccd2d8}p{line-height:1.65}'
        '@media(max-width:900px){.images{grid-template-columns:1fr}}</style></head><body>'
        '<h1>任意构型多足机器人步态优化——最终迭代报告</h1>'
        f'<p>{sum(bool(row.get("trajectory_success")) for row in rows)}/{len(rows)} 构型通过轨迹验收；'
        '严格接触安全结果单独列出。图片已嵌入本文件，无外部路径依赖。</p>'
        + "".join(cards) + '</body></html>',
        encoding="utf-8",
    )
    return output


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = collect()
    for path in (write_json(rows), write_pdf(rows), write_html(rows)):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
