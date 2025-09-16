import json
import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from gaico.visualize import plot_metric_comparison, plot_radar_comparison

PALETTE = "viridis"
DPI = 400

# Define a custom color cycle (fallback if seaborn theme not applied later)
color_cycle = sns.color_palette(PALETTE, 8)

mpl_params = {
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "figure.autolayout": False,  # we'll manage tight_layout explicitly
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "axes.titleweight": "semibold",
    "axes.linewidth": 1.0,
    "axes.edgecolor": "#222222",
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.9,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.facecolor": "white",
    "legend.edgecolor": "#CCCCCC",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "axes.prop_cycle": mpl.cycler(color=color_cycle),  # type: ignore
    "ps.fonttype": 42,
}


# Utility: consistent file save helper supporting multiple formats
def _save_fig_multi(fig, base_path: str, transparent=False):
    fig.savefig(f"{base_path}.png", dpi=DPI, bbox_inches="tight", transparent=transparent)


def show_textuals(REFERENCE_DIR, OUTPUTS_DIR, PIPELINES):
    """
    Summarize and compare model outputs (text, images, audio) across different pipelines.
    """
    # Load reference (if available) for contextual comparison
    reference_data = None
    with open(REFERENCE_DIR / "reference.json") as f:
        reference_data = json.load(f)
        print("Loaded reference baseline plan.")

    # Helper: safely load each pipeline's generated JSON output
    pipeline_json = {}
    for pipeline_name in PIPELINES:
        output_path = OUTPUTS_DIR / pipeline_name / "output.json"
        with open(output_path) as f:
            pipeline_json[pipeline_name] = json.load(f)

    # Build a quick comparative table (Day 1 snapshot) for itinerary text + budget
    rows = []
    for pipeline_name, data in pipeline_json.items():
        day1 = data["trip_plan"][0]
        rows.append(
            {
                "pipeline": pipeline_name,
                "day": 1,
                "plan_text_excerpt": (day1.get("day_plan_text", "")[:140] + "...")
                if len(day1.get("day_plan_text", "")) > 140
                else day1.get("day_plan_text", ""),
                "sequence_len": len(day1.get("day_plan_sequence", [])),
                "budget": day1.get("day_budget_euros", None),
            }
        )

    if reference_data:
        ref_day1 = reference_data["trip_plan"][0]
        rows.append(
            {
                "pipeline": "REFERENCE",
                "day": 1,
                "plan_text_excerpt": (ref_day1.get("day_plan_text", "")[:140] + "...")
                if len(ref_day1.get("day_plan_text", "")) > 140
                else ref_day1.get("day_plan_text", ""),
                "sequence_len": len(ref_day1.get("day_plan_sequence", [])),
                "budget": ref_day1.get("day_budget_euros", None),
            }
        )

    df_day1_snapshot = pd.DataFrame(rows).sort_values("pipeline")
    return reference_data, pipeline_json, df_day1_snapshot


def show_budgets(pipeline_json, reference_data):
    # Aggregate budget time series for quick glance
    budget_series_rows = []
    for pipeline_name, data in pipeline_json.items():
        for d in data["trip_plan"]:
            budget_series_rows.append(
                {
                    "pipeline": pipeline_name,
                    "day": d.get("day"),
                    "budget_euros": d.get("day_budget_euros"),
                }
            )
    if reference_data:
        for d in reference_data["trip_plan"]:
            budget_series_rows.append(
                {
                    "pipeline": "REFERENCE",
                    "day": d.get("day"),
                    "budget_euros": d.get("day_budget_euros"),
                }
            )

    if budget_series_rows:
        df_budget_glance = pd.DataFrame(budget_series_rows)
        pivot_budget = df_budget_glance.pivot_table(
            index="day", columns="pipeline", values="budget_euros"
        )
        return pivot_budget


def show_images(PIPELINES, OUTPUTS_DIR):
    max_images = 3  # Samples per pipeline
    image_exts = ("*.png", "*.jpg", "*.jpeg")
    audio_exts = ("*.mp3", "*.wav", "*.aac", "*.m4a")

    # Collect media file paths per pipeline
    media_summary = {}
    for pipeline_name in PIPELINES:
        p_dir = OUTPUTS_DIR / pipeline_name

        image_files = []
        for ext in image_exts:
            image_files.extend(sorted(p_dir.glob(ext)))
        audio_files = []
        for ext in audio_exts:
            audio_files.extend(sorted(p_dir.glob(ext)))

        media_summary[pipeline_name] = {
            "images": image_files[:max_images],
            "audio": audio_files[:1],  # Just first audio sample
        }

    # Display Images in a grid (row = pipeline)
    # Count pipelines that have any images

    table_rows = []
    for p in PIPELINES:
        row_html = f"<tr><th style='text-align:left;padding-right:8px'>{p}</th>"
        imgs = media_summary.get(p, {}).get("images", [])
        for i in range(max_images):
            if i < len(imgs):
                img_rel = os.path.relpath(imgs[i], start=os.getcwd())
                row_html += f"<td style='padding:4px'><img src='{img_rel}' style='max-width:180px;max-height:140px;object-fit:contain;border:1px solid #ccc;padding:2px;border-radius:4px' title='{imgs[i].name}'></td>"
            else:
                row_html += "<td style='padding:4px;color:#888;font-size:12px'>—</td>"
        row_html += "</tr>"
        table_rows.append(row_html)

    html_table = (
        "<h4>Image Samples (up to "
        + str(max_images)
        + " per pipeline)</h4><table style='border-collapse:collapse'>"
        + "".join(table_rows)
        + "</table>"
    )
    return html_table, media_summary


def show_audios(PIPELINES, media_summary):
    # Display Audio players
    audio_players = []

    for p in PIPELINES:
        auds = media_summary.get(p, {}).get("audio", [])
        if auds:
            aud_rel = os.path.relpath(auds[0], start=os.getcwd())
            audio_players.append(
                f"<div style='margin-bottom:8px'><strong>{p}</strong><br><audio controls src='{aud_rel}' style='width:300px'></audio><div style='font-size:11px;color:#555'>{os.path.basename(auds[0])}</div></div>"
            )
        else:
            audio_players.append(
                f"<div style='margin-bottom:8px'><strong>{p}</strong><br><em style='color:#888'>No audio found</em></div>"
            )

    return "<h4>Audio Samples (first file per pipeline)</h4>" + "".join(audio_players)


def create_plan_coherence_visuals(df_long, FIGURES_DIR):
    """Generates and saves radar and bar plots for plan coherence results.

    Enhancements:
    - Deterministic metric ordering
    - Adds median reference line per subplot
    - Annotates bars with background boxes
    - Returns dict of figures for further customization
    """
    print("\nGenerating visuals for Plan Coherence")

    # Defensive copy & ordering
    metric_order = [
        "ROUGE_rougeL",
        "BERTScore_f1",
        "PlanningLCS",
        "PlanningJaccard",
        "TimeSeriesDTW",
    ]
    df_long = df_long[df_long["metric_name"].isin(metric_order)].copy()
    df_long["metric_name"] = pd.Categorical(
        df_long["metric_name"], categories=metric_order, ordered=True
    )

    figs = {}

    # 1. Radar Plot
    print("Creating Plan Coherence Radar Plot...")
    fig_radar = plt.figure(figsize=(7.8, 7.8))
    radar_ax = plt.subplot(111, polar=True)
    plot_radar_comparison(
        df_long,
        axis=radar_ax,
        title="Plan Coherence: Overall Pipeline Performance",
    )
    radar_ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), frameon=True)
    radar_ax.set_facecolor("#FCFCFC")
    figs["radar"] = fig_radar
    _save_fig_multi(fig_radar, os.path.join(FIGURES_DIR, "plan_coherence_radar"))
    plt.show()

    # 2. Metric-by-metric Bars
    print("Creating Plan Coherence Bar Plots...")
    metrics = list(df_long["metric_name"].cat.categories)

    n_cols = 3
    n_rows = int(math.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(5.6 * n_cols, 4.2 * n_rows),
        sharey=False,
    )
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        plot_metric_comparison(
            df_long,
            metric_name=metric,
            axis=ax,
            title=metric.replace("_", " ") if metric else metric,
        )
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=12)
        ax.set_facecolor("#FFFFFF")

        # Dynamic y-limit padding
        vals = [p.get_height() for p in ax.patches]
        if vals:
            vmin, vmax = min(vals), max(vals)
            padding = (vmax - vmin) * 0.12 if vmax > vmin else vmax * 0.15 or 0.1
            ax.set_ylim(vmin * 0.98, vmax + padding)

        # Median line
        if vals:
            med = pd.Series(vals).median()
            ax.axhline(med, color="#444", linestyle="--", linewidth=0.9, alpha=0.55)
            ax.text(
                0.98,
                med,
                f"Median {med:.3f}",
                ha="right",
                va="bottom",
                fontsize=9,
                color="#222",
                alpha=0.85,
                transform=ax.get_yaxis_transform(),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888", alpha=0.8),
            )

        # Annotate bars
        for bar in ax.patches:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#111",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
            )

    # Remove unused axes
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        "Plan Coherence: Detailed Metric-by-Metric Breakdown",
        fontsize=20,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    figs["bars"] = fig
    _save_fig_multi(fig, os.path.join(FIGURES_DIR, "plan_coherence_bars"))
    plt.show()

    return figs


def create_modality_quality_visuals(df_long, FIGURES_DIR):
    """Generates and saves radar and bar plots for modality quality results.

    Enhancements mirror plan coherence function.
    """
    print("\nGenerating visuals for Modality Generation Quality")

    metric_order = [
        "ImageSSIM",
        "ImageAverageHash",
        "ImageHistogramMatch",
        "AudioSNR",
        "AudioSpectrogramDistance",
    ]
    df_long = df_long[df_long["metric_name"].isin(metric_order)].copy()
    df_long["metric_name"] = pd.Categorical(
        df_long["metric_name"], categories=metric_order, ordered=True
    )

    figs = {}

    # Radar
    print("Creating Modality Quality Radar Plot...")
    fig_radar = plt.figure(figsize=(7.8, 7.8))
    radar_ax = plt.subplot(111, polar=True)
    plot_radar_comparison(
        df_long,
        axis=radar_ax,
        title="Modality Generation Quality: Specialist Model Performance",
    )
    radar_ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), frameon=True)
    radar_ax.set_facecolor("#FCFCFC")
    figs["radar"] = fig_radar
    _save_fig_multi(fig_radar, os.path.join(FIGURES_DIR, "modality_quality_radar"))
    plt.show()

    # Bars
    print("Creating Modality Quality Bar Plots...")
    metrics = list(df_long["metric_name"].cat.categories)
    n_cols = 3
    n_rows = int(math.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(5.6 * n_cols, 4.2 * n_rows),
        sharey=False,
    )
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        plot_metric_comparison(
            df_long,
            metric_name=metric,
            axis=ax,
            title=metric,
        )
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=12)
        ax.set_facecolor("#FFFFFF")

        vals = [p.get_height() for p in ax.patches]
        if vals:
            vmin, vmax = min(vals), max(vals)
            padding = (vmax - vmin) * 0.12 if vmax > vmin else vmax * 0.15 or 0.1
            ax.set_ylim(vmin * 0.98, vmax + padding)
            med = pd.Series(vals).median()
            ax.axhline(med, color="#444", linestyle="--", linewidth=0.9, alpha=0.55)
            ax.text(
                0.98,
                med,
                f"Median {med:.3f}",
                ha="right",
                va="bottom",
                fontsize=9,
                color="#222",
                alpha=0.85,
                transform=ax.get_yaxis_transform(),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888", alpha=0.8),
            )

        for bar in ax.patches:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#111",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
            )

    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        "Modality Generation Quality: Detailed Metric-by-Metric Breakdown",
        fontsize=20,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    figs["bars"] = fig
    _save_fig_multi(fig, os.path.join(FIGURES_DIR, "modality_quality_bars"))
    plt.show()

    return figs
