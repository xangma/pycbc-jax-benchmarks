#!/usr/bin/env python3
"""Plot recorded diagnostic samples without inferring speedup or capacity."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics


REPOSITORY = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def series(data, key, median_key, label, scope, work):
    values = data[key]
    if len(values) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                               or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError(f"{key} must contain three finite positive timings")
    median = statistics.median(values)
    if not math.isclose(median, data[median_key], rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Recorded median disagrees with samples: {median_key}")
    return {"samples_seconds": values, "median_seconds": median,
            "display_median_seconds": f"{median:.2f}", "label": label,
            "scope": scope, "work": work, "source_key": key,
            "median_source_key": median_key}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path,
                        default=REPOSITORY / "reports/2026-10-09/measurements.json")
    parser.add_argument("--output", type=Path,
                        default=REPOSITORY / "reports/2026-10-09/timings.png")
    parser.add_argument("--manifest", type=Path,
                        default=REPOSITORY / "reports/2026-10-09/timings-plot.json")
    args = parser.parse_args(argv)
    input_sha = digest(args.measurements)
    data = json.loads(args.measurements.read_text())
    inspiral, live = data["inspiral"], data["live"]
    panels = {
        "inspiral": [
            series(inspiral, "full_process_wall_seconds", "full_process_median_seconds",
                   "Full observer process", data["full_process_scope"],
                   {"valid_detector_seconds": inspiral["valid_detector_seconds"],
                    "template_detector_seconds": inspiral["template_detector_seconds"]}),
            series(inspiral, "application_host_search_seconds", "application_host_search_median_seconds",
                   "Application host search timer\nNo additional device fence",
                   inspiral["application_timer_scope"],
                   {"valid_detector_seconds": inspiral["valid_detector_seconds"],
                    "template_detector_seconds": inspiral["template_detector_seconds"]}),
        ],
        "live": [
            series(live, "full_process_wall_seconds", "full_process_median_seconds",
                   f"Full observer process\n{live['valid_detector_seconds']['H1']} s per detector",
                   data["full_process_scope"],
                   {"valid_detector_seconds": live["valid_detector_seconds"],
                    "template_detector_seconds": live["template_detector_seconds"]}),
            series(live, "suffix_after_two_valid_blocks_seconds", "suffix_median_seconds",
                   f"Suffix after two valid blocks\n{live['suffix_valid_detector_seconds']['H1']} s per detector",
                   live["suffix_scope"],
                   {"valid_detector_seconds": live["suffix_valid_detector_seconds"],
                    "template_detector_seconds": live["suffix_template_detector_seconds"]}),
        ],
    }
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.6), sharey=False)
    colors, markers = ["#0072B2", "#D55E00", "#009E73"], ["o", "^", "s"]
    offsets = [-0.12, 0, 0.12]
    titles = {
        "inspiral": f"Inspiral · {inspiral['template_count']:,} templates\nH1: {inspiral['valid_detector_seconds']['H1']:,} valid seconds",
        "live": f"Live · {live['template_count']:,} templates\nH1 and L1 · different full/suffix work",
    }
    for ax, (name, values) in zip(axes, panels.items()):
        maximum = max(max(item["samples_seconds"]) for item in values)
        for x, item in enumerate(values):
            for offset, color, marker, value in zip(offsets, colors, markers, item["samples_seconds"]):
                ax.scatter(x + offset, value, color=color, marker=marker, s=58,
                           edgecolors="white", linewidths=0.6, zorder=3)
            median = item["median_seconds"]
            ax.plot([x - 0.24, x + 0.24], [median, median], color="#222222",
                    linewidth=2.2, zorder=2)
            ax.text(x, median + maximum * 0.065, f"median {median:.2f} s",
                    ha="center", va="bottom", fontsize=10, color="#222222")
        ax.set_title(titles[name], fontsize=12, pad=15)
        ax.set_ylabel("Elapsed time (s)")
        ax.set_xticks(range(2), [item["label"] for item in values], fontsize=10)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, maximum * 1.25)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"Diagnostic observer timings · {data['date']}\n"
                    "Three samples per scope · no CPU/GPU performance comparison", fontsize=15, y=0.98)
    legend = [Line2D([], [], color=color, marker=marker, linestyle="none", markersize=7,
                     label=f"Sample {i + 1}") for i, (color, marker) in enumerate(zip(colors, markers))]
    legend.append(Line2D([], [], color="#222222", linewidth=2.2, label="Median"))
    figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.865),
                  ncol=4, frameon=False, fontsize=10)
    figure.text(0.5, 0.065, "Separate prime and Nsight runs excluded. Observer processes include imports, checks and shutdown.\n"
                "Inspiral secondary timer excludes setup/output and has no added device fence.\n"
                "Live suffix covers less data; tracing/cache work continues. Neither secondary scope is qualified steady-state.",
                ha="center", va="center", fontsize=9, linespacing=1.55, color="#444444")
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.24, top=0.71, wspace=0.26)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor="white")
    plt.close(figure)
    try:
        input_name = args.measurements.resolve().relative_to(REPOSITORY).as_posix()
    except ValueError:
        input_name = args.measurements.name
    manifest = {
        "schema_version": 1, "input": input_name, "input_sha256": input_sha,
        "plot_sha256": digest(args.output), "source_commit": data["source_commit"],
        "script": "tools/plot_measurements.py", "script_sha256": digest(Path(__file__)),
        "python_version": platform.python_version(), "matplotlib_version": matplotlib.__version__,
        "unit": "seconds", "panels": panels,
        "qualification": {"qualified_speedup": None, "steady_state_capacity": None,
                          "steady_warm_verified": None},
        "excluded": ["prime", "profile", "single original CPU control"],
        "note": "Recorded observer timings. Full and secondary scopes differ; Live work counts also differ. No speedup or capacity is inferred.",
    }
    if digest(args.measurements) != input_sha:
        raise RuntimeError("Measurements changed while plotting")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Saved {args.output.name} and {args.manifest.name}")


if __name__ == "__main__":
    main()
