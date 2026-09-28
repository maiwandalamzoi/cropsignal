"""
STEP 6 - Three charts, saved as PNGs, that make the terminal tables visual.

Run:  python analysis/step6_charts.py

This produces figures/*.png. It reads only what earlier steps already wrote
(data/processed/, data/raw/, sites_resolved.json) - no new computation, no
Earth Engine call. Each chart has one job:

  fig1_algorithm_result.png     The headline finding: the phenology-aligned
                                detector against its calendar baseline, by
                                country, against a shared no-skill line.

  fig2_bimodal_seasons.png      Why it fails where it fails: Kenya's
                                two-peak growing-season climatology against
                                Afghanistan's single-peak one - the
                                assumption PhenoShift breaks on, made visible.

  fig3_sampling_audit.png       The finding that started the project: which
                                of the original 36 hand-picked points were
                                real farmland, next to this project's own
                                24 sites, all independently verified.

  fig4_site_map.png             Where those 24 sites actually are, on each
                                country's own outline - the map version of
                                fig3's second panel, for anyone who wants
                                to see the geography rather than the count.

Uses the validated categorical palette from the dataviz skill (blue/orange,
CVD-checked) for identity, and the fixed status palette (green/red) for the
pass/fail audit map, where color encodes verdict rather than identity.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FIG_DIR = REPO / "figures"

# Validated categorical palette (dataviz skill, references/palette.md) -
# slots 1/2 (blue/orange), the pair proven safe for a 2-series comparison.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
# Fixed status palette - reserved for pass/fail state, never identity.
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8878"
SURFACE = "#fcfcfb"
GRID = "#e4e3dd"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "savefig.facecolor": SURFACE,
})


def _no_top_right(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _header(fig, title: str, subtitle: str, top: float = 0.78, left: float = 0.13):
    """
    Bold title + gray subtitle, positioned in FIGURE-fraction coordinates.

    Axes-fraction coordinates (`transform=ax.transAxes`) seem like the
    natural choice for a subtitle sitting "just above the axes", but the
    same fraction maps to a different physical distance on every chart -
    it depends on that axes' own height, which changes with figsize and
    with how much room `subplots_adjust(top=...)` leaves it. A y-position
    tuned by eye on one chart collided with the title on the next chart
    that had a different aspect ratio. Figure-fraction coordinates are
    anchored to the whole canvas instead, so one pair of y-values is
    correct for every chart that reserves the same top margin - which is
    why `top` here and in `fig.subplots_adjust(top=...)` must match.

    `left` is the same story on the other axis: text starting at a fixed
    small x looked fine until a rotated y-axis label happened to reach
    further left than that x on this chart, and the two overlapped. Text
    starts at the axes' own left edge instead - `fig.subplots_adjust(left=)`
    reserves the margin the y-label lives in, so nothing placed at that
    same x can land inside it.
    """
    fig.text(left, 0.97, title, fontsize=13.5, fontweight="bold",
             color=INK_PRIMARY, ha="left", va="top")
    fig.text(left, 0.90, subtitle, fontsize=9.5, color=INK_SECONDARY,
             ha="left", va="top", linespacing=1.4)
    fig.subplots_adjust(top=top, left=left)


def fig1_algorithm_result():
    """Grouped bars: water-stress AUC, calendar vs phenology-aligned, by country."""
    val = pd.read_csv(REPO / "data" / "processed" / "validation_results.csv")
    countries = sorted(val["country"].unique())
    detectors = ["calendar", "phenology-aligned"]
    colors = {"calendar": BLUE, "phenology-aligned": ORANGE}
    labels = {"calendar": "Calendar (baseline)", "phenology-aligned": "Phenology-aligned (this project)"}

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(countries))
    width = 0.32

    for i, det in enumerate(detectors):
        vals = [val[(val.country == c) & (val.detector == det)]["water_auc"].iloc[0]
                for c in countries]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, color=colors[det], label=labels[det],
                      zorder=3, edgecolor=SURFACE, linewidth=1.5)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10, color=INK_PRIMARY, fontweight="bold")

    ax.axhline(0.5, color=INK_MUTED, linestyle="--", linewidth=1.3, zorder=2)
    # The empty gap between the two bar groups (x=0.5, between centers 0 and
    # 1) is the only space wide enough for this label not to run under a
    # bar - checked by looking, since the string length vs. margin width
    # isn't obvious from the code alone.
    ax.text(0.5, 0.512, "no independent\nsignal (AUC 0.5)",
            ha="center", va="bottom", fontsize=8.5, color=INK_MUTED, style="italic",
            linespacing=1.3)

    ax.set_xticks(x)
    ax.set_xticklabels(countries, fontsize=11)
    ax.set_ylabel("Water-stress AUC  (higher = tracks independent ET/LST signal)")
    ax.set_ylim(0, 0.80)
    _header(fig, "The phase-aligned detector helps in Afghanistan, loses in Kenya",
           "Validated against evapotranspiration + land surface temperature\n"
           "(independent of the NDVI the labels themselves are built from)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _no_top_right(ax)
    # Outside the plotting area, in the right margin - collision-proof
    # regardless of bar heights, unlike an inside corner.
    ax.legend(frameon=False, fontsize=9.5, loc="upper left",
             bbox_to_anchor=(1.01, 0.7))

    fig.subplots_adjust(right=0.72)
    out = FIG_DIR / "fig1_algorithm_result.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out}")


def fig2_bimodal_seasons():
    """
    Monthly NDVI climatology: Kenya's two peaks vs Afghanistan's one.

    Afghanistan is not purely single-peak either - it shows a smaller
    second bump in August (see phenology.py's module docstring, written
    from the same measurement). Framed honestly: Kenya is the extreme,
    all-12-sites case, not the only case, and the chart says so rather
    than letting a clean-looking two-line plot imply otherwise.
    """
    df = pd.read_csv(REPO / "data" / "raw" / "timeseries.csv", parse_dates=["period_start"])
    df["month"] = df["period_start"].dt.month
    clim = df.groupby(["country", "month"])["NDVI"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    colors = {"Afghanistan": BLUE, "Kenya": ORANGE}
    month_names = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    for country in ["Afghanistan", "Kenya"]:
        g = clim[clim.country == country].sort_values("month")
        ax.plot(g["month"], g["NDVI"], color=colors[country], linewidth=2.4,
                marker="o", markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=country, zorder=3)

    # Headroom above the tallest point so annotations never reach the
    # subtitle band - checked by rendering, not assumed from the numbers.
    ax.set_ylim(0.10, 0.66)

    kenya = clim[clim.country == "Kenya"].set_index("month")["NDVI"]
    for peak_month, tag in [(5, "long rains"), (12, "short rains")]:
        y = kenya.loc[peak_month]
        ax.annotate(tag, xy=(peak_month, y), xytext=(0, 11), textcoords="offset points",
                   ha="center", fontsize=9.5, color=ORANGE, fontweight="bold")

    afg = clim[clim.country == "Afghanistan"].set_index("month")["NDVI"]
    ax.annotate("smaller 2nd peak", xy=(8, afg.loc[8]), xytext=(18, 16),
               textcoords="offset points", fontsize=8.5, color=BLUE,
               arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=1))

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean NDVI (2019–2024)")
    _header(fig, "Bimodal cropping breaks the single-season assumption",
           "Kenya's two rains are comparably sized at all 12 sites; Afghanistan\n"
           "shows the same effect at lower amplitude - PhenoShift fits one season")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _no_top_right(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(1.01, 0.65))

    fig.subplots_adjust(right=0.80)
    out = FIG_DIR / "fig2_bimodal_seasons.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out}")


def fig3_sampling_audit():
    """
    Stacked bar: real-farmland vs not, per legacy country - plus a stat
    callout for this project's own sites.

    A lon/lat scatter was the first attempt here and it was wrong: the
    Netherlands, Afghanistan and New Zealand sites are thousands of
    kilometres apart, so one shared axis range collapses each country's
    12 points into an indistinguishable blob and the actual message - how
    many of each country's points are real - is illegible. The job is a
    per-category proportion, which a stacked bar shows directly; geographic
    position isn't part of the message this chart needs to carry.
    """
    from step3_audit_sampling_points import LEGACY_CSV, main as audit_main

    legacy = audit_main(LEGACY_CSV)
    legacy["ok"] = ~legacy["verdict"].str.startswith(("FAIL", "CHECK"))
    counts = legacy.groupby(["country", "ok"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[True, False], fill_value=0)
    countries = counts.index.tolist()

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(countries))
    ok_vals = counts[True].values
    fail_vals = counts[False].values

    ax.bar(x, ok_vals, 0.55, color=GOOD, label="real farmland", zorder=3,
          edgecolor=SURFACE, linewidth=1.5)
    ax.bar(x, fail_vals, 0.55, bottom=ok_vals, color=CRITICAL, label="not farmland",
          zorder=3, edgecolor=SURFACE, linewidth=1.5)

    for i, (ok, fail) in enumerate(zip(ok_vals, fail_vals)):
        if ok:
            ax.text(i, ok / 2, str(ok), ha="center", va="center", fontsize=11,
                    color="white", fontweight="bold")
        if fail:
            ax.text(i, ok + fail / 2, str(fail), ha="center", va="center", fontsize=11,
                    color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(countries, fontsize=11)
    ax.set_ylabel("Sampling points (of 12 per country)")
    ax.set_ylim(0, 14.5)
    _header(fig, "12 of the original 36 hand-picked points were not farmland",
           "A town, a river, or bare ground - confirmed against ESA WorldCover,\n"
           "not just the vegetation index that first raised the question")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _no_top_right(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(1.01, 0.65))

    # Stat callout, not a second chart: this project's own sites need no
    # bar of their own when the number is simply 24/24.
    n_own = len(json.loads((REPO / "data" / "sites_resolved.json").read_text(encoding="utf-8")))
    ax.text(1.01, 0.18,
           f"This project's own\n{n_own} sites (Afghanistan\n+ Kenya): {n_own}/{n_own}\n"
           f"verified ≥ 90% cropland\nby two independent\nland-cover products.",
           transform=ax.transAxes, fontsize=9, color=INK_SECONDARY,
           va="top", linespacing=1.5,
           bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef7ee",
                    edgecolor=GOOD, linewidth=1.2))

    fig.subplots_adjust(right=0.72)
    out = FIG_DIR / "fig3_sampling_audit.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out}")


def _add_geojson_outline(ax, geojson_path: Path, facecolor: str, edgecolor: str):
    """
    Draw a country outline from a small local GeoJSON file.

    Plain matplotlib polygon patches rather than geopandas/cartopy: the
    only thing needed here is "draw this one shape", and pulling in a
    geospatial stack (with its compiled GEOS/PROJ dependencies, notoriously
    fiddly to install on Windows) for that would be a heavy answer to a
    light question. Handles Polygon and MultiPolygon; holes (a ring after
    the first in a polygon's coordinate list) are cut out via evenodd fill.
    """
    import json

    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    geo = json.loads(geojson_path.read_text(encoding="utf-8"))
    geom = geo["features"][0]["geometry"]
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]

    for poly in polys:
        vertices, codes = [], []
        for ring in poly:
            vertices += ring
            codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 2) + [MplPath.CLOSEPOLY]
        path = MplPath(vertices, codes)
        ax.add_patch(PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,
                               linewidth=1.3, zorder=1))


def fig4_site_map():
    """Where the 24 verified sites actually are, on each country's own outline."""
    sites = pd.DataFrame(json.loads((REPO / "data" / "sites_resolved.json")
                                    .read_text(encoding="utf-8")))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    countries = [("Afghanistan", "AFG"), ("Kenya", "KEN")]

    # Panel titles as fig.text at one shared y, not per-axes ax.set_title.
    # set_aspect("equal") on a map gives the two panels different effective
    # box heights (Afghanistan's and Kenya's lon/lat extents have different
    # aspect ratios), so a title padded from each axes' own top - which is
    # where ax.set_title anchors - lands at a different figure-y for each
    # panel. Rendered and looked: Kenya's title sat visibly higher than
    # Afghanistan's. Figure-fraction coordinates put both at the same
    # height regardless of what the map underneath does.
    panel_x = [0.02, 0.52]

    for ax, (country, code), tx in zip(axes, countries, panel_x):
        boundary = REPO / "data" / "boundaries" / f"{code}.geojson"
        _add_geojson_outline(ax, boundary, facecolor="#eef2ea", edgecolor="#9a9a8f")

        sub = sites[sites.country == country]
        ax.scatter(sub["lon"], sub["lat"], c=GOOD, s=90, edgecolor="white",
                  linewidth=1.4, zorder=3)

        fig.text(tx, 0.66, f"{country}  (n={len(sub)})", fontsize=12,
                 fontweight="bold", color=INK_PRIMARY, ha="left", va="top")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Fixed margin around the country outline rather than autoscaling
        # to the 12 points - otherwise a tight cluster of sites (Kenya's,
        # which sit in two regional groups) zooms in past the country's own
        # shape and the map stops reading as "this country".
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        pad_x, pad_y = (x1 - x0) * 0.08, (y1 - y0) * 0.08
        ax.set_xlim(x0 - pad_x, x1 + pad_x)
        ax.set_ylim(y0 - pad_y, y1 + pad_y)

    _header(fig, "24 verified sites, on the ground",
           "Every point cleared ≥90% cropland against two independent land-cover "
           "products\n(ESA WorldCereal + WorldCover) before any satellite data was pulled",
           top=0.72, left=0.04)

    fig.subplots_adjust(wspace=0.15)
    out = FIG_DIR / "fig4_site_map.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_algorithm_result()
    fig2_bimodal_seasons()
    fig3_sampling_audit()
    fig4_site_map()
    print(f"\nAll figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
