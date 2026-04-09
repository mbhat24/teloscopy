"""Command-line interface for teloscopy.

Provides the ``teloscopy`` CLI with subcommands for single-image analysis,
batch processing, synthetic image generation, and report creation.

Usage::

    teloscopy analyze IMAGE_PATH [OPTIONS]
    teloscopy batch INPUT_DIR [OPTIONS]
    teloscopy generate [OPTIONS]
    teloscopy report CSV_PATH [OPTIONS]
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Rich console helpers (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table

    _console = Console()
    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False
    _console = None  # type: ignore[assignment]


def _info(msg: str) -> None:
    if _HAS_RICH:
        _console.print(f"[bold green]INFO[/]  {msg}")
    else:
        click.echo(f"INFO  {msg}")


def _warn(msg: str) -> None:
    if _HAS_RICH:
        _console.print(f"[bold yellow]WARN[/]  {msg}")
    else:
        click.echo(f"WARN  {msg}", err=True)


def _error(msg: str) -> None:
    if _HAS_RICH:
        _console.print(f"[bold red]ERROR[/] {msg}")
    else:
        click.echo(f"ERROR {msg}", err=True)


def _load_yaml(path: str) -> dict:
    """Load a YAML config file, handling missing ``pyyaml`` gracefully."""
    try:
        import yaml
    except ImportError:
        _error("PyYAML is not installed.  Install it with: pip install pyyaml")
        sys.exit(1)
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        _error(f"Config file {path} did not parse to a dict.")
        sys.exit(1)
    return data


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="0.1.0", prog_name="teloscopy")
def main() -> None:
    """Teloscopy: Telomere length analysis from microscopy images."""


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@main.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Pipeline configuration YAML file.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="output",
    help="Output directory for results.",
)
@click.option(
    "--method",
    "-m",
    type=click.Choice(["otsu_watershed", "cellpose"], case_sensitive=False),
    default="otsu_watershed",
    help="Chromosome segmentation method.",
)
@click.option(
    "--save-overlay/--no-overlay",
    default=True,
    help="Save an overlay image with detected spots.",
)
def analyze(
    image_path: str,
    config: str | None,
    output: str,
    method: str,
    save_overlay: bool,
) -> None:
    """Analyze a single qFISH microscopy image for telomere lengths."""
    from .analysis.statistics import create_results_dataframe
    from .telomere.pipeline import analyze_image, get_default_config, load_config
    from .visualisation.plots import plot_telomere_overlay

    _info(f"Analyzing image: {image_path}")

    # Configuration -------------------------------------------------------
    cfg = get_default_config()
    if config is not None:
        user_cfg = load_config(config)
        cfg.update(user_cfg)
    cfg["segmentation_method"] = method

    # Output directory ----------------------------------------------------
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline --------------------------------------------------------
    if _HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=_console,
        ) as progress:
            task = progress.add_task("Running pipeline...", total=None)
            result = analyze_image(str(image_path), config=cfg)
            progress.update(task, completed=1, total=1)
    else:
        result = analyze_image(str(image_path), config=cfg)

    spots = result["spots"]
    stats = result["statistics"]

    # Save CSV results ----------------------------------------------------
    image_name = Path(image_path).stem
    df = create_results_dataframe(spots, image_name=image_name)
    csv_path = out_dir / f"{image_name}_telomeres.csv"
    df.to_csv(csv_path, index=False)
    _info(f"Results saved to {csv_path}")

    # Save overlay --------------------------------------------------------
    if save_overlay:
        overlay_path = out_dir / f"{image_name}_overlay.png"
        plot_telomere_overlay(
            dapi=result["channels"]["dapi"],
            cy3=result["channels"]["cy3"],
            spots=spots,
            chromosomes=result.get("chromosomes"),
            labels=result.get("labels"),
            save_path=str(overlay_path),
        )
        _info(f"Overlay saved to {overlay_path}")

    # Print summary table -------------------------------------------------
    if _HAS_RICH:
        table = Table(title="Analysis summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Image", image_name)
        table.add_row("Telomeres detected", str(stats.get("n_telomeres", 0)))
        table.add_row("Mean intensity", f"{stats.get('mean_intensity', 0):.2f}")
        table.add_row("Median intensity", f"{stats.get('median_intensity', 0):.2f}")
        table.add_row("CV", f"{stats.get('cv', 0):.3f}")
        table.add_row("Short telomere %", f"{stats.get('short_telomere_pct', 0):.1f}%")
        _console.print(table)
    else:
        click.echo(f"  Telomeres detected : {stats.get('n_telomeres', 0)}")
        click.echo(f"  Mean intensity     : {stats.get('mean_intensity', 0):.2f}")
        click.echo(f"  Median intensity   : {stats.get('median_intensity', 0):.2f}")
        click.echo(f"  CV                 : {stats.get('cv', 0):.3f}")

    _info("Done.")


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


@main.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--config", "-c", type=click.Path(exists=True), default=None)
@click.option("--output", "-o", type=click.Path(), default="output")
@click.option("--pattern", "-p", default="*.tif", help="File glob pattern.")
def batch(input_dir: str, config: str | None, output: str, pattern: str) -> None:
    """Batch process a directory of qFISH images."""
    from .telomere.pipeline import analyze_batch, get_default_config, load_config

    cfg = get_default_config()
    if config is not None:
        user_cfg = load_config(config)
        cfg.update(user_cfg)

    in_dir = Path(input_dir)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(pattern))
    if not files:
        _warn(f"No files matching '{pattern}' found in {in_dir}")
        return

    _info(f"Found {len(files)} images in {in_dir}")

    if _HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=_console,
        ) as progress:
            task = progress.add_task("Batch processing...", total=len(files))
            analyze_batch(
                input_dir=str(in_dir),
                output_dir=str(out_dir),
                config=cfg,
                pattern=pattern,
            )
            progress.update(task, completed=len(files))
    else:
        analyze_batch(
            input_dir=str(in_dir),
            output_dir=str(out_dir),
            config=cfg,
            pattern=pattern,
        )

    _info(f"Processed {len(files)} images.  Combined CSV: {out_dir / 'combined_results.csv'}")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@main.command()
@click.option("--output", "-o", type=click.Path(), default="data/sample_images")
@click.option("--n-images", "-n", default=5, help="Number of images to generate.")
@click.option("--seed", "-s", default=42, help="Random seed for reproducibility.")
def generate(output: str, n_images: int, seed: int) -> None:
    """Generate synthetic qFISH test images."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    _info(f"Generating {n_images} synthetic qFISH images (seed={seed})...")

    try:
        from skimage.io import imsave
    except ImportError:
        _error("scikit-image is required for image generation. pip install scikit-image")
        sys.exit(1)

    for i in range(n_images):
        height, width = 512, 512

        # DAPI channel – diffuse blobs simulating chromosome spread
        dapi = np.zeros((height, width), dtype=np.float64)
        n_chroms = rng.integers(30, 50)
        for _ in range(n_chroms):
            cy, cx = rng.integers(50, height - 50), rng.integers(50, width - 50)
            length = rng.integers(15, 40)
            angle = rng.uniform(0, np.pi)
            for t in np.linspace(-length / 2, length / 2, num=int(length * 2)):
                yy = int(cy + t * np.sin(angle))
                xx = int(cx + t * np.cos(angle))
                if 0 <= yy < height and 0 <= xx < width:
                    dapi[max(0, yy - 2) : yy + 3, max(0, xx - 2) : xx + 3] += rng.uniform(300, 800)
        dapi += rng.normal(100, 20, size=(height, width))
        dapi = np.clip(dapi, 0, 65535)

        # Cy3 channel – bright point-like telomere spots at chromosome tips
        cy3 = rng.normal(80, 15, size=(height, width))
        n_spots = rng.integers(60, 100)
        for _ in range(n_spots):
            sy, sx = rng.integers(30, height - 30), rng.integers(30, width - 30)
            intensity = rng.uniform(500, 4000)
            sigma = rng.uniform(1.5, 3.0)
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    yy, xx = sy + dy, sx + dx
                    if 0 <= yy < height and 0 <= xx < width:
                        cy3[yy, xx] += intensity * np.exp(-(dy**2 + dx**2) / (2 * sigma**2))
        cy3 = np.clip(cy3, 0, 65535)

        # Stack as multi-channel TIFF (C, H, W)
        merged = np.stack([dapi.astype(np.uint16), cy3.astype(np.uint16)], axis=0)
        fname = out_dir / f"synthetic_qfish_{i:03d}.tif"
        imsave(str(fname), merged)
        _info(f"  Saved {fname}")

    _info("Synthetic image generation complete.")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@main.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="output")
def report(csv_path: str, output: str) -> None:
    """Generate analysis report from CSV results."""
    import pandas as pd

    from .visualisation.plots import plot_cell_comparison, plot_intensity_histogram

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    _info(f"Loading results from {csv_path}")
    df = pd.read_csv(csv_path)

    # Convert rows back to spot dicts
    spots = df.to_dict(orient="records")

    # Intensity histogram -------------------------------------------------
    hist_path = out_dir / "telomere_histogram.png"
    calibrated = "length_bp" in df.columns and df["length_bp"].notna().any()
    plot_intensity_histogram(spots, calibrated=calibrated, save_path=str(hist_path))
    _info(f"Histogram saved to {hist_path}")

    # Per-cell box plot (group by image column) ---------------------------
    images = df["image"].unique()
    cells_data: list[dict] = []
    cell_labels: list[str] = []
    for img_name in images:
        sub = df[df["image"] == img_name]
        valid_sub = sub[(sub["associated"] == True) & (sub["valid"] == True)]  # noqa: E712
        if valid_sub.empty:
            continue
        cells_data.append({"intensities": valid_sub["corrected_intensity"].values})
        cell_labels.append(str(img_name))

    if cells_data:
        box_path = out_dir / "cell_comparison.png"
        plot_cell_comparison(cells_data, labels=cell_labels, save_path=str(box_path))
        _info(f"Cell comparison saved to {box_path}")

    # Summary table -------------------------------------------------------
    if _HAS_RICH:
        table = Table(title="Report summary")
        table.add_column("Image", style="cyan")
        table.add_column("Telomeres", style="green")
        table.add_column("Mean intensity", style="magenta")
        for img_name in images:
            sub = df[df["image"] == img_name]
            valid_sub = sub[(sub["associated"] == True) & (sub["valid"] == True)]  # noqa: E712
            n = len(valid_sub)
            mean_i = valid_sub["corrected_intensity"].mean() if n > 0 else 0.0
            table.add_row(str(img_name), str(n), f"{mean_i:.2f}")
        _console.print(table)
    else:
        click.echo("--- Report Summary ---")
        for img_name in images:
            sub = df[df["image"] == img_name]
            valid_sub = sub[(sub["associated"] == True) & (sub["valid"] == True)]  # noqa: E712
            n = len(valid_sub)
            mean_i = valid_sub["corrected_intensity"].mean() if n > 0 else 0.0
            click.echo(f"  {img_name}: {n} telomeres, mean={mean_i:.2f}")

    _info("Report generation complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
