"""Telomere length estimation from sequencing data (BAM/FASTQ files).

Provides pure-Python FASTQ parsing and optional ``pysam``-based BAM reading
to estimate telomere content in whole-genome or targeted sequencing data.
The main metric is the fraction of reads containing ≥3 consecutive telomere
repeats (TTAGGG on the forward strand, CCCTAA on the reverse strand), along
with a simple per-read estimate of telomeric base content.

Gzipped FASTQ files (``.fastq.gz``) are handled transparently via the
:mod:`gzip` module.  BAM file support requires ``pysam`` as an optional
dependency; an informative :exc:`ImportError` is raised if it is missing.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Telomere repeat patterns
# ---------------------------------------------------------------------------

TELOMERE_FORWARD: str = "TTAGGG"
"""Canonical human telomere repeat on the G-rich (forward) strand."""

TELOMERE_REVERSE: str = "CCCTAA"
"""Canonical human telomere repeat on the C-rich (reverse) strand."""

TELOMERE_PATTERN: re.Pattern[str] = re.compile(r"(TTAGGG){3,}|(CCCTAA){3,}")
"""Regex matching ≥3 consecutive telomere repeats in either orientation."""


# ---------------------------------------------------------------------------
# Core counting function
# ---------------------------------------------------------------------------


def count_telomere_repeats(sequence: str) -> dict:
    """Count TTAGGG and CCCTAA repeats in a DNA sequence.

    Non-overlapping occurrences of each hexamer are counted independently.
    The ``is_telomeric`` flag is set when the sequence contains at least
    three *consecutive* repeats (matched by :data:`TELOMERE_PATTERN`).

    Parameters
    ----------
    sequence : str
        DNA sequence string (case-insensitive).

    Returns
    -------
    dict
        - ``forward_count`` (int): number of non-overlapping ``TTAGGG``.
        - ``reverse_count`` (int): number of non-overlapping ``CCCTAA``.
        - ``total_repeats`` (int): ``forward_count + reverse_count``.
        - ``telomeric_bases`` (int): total bases in telomeric repeats
          (``total_repeats * 6``).
        - ``is_telomeric`` (bool): whether ≥3 consecutive repeats are present.
    """
    seq = sequence.upper()
    forward_count: int = seq.count(TELOMERE_FORWARD)
    reverse_count: int = seq.count(TELOMERE_REVERSE)
    total_repeats: int = forward_count + reverse_count
    telomeric_bases: int = total_repeats * 6
    is_telomeric: bool = bool(TELOMERE_PATTERN.search(seq))

    return {
        "forward_count": forward_count,
        "reverse_count": reverse_count,
        "total_repeats": total_repeats,
        "telomeric_bases": telomeric_bases,
        "is_telomeric": is_telomeric,
    }


# ---------------------------------------------------------------------------
# FASTQ estimation (pure Python, no biopython needed)
# ---------------------------------------------------------------------------


def estimate_from_fastq(
    fastq_path: str,
    max_reads: int | None = None,
) -> dict:
    """Estimate telomere length from a FASTQ file.

    Reads the FASTQ file line-by-line, parsing standard 4-line records
    without any external dependencies.  Gzipped files (``.fastq.gz`` or
    ``.fq.gz``) are decompressed transparently.

    A read is classified as *telomeric* if it contains ≥3 consecutive
    ``TTAGGG`` or ``CCCTAA`` repeats.  The estimated mean telomere length
    is the average number of telomeric bases per telomeric read.

    Parameters
    ----------
    fastq_path : str
        Filesystem path to a ``.fastq`` or ``.fastq.gz`` file.
    max_reads : int or None
        If given, stop after processing this many reads.

    Returns
    -------
    dict
        - ``total_reads`` (int): Number of reads processed.
        - ``telomere_reads`` (int): Reads containing ≥3 consecutive repeats.
        - ``telomere_fraction`` (float): ``telomere_reads / total_reads``
          (0.0 if no reads).
        - ``estimated_mean_length_bp`` (float): Mean telomeric bases per
          telomeric read (0.0 if none found).
        - ``telomere_bases_total`` (int): Total telomeric bases across all
          reads.

    Raises
    ------
    FileNotFoundError
        If *fastq_path* does not exist.
    """
    path = Path(fastq_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTQ file not found: {fastq_path}")

    # Choose opener based on extension
    is_gzipped = (
        path.suffix == ".gz" or path.name.endswith(".fastq.gz") or path.name.endswith(".fq.gz")
    )
    opener = gzip.open if is_gzipped else open

    total_reads: int = 0
    telomere_reads: int = 0
    telomere_bases_total: int = 0

    with opener(fastq_path, "rt") as fh:
        while True:
            if max_reads is not None and total_reads >= max_reads:
                break

            # FASTQ 4-line record: @header, sequence, +, quality
            header = fh.readline()
            if not header:
                break  # EOF
            sequence = fh.readline().strip()
            _plus = fh.readline()  # '+' separator line
            _quality = fh.readline()  # quality scores

            # Validate we got a complete record
            if not sequence:
                break

            total_reads += 1
            result = count_telomere_repeats(sequence)

            if result["is_telomeric"]:
                telomere_reads += 1
            telomere_bases_total += result["telomeric_bases"]

    telomere_fraction: float = telomere_reads / total_reads if total_reads > 0 else 0.0
    estimated_mean_length_bp: float = (
        telomere_bases_total / telomere_reads if telomere_reads > 0 else 0.0
    )

    return {
        "total_reads": total_reads,
        "telomere_reads": telomere_reads,
        "telomere_fraction": telomere_fraction,
        "estimated_mean_length_bp": estimated_mean_length_bp,
        "telomere_bases_total": telomere_bases_total,
    }


# ---------------------------------------------------------------------------
# BAM estimation (requires pysam)
# ---------------------------------------------------------------------------


def estimate_from_bam(
    bam_path: str,
    max_reads: int | None = None,
) -> dict:
    """Estimate telomere length from a BAM file.

    Requires the optional ``pysam`` dependency.  If ``pysam`` is not
    installed, an informative :exc:`ImportError` is raised with
    installation instructions.

    Unmapped reads and reads without a query sequence are skipped.

    Parameters
    ----------
    bam_path : str
        Filesystem path to a ``.bam`` file (an index is not required;
        the file is read sequentially with ``until_eof=True``).
    max_reads : int or None
        If given, stop after processing this many mapped reads.

    Returns
    -------
    dict
        Same structure as :func:`estimate_from_fastq`:

        - ``total_reads`` (int)
        - ``telomere_reads`` (int)
        - ``telomere_fraction`` (float)
        - ``estimated_mean_length_bp`` (float)
        - ``telomere_bases_total`` (int)

    Raises
    ------
    ImportError
        If ``pysam`` is not installed.
    FileNotFoundError
        If *bam_path* does not exist.
    """
    try:
        import pysam  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "pysam is required for BAM file processing but is not installed. "
            "Install it with:  pip install pysam\n"
            "Alternatively, convert your BAM to FASTQ and use "
            "estimate_from_fastq() instead."
        )

    path = Path(bam_path)
    if not path.exists():
        raise FileNotFoundError(f"BAM file not found: {bam_path}")

    total_reads: int = 0
    telomere_reads: int = 0
    telomere_bases_total: int = 0

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(until_eof=True):
            if max_reads is not None and total_reads >= max_reads:
                break

            # Skip unmapped reads and reads without sequence
            if read.is_unmapped or read.query_sequence is None:
                continue

            total_reads += 1
            result = count_telomere_repeats(read.query_sequence)

            if result["is_telomeric"]:
                telomere_reads += 1
            telomere_bases_total += result["telomeric_bases"]

    telomere_fraction: float = telomere_reads / total_reads if total_reads > 0 else 0.0
    estimated_mean_length_bp: float = (
        telomere_bases_total / telomere_reads if telomere_reads > 0 else 0.0
    )

    return {
        "total_reads": total_reads,
        "telomere_reads": telomere_reads,
        "telomere_fraction": telomere_fraction,
        "estimated_mean_length_bp": estimated_mean_length_bp,
        "telomere_bases_total": telomere_bases_total,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_telomere_report(results: dict) -> str:
    """Generate a human-readable report from telomere estimation results.

    Parameters
    ----------
    results : dict
        Output from :func:`estimate_from_fastq` or :func:`estimate_from_bam`.

    Returns
    -------
    str
        Multi-line formatted report string suitable for printing or logging.
    """
    divider = "=" * 60
    lines = [
        divider,
        "  TELOMERE LENGTH ESTIMATION REPORT",
        divider,
        "",
        f"  Total reads analysed:      {results['total_reads']:>12,}",
        f"  Telomeric reads:           {results['telomere_reads']:>12,}",
        f"  Telomere fraction:         {results['telomere_fraction']:>12.4%}",
        f"  Total telomeric bases:     {results['telomere_bases_total']:>12,}",
        f"  Estimated mean length:     {results['estimated_mean_length_bp']:>12.1f} bp",
        "",
    ]

    # Interpret results
    fraction = results["telomere_fraction"]
    if fraction > 0.01:
        interpretation = "HIGH telomere content detected."
    elif fraction > 0.001:
        interpretation = "Moderate telomere content detected."
    elif fraction > 0:
        interpretation = "Low telomere content detected."
    else:
        interpretation = "No telomeric reads found."

    lines.extend(
        [
            f"  Interpretation:  {interpretation}",
            "",
            divider,
        ]
    )

    return "\n".join(lines)
