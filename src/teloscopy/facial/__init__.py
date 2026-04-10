"""Facial photo analysis for estimated genomic profiling.

Uses computer-vision techniques to extract facial features from regular
photographs and map them to *estimated* genomic risk profiles using
published research correlations.

**IMPORTANT**: Predictions from facial photos are statistical estimates
based on population-level correlations and phenotype-genotype research.
They are NOT equivalent to actual DNA sequencing or genotyping.  Results
must be interpreted as *indicative* only and should never replace
clinical genetic testing.
"""

from .enhanced_predictor import (
    EnhancedGenomicProfile,
    FacialShapeLocus,
    HIrisPlex_S_Result,
    compute_prediction_accuracy,
    generate_enhanced_profile,
    predict_facial_shape_loci,
    predict_hirisplex_s,
    summarise_profile,
)
