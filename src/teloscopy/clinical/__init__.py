"""
Clinical-grade modules for teloscopy analysis.

This package provides clinical validation, regulatory pathway assessment,
multi-site clinical trial coordination, and quality management tools
necessary for transitioning telomere length analysis from research use
only (RUO) to clinical diagnostic applications.

Modules:
    validation : Analytical validation pipelines and FDA regulatory pathway
                 assessment for Software as a Medical Device (SaMD) classification.
    trials     : Multi-institution clinical trial coordination, enrollment
                 management, group-sequential analysis, adverse-event tracking,
                 DSMB reporting, and the high-level ``TrialManager`` facade
                 with consent management, differential-privacy aggregation,
                 and FDA regulatory export capabilities.
    endpoints  : FastAPI REST API endpoints for clinical trial management,
                 exposing the ``TrialManager`` over HTTP.

All modules are designed in accordance with:
    - FDA Guidance: Software as a Medical Device (SaMD) — Clinical Evaluation (2017)
    - CLSI EP15-A3: User Verification of Precision and Estimation of Bias
    - ISO 13485:2016: Medical Devices — Quality Management Systems
    - IEC 62304:2006+A1:2015: Medical Device Software — Software Life Cycle Processes
    - ICH E6(R2): Guideline for Good Clinical Practice (2016)
    - ICH E9: Statistical Principles for Clinical Trials (1998)
    - 21 CFR Part 11: Electronic Records; Electronic Signatures
"""
