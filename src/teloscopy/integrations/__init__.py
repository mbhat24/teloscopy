"""External data integrations for Teloscopy.

This sub-package provides adapters and importers for connecting Teloscopy's
genomic analysis pipeline with external data sources and direct-to-consumer
(DTC) genomics services.

Supported integrations
----------------------
* **23andMe** — raw genotype data import (v3/v4/v5 chip formats).
* **AncestryDNA** — raw genotype data import (v1/v2 chip formats).
* **VCF 4.x** — Variant Call Format files produced by clinical or research
  sequencing pipelines.
* **Generic tab-delimited** — custom genotype files with auto-detection
  heuristics.

The primary entry point is :class:`~teloscopy.integrations.genotype_import.GenotypeImporter`,
which converts raw files from any supported DTC service into the
``{rsid: genotype}`` dictionary format expected by
:class:`~teloscopy.genomics.disease_risk.DiseasePredictor` and
:class:`~teloscopy.nutrition.diet_advisor.DietAdvisor`.

Example
-------
>>> from teloscopy.integrations.genotype_import import GenotypeImporter
>>> importer = GenotypeImporter()
>>> data = importer.parse_auto("my_23andme_raw.txt")
>>> variant_dict = importer.convert_to_variant_dict(data)
>>> # variant_dict is now ready for DiseasePredictor / DietAdvisor
"""

from teloscopy.integrations.genotype_import import (
    GenotypeData,
    GenotypeImporter,
    GenotypeRecord,
    ValidationReport,
)

__all__ = [
    "GenotypeImporter",
    "GenotypeData",
    "GenotypeRecord",
    "ValidationReport",
]
