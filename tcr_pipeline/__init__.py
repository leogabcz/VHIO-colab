"""
tcr_pipeline
============
Modular Python pipeline for TCR repertoire dynamics analysis.

Modules
-------
data_io         : load and convert between data formats
normalization   : library-size correction, CLR normalization and TMM normalization
clustering      : cluster non-neutral clonotype trajectories
"""

from tcr_pipeline import data_io, nb_glm_old, normalization, visualization, noise_model, statistics, utils, nb_glm_legacy

try:  # needs rpy2 + R + edgeR; only used for cross-checking against real edgeR
    from tcr_pipeline import edgeR_wrapper
except ImportError:
    edgeR_wrapper = None
