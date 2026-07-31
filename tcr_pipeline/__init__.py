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

from tcr_pipeline import data_io, normalization, visualization, noise_model, statistics, utils, edgeR_wrapper
