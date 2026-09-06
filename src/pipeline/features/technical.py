# src/pipeline/features/technical.py
"""
NOT NEEDED AS A SEPARATE MODULE YET. Technical indicator computation
already exists, is tested, and is leakage-safe:
DataProcessor.add_technical_indicators() (src/pipeline/processor.py),
called strictly after chronological splitting so indicators never cross
a train/val/test boundary.

This is a placeholder, not a duplicate implementation. A second
technical-indicator path here -- however similar to the original -- 
creates the exact silent-divergence risk this project has already hit
once (two config files disagreeing on date range; two evaluate.py
metric implementations quietly using different ddof conventions). If a
genuine need for a standalone module emerges (e.g. computing indicators
outside the train/val/test split flow, like live feature serving),
extract from DataProcessor deliberately at that point, with the existing
test suite as the safety net -- don't rewrite from scratch.
"""