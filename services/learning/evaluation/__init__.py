"""Offline policy evaluation for the GA promotion gate.

The GA currently promotes candidate policies on a bootstrap confidence
interval over replayed outcomes.  Doubly-robust (DR) estimation extracts
more signal from the same logged data: it combines an importance-sampling
term (unbiased, high variance) with a reward model (biased, low variance)
so the estimator stays consistent if EITHER the propensities or the model
are right -- hence "doubly robust".
"""
