"""Contextual bandits (Linear Thompson Sampling) for model routing.

Classic Thompson keeps one Beta posterior per model, blind to what the task
is.  This package implements Linear Thompson Sampling (LinTS): per-arm
Gaussian linear posteriors over a small task-feature vector, so the router
learns *which model wins for which kind of task* from the same outcome
ledger evidence.

Design constraints (same as the rest of the learning stack):

* numpy only -- no torch, no sklearn, no service dependency.
* File-based JSON state, atomic writes, backward-compatible loads.
* The Beta-Thompson service remains the source of truth for raw counts;
  LinTS is an opt-in policy that can be evaluated side-by-side
  (shadow mode) before it ever routes anything.
"""
