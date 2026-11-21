"""Racing algorithms (best-arm identification) for model onboarding.

Successive halving allocates a fixed evaluation budget across candidate arms
in brackets: every surviving arm gets the same number of probes per round,
then only the best half advance.  Compared with replaying every candidate on
every probe task, this identifies the strong arms with a fraction of the
cost -- the point for cold-starting new models, where each probe is a real
gateway call adjudicated by a real arbiter.
"""
