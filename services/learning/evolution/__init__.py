"""Evolution strategies for GA gene search (CMA-ES and friends).

The genetic algorithm mutates its policy genes with fixed-width random
noise.  CMA-ES learns the mutation distribution itself -- which directions
in gene space are productive and how far to step -- which is strictly a
better search for small-dimensional continuous problems like the four
Thompson policy genes.  Numpy only, pure functions, deterministic under a
seeded RandomState.
"""
