"""GA tuning framework for Claude Ensemble.

Target A (the Thompson selection policy) is implemented end to end in PR 1 and
runs report-only: it evaluates candidate policy parameters by replaying
recorded outcomes and never touches live routing unless a human (or an
explicitly enabled auto-apply unit) promotes a result.

Layout::

    schema.py    gene specs, canonical JSON, candidate hashing/validation
    corpus.py    bounded read-only ledger ingestion and chronological splits
    engine.py    replay fitness and the ``random`` search strategy
    gates.py     evidence / statistical / support gates and the bootstrap CI
    journal.py   flocked append-only change journal and rollback records
    cli.py       ``python -m learning.ga {run,report,apply,rollback}``
    targets/     one adapter per optimization target
"""

__all__ = ["schema", "corpus", "engine", "gates", "journal"]
