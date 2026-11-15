#!/usr/bin/env python3
"""
Anomaly detection for API costs and latency.
Identifies outliers, cost spikes, and performance regressions.
"""

import json
from pathlib import Path
from statistics import mean, stdev
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class Anomaly:
    """Detected anomaly record."""
    timestamp: str
    type: str  # 'cost_spike', 'latency_spike', 'error_rate', 'model_shift'
    model: str
    value: float
    baseline: float
    deviation: float  # in sigma

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'type': self.type,
            'model': self.model,
            'value': self.value,
            'baseline': self.baseline,
            'deviation': self.deviation,
        }

class AnomalyDetector:
    """Statistical anomaly detection using z-score method."""

    def __init__(self, log_file: str = 'cost_tracking/api_costs.jsonl', threshold: float = 2.5):
        """
        Args:
            log_file: Path to JSONL cost log
            threshold: Z-score threshold (2.5 = 1.2% tails)
        """
        self.log_file = Path(log_file)
        self.threshold = threshold
        self.records = self._load_records()

    def _load_records(self) -> List[Dict[str, Any]]:
        """Load all cost records from JSONL."""
        records = []
        if not self.log_file.exists():
            return records

        try:
            with open(self.log_file) as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass

        return records

    def detect_cost_anomalies(self) -> List[Anomaly]:
        """Detect unusual API costs per model."""
        anomalies = []

        # Group by model
        by_model = {}
        for record in self.records:
            model = record.get('model', 'unknown')
            cost = record.get('cost', 0.0)
            if model not in by_model:
                by_model[model] = []
            by_model[model].append({'cost': cost, 'timestamp': record.get('timestamp', '')})

        # Z-score analysis per model
        for model, entries in by_model.items():
            costs = [e['cost'] for e in entries]
            if len(costs) < 3:
                continue

            mu = mean(costs)
            sigma = stdev(costs) if len(costs) > 1 else 0
            if sigma == 0:
                continue

            for entry in entries:
                z = (entry['cost'] - mu) / sigma
                if z > self.threshold:
                    anomalies.append(Anomaly(
                        timestamp=entry['timestamp'],
                        type='cost_spike',
                        model=model,
                        value=entry['cost'],
                        baseline=mu,
                        deviation=z,
                    ))

        return sorted(anomalies, key=lambda a: a.deviation, reverse=True)

    def detect_latency_anomalies(self) -> List[Anomaly]:
        """Detect unusual response times."""
        anomalies = []

        # Group by model
        by_model = {}
        for record in self.records:
            model = record.get('model', 'unknown')
            latency = record.get('latency_ms', 0)
            if latency == 0:
                continue
            if model not in by_model:
                by_model[model] = []
            by_model[model].append({'latency': latency, 'timestamp': record.get('timestamp', '')})

        # Z-score analysis per model
        for model, entries in by_model.items():
            latencies = [e['latency'] for e in entries]
            if len(latencies) < 3:
                continue

            mu = mean(latencies)
            sigma = stdev(latencies) if len(latencies) > 1 else 0
            if sigma == 0:
                continue

            for entry in entries:
                z = (entry['latency'] - mu) / sigma
                if z > self.threshold:
                    anomalies.append(Anomaly(
                        timestamp=entry['timestamp'],
                        type='latency_spike',
                        model=model,
                        value=entry['latency'],
                        baseline=mu,
                        deviation=z,
                    ))

        return sorted(anomalies, key=lambda a: a.deviation, reverse=True)

    def detect_error_rate_anomalies(self) -> List[Anomaly]:
        """Detect unusual error rate spikes."""
        anomalies = []

        # Group by model
        by_model = {}
        for record in self.records:
            model = record.get('model', 'unknown')
            has_error = record.get('error', False)
            if model not in by_model:
                by_model[model] = {'errors': 0, 'total': 0, 'timestamp': record.get('timestamp', '')}
            by_model[model]['total'] += 1
            if has_error:
                by_model[model]['errors'] += 1

        # Compare error rates
        overall_error_rate = sum(m['errors'] for m in by_model.values()) / max(sum(m['total'] for m in by_model.values()), 1)

        for model, stats in by_model.items():
            model_error_rate = stats['errors'] / max(stats['total'], 1)
            if model_error_rate > overall_error_rate * 2:  # Double the average
                anomalies.append(Anomaly(
                    timestamp=stats['timestamp'],
                    type='error_rate',
                    model=model,
                    value=model_error_rate,
                    baseline=overall_error_rate,
                    deviation=(model_error_rate - overall_error_rate) / max(overall_error_rate, 0.001),
                ))

        return sorted(anomalies, key=lambda a: a.value, reverse=True)

    def detect_all(self) -> Dict[str, List[Anomaly]]:
        """Run all anomaly detectors."""
        return {
            'cost': self.detect_cost_anomalies(),
            'latency': self.detect_latency_anomalies(),
            'error_rate': self.detect_error_rate_anomalies(),
        }

    def report(self) -> str:
        """Generate anomaly report."""
        all_anomalies = self.detect_all()
        lines = ["Anomaly Detection Report", "=" * 40]

        for category, anomalies in all_anomalies.items():
            lines.append(f"\n{category.upper()}: {len(anomalies)} anomalies")
            for a in anomalies[:5]:  # Top 5
                lines.append(f"  {a.timestamp:20s} {a.model:15s} {a.value:8.2f} ({a.deviation:.1f}σ)")

        return '\n'.join(lines)


if __name__ == '__main__':
    detector = AnomalyDetector()
    print(detector.report())
