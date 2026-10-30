"""10 Representative RH Workflows for Cache Integration Testing"""
from dataclasses import dataclass
from typing import List

@dataclass
class TestCase:
    id: int
    name: str
    category: str
    expected_baseline_tokens: int
    expected_cached_tokens: int
    cache_reuse_count: int

TEST_CASES = [
    TestCase(1, "Critical Code Review", "Code Review", 18000, 4500, 4),
    TestCase(2, "Release Note Generation", "Documentation", 12000, 2000, 26),
    TestCase(3, "MR 1087 Arbiter-Worker Pattern", "Governance", 15000, 6000, 3),
    TestCase(4, "CPSEARCH-10981 Keyset Pagination", "Code Review", 16000, 5000, 2),
    TestCase(5, "Infrastructure Decision: AWX", "Infrastructure", 10000, 2500, 2),
    TestCase(6, "Sumo Logic Integration Setup", "Monitoring", 14000, 4000, 5),
    TestCase(7, "Google Workspace & Sheets", "Automation", 13000, 5500, 3),
    TestCase(8, "Confluence Publishing Workflow", "Documentation", 9000, 2500, 8),
    TestCase(9, "GitLab Workflow & Approval Gates", "Git/CI", 17000, 6500, 15),
    TestCase(10, "Deployment Tracking", "Operations", 8500, 1500, 30),
]

def get_test_cases(): return TEST_CASES
