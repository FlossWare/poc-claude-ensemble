#!/bin/bash
# Multi-pass ingestion — Pass 1a: high-value docs (priority-ordered)
# One stream at a time, per the rule of thumb.
set -u
TOOL=~/Development/FlossWare/poc-claude-ensemble-worktrees/ingest-perf/tools/bulk_ingest.py
GW=http://127.0.0.1:8080
LOG=~/.FlossWare/ledger/bulk-ingest/pass1a.log
mkdir -p "$(dirname $LOG)"

SOURCES=(
  "raw-camel          camel-docs        10000"
  "raw-quarkus        quarkus-docs       5000"
  "raw-spring-boot-ref spring-boot-ref   10000"
  "raw-spring-framework-ref spring-fw-ref 10000"
  "raw-kubernetes-io  kubernetes-docs   10000"
  "raw-mongodb        mongodb-docs      10000"
  "raw-docker         docker-docs        5000"
  "raw-rfcs-full      rfcs              12000"
  "raw-w3c-tr         w3c-specs          2000"
  "raw-arxiv-api      arxiv-ml          50000"
  "raw-rust           rust-docs          8000"
  "raw-golang         go-docs            5000"
  "raw-typescript     ts-docs            5000"
  "raw-swift          swift-docs         5000"
  "raw-jenkins        jenkins-docs       5000"
  "raw-sonarqube      sonarqube-docs     3000"
  "raw-puppet         puppet-docs        5000"
  "raw-chef           chef-docs          3000"
  "raw-salt           salt-docs          3000"
)

for entry in "${SOURCES[@]}"; do
  set -- $entry
  dir=$1; name=$2; limit=$3
  echo "=== [pass1a] $name ($(date +%H:%M:%S)) ===" | tee -a "$LOG"
  python3 "$TOOL" ~/.FlossWare/sources/web-scrape/$dir \
    --source $name --origin web --gateway $GW --limit $limit >> "$LOG" 2>&1
  echo "=== $name: $(tail -1 $LOG | grep -o 'Done:.*')" | tee -a "$LOG"
done
echo "=== PASS 1a COMPLETE $(date) ===" | tee -a "$LOG"
