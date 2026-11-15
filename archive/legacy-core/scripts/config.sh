#!/bin/bash
# Claude Ensemble Configuration Script
#
# Interactive configuration for tools and features.
# Users can run this anytime to reconfigure settings.
#
# Usage:
#   ./scripts/config.sh
#   or
#   ~/.claude/config.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CONFIG="$REPO_ROOT/settings.json.default"
USER_CONFIG="$HOME/.claude/settings.json"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is required but not installed"
    exit 1
fi

# If user config doesn't exist, copy defaults
if [ ! -f "$USER_CONFIG" ]; then
    mkdir -p "$(dirname "$USER_CONFIG")"
    cp "$DEFAULT_CONFIG" "$USER_CONFIG"
    echo "📝 Created new config at $USER_CONFIG"
fi

echo ""
echo "======================================================"
echo "  Claude Ensemble Configuration"
echo "======================================================"
echo ""
echo "Config file: $USER_CONFIG"
echo ""

python3 << 'PYTHON_CONFIG'
import json
import sys
import os

config_path = os.environ['USER_CONFIG']
default_path = os.environ['DEFAULT_CONFIG']

# Load current configuration
with open(config_path, 'r') as f:
    config = json.load(f)

# Load defaults for reference
with open(default_path, 'r') as f:
    defaults = json.load(f)

print("\n📦 TOOLS\n")
print("Select which tools to enable:\n")

tools = config.get('tools', {})
for tool_name, tool_config in sorted(tools.items()):
    enabled = tool_config.get('enabled', True)
    description = tool_config.get('description', '')
    status = "✓ ON " if enabled else "✗ OFF"

    print(f"  {status} {tool_name}")
    print(f"       {description}")

    while True:
        response = input(f"\n  Enable {tool_name}? [{'Y/n' if enabled else 'y/N'}]: ").strip().lower()
        if response in ['', 'y', 'n', 'yes', 'no']:
            break
        print("  Please enter 'y' or 'n'")

    if response in ['y', 'yes']:
        tools[tool_name]['enabled'] = True
    elif response in ['n', 'no']:
        tools[tool_name]['enabled'] = False
    print()

print("\n🎯 FEATURES\n")
print("Select which features to enable:\n")

features = config.get('features', {})
for feature_name, feature_config in sorted(features.items()):
    enabled = feature_config.get('enabled', True)
    description = feature_config.get('description', '')
    status = "✓ ON " if enabled else "✗ OFF"

    print(f"  {status} {feature_name}")
    print(f"       {description}")

    while True:
        response = input(f"\n  Enable {feature_name}? [{'Y/n' if enabled else 'y/N'}]: ").strip().lower()
        if response in ['', 'y', 'n', 'yes', 'no']:
            break
        print("  Please enter 'y' or 'n'")

    if response in ['y', 'yes']:
        features[feature_name]['enabled'] = True
    elif response in ['n', 'no']:
        features[feature_name]['enabled'] = False
    print()

# Write updated configuration
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("\n" + "="*54)
print("✓ Configuration saved to:")
print(f"  {config_path}")
print("="*54)
print()

# Summary
enabled_tools = [t for t, c in tools.items() if c.get('enabled', True)]
disabled_tools = [t for t, c in tools.items() if not c.get('enabled', True)]
enabled_features = [f for f, c in features.items() if c.get('enabled', True)]
disabled_features = [f for f, c in features.items() if not c.get('enabled', True)]

print("\n📊 SUMMARY\n")
print(f"Tools enabled ({len(enabled_tools)}):")
for tool in enabled_tools:
    print(f"  ✓ {tool}")

if disabled_tools:
    print(f"\nTools disabled ({len(disabled_tools)}):")
    for tool in disabled_tools:
        print(f"  ✗ {tool}")

print(f"\nFeatures enabled ({len(enabled_features)}):")
for feature in enabled_features:
    print(f"  ✓ {feature}")

if disabled_features:
    print(f"\nFeatures disabled ({len(disabled_features)}):")
    for feature in disabled_features:
        print(f"  ✗ {feature}")

print()

PYTHON_CONFIG
