#!/usr/bin/env bash
# Installs this repo's git hooks (currently just pre-commit) into .git/hooks/.
#
# Usage: ./scripts/install-git-hooks.sh
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
source_hook="$repo_root/.githooks/pre-commit"
target_hook="$repo_root/.git/hooks/pre-commit"

if [ ! -f "$source_hook" ]; then
    echo "Error: $source_hook not found." >&2
    exit 1
fi

cp "$source_hook" "$target_hook"
chmod +x "$target_hook"

echo "Installed pre-commit hook: $target_hook"
echo "Every commit will now run 'make verify' first."
