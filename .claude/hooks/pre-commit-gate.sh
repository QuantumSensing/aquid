#!/usr/bin/env bash
# Pre-commit gate — hard blocks git commit if checks fail.
# Invoked by Claude Code PreToolUse hook on every Bash tool call.
# Only activates when the command contains "git commit".
# Detects Python (uv/pyproject.toml) and Rust (Cargo.toml) projects automatically.

set -euo pipefail

TOOL_INPUT="$1"

# Only intercept git commit calls
if ! echo "$TOOL_INPUT" | grep -q "git commit"; then
  exit 0
fi

echo "=== Pre-commit gate ==="
FAILED=0

# ── Python checks (if pyproject.toml present) ─────────────────────────────────
if [ -f "pyproject.toml" ]; then
  echo "--- ruff check ---"
  if ! uv run ruff check . 2>&1; then
    echo "FAIL: ruff check"
    FAILED=1
  fi

  echo "--- ruff format ---"
  if ! uv run ruff format --check . 2>&1; then
    echo "FAIL: ruff format (run 'uv run ruff format .' to fix)"
    FAILED=1
  fi

  echo "--- ty check ---"
  if ! uv run ty check 2>&1; then
    echo "FAIL: ty check"
    FAILED=1
  fi

  echo "--- pytest ---"
  if ! uv run pytest --tb=short -q 2>&1; then
    echo "FAIL: pytest"
    FAILED=1
  fi
fi

# ── Rust checks (if Cargo.toml present) ───────────────────────────────────────
if [ -f "Cargo.toml" ]; then
  echo "--- cargo fmt ---"
  if ! cargo fmt --check 2>&1; then
    echo "FAIL: cargo fmt (run 'cargo fmt' to fix)"
    FAILED=1
  fi

  echo "--- cargo clippy ---"
  if ! cargo clippy -- -D warnings 2>&1; then
    echo "FAIL: cargo clippy"
    FAILED=1
  fi

  echo "--- cargo test ---"
  if ! cargo test 2>&1; then
    echo "FAIL: cargo test"
    FAILED=1
  fi
fi

# ── notes/ guard ──────────────────────────────────────────────────────────────
echo "--- notes/ check ---"
if git diff --cached --name-only | grep -q "^notes/"; then
  echo "FAIL: notes/ files are staged for commit."
  echo "notes/ is a private working directory and must not be committed."
  echo "Run: git restore --staged notes/"
  FAILED=1
fi

if [ "$FAILED" -eq 1 ]; then
  echo ""
  echo "=== COMMIT BLOCKED ==="
  echo "Fix all failures above before committing."
  exit 1
fi

echo ""
echo "=== All checks passed — commit allowed ==="
exit 0
