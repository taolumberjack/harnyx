---
name: run-local-eval
description: Run miner local batch evaluation and extract the next iteration signal.
---

# Run Local Eval

Goal: run `harnyx-miner-local-eval`, collect the reports, and decide the next move.

## Steps

1. Choose the mode:
   - `vs-champion` for the default comparison loop
   - `target-only` for a quicker isolated check
2. Run:

```bash
uv run --package harnyx-miner harnyx-miner-local-eval --agent-path ./agent.py
```

3. Read:
   - `local-eval-report-<batch-id>-<mode>.json`
   - `local-eval-report-<batch-id>-<mode>.md`
4. Extract:
   - score deltas
   - task-level wins / losses
   - cost regressions
   - retries / failures
5. Decide whether to:
   - keep iterating on the artifact
   - switch to `target-only`
   - submit the current artifact

## Output

- command used
- report paths
- key findings from the run
- next recommended action
