# Caster Subnet

This repository contains everything validator operators need to run a validator and everything miners need to build/test miner scripts and submit them to the platform.

## Packages

- `validator/` — validator runtime for operators
- `miner/` — miner-facing sandbox harness and local tooling
- `miner-sdk/` — SDK used by miner scripts
- `commons/` — shared utilities used by the above packages

## Entry points

- Validator operators: see `validator/README.md`
- Miner developers: see `miner/README.md`

## Local development

This repository is a UV workspace. To install dependencies:

```bash
uv sync --all-packages --dev
```
