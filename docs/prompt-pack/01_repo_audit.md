# Prompt 01 — Repository Documentation Audit

Use `repo-doc-writer` to audit this repository and propose a documentation plan.

Goals:
- identify the most important documentation gaps for both humans and coding agents
- reduce hidden assumptions and fragile interfaces
- recommend a minimum high-leverage set of docs to create first

Instructions:
1. Inspect the repo structure, entry points, artifacts, and major modules.
2. Tell me which docs already exist and which ones are still missing.
3. Prioritize docs that will most reduce accidental breakage or exploration cost.
4. You may ask me questions, but only if the answer materially changes architecture, interface, or workflow docs.
5. Ask at most 5 questions.
6. If the repo already contains enough information, state assumptions and proceed.

Required output:

## Findings
### Critical
- ...

### Important
- ...

### Nice-to-have
- ...

## Proposed docs to create
- `AGENTS.md`: why it matters
- `docs/architecture.md`: why it matters
- ...

## Assumptions
- ...

## Blocking questions
- None
