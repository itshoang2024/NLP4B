# Project Documentation Conventions

Use these conventions by default when writing docs for this repository family.

## Canonical source rule
Avoid duplicating truth across many files.

Recommended ownership:
- `README.md`: project overview and getting started
- `AGENTS.md`: coding-agent workflow and safe change rules
- `docs/architecture.md`: system map and artifact flow
- `docs/contracts/*.md`: stage IO contracts and compatibility notes
- module `README.md`: local responsibilities and commands
- runbooks: setup / troubleshooting / release procedures

## Command labeling
If a command is observed directly in code or docs, present it normally.
If it is inferred from imports, CLI structure, or file layout, label it as:
- `Inferred command:`

## Assumption labeling
Use a dedicated section:

## Assumptions
- ...

Do not scatter hidden assumptions into factual sections.

## Change-impact labeling
For docs that describe a boundary or artifact producer, include:

## Change impact
- If you modify X, inspect Y and Z.

## Schema documentation
For JSON / CSV artifacts, prefer documenting:
- purpose of the file
- top-level fields or columns
- ordering assumptions
- compatibility risks
- downstream consumers if identifiable
