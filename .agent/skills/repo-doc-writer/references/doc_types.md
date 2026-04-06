# Preferred Doc Types

When there is not enough time to document everything, prioritize in this order.

## 1. AGENTS.md
Create when the repo is large, multi-stage, or likely to be edited by coding agents.

Must include:
- purpose
- module map
- artifact map
- safe change rules
- what to inspect before refactors
- subsystem-specific change impact

## 2. docs/architecture.md
Create when the repository spans multiple modules, runtimes, or services.

Must include:
- text architecture diagram
- runtime boundaries
- artifact flow
- important entry points
- current limitations

## 3. Module README
Create for major folders that have their own logic or entry points.

Must include:
- responsibility
- non-responsibility
- key files
- commands / invocation pattern
- inputs and outputs
- what to test after changes

## 4. Contract docs
Create when a stage writes files, emits JSON/CSV, or acts as a boundary between modules.

Must include:
- upstream inputs
- downstream outputs
- naming conventions
- schema expectations
- compatibility risks
- validation checklist

## 5. Runbooks
Create when setup or troubleshooting repeatedly causes friction.

Must include:
- prerequisites
- exact steps
- expected outputs
- common failure cases
