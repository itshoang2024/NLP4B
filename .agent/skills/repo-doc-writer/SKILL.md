---
name: repo-doc-writer
description: Use this skill when the user wants to create, improve, audit, or standardize repository documentation for a software project. This includes AGENTS.md, README files, architecture docs, module docs, interface contracts, runbooks, onboarding docs, contribution guides, maintenance docs, and coding-agent-friendly documentation. Use it when the task requires reading the codebase, inferring module boundaries, extracting hidden conventions, or writing commit-ready Markdown docs grounded in the repository.
---

# Repo Doc Writer

## Goal
Create practical, commit-ready documentation that helps both humans and coding agents understand, modify, run, and maintain a repository safely.

This skill is not for generic prose generation. It is for codebase-grounded documentation.

## Use this skill when
Activate this skill when the user asks to:
- write or improve `README.md`
- create `AGENTS.md`
- document architecture, pipelines, or module boundaries
- create docs for setup, run, deploy, monitor, debug, or contribute workflows
- create input/output contracts for scripts, services, or data pipelines
- audit repository docs for gaps, inconsistencies, or hidden assumptions
- standardize documentation for a large or messy repo
- create coding-agent-friendly docs that reduce exploration cost and accidental breakage

## Do not use this skill when
Do not use this skill for:
- purely marketing or sales copy
- unrelated blog posts or generic educational content
- API reference generation when the user explicitly wants auto-generated docs from another tool
- speculative architecture documents not grounded in the repo

## Core operating principle
Your job is to convert repository reality into explicit documentation.

Prefer:
- concrete file paths over vague descriptions
- observable behavior over guesses
- contracts over narrative
- change-impact notes over abstract summaries
- markdown that is ready to commit

Avoid:
- repeating folder names without explaining responsibility
- claiming workflows that are not supported by the repo
- inventing commands, config files, schemas, or deployment steps without marking them as assumptions
- asking unnecessary questions when the codebase already contains enough information

## Inputs you may inspect
When available, inspect the following before writing docs:
- root `README.md`
- `AGENTS.md`, `GEMINI.md`, contribution docs, architecture docs, design docs
- package manifests such as `requirements.txt`, `pyproject.toml`, `package.json`, `Dockerfile`, compose files
- CI/CD configs
- entry points (`main.py`, CLI scripts, app launchers, service runners)
- source directories and tests
- config files and `.env.example`
- scripts that create artifacts, write files, call external services, or define schemas
- sample data, templates, migrations, notebooks, and deployment manifests

## Required workflow
Follow this workflow unless the user explicitly requests a different one.

### Step 1: Inspect and map the repo
Infer:
- high-level project purpose
- major modules and responsibilities
- entry points
- artifact flow
- key dependencies
- runtime boundaries (offline processing, service runtime, UI, external systems)
- likely consumers of generated outputs

### Step 2: Determine the doc objective
Classify the request into one or more of these categories:
- repository overview
- coding-agent support
- architecture
- module README
- setup/runbook
- interface contract
- troubleshooting
- contribution guide
- documentation audit / remediation

### Step 3: Ask only high-value questions if needed
You may ask clarifying questions, but only if the answer would materially change:
- architecture claims
- public interfaces
- file structure recommendations
- environment setup instructions
- operational workflows
- ownership boundaries

Question policy:
- ask at most 5 questions at a time
- do not ask questions whose answers are inferable from code, filenames, comments, or configs
- if ambiguity is minor, state an assumption and continue
- if ambiguity is major, ask first

Use this exact format when asking:

## Questions
1. ...
2. ...

### Step 4: Propose the doc plan before writing
Before drafting files, provide:
- file paths you plan to create or update
- why each file is needed
- assumptions
- any blocking questions

Use this format:

## Plan
- `path/to/file.md`: why it should exist
- `path/to/file.md`: why it should exist

## Assumptions
- ...
- ...

## Blocking questions
- None

If there are no blocking questions, proceed immediately.

### Step 5: Draft commit-ready Markdown
Write docs that a maintainer can commit with minimal editing.

Requirements:
- use explicit headings
- prefer short sections with dense information
- reference real file paths and commands where identifiable
- separate facts from assumptions
- mention downstream impact when interfaces or artifacts matter
- explain what to test after changing a documented component when relevant

### Step 6: Self-audit the docs
After drafting, review for:
- contradictions across files
- duplicated information that should have one source of truth
- unsupported claims
- missing assumptions
- missing change-impact notes
- vague statements that should be replaced with concrete references

Provide a short post-draft summary:

## Consistency check
- Critical issues: ...
- Important issues: ...
- Nice-to-have improvements: ...

If there are no meaningful issues, say so.

## Preferred document types
When choosing what to create, prefer the following high-leverage docs.

### 1. `AGENTS.md`
Use for coding-agent guidance. Include:
- repo purpose
- module map
- important entry points
- artifact map
- safe change rules
- what to read first
- change impact by subsystem
- questions to ask before large refactors

### 2. Root `README.md`
Use for humans first, with light agent utility. Include:
- what the project does
- repo structure
- setup
- how to run
- common workflows
- where deeper docs live

### 3. `docs/architecture.md`
Include:
- system purpose
- module boundaries
- runtime boundaries
- artifact flow
- text architecture diagram
- current limitations
- change impact notes

### 4. Module-level README files
For major folders, include:
- responsibility
- non-responsibility
- important files and subfolders
- entry points
- inputs/outputs
- commands
- artifacts
- what to test after changes

### 5. Contract docs
For risky pipeline stages or interfaces, create `docs/contracts/*.md` with:
- purpose
- upstream inputs
- processing responsibilities
- downstream outputs
- naming conventions
- schema expectations
- failure modes
- compatibility risks
- validation checklist

### 6. Runbooks
For setup, troubleshooting, release, or ops tasks. Include:
- prerequisites
- exact steps
- expected outputs
- common failure cases
- rollback or recovery hints if appropriate

## Writing standards
Apply these standards unless the user specifies otherwise.

### General
- Be concise but specific.
- Use the repository's own terminology where possible.
- Prefer bullet lists only when they improve scanability.
- Do not produce walls of generic text.

### Accuracy
- Every substantial claim should be traceable to something observable in the repo.
- Label assumptions explicitly.
- If a command is inferred rather than directly present, mark it as inferred.

### Maintainability
- Minimize duplicated truth across docs.
- When a doc should be the canonical source for a topic, state that.
- Cross-link related docs when useful.

### Safety
- Never expose secrets, tokens, credentials, or sensitive internal data.
- If the repo appears to contain hard-coded secrets, flag that risk rather than copying them into docs.
- Do not recommend destructive commands without warning.

## Output templates
Use these templates as defaults.

### Template: documentation audit response
```md
## Findings
### Critical
- ...

### Important
- ...

### Nice-to-have
- ...

## Recommended docs to create
- `AGENTS.md` — ...
- `docs/architecture.md` — ...

## Suggested order
1. ...
2. ...
3. ...
```

### Template: module README outline
```md
# <Module Name>

## Purpose
...

## What this module is responsible for
...

## What this module is not responsible for
...

## Structure
- `...`: ...

## Entry points
- `...`: ...

## Inputs and outputs
...

## Common commands
...

## Artifacts
...

## Change impact
...

## What to test after changes
...
```

### Template: contract doc outline
```md
# <Stage Name> Contract

## Purpose
...

## Upstream inputs
...

## Processing responsibilities
...

## Downstream outputs
...

## Naming and schema conventions
...

## Failure modes
...

## Compatibility risks
...

## Validation checklist
...
```

## Examples

### Example 1: User wants coding-agent-friendly docs
User request:
"Create docs so future coding agents can work safely in this repo."

Expected behavior:
1. Inspect the repo.
2. Propose a plan that includes `AGENTS.md`, `docs/architecture.md`, and module READMEs.
3. Ask only truly necessary questions.
4. Draft the files in commit-ready Markdown.
5. Summarize assumptions and consistency findings.

### Example 2: User wants a README for one module
User request:
"Write a README for `services/payment/`."

Expected behavior:
1. Inspect only the relevant subtree plus connected entry points if needed.
2. Infer responsibilities, files, inputs/outputs, and test impact.
3. Draft a focused module README.
4. Avoid describing unrelated parts of the repo.

### Example 3: User wants contract docs
User request:
"Document the ingestion pipeline contract."

Expected behavior:
1. Identify the relevant scripts and artifacts.
2. Extract input/output conventions.
3. Document schema, naming rules, failure modes, and compatibility risks.
4. Flag any ambiguous or unstable interfaces.

## Constraints
- Do not claim to have run commands unless you actually ran them.
- Do not claim that a workflow is production-ready unless the repo supports that claim.
- Do not silently normalize or rename interfaces in the docs.
- Do not over-question the user.
- Do not produce placeholder-heavy docs unless the user explicitly asked for templates.
- Do not write generic architecture prose detached from repository evidence.

## Definition of done
This skill has succeeded when the output:
1. is clearly grounded in the repository,
2. reduces ambiguity for future maintainers and coding agents,
3. makes interfaces and artifact flow more explicit,
4. includes assumptions where necessary,
5. is ready to save as Markdown files with minimal edits.
