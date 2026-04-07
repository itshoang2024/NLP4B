# Documentation Style Reference

Use this as the default style guide for docs written by `repo-doc-writer`.

## Primary objective
Write documentation that helps developers and coding agents understand, modify, and run the repository safely.

## Style priorities
1. Concrete over generic
2. Operational over descriptive
3. Repository-grounded over speculative
4. Contracts over storytelling
5. Maintainable over exhaustive

## Preferred writing style
- Use short sections with informative headings.
- Prefer direct language.
- Mention exact file paths when relevant.
- State assumptions explicitly.
- Avoid filler such as “this module is designed to...” unless followed by concrete details.
- Avoid repeating the same repository facts across multiple docs unless one is a deliberate canonical source.

## What good docs look like
Good docs answer questions like:
- Which file do I run?
- What does this module produce?
- What depends on that output?
- What breaks if I rename this file or change this field?
- What should I test after editing this component?

## What bad docs look like
Avoid docs that:
- merely restate folder names
- describe architecture without artifact flow
- include commands that were not verified or clearly labeled as inferred
- mix facts and guesses without labeling assumptions
- hide risky interface changes behind vague wording

## Voice
- calm
- precise
- non-marketing
- commit-ready
