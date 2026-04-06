# Prompt 04 — Write a Contract Doc for a Pipeline Stage

Use `repo-doc-writer` to create a contract doc for this pipeline stage:
- stage name: `[STAGE_NAME]`
- target file: `[TARGET_DOC_PATH]`

The contract doc must include:
- stage purpose
- upstream inputs
- CLI arguments or runtime inputs
- processing responsibilities
- downstream outputs
- filename and folder conventions
- schema expectations if applicable
- failure modes visible from code
- backward compatibility risks
- validation checklist after modifying this stage

Rules:
- do not generate a generic template
- infer as much as possible from implementation
- ask only high-value questions if critical details are truly ambiguous
- otherwise proceed with clearly labeled assumptions

Before drafting, show:
- assumptions
- any blocking questions

Then write the contract doc.
