# Failure Cases and Fixes

Failure-mode history is valuable paper material. It shows which risks were
identified, how they were fixed, and which invariants are now tested.

## F1. Evidence quote re-entered LLM history

- Symptom: collapsible evidence text was appended to the assistant history.
- Risk: a later LLM turn could see prior quotes and paraphrase, combine, or
  misattribute them.
- Fix: split user-visible evidence emission from provider-visible history.
- Test: provider messages are checked for absence of quote body.
- Paper value: motivates provider-isolated evidence rendering.

## F2. User-facing answer looked more authoritative than intended

- Symptom: evidence summary could be read as final design judgment.
- Risk: users may treat a RAG-backed explanation as a compliance decision.
- Fix: add advisory label that the evidence is current-corpus reference and
  not final design judgment.
- Test: summary text checked in chat-tool tests.
- Paper value: supports responsible UX framing.

## F3. Strength issue type was too coarse

- Symptom: axial, flexure, and interaction failures were routed through one
  generic `strength_exceeded` bucket.
- Risk: retrieval query had weak limit-state signal.
- Fix: split axial, flexure, interaction, shear, and drift issue types.
- Test: routing and mapping tests.
- Paper value: supports limit-state-aware retrieval routing.

## F4. AISC temporary proxy could be confused with KDS evidence

- Symptom: early corpus gaps used AISC temporary references.
- Risk: users could mistake proxy evidence for KDS compliance evidence.
- Fix: ingest KDS 14 31 10 sample corpus and remove AISC proxy hits in the
  tested golden set.
- Test: R1 benchmark reports AISC proxy 0/5.
- Paper value: useful as a corpus-quality maturation example.

## F5. Only governing violation was shown

- Symptom: when multiple ratios exceeded 1.0, the answer emphasized only the
  governing ratio.
- Risk: a non-governing but still failed limit state could disappear from the
  explanation and audit trail.
- Fix: list all exceeded ratios in summary and audit trigger.
- Test: multi-violation handler and audit tests.
- Paper value: practical explainability improvement.

## F6. Audit endpoint exposed too much by analysis id

- Symptom: early audit endpoint allowed reading provenance metadata by knowing
  `analysis_id`.
- Risk: cross-session metadata leakage and quote exposure through
  `include_quotes`.
- Fix: require operator token or server-stamped `record.session_id` ownership;
  redact quotes by default.
- Test: token/session access matrix in router tests.
- Paper value: shows transition from research/debug tooling to shared
  deployment readiness.

## F7. JSONL audit trail was unbounded

- Symptom: durable audit log appended forever.
- Risk: disk growth and indefinite exposure window for quote-bearing records.
- Fix: size rotation, backup count, retention-days sweep for numeric backups.
- Test: rotation and retention tests.
- Paper value: operationalization detail, likely appendix material.

## F8. Earlier empty-prefix observation did not reproduce

- Symptom: informal observation suggested qwen2.5:14b often returned empty
  prefixes.
- Measurement: fixed shear-NG probe showed kept 36/36.
- Decision: no prompt relaxation and no guard weakening.
- Paper value: a good example of measuring before changing guardrails.
