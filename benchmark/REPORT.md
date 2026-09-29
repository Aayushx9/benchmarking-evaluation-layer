# Benchmark Report: Evaluating the Evaluation Layer

Generated: 2026-09-29T04:50:18.429289+00:00  
Evaluator mode for this run: **deterministic** (LLM judge not available; score sources = 40 deterministic, 0 deterministic+llm)  
Detection threshold: a dimension is *flagged* when the evaluator scores it < 0.9.

> **Honesty note:** these metrics measure the evaluator only on this small,
> constructed, controlled benchmark. They do **not** prove the evaluator is
> generally reliable on unseen agent outputs.

## 1. Research question

> How might we tell whether the systems that judge AI agents are themselves any good?

## 2. Benchmark design

- Each case starts from a known-correct existing agent trace (or a minimal variation of it).
- Exactly one error type is injected per case, except case H which combines three.
- Every mutation is documented in the `mutation` field of `benchmark/cases.json` (no random mutations).
- Ground-truth labels cover all five evaluator dimensions, each `correct` or `incorrect`.
- `expected_pass` is true only when every dimension in the case is labeled `correct`.
- The runner calls the **existing** `src.evaluator.evaluate_trace`; no second evaluator exists.

## 3. Benchmark cases

| Case | Injected error | Question | Expected pass |
|---|---|---|---|
| case_A_correct_answer | none | Which model has the highest average accuracy? | True |
| case_B_incorrect_number | incorrect_numerical_value | Which model has the highest average accuracy? | False |
| case_C_wrong_method | incorrect_analytical_method | Which task appears to be the most difficult? | False |
| case_D_incomplete_answer | incomplete_answer | Which model has the highest average accuracy? | False |
| case_E_hallucinated_column | hallucinated_column | Which task appears to be the most difficult? | False |
| case_F_unsupported_claim | unsupported_claim | Which task appears to be the most difficult? | False |
| case_G_verbose_correct | unnecessary_wording | Which model has the highest average accuracy? | True |
| case_H_multiple_errors | multiple_errors | Which task appears to be the most difficult? | False |

## 4. Ground-truth labels

| Case | numerical correctness | data grounding | method correctness | completeness | hallucination free | pass |
|---|---|---|---|---|---|---|
| case_A_correct_answer | correct | correct | correct | correct | correct | True |
| case_B_incorrect_number | incorrect | correct | correct | correct | incorrect | False |
| case_C_wrong_method | correct | correct | incorrect | correct | correct | False |
| case_D_incomplete_answer | correct | correct | correct | incorrect | correct | False |
| case_E_hallucinated_column | correct | incorrect | correct | correct | incorrect | False |
| case_F_unsupported_claim | correct | correct | correct | correct | incorrect | False |
| case_G_verbose_correct | correct | correct | correct | correct | correct | True |
| case_H_multiple_errors | incorrect | incorrect | incorrect | correct | incorrect | False |

## 5. Evaluator results

| Case | numerical correctness | data grounding | method correctness | completeness | hallucination free | overall | pass | verdict ok |
|---|---|---|---|---|---|---|---|---|
| case_A_correct_answer | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | True | yes |
| case_B_incorrect_number | 0.0 | 1.0 | 1.0 | 1.0 | 0.5 | 0.7 | False | yes |
| case_C_wrong_method | 1.0 | 1.0 | 0.4 | 1.0 | 1.0 | 0.88 | True | NO |
| case_D_incomplete_answer | 1.0 | 1.0 | 1.0 | 0.8 | 1.0 | 0.96 | True | NO |
| case_E_hallucinated_column | 1.0 | 0.0 | 1.0 | 1.0 | 0.5 | 0.7 | True | NO |
| case_F_unsupported_claim | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | True | NO |
| case_G_verbose_correct | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | True | yes |
| case_H_multiple_errors | 0.5 | 0.5 | 0.4 | 1.0 | 0.0 | 0.48 | False | yes |

## 6. Per-dimension performance

Aggregated over 40 dimension comparisons (8 cases x 5 dimensions). Positive class = ground truth `incorrect` (an error was injected in that dimension).

| Dimension | TP | FP | TN | FN | accuracy | precision | recall |
|---|---|---|---|---|---|---|---|
| numerical_correctness | 2 | 0 | 6 | 0 | 1.0 | 1.0 | 1.0 |
| data_grounding | 2 | 0 | 6 | 0 | 1.0 | 1.0 | 1.0 |
| method_correctness | 2 | 0 | 6 | 0 | 1.0 | 1.0 | 1.0 |
| completeness | 1 | 0 | 7 | 0 | 1.0 | 1.0 | 1.0 |
| hallucination_free | 3 | 0 | 4 | 1 | 0.875 | 1.0 | 0.75 |
| **ALL** | 10 | 0 | 29 | 1 | **0.975** | - | - |

Score landed inside the pre-registered expected range in 40/40 dimension comparisons (1.0).

Case-level pass/fail agreement: 4/8 (0.5).

## 7. False positives

- None. The evaluator did not flag any dimension that ground truth labeled `correct`.

## 8. False negatives (missed errors)

- **case_F_unsupported_claim** / hallucination_free: scored 1.0 (>= 0.9) even though the error was injected.

Case-level misses (evaluator PASSED a case that should have failed): case_C_wrong_method, case_D_incomplete_answer, case_E_hallucinated_column, case_F_unsupported_claim

## 9. Cases the evaluator handled correctly

- **case_A_correct_answer**: none — all dimension detections and the pass/fail verdict matched ground truth.
- **case_B_incorrect_number**: incorrect_numerical_value — all dimension detections and the pass/fail verdict matched ground truth.
- **case_G_verbose_correct**: unnecessary_wording — all dimension detections and the pass/fail verdict matched ground truth.
- **case_H_multiple_errors**: multiple_errors — all dimension detections and the pass/fail verdict matched ground truth.

## 10. Cases where the evaluator failed

- **case_C_wrong_method**: pass/fail verdict disagreed with ground truth.
- **case_D_incomplete_answer**: pass/fail verdict disagreed with ground truth.
- **case_E_hallucinated_column**: pass/fail verdict disagreed with ground truth.
- **case_F_unsupported_claim**: dimension detection wrong on: hallucination_free; pass/fail verdict disagreed with ground truth.

## 11. Limitations

- Small, constructed benchmark built from one 18-row fixture CSV and a handful of existing traces.
- Cases were created by the same team that wrote the evaluator, so shared assumptions can hide evaluator blind spots.
- Ground truth is rule-based (injected error => dimension incorrect), not independent human annotation.
- Only one non-numeric unsupported-claim case (F) exists; hallucination coverage beyond column/number errors is thin.
- The run above used mode `deterministic`; LLM-judged dimensions may behave differently when a key is present.
- Correct cases are all variations of two base traces, so the true-positive side is narrow.

## 12. What the results mean

- On this benchmark the evaluator flagged 10 of 11 injected dimension errors (recall 90.9%) with 0 false positives.
- Case-level pass/fail verdicts agreed with ground truth on 4/8 cases.
- This is evidence that the evaluator handled *these constructed cases* correctly (or identifies exactly where it did not).
- It is **not** evidence that the evaluator is generally reliable; the benchmark is too small and too closely tied to the evaluator's own design.
- Case F is the expected structural gap: deterministic checks cannot see non-numeric unsupported claims; that is what the optional LLM judge is for.
- Averaging five dimensions means a single-dimension error can still leave overall >= 0.7, so `pass` alone under-reports problems — the per-dimension scores are the sharper signal.

## 13. What should be tested next

- More unsupported-claim cases (causal, comparative, and speculative statements) with and without the LLM judge.
- Adversarial traces: subtly wrong numbers within tolerance, near-miss columns (`latency` vs `latency_ms`), off-by-one rankings.
- Correct-but-unusual outputs (different phrasings, reordered answers) to hunt for false positives.
- Independent human labels on natural agent traces (not just injected mutations).
- Correlation between evaluator scores and human ratings across a larger trace set.
- Deterministic-vs-LLM agreement study: same cases under both modes, per-dimension delta.

