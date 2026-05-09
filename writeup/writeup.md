# ARK — Interpretability — MAIA AISF Hack S26

## Research Question
Do LLM models exhibit Illusion of Explanatory Depth (IOED) on mechanistic and complex queries, and can fine-tuning models to align with confidence metrics result in models that perform less IOED?

## Background
The Illusion of Explanatory Depth is a well-documented human cognitive bias, first systematically characterized by Rozenblit and Keil (2002): people consistently overestimate how deeply they understand causal mechanisms — until they're asked to actually explain them. Someone who feels confident they understand how a zipper works will often stumble when pressed to describe the wedge geometry of the slider. The gap between felt understanding and articulable understanding is the core of IOED.

We're wondering if we can translate this to model the behavior of LLMs. This matters for AI safety for a fairly direct reason: if LLMs exhibit an analogous pattern — expressing high confidence in mechanistic explanations that are actually shallow or wrong — then model-expressed confidence scores become misleading signals. Downstream systems, humans using models as reasoning assistants, and alignment techniques that rely on model self-evaluation all break down if a model's stated confidence doesn't track its actual accuracy.

It's also interesting from a purely intellectual perspective — if the same phenomenon that has been observed in humans can also be observed in LLMs, how much more of human psychological research can we translate to LLMs?

## Method

**Datasets.** We built two mechanistic question-answering datasets. The first (`mechanism_qa.jsonl`, 80 items) covers everyday mechanisms across three topic buckets — devices (zippers, toilets, pianos), biological/physical processes (blood clotting, rainbows, lightning), and algorithms (binary search, TCP handshake, JPEG compression). Each item includes a reference answer and rubric notes flagging what a shallow explanation misses. The second dataset (`mechanism_qa_harder.jsonl`, 40 items) targets specialty technical domains — scientific instrumentation, electrochemistry, manufacturing processes, molecular biology, geophysics, and engineering — chosen specifically because Qwen 2.5's instruction-tuning data is unlikely to cover them deeply. Critically, difficulty was assigned a priori by domain selection, not by querying the model, so we couldn't accidentally select questions the model already got right.

A separate math dataset provides calibration anchors: simple arithmetic and factual questions where correct confidence should be high, and hard proof-style questions (Fermat's Last Theorem, Gödel's theorem) where a well-calibrated model should express uncertainty.

**IOED Evaluation Protocol.** We ran Qwen2.5-1.5B-Instruct through a four-turn protocol designed to mirror the original IOED psychology experiments:

- **T1 (pre-explanation confidence):** The model is asked for its confidence (0–100) that it can explain the mechanism correctly, before generating any explanation.
- **T2 (explanation + confidence):** The model produces a step-by-step mechanistic explanation and re-rates its confidence.
- **T3 (diagnostic probe):** The model is shown a targeted follow-up question highlighting a specific deep feature its explanation likely missed (e.g., for a ballpoint pen: the thixotropic ink and pressure-equalization mechanism). It is asked to update its explanation and re-rate confidence.
- **T4 (post-reference confidence):** The model is shown the reference answer and asked to give a final confidence score.

The T1→T4 trajectory is the core measurement. A model with IOED should show confidence staying high at T1/T2/T3 even when its explanation is objectively shallow, and then dropping at T4 when shown what it missed.

**Calibration Fine-tuning.** We constructed a supervised fine-tuning dataset (`qwen25_ioed_math_mech_calibration.jsonl`) pairing instructions with model outputs that express calibrated uncertainty. For complex mechanism questions, the target response explicitly acknowledges the IOED pattern ("I can offer a surface-level explanation, but my actual mechanistic understanding is limited..."). For simple factual and arithmetic queries, the target response expresses high confidence. The goal was to train the model to differentiate between questions it can answer reliably and those where it only has surface coverage.

## Results

We evaluated Qwen2.5-1.5B-Instruct before and after calibration fine-tuning on two domains. D1–D4 refer to the four rounds of the IOED protocol: D1 = pre-explanation confidence, D2 = confidence after generating an explanation, D3 = confidence after the diagnostic probe, D4 = confidence after seeing the reference answer. The metric is the model's self-reported confidence score (0–100) at each round.

![Baseline vs Fine-tuned Across D-levels](../results/baseline_vs_finetuned.png)

**Math confidence by round:**

| Round | Baseline | Fine-tuned |
|-------|----------|------------|
| D1 (pre-explanation) | 70.0 | 81.1 |
| D2 (post-explanation) | 37.4 | 7.0 |
| D3 (post-diagnostic) | 32.5 | 0.0 |
| D4 (post-reference) | 31.0 | 2.0 |

**Mechanistic confidence by round:**

| Round | Baseline | Fine-tuned |
|-------|----------|------------|
| D1 (pre-explanation) | 44.0 | 19.1 |
| D2 (post-explanation) | 56.3 | 10.0 |
| D3 (post-diagnostic) | 66.5 | 14.0 |
| D4 (post-reference) | 55.2 | 10.2 |

The mechanistic baseline is the clearest finding. Confidence rises from D1 to D3 (44 → 56.3 → 66.5) — the model becomes *more* confident after generating an explanation, and *more* confident still after being shown a diagnostic question highlighting what it missed. It only partially recovers at D4 when directly shown the reference answer (55.2). This is the IOED signature: the act of producing an explanation, even a shallow one, inflates rather than deflates confidence.

Math tells a different story. The baseline model's confidence drops sharply after explanation (D1→D2: 70 → 37.4) and continues declining through D3 and D4. The model correctly recognizes when its math explanation is inadequate — no IOED on quantitative reasoning.

The fine-tuned model suppresses mechanistic confidence across all rounds (10–19), eliminating the D1→D3 inflation. On math, fine-tuning overcorrects: starting confidence is slightly higher (81.1 vs 70) but collapses to near-zero at D2–D4 (7, 0, 2).

## Interpretation

The core finding is clean: the baseline model shows IOED specifically on mechanistic questions, not on math. This domain-specificity is itself meaningful — the model has enough self-awareness to revise its math confidence downward after explaining, but not enough to do the same for mechanism explanations. Being asked to explain *how a zipper works* does not trigger the same epistemic self-check as being asked to explain *why 3x = 15*.

Fine-tuning successfully eliminates the mechanistic IOED pattern. The model no longer gains confidence through the explanation rounds.

The math results are worth being honest about. The near-zero D2–D4 confidence (0–7) is not ideal calibration — it looks like the fine-tuning trained a coarse heuristic: suppress confidence after any explanation attempt. The model is not cleanly distinguishing "I cannot explain mechanisms deeply" from "I cannot do math." Separating those two would require checking whether post-explanation confidence tracks actual accuracy, which we did not measure in this run.

Still, the main claim holds: LLMs exhibit a measurable IOED analog on mechanistic reasoning, and it can be reduced through targeted calibration fine-tuning. The open question is whether the reduction reflects genuine domain-aware uncertainty or learned surface-level hedging.

## If we had two more weeks

First, we'd run the full evaluation pipeline across both datasets (80 + 40 items) and compute confidence trajectory statistics by topic bucket. That would tell us whether IOED is stronger in some domains (algorithms vs. biology vs. geophysics) or uniformly distributed.

Second, we'd properly evaluate the fine-tuned model on held-out eval items and compare confidence calibration curves (reliability diagrams) before and after fine-tuning. The key question is whether training on the calibration dataset reduces the T1–T3 overconfidence gap on *unseen* mechanism questions, not just the ones in the training set.

Third, we'd scale model size. Qwen2.5-1.5B is small enough that its explanations are often wrong for the wrong reasons. Testing 7B or 14B would let us separate "model is overconfident because it's small" from "IOED is a structural property of how these models are trained." If larger models show less IOED, that's actually reassuring — it suggests scale helps. If IOED persists at scale, it becomes more interesting as a safety-relevant phenomenon.
