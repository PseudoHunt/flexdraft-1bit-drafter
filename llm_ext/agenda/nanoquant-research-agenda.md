# Research directions for improving NanoQuant

**Recommendation.** Build the paper around a demonstrated failure mechanism in binary-factor tuning, then introduce one method that fixes it under a controlled storage and compute budget. The strongest first investigation is **whether immediate freezing of each projection and uncontrolled function changes from sign updates cause fragile compensation**. Delayed finalization and influence-aware sign updates provide concrete interventions. Rank allocation and more calibration data are strong baselines and useful components, but neither is sufficiently novel by itself.

This agenda contains 64 proposals covering optimization, objectives, calibration, allocation, representation, initialization, distillation, and deployment. It is a systematic research map, not a claim that every possible idea is listed or that any candidate is unpublished. “Candidate core” means that a sufficiently distinct formulation and convincing evidence could support a paper; it does not certify novelty. Literature reviewed includes work available through September 9, 2026. Recent preprints establish relevant overlap, not independently verified performance.

**Experimental scope and provenance**

The experimental record is pinned to branch llm-ext-cov, commit 1d36d8d79b99986029aa98e98ea15dd9b1bae367, with NanoQuant changes against a9e0a43. The local checkout had the same HEAD; rank-allocation JSON files still contained only the four pre-KD results. Running or modified logs are not treated as completed evidence. No model training was performed for this report. The reported experiments use Qwen3-0.6B-Base, WikiText2, sequence length 2048, and roughly 0.973 bits per original weight over the factorized matrices. FP-reference PPL is 12.669. Embeddings, norms, and the output head are outside that factorized-matrix budget.[^1][^2][^3][^6]

The experiment series is valuable because it exposes a mismatch between the local metrics being optimized and held-out model quality. It does not yet identify one exclusive cause of that mismatch or establish a general result across LLM families.

**1. What the existing experiments establish**

| Experiment | Observed result | Defensible interpretation | Missing evidence |
|---|---|---|---|
| E1: ADMM only, all blocks | Mean covariance-weighted output error 0.1571 → 0.0957; weight error 0.3124 → 0.3253. PPL approximately 7.7e8 versus 9707. | Covariance improves this local proxy; initialization alone is inadequate in both arms. | Whether improved initialization can reduce tuning compute or help another regime. |
| E2: block 13 only, tuned | PPL 12.8693 diagonal versus 12.8411 covariance. | A small single-run benefit under a mostly-FP model. | Replication and survival when other blocks are quantized. |
| E3: full model and KD, two runs per arm | Diagonal 28.105/33.695; covariance 30.817/35.902. | No demonstrated covariance advantage; both paired outcomes favor diagonal. | Enough runs for an effect estimate or equivalence claim. |
| E4/E7: per-block curves and reruns | Large divergence around block 3; later offsets persist. | Early tuning trajectories are an important diagnostic target. | Shared-prefix interventions separating inherited variation from block-local variation. |
| E5: correlation shrinkage | Better-conditioned correlation matrices and smoother local errors do not give monotone PPL. | A simple conditioning explanation is insufficient in this screen. | Other conditioning/optimization effects and replicated selection comparisons. |
| E6: projection subsets | Block-2 benefit is associated with the MLP; MLP-only is poor by block 9. | Projection interactions and downstream survival matter. | Within-MLP localization and repeated group interventions. |
| E8: validation-selected beta | Ten-block PPL 17.688; five candidates per block. | Validation selection is operationally feasible. | Fixed-beta best-of-five control, search-budget fairness, full-model/KD outcome. |
| E9: activation analysis | Four q-projection input channels account for 37% of block-3 energy and 45% by block 5. | Energy concentration motivates better diagnostics. | Their share of error, gradients, and causal task impact. |
| E10: 512 samples, matched steps | Ten-block diagonal PPL 16.32/16.46; covariance 15.87/16.87. | More-data configuration helps and reduces observed spread, especially diagonal. | Separate effects of recollected statistics and tuning-data diversity; independent calibration seeds. |
| E11: more accumulation | Block-0 PPL 33.9/36.9 in the screen. | This accumulation/schedule configuration is worse. | Matched optimizer updates and retuned learning rates. |
| E12: inverse second-moment channel weighting | Block-0 PPL around 20.1/21.1 versus 17.7/17.0; stopped after block 1. | This replacement loss harms early-block tuning. | Additive losses, coefficient controls, or a targeted block-3 test. |
| E13: rank allocation, gamma 0.15 | Pre-KD 28.234/28.541 versus uniform 30.126/32.707, at 0.9722 versus 0.9729 bpw. | Promising equal-budget pre-KD improvement. | Completed post-KD runs, more seeds, direct comparison with DBF allocation. |
| E14/E15 | Stronger allocation and default-data-setting allocations pending at the cited commit. | No completed result to interpret. | Final JSON/checkpoints and matched evaluation. |

These values come from the branch ledger and result tables; the allocation result is also explicitly marked as salvaged after a KD-memory failure in the raw JSON.[^1][^2][^44] The arithmetic mean pre-KD improvement for allocation is 3.029 PPL (31.4165 → 28.3875), calculated from the four reported values. This is descriptive, not a confidence interval or a predicted final gain.

**2. Corrections that matter for a paper**

**A low relative error is not evidence of a flat training objective.** The logged block error is divided by teacher-output energy. The actual weighted MSE training function sums squared differences times importance; it does not divide by teacher energy. A large denominator can make the reported ratio small without reducing the training gradient. Measure absolute error, error on the high-energy channels, error on the remaining channels, and their respective gradient contributions before claiming objective saturation.[^4][^7]

**The current observations do not prove ADMM is bit-reproducible.** Matching aggregate errors to four decimals is weaker than equality of factor tensors or packed hashes. In the tuned pipeline, ADMM receives weights already modified by tune_nonfact. Hash the inputs, random state, output signs, scales, and latent factors separately. Also, per-block curves already show some earlier variation, so “all variance is injected at one block” is too strong.[^2][^7][^8]

**More data and new statistics are currently bundled.** The 512-sample screen recollects importance and covariance and changes the sequences seen during tuning. Use a 2×2 experiment: statistics from 128 or 512 samples crossed with tuning data from 128 or 512 samples. Match update budgets. A separate fixed-epoch comparison answers a different question about the quality attainable with more compute.[^2][^3]

**Count updates per phase.** In the recorded default, nonfactorized tuning accumulates four examples and factorized tuning accumulates one. At 128×8, that is 256 and 1024 updates respectively per projection, assuming the ordinary complete batches. At 512×2 those counts remain the same. The accumulation screen's settings of 16 and 8 reduce these counts by 4× and 8×, respectively; it does not isolate the effect of gradient noise.[^2][^6][^7]

**Rank-allocation scores are not causal sensitivity measurements.** The script uses log accumulated reconstruction error against the FP chain. This includes inherited drift and changes in teacher-output magnitude. A block with high accumulated error need not be the best location for one more rank unit. Direct rank perturbations, under a fixed prefix and tuning protocol, estimate the relevant marginal benefit more closely.[^4][^5]

**The covariance objective is a regularized proxy.** The implementation uses clipped input second moments, shrinkage, and output importance; the diagnostic uses its own covariance-weighted error. E[xxᵀ] is an uncentered second moment, not centered covariance. It equals an expected linear-output quadratic under the corresponding input distribution and weighting, not automatically held-out task error. Matching diagonals verifies one consistency property, not the whole causal comparison.[^3][^4][^10]

**There are existing capabilities to account for.** The linear module already supports scale_mid. It finalizes the current projection after tuning, deletes its latent factors, and freezes parameters. Global KD selects scales. The module stores training parameters in BF16, while the optimizer supports automatic Kahan accumulation for low-precision parameters. Thus “add middle scales,” “preserve FP32 latent updates,” and “tune all signs globally” require precise baseline audits, not assumptions about missing features.[^7][^8][^9][^43]

**Two runs do not establish equivalence, variance elimination, or a universal failure.** Report each run and label mean ± half-range as such; it is not a standard error. Use separate factorization/tuning and calibration-data seeds. Test-set curves have already informed hypothesis generation, so lock a final evaluation protocol and use additional held-out domains for confirmation.

**3. Prior work that constrains novelty**

| Proposed generic contribution | Closest established direction | What must be different for a stronger paper |
|---|---|---|
| Nonuniform binary-factor rank | DBF already includes nonuniform layer-wise compression.[^12] | Demonstrated interaction effects, reliable marginal estimates, or joint optimization with actual compute/latency. |
| Replace STE / stabilize binary training | PV-Tuning, oscillation work, QuEST, and progressive rounding.[^13][^14][^15][^16] | A method exploiting the two-factor binary geometry with measurable advantages over these controls. |
| Weight reconstruction by end loss | GuidedQuant, YAQA, KronQ.[^17][^18][^19] | Cheaper or more reliable curvature for this binary parameterization; simply adding Fisher weights is insufficient. |
| Look ahead / tune multiple blocks | Look-ahead PTQ and CBQ.[^20][^21] | A discovered binary-specific freezing failure and an adaptive intervention that beats equal-compute multi-block baselines. |
| Correct upstream activation mismatch | QEP and CoreQ; recent low-rank calibration correction.[^22][^23][^31] | Robustness across a distribution of plausible quantized prefixes rather than one point estimate, with direct controls. |
| Regularize calibration drift | SARQC.[^24] | A representation- or trajectory-specific constraint with a benefit beyond generic weight anchoring. |
| Better calibration samples | FAQ, outlier-coverage selection, and multiscale calibration.[^26][^27][^28] | Selection based on uncertainty in binary decisions that improves quality per total collection/scoring/training cost. |
| Low-rank or sparse high-precision correction | CALDERA, Preserve-Then-Quantize, SpQR, SqueezeLLM.[^33][^34][^35][^36] | A functional subspace with strict storage accounting and an efficient binary-compatible kernel. |
| Rotate before quantization | QuaRot and SpinQuant.[^37][^38] | An objective specifically suited to binary factor products and legal network transformations. |
| Multiple deployable precisions | MatQuant and MatGPTQ.[^39][^40] | Nested factor ranks with useful compute reuse and a measured deployment advantage. |
| Drafting-oriented compression | Speculative sampling and direct draft alignment.[^41][^42] | Joint binary rank/latency/acceptance design, not the acceptance/TV identity itself. |

Recent papers also examine calibration/rank drift and non-additive allocation effects. These make “refresh the ranks” or “include interactions” broad motivations rather than automatic novelty claims.[^31][^32] The next literature pass should focus on the exact algorithm chosen, its closest implementation baselines, and differences in assumptions.

**4. Complete idea catalog**

Cost labels describe the relative cost of a small screen, not a measured GPU-hour forecast. “First” identifies high-value experiments or necessary controls; it is not a claim of high novelty. All mechanisms and expected benefits below are proposals.

**A. Binary decisions and optimization**

Relevant prior work and implementation: [^12][^13][^14][^15][^16]

**1. Function-space trust regions for sign updates**

Budget sign changes by their predicted effect on outputs, using the other factor and current activations, rather than by latent-coordinate distance. Accept structured flip groups only when measured loss agrees with the local prediction.

First experiment: Instrument block 3 and a late block; compare STE, clipping, PV-style updates, and function-space flip budgets at equal steps and time. Assessment: **Candidate core**. Screen cost: Medium. Priority: First.

**2. Delay finalization across projections**

Keep previously factorized projections trainable until the attention or MLP group is complete, so later quantization can revise earlier signs. Optionally keep one adjacent block active.

First experiment: Compare immediate finalization, scales-only reopening, sign reopening, and delayed finalization from identical initial factors. Assessment: **Candidate core**. Screen cost: Medium. Priority: First.

**3. Influence-weighted sign margins**

Penalize unstable near-zero latent entries in proportion to the damage caused by flipping their represented binary weights. Ramp the penalty late to avoid freezing bad initialization.

First experiment: Measure near-threshold mass, flip-back rates, held-out NLL, and loss caused by controlled one-bit perturbations. Assessment: **Candidate component**. Screen cost: Low–medium. Priority: First.

**4. Alternating factor and scale updates**

Alternate updating U, V, and scales rather than moving both coupled sign factors simultaneously. Use an output-based step budget to limit destructive co-adaptation.

First experiment: Compare simultaneous versus alternating updates with identical forward/backward budgets; test whether covariance initialization benefits more. Assessment: **Established adaptation**. Screen cost: Low. Priority: Second.

**5. Curvature-scored binary coordinate search**

Score a restricted candidate set of sign flips using loss gradients plus low-rank curvature, then verify candidate groups with actual forwards.

First experiment: Run after ordinary tuning on one block; report gain per evaluated flip group and compare a PV-Tuning adaptation. Assessment: **Candidate component**. Screen cost: Medium–high. Priority: Second.

**6. Stochastic binary factors with annealing**

Train Bernoulli sign probabilities, anneal uncertainty, and finalize a deterministic sample or mode. Test whether uncertainty measures identify poorly calibrated decisions.

First experiment: Compare deterministic STE and annealing at the same compute; separately evaluate selection cost and final packed-model accuracy. Assessment: **Established adaptation**. Screen cost: Medium. Priority: Later.

**7. Symmetry-aware factor alignment and consensus**

Align factor permutations and paired sign symmetries before comparing seeds or combining checkpoints; select stable binary components using held-out function error.

First experiment: First compare represented-weight distance with raw latent distance. Try aligned consensus only if independent runs contain complementary errors. Assessment: **Candidate component**. Screen cost: Medium. Priority: Later.

**8. Training precision and optimizer-state ablations**

Separate latent precision, optimizer implementation, and backward-kernel nondeterminism. BF16 parameters alone do not prove update loss because the optimizer supports Kahan accumulation.

First experiment: Hold a pre-block checkpoint fixed; compare BF16/Kahan and FP32 master-state configurations with sign hashes and per-stage gradients. Assessment: **Diagnostic / baseline**. Screen cost: Low–medium. Priority: First.


**B. Objectives and downstream behavior**

Relevant prior work and implementation: [^17][^18][^19][^20][^21][^22][^23][^24]

**9. Auxiliary loss through the next RMSNorm**

Retain the current weighted MSE and add teacher–student error after the next block's actual RMSNorm. Test whether normalized feature error discriminates solutions better.

First experiment: Use shared pre-block-3 inputs; compare raw-only, normalized-only, and additive losses, including gradient-scale controls. Assessment: **Established adaptation**. Screen cost: Low. Priority: First.

**10. Adaptive look-ahead depth**

Use one or two frozen downstream blocks only where a local error proxy fails to predict held-out NLL. Allocate look-ahead computation based on measured disagreement.

First experiment: Compare fixed-depth look-ahead with the same total compute concentrated on diagnosed blocks; verify the gain survives later quantization. Assessment: **Candidate component**. Screen cost: Medium–high. Priority: Second.

**11. End-loss curvature for binary-factor tuning**

Approximate the downstream loss metric on block outputs using gradients or low-rank Fisher sketches, then tune binary factors in that metric.

First experiment: Compare plain MSE, diagonal output weighting, GuidedQuant/YAQA-style weighting, and the proposed sketch at matched calibration cost. Assessment: **Established adaptation**. Screen cost: Medium–high. Priority: Second.

**12. Joint attention-output reconstruction**

Match the result of attention after Q/K, RoPE, softmax, and V interaction instead of independently emphasizing projection errors. Use sampled query positions to control memory.

First experiment: Localize Q/K versus V/O effects by head and token position; require gains in language loss beyond attention-map similarity. Assessment: **Established adaptation**. Screen cost: Medium. Priority: Second.

**13. Joint gated-MLP product reconstruction**

Tune gate and up factors together against the nonlinear product and down-projected update, accounting for their interaction.

First experiment: Start at block 2, where the covariance gain is MLP-associated; compare separate losses, product loss, and joint sign tuning. Assessment: **Candidate component**. Screen cost: Medium. Priority: First.

**14. Residual-update loss plus stream anchoring**

Separate reconstruction of attention/MLP updates from reconstruction of the full residual stream. Keep a stream anchor so matching an update does not ignore inherited prefix error.

First experiment: Compare teacher–student update error, accumulated stream error, and held-out NLL across early and late blocks. Assessment: **Established adaptation**. Screen cost: Low. Priority: Second.

**15. Joint magnitude and direction preservation**

Decompose token-level feature mismatch into norm and angular components, retaining both rather than replacing the loss with inverse channel variance.

First experiment: Compare raw MSE with an explicitly balanced norm/direction objective only at a fixed failing block; retune coefficient scales. Assessment: **Established adaptation**. Screen cost: Low. Priority: Second.

**16. Task-aware treatment of rare sensitive tokens**

Weight reconstruction using disagreement or curvature while capping individual weights, so rare important tokens matter without letting outliers consume the budget.

First experiment: Stratify held-out metrics by token surprisal, position, and activation regime; compare uniform and importance-corrected sampling. Assessment: **Established adaptation**. Screen cost: Medium. Priority: Later.


**C. Calibration and generalization**

Relevant prior work and implementation: [^23][^24][^26][^27][^28][^31]

**17. Fresh sequences at a fixed tuning-step budget**

Replace repeated passes over a small fixed set with newly sampled sequences, refreshing student inputs as needed. Separate data diversity from additional optimization.

First experiment: Cross 128/512/1024 samples with fixed optimizer steps and with fixed epochs; keep the statistics cache fixed in one ablation. Assessment: **Baseline / enabling**. Screen cost: Low–medium. Priority: First.

**18. Robust tuning across quantized-prefix trajectories**

Train each vulnerable block against several realistic prefix-error trajectories, keeping the same FP teacher target for each sequence. Use actual alternate prefixes or validated structured perturbations.

First experiment: Compare ordinary more-data training, isotropic noise, covariance-shaped noise, and real prefix replay at equal token/compute budgets. Assessment: **Candidate core**. Screen cost: Medium–high. Priority: First.

**19. Binary-decision disagreement for data selection**

Select calibration examples where plausible binary-factor solutions disagree in downstream behavior, combined with a diversity constraint and random anchors.

First experiment: Compare random, activation-coverage, entropy, and sign-disagreement selection; count every scoring forward in the budget. Assessment: **Candidate component**. Screen cost: Medium–high. Priority: Second.

**20. Coverage of massive-activation regimes**

Ensure calibration covers the positions and token contexts that induce large activations, rather than merely selecting high-energy sequences.

First experiment: Compare random 128/512 samples with regime-stratified sets; use per-projection inputs rather than assuming residual channel IDs transfer. Assessment: **Established adaptation**. Screen cost: Medium. Priority: Second.

**21. Same-family generated calibration data**

Mix original text with text generated by an appropriate teacher, keeping real-data anchors and reporting generation cost.

First experiment: Compare equal-sized real, generated, and mixed sets across in-domain and out-of-domain evaluation. Assessment: **Established adaptation**. Screen cost: Medium–high. Priority: Later.

**22. Refresh input statistics on the compressed prefix**

Recollect local moments as earlier blocks and surviving full-precision weights change, instead of using only the initial FP cache.

First experiment: Cross fixed/recomputed statistics with fixed/recomputed tuning data; test whether covariance's gain becomes more transferable. Assessment: **Established adaptation**. Screen cost: Medium. Priority: First.

**23. Held-out calibration and early stopping**

Reserve validation documents or use cross-fitting to stop block tuning and select checkpoints. Separate this split from the final reporting set.

First experiment: Compare last iterate and selected iterate with an explicit equal-compute control; measure correlation between proxy and NLL. Assessment: **Baseline / enabling**. Screen cost: Low–medium. Priority: First.

**24. Domain- and length-robust calibration**

Use multiple domains and sequence lengths, optionally with a worst-group objective, to reduce specialization to WikiText2 at length 2048.

First experiment: Report per-domain and per-length NLL; compare against simply increasing the random calibration set. Assessment: **Established adaptation**. Screen cost: Medium. Priority: Second.


**D. Rank, precision, and compute allocation**

Relevant prior work and implementation: [^5][^12][^29][^30][^31][^32]

**25. Marginal validation benefit per storage bit**

Probe legal rank changes and allocate by measured reduction in held-out NLL per added packed bit, not accumulated hidden-state error.

First experiment: Compare uniform, current log-error heuristic, DBF allocation, and finite-difference allocation with search cost included. Assessment: **Established adaptation**. Screen cost: Medium–high. Priority: First.

**26. Interaction-aware rank exchanges**

Estimate whether changing two neighboring or coupled projections together helps beyond their isolated effects. Use a sparse interaction graph and budget-preserving exchanges.

First experiment: Measure rank-response interactions at matched prefixes and compare with an additive allocator using the same number of probes. Assessment: **Candidate core**. Screen cost: High. Priority: First.

**27. Joint allocation of rank and tuning steps**

Choose whether each unit of training compute should buy more optimization at the current rank or support a different rank allocation.

First experiment: Build small rank-by-step response surfaces; compare against uniform steps and the strongest rank-only baseline. Assessment: **Candidate core**. Screen cost: High. Priority: Second.

**28. Uncertainty-aware rank allocation**

Use repeated short probes or bootstrap document splits to avoid spending bits on noisy sensitivity estimates; optimize conservative expected benefit.

First experiment: Hold the search budget fixed; compare mean-only allocation with lower-confidence benefit estimates and random exchanges. Assessment: **Candidate component**. Screen cost: Medium–high. Priority: Second.

**29. Projection-specific capacity within blocks**

Allocate independently to Q/K/V/O and gate/up/down, respecting kernel rank multiples and unequal matrix shapes.

First experiment: Test a few coupled groups first; compare with block-only allocation at exact packed bytes. Assessment: **Established adaptation**. Screen cost: Medium. Priority: First.

**30. Joint rank and protected-residual budgeting**

Trade binary rank against a small high-precision or low-bit correction path under one storage and latency budget.

First experiment: Compare extra rank, protected columns, and residual factors at the same complete byte count. Assessment: **Established adaptation**. Screen cost: Medium–high. Priority: Second.

**31. Allocation by measured inference latency**

Treat legal rank configurations as hardware-dependent choices, jointly constrained by packed bytes and profiled prefill/decode latency.

First experiment: Benchmark real packed kernels; compare memory-only and latency-aware allocation across at least two batch regimes. Assessment: **Candidate systems component**. Screen cost: High. Priority: Second.

**32. Learned nested binary ranks**

Train ordered groups of binary components so prefixes of the rank dimension remain usable at several budgets.

First experiment: Compare separately trained models with one nested model at 0.5/0.75/1/1.25 nominal bits, counting padding and scales. Assessment: **Candidate core**. Screen cost: High. Priority: Later.


**E. Representations and protected subspaces**

Relevant prior work and implementation: [^9][^12][^25][^33][^34][^35][^36][^37][^38][^39][^40]

**33. Protected input columns under a strict budget**

Keep a small set of damaging columns in higher precision and subtract their storage from binary ranks. Select on quantization damage, not activation energy alone.

First experiment: Compare energy, residual-error, and end-loss selection against equal-byte extra binary rank; test sparse-kernel cost. Assessment: **Established adaptation**. Screen cost: Medium. Priority: Second.

**34. Protected functional subspace plus binary complement**

Reserve a tiny path for stable, high-impact directions and factorize the complement. Learn or select the subspace using downstream error rather than only top input eigenvalues.

First experiment: Compare coordinate columns, activation PCs, residual SVD, and task-sensitive directions at equal bytes and time. Assessment: **Candidate core**. Screen cost: High. Priority: Second.

**35. Per-component intermediate scales**

Use the already-supported scale_mid path to weight binary rank components independently; include its storage in allocation and compare joint versus scale-only fitting.

First experiment: Confirm whether baseline export populates scale_mid; then ablate with rank reduced to pay for the new scales. Assessment: **Existing code path / baseline**. Screen cost: Low. Priority: First.

**36. Shared input binary bases across projections**

Factor Q/K/V or gate/up with a shared input basis and private output bases, spending saved storage on additional rank or private residual components.

First experiment: Start with gate/up versus Q/K/V sharing; compare equal bytes and actual reused intermediate computation. Assessment: **Candidate core**. Screen cost: High. Priority: Second.

**37. Multiple binary groups with separate scales**

Partition rank components into a few groups with independent input/output scales, increasing representational flexibility without full dense correction.

First experiment: Compare one group and several groups with all repeated scales charged; benchmark launch overhead. Assessment: **Candidate component**. Screen cost: Medium–high. Priority: Later.

**38. Binary-aware orthogonal preprocessing**

Optimize legal network rotations for the approximation class of two binary factors, not only for outlier reduction or scalar quantization.

First experiment: Compare no rotation, random legal rotation, SpinQuant-style rotation, and a binary-factor-aware objective. Assessment: **Candidate component**. Screen cost: High. Priority: Later.

**39. Structured bias or mean correction**

Separate stable mean errors from token-varying errors and fit small bias or affine corrections where the model's computation permits it.

First experiment: Evaluate global and position-conditioned mean correction; include metadata and avoid assuming massive activations are constant for all tokens. Assessment: **Established adaptation**. Screen cost: Low–medium. Priority: Second.

**40. Hybrid low-bit latent factors**

Use binary factors for most components and a small number of ternary or 2-bit components where they buy more quality per byte than rank growth.

First experiment: Compare against more binary rank and a low-rank residual; require an implemented storage format and kernel path. Assessment: **Candidate component**. Screen cost: High. Priority: Later.


**F. Initialization and covariance estimation**

Relevant prior work and implementation: [^4][^10][^17][^18][^19][^22][^23]

**41. Asymmetric covariance factorization**

Fit teacher outputs on teacher inputs using quantized-prefix inputs as regressors, incorporating cross-moments rather than only E[xx^T].

First experiment: Compare the current covariance objective with direct asymmetric regression at identical input streams and rank. Assessment: **Established adaptation**. Screen cost: Medium. Priority: First.

**42. Two-sided curvature-aware ADMM**

Add structured output sensitivity as well as input covariance, using diagonal-plus-low-rank factors to bound solve cost.

First experiment: Compare to GuidedQuant/YAQA/KronQ-inspired baselines; measure final NLL, not just the newly optimized quadratic. Assessment: **Established adaptation**. Screen cost: High. Priority: Later.

**43. Domain-robust covariance objectives**

Optimize across domain- or bootstrap-specific moments, or a bounded uncertainty set, rather than one covariance estimate.

First experiment: Use held-out domains and a fixed data budget; compare robust fitting with simple shrinkage and more samples. Assessment: **Candidate component**. Screen cost: Medium–high. Priority: Second.

**44. Low-rank-plus-diagonal covariance approximation**

Preserve only the correlation modes that materially affect reconstruction; use a sketch and efficient solves for the rest.

First experiment: Measure full-covariance objective error, final quality, peak memory, and wall time as sketch rank changes. Assessment: **Established adaptation / efficiency**. Screen cost: Medium. Priority: Later.

**45. Data-adaptive correlation shrinkage**

Choose shrinkage from held-out prediction or estimation uncertainty by layer, rather than a fixed global coefficient or a five-way full tuning search.

First experiment: Compare statistically selected beta with fixed beta and validation-PPL selection at matched end-to-end cost. Assessment: **Established adaptation**. Screen cost: Low–medium. Priority: Later.

**46. Reproducible spectral or residual initialization**

Initialize factors from structured projections or sequential residual fits, and audit whether better starts reduce subsequent sign churn.

First experiment: Compare total compute to random initialization and short multi-starts; preserve the same binary representation. Assessment: **Established adaptation**. Screen cost: Medium. Priority: Later.

**47. ADMM selected by downstream usefulness**

Choose ADMM checkpoints or penalty schedules by a cheap downstream proxy and early fine-tuning response, rather than factorization residual alone.

First experiment: Track ADMM residuals, held-out feature error, and short-tuning NLL; compare fixed 400 iterations with budget-matched selection. Assessment: **Candidate component**. Screen cost: Medium. Priority: Second.

**48. Order-invariant calibration-statistics estimation**

Audit the online clipping estimator against a fixed-threshold two-pass estimator and raw moments. Multiplying old aggregates when thresholds rise does not reconstruct earlier unclipped samples exactly.

First experiment: Permute the calibration order, compare diagonal and full moments, and repeat only promising settings through block tuning. Assessment: **Diagnostic / baseline**. Screen cost: Low–medium. Priority: First.


**G. Block scheduling and global recovery**

Relevant prior work and implementation: [^7][^8][^9][^13][^16][^20][^21]

**49. Joint polishing after all projections are binary**

After sequential initialization, reopen all factors in one block for a short joint pass so all projections adapt to the final quantized environment.

First experiment: Compare additional joint tuning with equal extra sequential tuning, and with scales-only polishing. Assessment: **Established adaptation**. Screen cost: Medium. Priority: First.

**50. Overlapping block-window recovery**

Reoptimize two or three adjacent quantized blocks together to permit compensation across the residual path.

First experiment: Start with blocks 2–4 and 25–27; compare equal-time one-block extra tuning and fixed look-ahead. Assessment: **Established adaptation**. Screen cost: High. Priority: Second.

**51. Quantization order selected by coupling**

Quantize tightly interacting projections as groups, or choose an order based on how strongly remaining weights must compensate.

First experiment: Evaluate several prespecified orders from identical states; measure both transient recovery and final binary quality. Assessment: **Candidate component**. Screen cost: Medium. Priority: Second.

**52. Constrain temporary FP compensation**

Regularize surviving FP weights during tune_nonfact so they do not absorb errors in a way that later binarization cannot preserve.

First experiment: Measure temporary weight drift and damage when each compensated projection is factorized; compare generic weight regularization. Assessment: **Candidate component**. Screen cost: Low–medium. Priority: First.

**53. Continuation from easier ranks to target ranks**

Start from a larger binary rank or softened constraints, progressively remove components, and refit at the final budget.

First experiment: Compare with direct target-rank training using identical total GPU time; include all higher-rank work. Assessment: **Established adaptation**. Screen cost: Medium–high. Priority: Later.

**54. Selective end-to-end binary recovery**

Extend global KD beyond scales by allowing only high-impact unstable factor bits or a few selected blocks to change.

First experiment: Compare scale-only KD, selective sign KD, and uniform sign updates with the same trainable-bit and compute budgets. Assessment: **Candidate component**. Screen cost: High. Priority: Second.

**55. Online or chunked teacher distillation**

Avoid a complete samples-by-sequence-by-vocabulary cache; stream teacher outputs or compute exact soft targets in manageable chunks.

First experiment: Verify loss/gradient parity on a small batch and use freed memory to compare larger KD datasets at matched steps. Assessment: **Engineering / enabling**. Screen cost: Medium. Priority: First.

**56. Scale-only polishing with more data and stronger selection**

Improve the existing scale-only global recovery by separating data size, learning rate, early stopping, and distillation target choices.

First experiment: Use identical pre-KD checkpoints to isolate why KD gains differ between arms; compare pre/post-KD NLL and task accuracy. Assessment: **Baseline / enabling**. Screen cost: Low–medium. Priority: First.


**H. Systems, deployment, and research diagnostics**

Relevant prior work and implementation: [^3][^6][^9][^11][^39][^40][^41][^42]

**57. Accelerate tuning without changing optimizer steps**

Reduce launch and data-transfer overhead while preserving microbatch semantics, rather than increasing accumulation and losing updates.

First experiment: Profile block forwards, backward, optimizer, and evaluation separately; verify matching update counts and quality. Assessment: **Systems component**. Screen cost: High. Priority: Second.

**58. Rank-specific packed-kernel optimization**

Specialize kernels for the nonuniform ranks actually selected; fuse scale operations and profile intermediate traffic.

First experiment: Measure prefill and decode latency, occupancy, and peak memory for representative selected ranks. Assessment: **Systems component**. Screen cost: High. Priority: Later.

**59. Whole-model storage optimization**

Quantize or otherwise compress embeddings and the output head when they dominate total storage, especially for small vocabulary-heavy models.

First experiment: Report matrix bpw and whole-checkpoint bytes separately; evaluate the quality cost of head/embedding compression. Assessment: **Established adaptation / systems**. Screen cost: Medium–high. Priority: Second.

**60. Compression under serving-memory budgets**

Allocate between weights and KV cache using workload-specific total memory and latency, treating weight-only compression as one component.

First experiment: Compare short and long context workloads, multiple batch sizes, and the full serving-memory breakdown. Assessment: **Candidate systems component**. Screen cost: High. Priority: Later.

**61. Adaptive rank during inference**

Use nested binary ranks to spend more computation on difficult tokens or requests; account for dispatch and batching overhead.

First experiment: Compare static models on the true throughput–quality frontier; test whether routing saves time after overhead. Assessment: **Candidate core**. Screen cost: High. Priority: Later.

**62. Causal study of binary instability and generalization**

Build a regime map across model size, rank, data diversity, optimizer seed, and precision to identify when reconstruction ceases to predict language quality.

First experiment: Use fixed-prefix interventions, factor hashes, train/held-out feature losses, and confidence intervals; test whether a simple intervention follows from the mechanism. Assessment: **Candidate analysis paper**. Screen cost: Medium–high. Priority: First.

**63. A validated cheap selection metric**

Develop a held-out metric that predicts final post-KD quality across ranks, seeds, and objectives, enabling cheaper model selection.

First experiment: Fit on some models/blocks and evaluate ranking on unseen models and completed quantized chains; compare NLL, raw MSE, normalized MSE, and KL. Assessment: **Candidate component**. Screen cost: Medium–high. Priority: Second.

**64. Acceptance-aware binary draft compression**

If speculative decoding is the intended application, optimize binary draft capacity for accepted tokens per unit time against a fixed verifier rather than draft PPL alone.

First experiment: Hold verifier and sampler fixed; compare equal-byte NanoQuant drafts, acceptance, accepted length, end-to-end speed, and target-distribution fidelity. Assessment: **Candidate application paper**. Screen cost: High. Priority: Second.


**5. Six coherent paper candidates**

**Candidate 1: Function-preserving optimization of binary factors — strongest initial methods direction**

Hypothesis: raw latent updates are a poor measure of model change because the effect of one sign flip depends on the other factor, scales, and the input distribution. Tiny perturbations around a latent threshold can trigger a finite function change. Ordinary STE can therefore make unstable or poorly calibrated updates even when a local loss looks good.

For clarity, write the representation without an optional middle scale as
\[
\widehat W=D_oUV D_i,\qquad U,V\in\{-1,+1\}.
\]
Flipping \(U_{ik}\) changes the represented matrix by
\[
\Delta\widehat W=-2U_{ik}D_o e_i e_k^\top V D_i.
\]
Flipping \(V_{kj}\) changes it by
\[
\Delta\widehat W=-2V_{kj}D_o Ue_k e_j^\top D_i.
\]
These are exact algebraic changes for one flip, with scales and the other factor held fixed. A middle scale modifies the same expressions by its corresponding component weight.

Use activation sketches to estimate function change and a small loss-gradient/curvature model to prioritize candidates. Constrain the total predicted output change per update, then compare predicted and measured loss reduction. Update signs and scales on different schedules if needed. This is a proposed specialization of discrete optimization, not a claim that trust regions or curvature are new.

The paper needs to show that this score predicts harmful changes better than latent margin or gradient norm, that controlling these changes improves generalization and seed stability, and that benefits hold at equal compute. Compare against vanilla STE, simple clipping, influence-free sign freezing, alternating factors, and a PV-style adaptation. Analyze stable parts of the mechanism across at least two model families. Kill or narrow the project if ordinary learning-rate/precision controls explain the result. Relevant overlap: PV-Tuning, QuEST, and oscillation mitigation.[^13][^14][^15]

**Candidate 2: Quantization commitment and delayed binary finalization — strongest causal mechanism experiment**

The current implementation tunes and then finalizes one projection at a time. Once a projection is finalized, its latent binary variables are deleted. Later quantization may change the environment in which those signs were optimized.[^7][^9]

Hypothesis: early binary decisions compensate for weights that will not remain in their current state. After the remaining projections are quantized, previously adequate decisions become poor, but the normal loop cannot revise their signs.

Preserve latent factors until an attention or MLP group is fully binary. Compare: immediate freezing; scales-only reopening; joint sign-and-scale reopening; and delayed freezing with the same total updates. Instrument the increase in held-out loss caused by each later projection's quantization and whether reopening earlier factors repairs it. Also measure how much temporary FP compensation tune_nonfact creates.

A stronger contribution is an adaptive commitment rule: finalize a group when its signs remain stable under plausible quantization of the unfinished group. This connects ideas 2, 3, 13, 49, and 52 without requiring a large global training run. The basic delayed/joint optimization idea overlaps multi-block reconstruction; the paper must establish and exploit a distinct binary commitment mechanism. Failure criterion: the effect disappears under equal extra tuning compute or scales-only correction.[^20][^21]

**Candidate 3: Robust compression across prefix uncertainty — strong generalization direction, substantial prior-art risk**

For a sequence, let \(h_l^T\) be the FP-prefix hidden state and \(h_l^{Q,k}\) its hidden state under plausible quantized prefix \(k\). The existing asymmetric reconstruction uses one prefix state. A proposed robust variant minimizes
\[
\frac{1}{K}\sum_k
\ell\big(B_l^Q(h_l^{Q,k}),B_l^T(h_l^T)\big)
\]
or a controlled worst-trajectory/variance penalty, while retaining diverse real calibration sequences.

The teacher target stays associated with the same sequence. Do not independently shuffle hidden states and targets. Real alternative prefixes are the strongest diagnostic; low-rank perturbations estimated from their differences may make training affordable. Recollect the local statistics where the prefix changes.

The claim would be that robustness to the *distribution of binary prefix errors* reduces sensitivity to initialization and calibration selection better than more examples, simple isotropic noise, or point-estimate mismatch correction. QEP and CoreQ are mandatory controls; recent low-rank calibration-drift work is relevant too.[^22][^23][^31] Limitations include expensive trajectory generation, perturbations that leave the model's valid activation manifold, and overconservative objectives. Stop if realistic perturbations provide no gain beyond equal-cost data diversity.

**Candidate 4: Joint optimization of binary rank, interactions, and tuning compute — strongest extension of the positive result**

The proposed problem is
\[
\min_{\{r_l,t_l\}} \widehat{\mathcal L}_{val}(\{r_l,t_l\})
\quad\text{subject to}\quad
\sum_l C_l(r_l)\le B,\quad
\sum_l T_l(r_l,t_l)\le T.
\]
Here \(r_l\) is a legal rank and \(t_l\) is a tuning budget. With the current two-scale representation, idealized matrix storage is
\[
C_l(r)=r(m_l+n_l)+16(m_l+n_l)\quad\text{bits},
\]
before packing padding, extra metadata, and any added scales. A legal +32 rank move adds \(32(m_l+n_l)\) binary bits if other storage stays unchanged.

Start with finite-difference response surfaces. For rank changes on two projections, estimate an interaction
\[
I_{ij}=L_{ij}-L_i-L_j+L_0,
\]
where all losses use the same reference configuration, validation data, and controlled tuning protocol. Nonzero interactions show when independent sensitivity scores fail. They do not automatically yield a tractable global model.

A candidate algorithm uses a sparse interaction graph, conservative benefit estimates, and budget-preserving exchanges. The distinguishing evidence must show why DBF-style allocation, the current heuristic, and a strong additive marginal allocator leave quality on the table. Include search GPU hours and repeated end-to-end compression. HAWQ-style sensitivity, adaptive rank/bit methods, calibration/rank correction, and existing work on non-additivity make the generic idea crowded.[^12][^29][^30][^31][^32]

**Candidate 5: Functional-subspace protection under a sub-bit budget — higher-risk representation direction**

Represent each matrix as
\[
\widehat W=W_{\mathrm{binary}}+CP^\top,
\]
where a very small correction protects directions that consistently cause large downstream loss when distorted. A coordinate subset is a constrained special case. The key question is whether a task-sensitive subspace saves more quality per byte and per unit latency than increasing binary rank.

Compare energy-selected columns, activation PCs, residual SVD, and task-sensitive directions. Retain the correction only if its gains survive joint refitting of the binary complement and strict budget compensation. For a FP16 rank-\(k\) correction, naive storage is \(16k(m+n)\) bits; this is substantial in a roughly one-bit budget. Four FP16 columns of a matrix with 1024 inputs add \(16\times4/1024=0.0625\) bpw before indices or offsets if added alongside the existing representation. They are not automatically negligible.

The paper must go beyond generic sparse exceptions or low-rank corrections. Those already exist; recent preserve-then-quantize work is especially close.[^33][^34][^35][^36] A plausible contribution is a stable functional-subspace selection criterion with a fused kernel and a demonstrably better sub-bit frontier. Stop if equal-byte extra binary rank performs as well or the extra path erases the runtime benefit.

**Candidate 6: Binary drafts optimized for accepted tokens per second — distinct application paper**

If speculative decoding is the ultimate objective, draft PPL is only a proxy. With an exact rejection-sampling verifier, the expected one-step acceptance at a fixed prefix is distribution overlap:
\[
\alpha=\sum_v\min(p_T(v),p_D(v))=1-\mathrm{TV}(p_T,p_D).
\]
This identity and draft alignment objectives are established.[^41][^42]

A new project could jointly allocate binary rank and tune the draft to maximize measured accepted tokens per wall-clock second under a fixed verifier. A better draft can be slower; a smaller draft can lose acceptance. Measure accepted length, draft cost, verification cost, batch behavior, and total speed. The identity above does not by itself predict multi-token throughput, which depends on changing prefixes and the serving implementation.

Compare the same NanoQuant draft compressed for language-model PPL, teacher KL, and acceptance-oriented objectives, then compare against independently selected draft sizes and higher-bit drafts at the same total memory. This route is attractive if the repository already has a working speculative stack. It is not established by the present Qwen3-0.6B WikiText2 experiments.

**6. Experiments that most efficiently distinguish the candidates**

**Stage A: Make the diagnosis causal.** Save a checkpoint just before block 3, teacher and student inputs, random states, calibration sample IDs, and hashes of all factors/scales. Repeat only the local intervention. Do the same for one late block. Measure loss before and after each tune_nonfact, ADMM, tune_fact, and finalize call. Finalize should not introduce an unexplained numerical discontinuity relative to the effective binary training forward; any discrepancy requires implementation investigation first.

Record absolute and relative hidden-state errors, normalized-feature errors, downstream NLL, sign flip/back-flip rates, latent margins, parameter-update norms, and separated gradient contributions from massive and ordinary channels. Repeat the key screen on an independent calibration split. A tensor hash locates reproducibility failures; an aggregate metric does not.

**Stage B: Establish a strong baseline.** Keep both the published-setting baseline and a strengthened 512-sample baseline. Use the 2×2 statistics/tuning-data control. Complete the pending rank-allocation KD runs. Audit scalar hyperparameters, accumulation counts, and teacher-cache precision. Separate repeated executions of the same seed from runs using new seeds.

**Stage C: Run three small causal screens.**

| Screen | Main comparison | What it resolves |
|---|---|---|
| Commitment | Immediate finalization vs delayed group finalization vs equal-time extra sequential tuning vs scales-only reopening | Whether freezing earlier signs creates an avoidable constraint. |
| Binary dynamics | STE vs alternating factors vs influence-free flip limit vs function-space flip limit | Whether binary-factor geometry matters beyond a smaller effective step. |
| Prefix uncertainty | Single prefix vs more real text vs realistic prefix replay vs matched covariance noise | Whether trajectory robustness matters beyond data diversity and generic regularization. |

Use at least three tuning seeds for screening, then five or more finalist seeds if resources permit, with calibration-seed variation in the confirmation stage. These are suggested starting counts, not a power calculation. Choose the final sample size from observed paired variance and the smallest practically meaningful gain.

**Stage D: Confirm a single method before combining improvements.** Run all 28 blocks and identical KD, using uniform rank first. Then cross the winning method with the strongest rank allocation. A method must improve completed quantized models, not only partial models with a long FP suffix. Evaluate whether a local improvement survives later quantization and whether KD helps or reverses it.

**Stage E: Scale only the credible winner.** Begin with Qwen3-0.6B-Base for diagnosis. Add a larger model in the same family and at least one different architecture/family, then a 7–8B-class model if feasible. Use supported existing checkpoints and record exact revisions. A compact-model-only paper needs an explicit scope; do not infer large-model conclusions from one small model.

**7. Evaluation needed for a publishable result**

Report nominal bits, realized factorized-matrix bpw, complete packed-checkpoint bytes, peak inference memory, and peak compression memory separately. Include scales, residuals, indices, padding, shared-basis metadata, embeddings, output head, and runtime workspace as applicable. Changes to scale_mid, protected columns, shared bases, or low-bit factors invalidate the current driver's simple accounting unless it is updated.[^6][^9]

Use validation NLL/KL for model selection and reserve final test metrics. Report WikiText2 plus at least one genuinely different text distribution and several downstream tasks matched to model capability. Show per-task results rather than only an average. Use longer contexts than the training window for confirmation if long-context behavior is part of the claim.

Measure both prefill and decode at realistic batch sizes using the exported packed kernels. A training-time fake-quant model or nominal operation count cannot establish inference speed. Compare against the original NanoQuant configuration, the strengthened baseline, DBF where relevant, and the closest algorithmic baseline. Higher-bit models belong on a quality–storage–latency frontier; they are not equal-bit comparisons.

For inference about average quality, use paired differences on NLL and per-task accuracy, report all model-seed outcomes, and describe confidence intervals. Bootstrap text at document or sequence-block level rather than pretending adjacent tokens are independent. Separate variation from calibration selection and tuning randomness. Count failed runs and the full model-selection budget.

Do not turn a larger data budget, more restarts, a broader search, or extra tuning into an unacknowledged algorithmic gain. Report matched-step and matched-wall-clock comparisons when they answer different questions. For a representation change, storage and deployment latency must also be controlled.

**8. What to pursue and what to defer**

The recommended first paper path is **binary commitment and function-space update control**: ideas 1, 2, 3, 13, 49, and 52. Begin with delayed-finalization and sign-influence diagnostics before implementing a sophisticated solver. If the failure mechanism is absent, switch to prefix robustness or rank/compute allocation rather than forcing the narrative.

For the best practical NanoQuant configuration, independently maintain more-data tuning, sound checkpoint selection, memory-efficient KD, and the current rank-allocation result. These make the baseline stronger. Treat them as controls or supporting components unless a separate, generalizable mechanism emerges.

Defer a full covariance parameter sweep, best-of-20 search, generic per-channel loss replacements, or a large representation rewrite until small causal screens justify them. Covariance could still matter at lower ranks, shorter tuning budgets, different models, or with refreshed statistics. The current result deprioritizes it as the sole headline, not as a universally useless technique.

A compelling eventual claim would have this structure: **identify a reproducible binary-specific failure; derive a targeted intervention; demonstrate better quality or lower compression cost at equal deployed storage; show the effect across models and independent evaluation data.** No performance gain or acceptance at a venue can be promised before these experiments.

**Sources**

The sources below link to the exact repository snapshot or original research. Publication dates refer to the cited work's first year unless a revision is explicitly mentioned.

[^1]: PseudoHunt. [Experiment summary](https://github.com/PseudoHunt/flexdraft-1bit-drafter/blob/1d36d8d/llm_ext/SUMMARY.md). 2026, commit 1d36d8d.

[^2]: PseudoHunt. [Experimental results, sections 1–6](https://github.com/PseudoHunt/flexdraft-1bit-drafter/blob/1d36d8d/llm_ext/RESULTS.md). 2026, commit 1d36d8d.

[^3]: PseudoHunt. [Reproduction instructions](https://github.com/PseudoHunt/flexdraft-1bit-drafter/blob/1d36d8d/llm_ext/README.md). 2026, commit 1d36d8d.

[^4]: PseudoHunt. [NanoQuant experiment patch](https://github.com/PseudoHunt/flexdraft-1bit-drafter/blob/1d36d8d/llm_ext/nanoquant_cov.patch). 2026, commit 1d36d8d.

[^5]: PseudoHunt. [Rank-allocation implementation](https://github.com/PseudoHunt/flexdraft-1bit-drafter/blob/1d36d8d/llm_ext/block_bits.py). 2026, commit 1d36d8d.

[^6]: PseudoHunt. [Experiment driver](https://github.com/PseudoHunt/flexdraft-1bit-drafter/blob/1d36d8d/llm_ext/run_llm_ext.py). 2026, commit 1d36d8d.

[^7]: SamsungLabs. [NanoQuant block tuning implementation](https://github.com/SamsungLabs/NanoQuant/blob/a9e0a43/src/nanoquant/core/compress_block.py). commit a9e0a43.

[^8]: SamsungLabs. [NanoQuant model compression and KD implementation](https://github.com/SamsungLabs/NanoQuant/blob/a9e0a43/src/nanoquant/core/compress_model.py). commit a9e0a43.

[^9]: SamsungLabs. [NanoQuant binary linear module](https://github.com/SamsungLabs/NanoQuant/blob/a9e0a43/src/nanoquant/modules/linear.py). commit a9e0a43.

[^10]: SamsungLabs. [NanoQuant importance statistics](https://github.com/SamsungLabs/NanoQuant/blob/a9e0a43/src/nanoquant/core/importance.py). commit a9e0a43.

[^11]: Chong et al.. [NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](https://arxiv.org/abs/2602.06694). 2026.

[^12]: Boža and Macko. [Addition is almost all you need: Compressing large language models with double binary factorization](https://arxiv.org/abs/2505.11076). 2025.

[^13]: Malinovskii et al.. [PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression](https://arxiv.org/abs/2405.14852). 2024.

[^14]: Nagel et al.. [Overcoming Oscillations in Quantization-Aware Training](https://arxiv.org/abs/2203.11086). 2022.

[^15]: Panferov et al.. [QuEST: Stable Training of LLMs with 1-Bit Weights and Activations](https://arxiv.org/abs/2502.05003). 2025.

[^16]: Li and Panda. [TesseraQ: Ultra Low-Bit LLM Post-Training Quantization with Block Reconstruction](https://arxiv.org/abs/2410.19103). 2024.

[^17]: Kim et al.. [GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance](https://arxiv.org/abs/2505.07004). 2025.

[^18]: YAQA authors. [Model-Preserving Adaptive Rounding (YAQA)](https://arxiv.org/abs/2505.22988). 2025.

[^19]: KronQ authors. [KronQ: LLM Quantization via Kronecker-Factored Hessian](https://arxiv.org/abs/2607.07964). 2026.

[^20]: Shabanovi et al.. [Interactions Across Blocks in Post-Training Quantization of Large Language Models](https://arxiv.org/abs/2411.03934). 2024.

[^21]: CBQ authors. [CBQ: Cross-Block Quantization for Large Language Models](https://arxiv.org/abs/2312.07950). 2023.

[^22]: Arai and Ichikawa. [Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](https://arxiv.org/abs/2504.09629). 2025.

[^23]: Cha et al.. [CoreQ: Learning-Free Mismatch Correction and Successive Rounding for Quantization](https://arxiv.org/abs/2602.05902). 2026; updated title in v2.

[^24]: Zhao et al.. [Saliency-Aware Regularized Quantization Calibration for Large Language Models](https://arxiv.org/abs/2605.05693). 2026.

[^25]: Sun et al.. [Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762). 2024.

[^26]: Xiao et al.. [FAQ: Mitigating Quantization Error via Regenerating Calibration Data with Family-Aware Quantization](https://arxiv.org/abs/2601.11200). 2026.

[^27]: COVERCAL authors. [Coverage-Based Calibration for Post-Training Quantization via Weighted Set Cover over Outlier Channels](https://arxiv.org/abs/2604.24008). 2026.

[^28]: MaCa authors. [On the Importance of a Multi-Scale Calibration for Quantization](https://arxiv.org/abs/2602.07465). 2026.

[^29]: HAWQ-V2 authors. [HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks](https://arxiv.org/abs/1911.03852). 2019.

[^30]: Zhou et al.. [Efficient Fine-Tuning of Quantized Models via Adaptive Rank and Bitwidth](https://arxiv.org/abs/2505.03802). 2025.

[^31]: Authors of arXiv:2608.08506. [Understanding Calibration and Truncation Error Propagation in Training-Free Low-Rank Compression for LLMs](https://arxiv.org/abs/2608.08506). 2026.

[^32]: Authors of arXiv:2607.12266. [Saturation Makes Quantization Error Additive: A Coverage Model with a Certificate](https://arxiv.org/abs/2607.12266). 2026.

[^33]: CALDERA authors. [Compressing Large Language Models using Low Rank and Low Precision Decomposition (CALDERA)](https://arxiv.org/abs/2405.18886). 2024.

[^34]: Cho et al.. [Preserve-Then-Quantize: Balancing Rank Budgets for Quantization Error Reconstruction in LLMs](https://arxiv.org/abs/2602.02001). 2026.

[^35]: Dettmers et al.. [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078). 2023.

[^36]: SqueezeLLM authors. [SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/abs/2306.07629). 2023.

[^37]: Ashkboos et al.. [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456). 2024.

[^38]: Liu et al.. [SpinQuant: LLM quantization with learned rotations](https://arxiv.org/abs/2405.16406). 2024.

[^39]: MatQuant authors. [Matryoshka Quantization](https://arxiv.org/abs/2502.06786). 2025.

[^40]: MatGPTQ authors. [MatGPTQ: Accurate and Efficient Post-Training Matryoshka Quantization](https://arxiv.org/abs/2602.03537). 2026.

[^41]: Leviathan et al.. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). 2023.

[^42]: Authors of arXiv:2403.00858. [Direct Alignment of Draft Model for Speculative Decoding with Chat-Fine-Tuned LLMs](https://arxiv.org/abs/2403.00858). 2024.

[^43]: SamsungLabs / Benjamin Warner. [NanoQuant AdamW implementation](https://github.com/SamsungLabs/NanoQuant/blob/a9e0a43/src/nanoquant/optimi/adamw.py). commit a9e0a43.

[^44]: PseudoHunt. [Rank allocation pre-KD raw result, replicate 1](https://github.com/PseudoHunt/flexdraft-1bit-drafter/blob/1d36d8d/llm_ext/results/q06_ra_alloc_r1_preKD.json). 2026, commit 1d36d8d.

