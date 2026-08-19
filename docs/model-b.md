# Model B — Word Diffusion

The README hinted at this one: "for 'model B' (the generalization companion):
replace `LAYOUTS.values()` with a few thousand 5-letter words, set `N`
accordingly — nothing else changes." That's exactly what
`model_b_word_diffusion.py` does, and the run documented below is the
Kaggle Tesla-T4 training run captured in `model-b-word-diffusion.log`.

Same recipe. Same mechanism. A much harder hypothesis space.

## What's different from the keyboard model

| | Keyboard (`model.py`) | Model B (`model_b_word_diffusion.py`) |
|---|---|---|
| Vocab | 38 chars + MASK | a–z + MASK (27) |
| `N` | 40 | 5 |
| Fixed context | number row (10 positions) | none — prompt is fully masked |
| Training data | 4 layouts (memorize one of four) | 8000 5-letter words |
| `dim` | 128 | 256 |
| `layers` | 4 | 6 |
| `ff` | 512 | 1024 |
| Params | ~0.8M | 4.75M |
| Steps | 3000 | 8000 |
| Batch | 64 | 1024 |

Everything else — bidirectional TransformerEncoder, `t ~ U[0.05, 1]`
masking, `1/t`-weighted CE on masked positions, AdamW lr=3e-4 wd=0.01,
the confidence-commit sampler, the analytic-Bayes sanity check — is
identical in shape. This is the point: the recipe is the story.

## The training run (Tesla T4, 8000 steps, ~660s)

| step | loss | masked-acc |
|---|---|---|
| 1000 | 3.123 | 19.9% |
| 2000 | 2.553 | 25.8% |
| 3000 | 2.494 | 24.7% |
| 4000 | 2.532 | 25.3% |
| 5000 | 2.312 | 26.7% |
| 6000 | 2.272 | 26.7% |
| 7000 | 2.377 | 27.9% |
| 8000 | 2.211 | 27.9% |

Loss settles in the 2.2–2.5 range and masked-acc plateaus around 27%.
**This is not undertraining.** The keyboard model's README footnote
applies verbatim: when the prompt is fully masked, heavily masked
positions are genuinely ambiguous — for any one letter slot, hundreds
or thousands of words in the training set disagree. The Bayes-optimal
answer is a spread distribution, and the model converging to ~the
unigram mixture *is* the "transformer learned Bayes" punchline, just
on a hypothesis space of 8000 instead of 4.

(The keyboard model gets away with ~26% masked-acc on a far smaller
hypothesis space partly because its prompt is *informative* — the
number row is fixed context. See "What to try next" below.)

## Sampling modes and the result that matters

Four samplers were compared, 256 samples each (greedy uses 64):

| mode | valid_train | valid_english | unique |
|---|---|---|---|
| `parallel` (predict all 5 positions at once from the all-masked prompt) | 1.6% | 2.0% | 256/256 |
| `k2` (iterative, commit top-2 confident masked tokens per step) | 37.9% | 39.5% | 256/256 |
| `ancestral` (iterative, commit top-1 each step, re-condition) | **66.4%** | **68.4%** | 254/256 |
| `greedy_ancestral` (same as ancestral but `argmax` not multinomial) | 100.0% | — | **1/64** |

The single number that matters: **66.4% → 100% with the same sampler and
the only change being `multinomial → argmax`.** Two facts hiding in that
gap:

1. **Ancestral > k2 > parallel.** With no fixed context, the only way to
   get usable outputs is to commit tokens one (or two) at a time, *then
   re-condition* the predictions on what was just committed. That's the
   whole mechanism the keyboard demo visualizes. Parallel prediction
   from a fully masked prompt is intentionally a near-empty baseline —
   it shows what "no information yet" looks like.
2. **Greedy collapses to one word** ("bales" — 64/64). This is the
   exact failure mode the keyboard README warns about: when there's no
   prompt to bias the dice, `argmax` commits the model's already-strongest
   preference, the posterior concentrates on it, and everything else
   gets killed. Stochastic sampling keeps diversity (254 unique strings
   out of 256). The keyboard model dodges the worst of this because its
   number-row prompt loads the dice; Model B has no such prompt by
   design.

## Analytic Bayes check

The script prints `mean TV(model vs analytic unigram marginals): 0.0374`
at the end — small, so the model matches the unigram prior closely.
**Caveat:** this is measured from the all-masked state, where the
model's output should literally be the unigram marginals. Of course
those match — it's not a strong test of the model learning Bayes.

A stronger test (proposed, not yet run): for a fixed prefix, compare the
model's per-position distributions to exact conditional counts over the
8000-word library. That's the test that would actually substantiate the
keyboard demo's "net ≈ Bayes" claim for Model B.

## Iteration 2 — decoding study + v3 retrain

**The follow-up experiment** (`model_b_word_diffusion_v3.py`, Kaggle T4):
pull the published v2 weights from HF, sweep decoding strategies on the
frozen weights, then retrain a deeper v3 (8 layers, 12k steps, cosine LR,
6.33M params) and re-run the best decoders on it. 512 samples per cell.

### Sampler study on frozen v2 weights (zero retraining)

| decoding | valid english | unique/512 | passes/sample |
|---|---|---|---|
| ancestral T=1.0 (iteration 1 baseline) | 68.8% | 501 | 5 |
| ancestral T=0.7 | 85.7% | 479 | 5 |
| **ancestral T=0.5** | **95.5%** | 409 | 5 |
| threshold 0.9 (LLaDA-style parallel-commit) | 68.8% | 501 | 5 |
| k=2 | 40.0% | 507 | 3 |
| revision-capable (re-mask weak commits, T=0.7) | 77.1% | 451 | 22.5 |

**The headline: temperature alone bought +27 points (68.8% → 95.5%) on
the exact same weights, at the same 5 forward passes.** Iteration 1's
lesson — "the sampler is half the model" — now has a second act: a single
scalar knob was worth more than any architecture change tested.

### v3 retrain + best decoders

| decoding | valid english | unique/512 |
|---|---|---|
| ancestral T=1.0 | 82.6% | 499 |
| **ancestral T=0.5** | **98.4%** | 428 |

v3 at T=1.0 beats v2 at T=1.0 by 14 points — but v2 at T=0.5 beats v3 at
T=1.0. **Decoding strategy > extra parameters**, on both axes measured.

Unigram TV vs exact Bayes: 0.0135 (v2: 0.0374) — the deeper model tracks
analytic statistics 2.7× more closely. At T=0.5 the top repeated words are
real English (`oases`, `vivas`, `bulla`) — the mode collapse that gave v2
greedy "bales" now lands on valid words.

### Prompting (fixed first letter, v3, T=0.5)

95.9% valid English overall, 32 samples per letter. Confirms the
keyboard demo's "prompt loads the dice" claim at word scale.

### Honest negatives

- **Revision sampling isn't worth it at N=5**: 77–78% validity (worse
  than plain T=0.5) at 4.5× the forward passes. With only 5 positions,
  un-committing early mistakes costs more than it recovers. May differ
  at longer sequences — that's an open question, not a closed one.
- **Prefix-Bayes tracking is density-dependent**: common prefixes track
  exact conditional Bayes closely (`br` TV 0.024, `st` 0.072, `ch`
  0.070), rare ones degrade (`ze` TV 0.29). Net ≈ Bayes holds where the
  training data is dense — exactly where you'd expect a learning
  approximation to break first.

## Artifacts

- `model_b_word_diffusion.py` — iteration 1 training/eval script (verbatim
  as run on the T4)
- `model_b_word_diffusion_v3.py` — iteration 2: sampler study + v3 retrain
- `model-b-word-diffusion.log` — iteration 1 full training/eval stdout
- `modelb_v2.pt` — iteration 1 weights (4.75M params, 19 MB)
- `modelb_v3.pt` — iteration 2 weights (6.33M params, 8 layers, 25 MB)
- `results-v3.json` — iteration 2 full results (all sampler tables,
  prompting study, Bayes checks)

Both checkpoints and the model card live on Hugging Face at
[PastelRuntime/KB-Diffusion-ModelB](https://huggingface.co/PastelRuntime/KB-Diffusion-ModelB).
GitHub public forks can't upload Git LFS objects (platform policy on fork
networks), so HF is the canonical home for the weights.

**Verification status: unverified locally.** The sampler numbers above
come from the Kaggle run itself; the weights have not been re-loaded
and re-evaluated on this machine. If reproducibility matters, the
natural first check is `torch.load("modelb_v2.pt", map_location="cpu")`
into a `Net(dim=256, heads=4, layers=6, ff=1024)` and confirm the
parameter count is 4.75M and the loss lands near 2.2 at step 8000.

### Where things live now

The fork (`PastelRuntime/KB-Diffusion-Optimized`) carries code, log,
and docs. Weights live on Hugging Face (see *Artifacts* above). A
simplified checkout:

    git clone https://github.com/PastelRuntime/KB-Diffusion-Optimized
    wget https://huggingface.co/PastelRuntime/KB-Diffusion-ModelB/resolve/main/modelb_v2.pt

Historical note: the first push attempt used Git LFS inside the fork,
which GitHub rejects for public forks ("can not upload new objects to
public fork"). That's why HF hosts the weights.

## What to try next

1. **N=10 (longer sequences).** Every trend here — temperature gains,
   iteration gains, revision costs — was measured at N=5. Longer
   sequences are where the revision sampler's tradeoff could flip, and
   where the parallel-vs-ancestral gap should widen. This is the single
   most informative next axis.
2. **Temperature/quality frontier, properly.** T=0.5 won here, but the
   sweep was coarse (1.0 / 0.7 / 0.5). A fine sweep plus a
   validity-vs-uniqueness Pareto plot would make the tradeoff curve
   exact.
3. **Train on `WORDS ∪ EVAL_WORDS`** and re-measure the
   `valid_train` vs `valid_english` gap to separate "learned English
   letter statistics" from "memorized 8000 strings."
4. **Browser export.** A 5-position Model B is small enough for a cute
   live demo — type a one-letter prefix, watch words denoise in, with
   the temperature slider front and center. The temperature finding
   deserves to be *visible*.
