"""Confidence-based unmasking — same sampler the web demo runs.
Digits are fed as fixed context (like the web demo's number row / a prompt).

  python sample.py --show-steps            # ASCII board, step by step
  python sample.py --runs 8                # divergence demo: same start, different endings
  python sample.py --runs 8 --seed 42      # reproducible take for filming
"""
import argparse
import torch
from model import MaskedDiffusionLM, LAYOUTS, MASK_ID, N, ITOS, DIGITS, STOI

CTX_END = len(DIGITS)

@torch.no_grad()
def probs_for(model, tokens):
    logits = model(tokens.unsqueeze(0))[0]
    logits[:, MASK_ID] = float("-inf")          # the mask token is never an answer
    return logits.softmax(-1)                   # [N, V] — independent per-position marginals

def board(tokens, fresh=()):
    rows = []
    for r in range(4):
        rows.append("".join(
            f"[{ITOS.get(int(tokens[i]), '·')}]" if i in fresh
            else f" {ITOS.get(int(tokens[i]), '·')} "
            for i in (r * 10 + c for c in range(10))))
    return "\n".join(rows)

def posterior(tokens):
    """Exact Bayes over the layout library — model-free bookkeeping.
    The trained net's distributions should track this. Net ≈ Bayes."""
    w = [float(all(int(tokens[i]) == MASK_ID or ITOS[int(tokens[i])] == s[i]
                   for i in range(N))) for s in LAYOUTS.values()]
    z = sum(w) or 1.0
    return {n: x / z for n, x in zip(LAYOUTS, w)}

@torch.no_grad()
def generate(model, show=False):
    tokens = torch.full((N,), MASK_ID, dtype=torch.long)
    for i, ch in enumerate(DIGITS):
        tokens[i] = STOI[ch]                    # digits = fixed context, never generated
    step = 0
    while (tokens == MASK_ID).any():
        step += 1
        p = probs_for(model, tokens)
        masked = (tokens == MASK_ID).nonzero().squeeze(1).tolist()
        conf = {i: p[i].max().item() for i in masked}
        # position selection is parallel (that part is safe)...
        order = sorted(masked, key=lambda i: conf[i] + 0.15 * torch.rand(1).item(),
                       reverse=True)
        sure = sum(c >= 0.99 for c in conf.values())
        k = min(len(masked), max(sure, 2 + int(torch.randint(0, 3, (1,)))))
        fresh = []
        for i in order[:k]:
            p = probs_for(model, tokens)            # ...but values are sampled
            tokens[i] = torch.multinomial(p[i], 1)  # ancestrally: re-condition after
            fresh.append(i)                         # each commit. Skipping this refresh
                                                    # is the independent-marginals bug.
        if show:
            post = "  ".join(f"{n} {v:.0%}" for n, v in posterior(tokens).items() if v)
            done = N - CTX_END - int((tokens == MASK_ID).sum())
            print(f"\nstep {step}  committed {done}/{N - CTX_END}  posterior: {post or '—'}")
            print(board(tokens, set(fresh)))
    return tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-steps", action="store_true")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    if a.seed is not None:
        torch.manual_seed(a.seed)
    model = MaskedDiffusionLM()
    model.load_state_dict(torch.load("keyboard_diffusion.pt", map_location="cpu"))
    model.eval()
    for r in range(a.runs):
        toks = generate(model, show=a.show_steps)
        text = "".join(ITOS.get(int(t), "?") for t in toks)
        name = max(posterior(toks).items(), key=lambda kv: kv[1])[0]
        exact = text in LAYOUTS.values()
        print(f"run {r + 1}: converged to {name}   exact reproduction: {exact}")

if __name__ == "__main__":
    main()
