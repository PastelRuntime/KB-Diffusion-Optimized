"""The full LLaDA training recipe. The five lines that matter are marked.

Trains in well under a minute on CPU; your RTX Pro 6000 will mostly be
deciding whether the job is worth spinning the fans for.

For 'model B' (the generalization companion): replace LAYOUTS.values()
with a few thousand 5-letter words, set N accordingly — nothing else changes.
"""
import torch
import torch.nn.functional as F
from model import MaskedDiffusionLM, LAYOUTS, encode, MASK_ID, N, V

def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.stack([encode(s) for s in LAYOUTS.values()]).to(dev)  # [4, N]
    model = MaskedDiffusionLM().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    steps, bs = 3000, 64
    for step in range(1, steps + 1):
        x0 = data[torch.randint(0, len(data), (bs,), device=dev)]
        t = torch.rand(bs, 1, device=dev).clamp(min=0.05)        # (1) ratio ~ U(0,1)
        mask = torch.rand(bs, N, device=dev) < t                 # (2) mask indep. w.p. t
        mask[mask.sum(1) == 0, 0] = True                         #     ensure ≥1 masked
        xt = torch.where(mask, torch.full_like(x0, MASK_ID), x0) # (3) corrupt
        logits = model(xt)                                       # (4) one fwd pass
        ce = F.cross_entropy(logits.view(-1, V), x0.view(-1),
                             reduction="none").view(bs, N)
        loss = ((ce * mask).sum(1) / t.squeeze(1)).mean() / N    # (5) CE on masked, 1/t wt
        # The 1/t weight is what upgrades BERT-style infilling into a proper
        # generative model: this loss upper-bounds the NLL of the data (LLaDA/RADD).
        # t is clamped at 0.05 because the 1/t weight blows up variance as t -> 0.
        opt.zero_grad(); loss.backward(); opt.step()

        if step == 1 or step % 300 == 0:
            with torch.no_grad():
                acc = ((logits.argmax(-1) == x0) & mask).float().sum() / mask.sum()
            print(f"step {step:4d}  loss {loss.item():7.4f}  masked-acc {acc:6.1%}")
    # masked-acc plateaus well below 100% — that is NOT undertraining. Heavily
    # masked home-row positions are genuinely ambiguous (4 layouts disagree),
    # so the Bayes-optimal answer is a split distribution. The model converging
    # to ~the analytic mixture IS the 'transformer learned Bayes' punchline.
    torch.save(model.state_dict(), "keyboard_diffusion.pt")
    print("saved keyboard_diffusion.pt")

if __name__ == "__main__":
    main()
