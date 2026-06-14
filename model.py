"""Micro masked-diffusion language model — the LLaDA-style core.

Vanilla bidirectional transformer (~0.8M params). The entire 'diffusion'
character of this model lives in two places that are NOT in this file:
  - train.py: random masking ratio t ~ U(0,1) with 1/t-weighted CE
  - sample.py: iterative confidence-based unmasking
The network itself is just a mask-predictor: tokens in, per-position
distributions out. No causal mask anywhere.
"""
import torch
import torch.nn as nn

DIGITS = "1234567890"
LAYOUTS = {
    "QWERTY":  DIGITS + "QWERTYUIOPASDFGHJKL;ZXCVBNM,./",
    "Dvorak":  DIGITS + "',.PYFGCRLAOEUIDHTNS;QJKXBMWVZ",
    "Colemak": DIGITS + "QWFPGJLUY;ARSTDHNEIOZXCVBKM,./",
    "AZERTY":  DIGITS + "AZERTYUIOPQSDFGHJKLMWXCVBN,;:!",
    # spicy variant: replace AZERTY's DIGITS prefix with '&é"\'(-è_çà'
    # (number row becomes 75/25 contested; one sampled é decides everything)
}
N = 40
MASK_ID = 0
CHARS = sorted(set("".join(LAYOUTS.values())))     # deterministic vocab order
STOI = {c: i + 1 for i, c in enumerate(CHARS)}     # id 0 reserved for [MASK]
ITOS = {i + 1: c for i, c in enumerate(CHARS)}
V = len(CHARS) + 1

def encode(s):  return torch.tensor([STOI[c] for c in s], dtype=torch.long)
def decode(t):  return "".join(ITOS.get(int(i), "·") for i in t)

class MaskedDiffusionLM(nn.Module):
    def __init__(self, vocab=V, n=N, dim=128, heads=4, layers=4, ff=512):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, n, dim))
        block = nn.TransformerEncoderLayer(dim, heads, ff, dropout=0.0,
                  activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, tokens):                  # [B, N] int64
        h = self.tok(tokens) + self.pos         # bidirectional: every position
        h = self.blocks(h)                      # attends to every other, no mask
        return self.head(self.norm(h))          # [B, N, V]

if __name__ == "__main__":
    p = sum(x.numel() for x in MaskedDiffusionLM().parameters())
    print(f"vocab {V}, seq len {N}, params {p/1e6:.2f}M")
