"""Export weights + a self-test vector to weights.json for the pure-JS engine.
No ONNX, no runtime versioning, nothing to go stale. The JS engine verifies
itself against the baked-in test logits on load, so it either provably
matches PyTorch or refuses and falls back to analytic Bayes."""
import base64, json
import numpy as np
import torch
from model import MaskedDiffusionLM, CHARS, MASK_ID, N, V

def pack(t):
    a = t.detach().cpu().numpy().astype(np.float32)
    return {"shape": list(a.shape),
            "data": base64.b64encode(a.tobytes()).decode()}

def main():
    model = MaskedDiffusionLM()
    model.load_state_dict(torch.load("keyboard_diffusion.pt", map_location="cpu"))
    model.eval()

    tensors = {k: pack(v) for k, v in model.state_dict().items()}

    test_tokens = [(i * 7) % V for i in range(N)]   # fixed, covers MASK + chars
    with torch.no_grad():
        test_logits = model(torch.tensor([test_tokens], dtype=torch.long))

    blob = {"mask_id": MASK_ID, "chars": CHARS, "n": N,
            "dim": 128, "heads": 4, "layers": 4, "ff": 512,
            "tensors": tensors,
            "test": {"tokens": test_tokens, "logits": pack(test_logits)}}
    with open("weights.json", "w") as f:
        json.dump(blob, f)
    mb = len(json.dumps(blob)) / 1e6
    print(f"wrote weights.json ({mb:.1f} MB) — drop it next to index.html")

if __name__ == "__main__":
    main()
