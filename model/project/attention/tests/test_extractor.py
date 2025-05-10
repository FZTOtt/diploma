import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
from attention.models.extractor import Extractor

def test_extractor_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Extractor().to(device)
    
    # подаём на вход стего-изображение
    stego = torch.randn(2, 3, 256, 256).to(device)
    out = model(stego)
    
    assert out.shape == (2, 3, 256, 256), \
        f"Unexpected output shape: {out.shape}"
    print("✅ Extractor forward pass OK. Output shape:", out.shape)

if __name__ == "__main__":
    test_extractor_forward()
