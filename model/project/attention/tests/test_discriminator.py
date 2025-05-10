import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
from attention.models.discriminator import Discriminator

def test_discriminator_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Discriminator().to(device)

    cover = torch.randn(2, 3, 256, 256).to(device)
    stego = torch.randn(2, 3, 256, 256).to(device)

    logits = model(cover, stego)
    assert logits.shape == (2, 1), f"Unexpected logits shape: {logits.shape}"
    print("✅ Discriminator forward pass OK. Logits shape:", logits.shape)

if __name__ == "__main__":
    test_discriminator_forward()