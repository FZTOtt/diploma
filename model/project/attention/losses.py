import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 1. Concealment (hiding) loss
# ----------------------------
# L₂-расстояние между cover и stego
mse_loss = nn.MSELoss()

def concealment_loss(cover: torch.Tensor, stego: torch.Tensor) -> torch.Tensor:
    """
    LC = ||cover - stego||₂²
    """
    return mse_loss(stego, cover)


# ----------------------------
# 2. Revealing (extraction) loss
# ----------------------------
# L₂-расстояние между secret и reconstructed secret
def reveal_loss(secret: torch.Tensor, extracted: torch.Tensor) -> torch.Tensor:
    """
    LR = ||secret - extracted||₂²
    """
    return mse_loss(extracted, secret)


# ----------------------------
# 3. Adversarial losses (standard GAN)
# ----------------------------
bce_loss = nn.BCEWithLogitsLoss()

def generator_adversarial_loss(disc_logits_fake: torch.Tensor) -> torch.Tensor:
    """
    LG = BCE(D(stego), 1)
    Стимулирует генератор делать стего настолько реалистичным,
    чтобы дискриминатор ставил метку «реальное» (1).
    """
    target_real = torch.ones_like(disc_logits_fake)
    return F.binary_cross_entropy_with_logits(disc_logits_fake, target_real)

def discriminator_adversarial_loss(
    disc_logits_real: torch.Tensor,
    disc_logits_fake: torch.Tensor
) -> torch.Tensor:
    """
    LD = BCE(D(cover), 1) + BCE(D(stego).detach(), 0)
    Дискриминатор учится ставить «реальное» (1) для cover
    и «фейковое» (0) для stego.
    """
    target_real = torch.ones_like(disc_logits_real)
    target_fake = torch.zeros_like(disc_logits_fake)
    loss_real = F.binary_cross_entropy_with_logits(disc_logits_real, target_real)
    loss_fake = F.binary_cross_entropy_with_logits(disc_logits_fake, target_fake)
    return loss_real + loss_fake


# ----------------------------
# 4. Composite generator loss
# ----------------------------
def total_generator_loss(
    cover: torch.Tensor,
    stego: torch.Tensor,
    secret: torch.Tensor,
    extracted: torch.Tensor,
    disc_logits_fake: torch.Tensor,
    lambda_conceal: float = 0.7,
    lambda_adv: float   = 0.001
) -> torch.Tensor:
    """
    L_total = λ1·LC + LR + λ2·LG
    По умолчанию: λ1=0.7, λ2=0.001
    """
    lc = concealment_loss(cover, stego)
    lr = reveal_loss(secret, extracted)
    lg = generator_adversarial_loss(disc_logits_fake)
    return lambda_conceal * lc + lr + lambda_adv * lg
