import torch.nn.functional as F

def hiding_loss(stego, cover):
    """ Потеря скрытия: L_hid """
    print('L_hid', stego.shape, cover.shape)
    return F.mse_loss(stego, cover)

def reconstruction_loss(secret, secret_recon):
    """ Потеря восстановления: L_rec """
    print('L_rec', secret.shape, secret_recon.shape)
    return F.l1_loss(secret, secret_recon)