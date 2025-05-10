import matplotlib.pyplot as plt

def visualize_progress(cover, stego, secret, secret_recon):
    """Visualization of intermediate results"""
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    # Cover vs Stego
    axs[0,0].imshow(cover.squeeze(), cmap='gray')
    axs[0,0].set_title('Cover Y')
    axs[0,1].imshow(stego.squeeze(), cmap='gray')
    axs[0,1].set_title('Stego Y')
    
    # Secret vs Reconstructed
    axs[1,0].imshow(secret.permute(1,2,0))
    axs[1,0].set_title('Original Secret')
    axs[1,1].imshow(secret_recon.permute(1,2,0))
    axs[1,1].set_title('Reconstructed Secret')
    
    for ax in axs.flatten():
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()