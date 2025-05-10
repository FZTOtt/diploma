import torch

class ChaosPermutation:
    def __init__(self, mu=3.9, x0=0.1, iterations=1000):
        self.mu = mu          
        self.x0 = x0          
        self.iterations = iterations
        
    def generate_chaos_sequence(self, size):
        seq = []
        x = self.x0
        for _ in range(self.iterations):
            x = self.mu * x * (1 - x)
        for _ in range(size):
            x = self.mu * x * (1 - x)
            seq.append(x)
        return torch.tensor(seq)

    def permute(self, image):
        c, h, w = image.shape
        sub_imgs = [
            image[:, 0:h:2, 0:w:2],  # (odd, odd)
            image[:, 0:h:2, 1:w:2],  # (odd, even)
            image[:, 1:h:2, 0:w:2],  # (even, odd)
            image[:, 1:h:2, 1:w:2]   # (even, even)
        ]
        
        permuted_subimgs = []
        for img in sub_imgs:
            seq = self.generate_chaos_sequence(img.numel())
            indices = torch.argsort(seq)
            flat = img.flatten()
            permuted = flat[indices].view_as(img)
            permuted_subimgs.append(permuted)
        
        permuted_subimgs = [permuted_subimgs[i] for i in [1, 3, 0, 2]]  # Example permutation
        
        output = torch.zeros_like(image)
        output[:, 0:h:2, 0:w:2] = permuted_subimgs[0]
        output[:, 0:h:2, 1:w:2] = permuted_subimgs[1]
        output[:, 1:h:2, 0:w:2] = permuted_subimgs[2]
        output[:, 1:h:2, 1:w:2] = permuted_subimgs[3]
        
        return output