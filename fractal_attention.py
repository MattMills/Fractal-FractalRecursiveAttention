import numpy as np
import torch
import torch.nn.functional as F

class FractalAttention:
    def __init__(self, X, level=0, max_depth=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.level = level
        self.max_depth = max_depth
        self.dim = X.shape[-1]
        
    def __call__(self, X=None):
        if X is None:
            X = self.X
        return self._fractal_attention(X, self.level, self.max_depth)
    
    def _fractal_attention(self, X, level=0, max_depth=None):
        if max_depth and level >= max_depth:
            return self._base_attention(X)
        
        # Normalize input
        X = F.normalize(X, p=2, dim=-1)
        
        # Compute attention with improved stability
        QK = X @ X.T / np.sqrt(X.shape[-1])
        QK = QK - torch.max(QK, dim=-1, keepdim=True)[0]
        scores = F.softmax(QK, dim=-1)
        
        # Apply attention with residual connection
        attended = scores @ X
        attended = attended + X  # Residual connection
        
        # Recursive call with normalized input
        if max_depth is None or level < max_depth:
            compressed = self._recompress(attended, scores, level)
            sub_attention = self._fractal_attention(compressed, level+1, max_depth)
            return self._reconstruct(sub_attention, level)
        
        return attended
    
    def _base_attention(self, X):
        return F.scaled_dot_product_attention(X, X, X)
    
    def _geometric_gradient(self, X, level):
        # Implement Riemannian gradient on manifold
        metric = torch.eye(X.shape[-1]) * (1.0 / (level + 1))
        grad = X @ metric
        return grad
    
    def _recompress(self, X, scores, level):
        # Compute optimal compression dimension
        D = self.fractal_dimension(X, level)
        k = max(int(X.shape[-1] * D), 1)
        
        # SVD-based compression
        U, S, V = torch.svd(X)
        compressed = U[:, :k] @ torch.diag(S[:k]) @ V[:, :k].T
        
        return compressed
    
    def _reconstruct(self, X, level):
        # Inverse transform using level-specific scaling
        scale = 1.0 / (level + 1)
        return X * scale
    
    def fractal_dimension(self, X, level):
        # Box-counting dimension estimation
        eps = 1e-6
        scales = torch.logspace(-1, 1, 10)
        counts = []
        
        for scale in scales:
            boxes = torch.floor(X / scale)
            unique_boxes = torch.unique(boxes, dim=0)
            counts.append(len(unique_boxes))
        
        # Linear regression for dimension estimation
        x = torch.log(1/scales)
        y = torch.log(torch.tensor(counts, dtype=torch.float32))
        D = (x * y).mean() / (x * x).mean()
        
        return D.item()
    
    def get_information_content(self):
        # Estimate information content using entropy with proper normalization
        attention_output = self(self.X)
        
        # Ensure numerical stability
        eps = 1e-10
        attention_output = torch.clamp(attention_output, min=eps)
        
        # Normalize the output
        probs = F.softmax(attention_output, dim=-1)
        probs = torch.clamp(probs, min=eps, max=1.0)
        
        # Use log2 for better interpretability
        entropy = -torch.sum(probs * torch.log2(probs)) / torch.log2(torch.tensor(probs.shape[-1]))
        
        # Normalize to [0, 1] range
        normalized_entropy = entropy / self.dim
        
        return normalized_entropy.item()
    
    def verify_attention_conservation(self):
        # Get attention patterns
        X_norm = F.normalize(self.X, p=2, dim=-1)
        
        # Input attention
        QK_input = X_norm @ X_norm.T / np.sqrt(self.dim)
        attention_input = F.softmax(QK_input, dim=-1)
        
        # Output attention
        attention_output = self(self.X)
        attention_output = F.normalize(attention_output, p=2, dim=-1)
        
        # Row-wise normalization
        attention_input = F.normalize(attention_input, p=1, dim=-1)
        attention_output = F.normalize(attention_output, p=1, dim=-1)
        
        # Compare using Frobenius norm
        diff = torch.norm(attention_input - attention_output, p='fro')
        
        return diff < 1e-3  # Using Frobenius norm for comparison
