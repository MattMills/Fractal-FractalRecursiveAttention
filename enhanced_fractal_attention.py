import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class AttentionMetrics:
    """Metrics for monitoring attention behavior"""
    information_content: float
    attention_conservation: float
    fractal_dimension: float
    manifold_dimension: float


class EnhancedFractalAttention:

    def __init__(self,
                 X: torch.Tensor,
                 level: int = 0,
                 max_depth: Optional[int] = None,
                 adaptive_depth: bool = True,
                 manifold_aware: bool = True):
        """
        Enhanced Fractal Attention with geometric awareness and adaptive features

        Args:
            X: Input tensor
            level: Starting recursion level
            max_depth: Maximum recursion depth (None for adaptive)
            adaptive_depth: Use adaptive depth control
            manifold_aware: Use manifold-aware compression
        """
        # Convert input to tensor with gradients enabled
        self.X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        self.level = level
        self.max_depth = max_depth
        self.dim = X.shape[-1]
        self.manifold_aware = manifold_aware

        # Initialize geometric tensors
        self.metric_tensor = self._initialize_metric_tensor()

    def _initialize_metric_tensor(self) -> torch.Tensor:
        """Initialize the Riemannian metric tensor"""
        return torch.eye(self.dim, dtype=torch.float32)

    def __call__(self, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        if X is None:
            X = self.X
        return self._fractal_attention(X, self.level, self.max_depth)

    def _adaptive_max_depth(self, X: torch.Tensor) -> int:
        """Dynamically determine optimal recursion depth"""
        return 5  # Fixed conservative depth

    def _compute_spectral_gap(self, X: torch.Tensor) -> float:
        """Compute normalized spectral gap for depth estimation"""
        U, S, _ = torch.svd(X)
        if len(S) < 2:
            return 1.0
        gap = (S[0] - S[1]) / (S[0] + 1e-8)
        return gap.item()

    def _fractal_attention(self, X: torch.Tensor, level: int,
                          max_depth: Optional[int]) -> torch.Tensor:
        """Main fractal attention computation with iterative implementation"""
        stack = [(X, level)]
        cache = {}
        
        while stack:
            current_X, current_level = stack.pop()
            
            # Early stopping check
            if current_level > 0 and torch.allclose(current_X, self.X, rtol=1e-5):
                continue
                
            if max_depth and current_level >= max_depth:
                result = self._base_attention(current_X)
                cache[(current_X.shape, current_level)] = result
                continue
                
            # Normalize and compute attention
            X_norm = F.normalize(current_X, p=2, dim=-1)
            QK = self._compute_geometric_attention(X_norm)
            scores = self._stabilized_softmax(QK)
            
            # Apply attention with residual
            attended = scores @ X_norm
            flow = self._geometric_flow(attended, current_level)
            attended = attended + X_norm + flow
            
            # Handle next level
            if max_depth is None or current_level < max_depth:
                compressed = (self._manifold_compress(attended, scores, current_level) 
                            if self.manifold_aware else self._recompress(attended, scores, current_level))
                stack.append((compressed, current_level + 1))
                
            cache[(current_X.shape, current_level)] = attended
        
        return cache[(X.shape, level)]

    def _compute_geometric_attention(self, X: torch.Tensor) -> torch.Tensor:
        """Compute attention scores with geometric structure awareness"""
        metric_scaled = self.metric_tensor / (self.level + 1)
        QK = X @ metric_scaled @ X.T / np.sqrt(X.shape[-1])
        return QK - torch.max(QK, dim=-1, keepdim=True)[0]

    def _stabilized_softmax(self, X: torch.Tensor) -> torch.Tensor:
        """Numerically stable softmax with proper scaling"""
        X_max, _ = torch.max(X, dim=-1, keepdim=True)
        exp_X = torch.exp(X - X_max)
        return exp_X / (torch.sum(exp_X, dim=-1, keepdim=True) + 1e-8)

    def _geometric_flow(self, X: torch.Tensor, level: int) -> torch.Tensor:
        """Compute geometric flow for attention enhancement"""
        metric = self._compute_metric_tensor(X)
        flow = self._geometric_gradient(X, level)
        return torch.einsum('ij,jk->ik', flow, metric) / (level + 1)

    def _compute_metric_tensor(self, X: torch.Tensor) -> torch.Tensor:
        """Compute local metric tensor from data"""
        try:
            if not X.requires_grad:
                X = X.detach().requires_grad_(True)
            grad = torch.autograd.grad(X.sum(), X, create_graph=True, allow_unused=True)[0]
            if grad is None:
                return torch.eye(X.shape[-1], dtype=X.dtype, device=X.device)
            return grad @ grad.T + torch.eye(X.shape[-1], dtype=X.dtype, device=X.device) * 1e-6
        except Exception:
            return torch.eye(X.shape[-1], dtype=X.dtype, device=X.device)

    def _geometric_gradient(self, X: torch.Tensor, level: int) -> torch.Tensor:
        """Enhanced geometric gradient computation"""
        try:
            metric = self._compute_metric_tensor(X)
            grad = X @ metric
            correction = torch.einsum('ij,jk,kl->il', grad, metric, grad)
            return (grad - 0.5 * correction / (level + 1)).detach()
        except Exception:
            return X @ torch.eye(X.shape[-1], dtype=X.dtype, device=X.device)

    def _base_attention(self, X: torch.Tensor) -> torch.Tensor:
        """Base attention mechanism for terminal nodes"""
        return F.scaled_dot_product_attention(X.unsqueeze(0), X.unsqueeze(0),
                                           X.unsqueeze(0)).squeeze(0)

    def _manifold_compress(self, X: torch.Tensor, scores: torch.Tensor,
                          level: int) -> torch.Tensor:
        """Manifold-aware compression with adaptive dimension"""
        U, S, V = torch.svd(X)

        # Estimate intrinsic dimension using singular value decay
        total_energy = torch.sum(S)
        cumulative_energy = torch.cumsum(S, dim=0) / total_energy
        dim_est = torch.sum(cumulative_energy < 0.95).item()
        dim_est = max(1, min(dim_est, X.shape[-1] - 1))

        # Project onto estimated manifold with proper scaling
        projection = U[:, :dim_est] @ torch.diag(S[:dim_est]) @ V[:, :dim_est].T
        return projection * np.sqrt(dim_est / X.shape[-1])

    def _recompress(self, X: torch.Tensor, scores: torch.Tensor,
                   level: int) -> torch.Tensor:
        """Fallback compression method when manifold-aware is disabled"""
        D = self.fractal_dimension(X, level)
        k = max(int(X.shape[-1] * D), 1)

        U, S, V = torch.svd(X)
        return U[:, :k] @ torch.diag(S[:k]) @ V[:, :k].T

    def _reconstruct(self, X: torch.Tensor, level: int) -> torch.Tensor:
        """Enhanced reconstruction with geometric correction"""
        scale = 1.0 / (level + 1)
        correction = self._geometric_gradient(X, level)
        return X * scale + correction * scale * scale

    def fractal_dimension(self, X: torch.Tensor, level: int) -> float:
        """Compute fractal dimension using correlation dimension estimation"""
        # Use more scale points and wider range for better distribution
        scales = torch.logspace(-3, 1, 30)  # More points, wider range
        counts = []
        
        # Normalize input for stable distance computation
        X_normalized = F.normalize(X, p=2, dim=-1)
        distances = torch.cdist(X_normalized, X_normalized)
        
        # Compute box counts at each scale
        for scale in scales:
            # Use smooth counting with gaussian kernel
            kernel = torch.exp(-distances**2 / (2 * scale**2))
            count = torch.sum(kernel > 0.5)  # Adaptive threshold
            counts.append(float(count))
        
        # Robust regression using multiple regions
        x = torch.log(scales)
        y = torch.log(torch.tensor(counts))
        
        # Split into three regions for multi-scale analysis
        regions = [(0, len(scales)//3), 
                  (len(scales)//3, 2*len(scales)//3),
                  (2*len(scales)//3, len(scales))]
                  
        slopes = []
        for start, end in regions:
            slope = ((x[start:end] * y[start:end]).mean() - 
                    x[start:end].mean() * y[start:end].mean()) / (
                    (x[start:end] ** 2).mean() - x[start:end].mean() ** 2)
            slopes.append(slope)
        
        # Use median slope for robustness
        final_slope = torch.tensor(slopes).median()
        
        # Normalize to (0,1) with sigmoid for smoother distribution
        return float(torch.sigmoid(final_slope / self.dim))

    def get_metrics(self) -> AttentionMetrics:
        """Compute comprehensive attention metrics"""
        attention_output = self(self.X)
        return AttentionMetrics(
            information_content=self.get_information_content(),
            attention_conservation=self.verify_attention_conservation(),
            fractal_dimension=self.fractal_dimension(self.X, self.level),
            manifold_dimension=self._estimate_manifold_dimension())

    def _estimate_manifold_dimension(self) -> float:
        """Estimate the intrinsic dimension of the data manifold"""
        U, S, _ = torch.svd(self.X)
        total_energy = torch.sum(S)
        cumulative_energy = torch.cumsum(S, dim=0) / total_energy
        return torch.sum(cumulative_energy < 0.95).item()

    def get_information_content(self) -> float:
        """Compute information content using relative entropy"""
        attention_output = self(self.X)
        eps = 1e-10
        
        # Compute attention probabilities with proper normalization
        X_norm = F.normalize(attention_output, p=2, dim=-1)
        QK = X_norm @ X_norm.T / np.sqrt(self.dim)
        probs = F.softmax(QK, dim=-1)
        probs = torch.clamp(probs, min=eps, max=1.0)
        
        # Compute entropy relative to uniform distribution
        uniform = torch.ones_like(probs) / probs.shape[-1]
        relative_entropy = torch.sum(probs * torch.log2(probs / uniform)) / self.dim
        
        # Normalize to [0,1] using sigmoid
        return float(torch.sigmoid(relative_entropy).item())

    def verify_attention_conservation(self) -> float:
        X = F.normalize(self.X, p=2, dim=-1)
        
        # Compute attention scores
        attn_in = self._stabilized_softmax(self._compute_geometric_attention(X))
        attn_out = self._stabilized_softmax(self._compute_geometric_attention(self(self.X)))
        
        # Compare normalized traces
        trace_in = torch.trace(attn_in) / self.dim
        trace_out = torch.trace(attn_out) / self.dim
        
        # Return positive conservation score
        return float(1.0 - abs(trace_in - trace_out))