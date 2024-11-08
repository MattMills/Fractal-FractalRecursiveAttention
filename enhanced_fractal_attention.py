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
        self.X = torch.tensor(X, dtype=torch.float32)
        self.level = level
        self.max_depth = max_depth if not adaptive_depth else None
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
        depth = self._adaptive_max_depth(
            X) if self.max_depth is None else self.max_depth
        return self._fractal_attention(X, self.level, depth)

    def _adaptive_max_depth(self, X: torch.Tensor) -> int:
        """Dynamically determine optimal recursion depth"""
        info_content = self.get_information_content()
        spectral_gap = self._compute_spectral_gap(X)

        # Combine information content and spectral properties
        optimal_depth = max(1, int(-np.log2(info_content) * spectral_gap))
        return min(optimal_depth,
                   10)  # Cap maximum depth for computational feasibility

    def _compute_spectral_gap(self, X: torch.Tensor) -> float:
        """Compute normalized spectral gap for depth estimation"""
        U, S, _ = torch.svd(X)
        if len(S) < 2:
            return 1.0
        gap = (S[0] - S[1]) / (S[0] + 1e-8)
        return gap.item()

    def _fractal_attention(self, X: torch.Tensor, level: int,
                           max_depth: Optional[int]) -> torch.Tensor:
        """Main fractal attention computation with enhanced stability"""
        if max_depth and level >= max_depth:
            return self._base_attention(X)

        # Normalize input with improved numerical stability
        X = F.normalize(X, p=2, dim=-1)

        # Compute attention scores with geometric awareness
        QK = self._compute_geometric_attention(X)
        scores = self._stabilized_softmax(QK)

        # Apply attention with residual and geometric flow
        attended = scores @ X
        flow = self._geometric_flow(attended, level)
        attended = attended + X + flow  # Enhanced residual connection

        # Recursive call with manifold-aware compression
        if max_depth is None or level < max_depth:
            compressed = (self._manifold_compress(attended, scores, level)
                          if self.manifold_aware else self._recompress(
                              attended, scores, level))

            sub_attention = self._fractal_attention(compressed, level + 1,
                                                    max_depth)
            return self._reconstruct(sub_attention, level)

        return attended

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
        grad = torch.autograd.grad(X.sum(), X, create_graph=True)[0]
        return grad @ grad.T + torch.eye(X.shape[-1]) * 1e-6

    def _geometric_gradient(self, X: torch.Tensor, level: int) -> torch.Tensor:
        """Enhanced geometric gradient computation"""
        metric = self._compute_metric_tensor(X)
        grad = X @ metric
        # Add curvature correction
        correction = torch.einsum('ij,jk,kl->il', grad, metric, grad)
        return grad - 0.5 * correction / (level + 1)

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
        projection = U[:, :dim_est] @ torch.diag(
            S[:dim_est]) @ V[:, :dim_est].T
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
        """Improved fractal dimension estimation"""
        eps = 1e-6
        scales = torch.logspace(-1, 1, 20)
        counts = []

        X_normalized = F.normalize(X, p=2, dim=-1)
        for scale in scales:
            boxes = torch.floor(X_normalized / scale)
            unique_boxes = torch.unique(boxes, dim=0)
            counts.append(len(unique_boxes))

        x = torch.log(1 / scales)
        y = torch.log(torch.tensor(counts, dtype=torch.float32))

        # Robust linear regression with outlier handling
        mask = torch.abs(y - y.mean()) < 2 * y.std()
        D = ((x[mask] * y[mask]).mean() / (x[mask] * x[mask]).mean()).item()
        return max(0.0, min(D, 1.0))

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
        """Enhanced information content estimation"""
        attention_output = self(self.X)
        eps = 1e-10

        # Compute entropy with improved numerical stability
        probs = F.softmax(attention_output, dim=-1)
        probs = torch.clamp(probs, min=eps, max=1.0)
        entropy = -torch.sum(probs * torch.log2(probs)) / np.log2(
            probs.shape[-1])

        return entropy.item()

    def verify_attention_conservation(self) -> float:
        """Verify attention pattern conservation with improved metrics"""
        X = F.normalize(self.X, p=2, dim=-1)

        # Input attention
        QK_input = self._compute_geometric_attention(X)
        attention_input = self._stabilized_softmax(QK_input)

        # Output attention
        attention_output = F.normalize(self(self.X), p=2, dim=-1)

        # Compute relative error with proper scaling
        trace_input = torch.trace(attention_input) / self.dim
        trace_output = torch.trace(attention_output) / self.dim

        relative_error = torch.abs(trace_input - trace_output) / (
            torch.abs(trace_input) + 1e-8)
        return 1.0 - relative_error.item(
        )  # Return conservation score between 0 and 1
