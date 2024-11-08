import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple
import plotly.graph_objects as go
from enhanced_fractal_attention import EnhancedFractalAttention, AttentionMetrics

@dataclass
class ComparisonResult:
    similarity_score: float
    pattern_differences: torch.Tensor
    metric_differences: dict
    convergence_rate: float

class AttentionComparator:
    def __init__(self):
        self.eps = 1e-8

    def compare_attention_patterns(self, 
                                 pattern1: torch.Tensor, 
                                 pattern2: torch.Tensor) -> ComparisonResult:
        """Compare two attention patterns and compute similarity metrics"""
        if isinstance(pattern1, np.ndarray):
            pattern1 = torch.tensor(pattern1)
        if isinstance(pattern2, np.ndarray):
            pattern2 = torch.tensor(pattern2)
            
        # Compute pattern similarity using cosine similarity
        similarity = F.cosine_similarity(
            pattern1.flatten().unsqueeze(0),
            pattern2.flatten().unsqueeze(0)
        ).item()
        
        # Compute pattern differences
        differences = torch.abs(pattern1 - pattern2)
        
        # Compute convergence rate using the ratio of eigenvalues
        conv_rate = self._compute_convergence_rate(pattern1, pattern2)
        
        return ComparisonResult(
            similarity_score=similarity,
            pattern_differences=differences,
            metric_differences={},  # Will be populated by compare_metrics
            convergence_rate=conv_rate
        )
    
    def compare_metrics(self, 
                       metrics1: AttentionMetrics, 
                       metrics2: AttentionMetrics) -> dict:
        """Compare two sets of attention metrics"""
        return {
            'information_content_diff': abs(metrics1.information_content - metrics2.information_content),
            'attention_conservation_diff': abs(metrics1.attention_conservation - metrics2.attention_conservation),
            'fractal_dimension_diff': abs(metrics1.fractal_dimension - metrics2.fractal_dimension),
            'manifold_dimension_diff': abs(metrics1.manifold_dimension - metrics2.manifold_dimension)
        }
    
    def _compute_convergence_rate(self, 
                                pattern1: torch.Tensor, 
                                pattern2: torch.Tensor) -> float:
        """Compute the convergence rate between two patterns"""
        try:
            # Use singular values to estimate convergence
            s1 = torch.linalg.svdvals(pattern1)
            s2 = torch.linalg.svdvals(pattern2)
            
            # Compute ratio of largest singular values
            ratio = (s1[0] + self.eps) / (s2[0] + self.eps)
            return float(torch.abs(1 - ratio))
        except:
            return 1.0

def plot_comparison(result: ComparisonResult) -> go.Figure:
    """Create a comparative visualization of attention patterns"""
    fig = go.Figure()
    
    # Add difference heatmap
    fig.add_trace(go.Heatmap(
        z=result.pattern_differences.numpy(),
        colorscale='RdBu',
        zmid=0,
        name='Pattern Differences'
    ))
    
    # Add annotations for metrics
    annotations = [
        f"Similarity Score: {result.similarity_score:.3f}",
        f"Convergence Rate: {result.convergence_rate:.3f}"
    ]
    
    for i, metric in enumerate(result.metric_differences.items()):
        name, value = metric
        annotations.append(f"{name}: {value:.3f}")
    
    fig.update_layout(
        title='Attention Pattern Comparison',
        xaxis_title='Token Position',
        yaxis_title='Token Position',
        annotations=[
            dict(
                text='<br>'.join(annotations),
                xref="paper",
                yref="paper",
                x=1.02,
                y=0.98,
                showarrow=False,
                font=dict(size=12)
            )
        ],
        width=800,
        height=600
    )
    
    return fig

def analyze_temporal_evolution(attention_patterns: List[torch.Tensor]) -> Tuple[List[float], List[float]]:
    """Analyze the temporal evolution of attention patterns"""
    comparator = AttentionComparator()
    
    similarities = []
    convergence_rates = []
    
    for i in range(len(attention_patterns) - 1):
        result = comparator.compare_attention_patterns(
            attention_patterns[i],
            attention_patterns[i + 1]
        )
        similarities.append(result.similarity_score)
        convergence_rates.append(result.convergence_rate)
    
    return similarities, convergence_rates

def plot_evolution_metrics(similarities: List[float], 
                         convergence_rates: List[float]) -> go.Figure:
    """Plot the evolution of comparison metrics over time"""
    fig = go.Figure()
    
    # Add similarity trace
    fig.add_trace(go.Scatter(
        y=similarities,
        name='Pattern Similarity',
        line=dict(color='blue')
    ))
    
    # Add convergence rate trace
    fig.add_trace(go.Scatter(
        y=convergence_rates,
        name='Convergence Rate',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Temporal Evolution of Attention Patterns',
        xaxis_title='Time Step',
        yaxis_title='Metric Value',
        yaxis_range=[0, 1],
        width=800,
        height=400,
        showlegend=True
    )
    
    return fig
