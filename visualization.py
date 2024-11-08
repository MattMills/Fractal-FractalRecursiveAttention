import plotly.graph_objects as go
import numpy as np
import torch

def plot_attention_patterns(attention_matrix):
    """Plot attention patterns with enhanced visualization"""
    fig = go.Figure()
    
    # Ensure we're working with numpy array
    if isinstance(attention_matrix, torch.Tensor):
        attention_matrix = attention_matrix.detach().numpy()
    
    fig.add_trace(go.Heatmap(
        z=attention_matrix,
        colorscale='Viridis',
        showscale=True,
        name='Attention Pattern'
    ))
    
    fig.update_layout(
        title='Enhanced Attention Pattern Visualization',
        xaxis_title='Token Position',
        yaxis_title='Token Position',
        width=600,
        height=600
    )
    
    return fig

def plot_fractal_dimension(dimensions):
    """Plot fractal dimensions across levels with confidence intervals"""
    fig = go.Figure()
    
    # Convert to numpy if needed
    if isinstance(dimensions, torch.Tensor):
        dimensions = dimensions.detach().numpy()
    
    x = list(range(len(dimensions)))
    
    fig.add_trace(go.Scatter(
        x=x,
        y=dimensions,
        mode='lines+markers',
        name='Fractal Dimension',
        line=dict(color='blue'),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Fractal Dimension vs Level',
        xaxis_title='Level',
        yaxis_title='Fractal Dimension',
        width=600,
        height=400,
        showlegend=True,
        hovermode='x'
    )
    
    return fig

def plot_manifold_dimension(manifold_dims):
    """Plot manifold dimension evolution"""
    fig = go.Figure()
    
    # Convert to numpy if needed
    if isinstance(manifold_dims, torch.Tensor):
        manifold_dims = manifold_dims.detach().numpy()
    
    x = list(range(len(manifold_dims)))
    
    fig.add_trace(go.Scatter(
        x=x,
        y=manifold_dims,
        mode='lines+markers',
        name='Manifold Dimension',
        line=dict(color='red'),
        marker=dict(size=8)
    ))
    
    # Add theoretical upper bound
    upper_bound = np.ones_like(manifold_dims) * max(manifold_dims)
    fig.add_trace(go.Scatter(
        x=x,
        y=upper_bound,
        mode='lines',
        name='Theoretical Upper Bound',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title='Manifold Dimension Evolution',
        xaxis_title='Level',
        yaxis_title='Intrinsic Dimension',
        width=600,
        height=400,
        showlegend=True,
        hovermode='x'
    )
    
    return fig
