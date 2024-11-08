import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import torch
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

def plot_attention_patterns(attention_matrix):
    """Plot attention patterns with enhanced visualization"""
    if isinstance(attention_matrix, torch.Tensor):
        attention_matrix = attention_matrix.detach().numpy()
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add enhanced heatmap
    fig.add_trace(go.Heatmap(
        z=attention_matrix,
        colorscale='Viridis',
        showscale=True,
        name='Attention Pattern',
        customdata=np.round(attention_matrix, 3),
        hovertemplate='Row: %{y}<br>Column: %{x}<br>Value: %{customdata}<extra></extra>'
    ))
    
    # Add contour patterns to highlight attention flow
    fig.add_trace(go.Contour(
        z=attention_matrix,
        colorscale='RdBu',
        showscale=False,
        opacity=0.3,
        name='Attention Flow'
    ))
    
    fig.update_layout(
        title={
            'text': 'Enhanced Attention Pattern Visualization',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Token Position',
        yaxis_title='Token Position',
        width=800,
        height=800,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def plot_3d_attention(attention_matrix):
    """Create 3D visualization of attention patterns"""
    if isinstance(attention_matrix, torch.Tensor):
        attention_matrix = attention_matrix.detach().numpy()
    
    x, y = np.meshgrid(
        np.arange(attention_matrix.shape[0]),
        np.arange(attention_matrix.shape[1])
    )
    
    fig = go.Figure(data=[
        go.Surface(
            x=x,
            y=y,
            z=attention_matrix,
            colorscale='Viridis',
            name='Attention Surface'
        )
    ])
    
    fig.update_layout(
        title='3D Attention Pattern Surface',
        scene={
            'xaxis_title': 'Token Position X',
            'yaxis_title': 'Token Position Y',
            'zaxis_title': 'Attention Strength',
            'camera': dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        },
        width=800,
        height=800
    )
    
    return fig

def plot_hierarchical_attention(attention_matrix):
    """Plot attention patterns with hierarchical clustering"""
    if isinstance(attention_matrix, torch.Tensor):
        attention_matrix = attention_matrix.detach().numpy()
    
    # Compute linkage matrix for hierarchical clustering
    distance_matrix = pdist(attention_matrix)
    linkage_matrix = hierarchy.linkage(distance_matrix, method='ward')
    
    # Get ordering of points from hierarchical clustering
    dendro_idx = hierarchy.dendrogram(linkage_matrix, no_plot=True)['leaves']
    
    # Reorder the attention matrix
    reordered_matrix = attention_matrix[dendro_idx][:, dendro_idx]
    
    fig = go.Figure()
    
    # Add hierarchically clustered heatmap
    fig.add_trace(go.Heatmap(
        z=reordered_matrix,
        colorscale='Viridis',
        showscale=True,
        name='Clustered Attention',
        customdata=np.round(reordered_matrix, 3),
        hovertemplate='Cluster Row: %{y}<br>Cluster Column: %{x}<br>Value: %{customdata}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Hierarchically Clustered Attention Patterns',
        xaxis_title='Clustered Token Position',
        yaxis_title='Clustered Token Position',
        width=800,
        height=800,
        template='plotly_white'
    )
    
    return fig

def plot_attention_evolution(attention_matrices):
    """Plot time evolution of attention patterns"""
    if isinstance(attention_matrices, torch.Tensor):
        attention_matrices = attention_matrices.detach().numpy()
    
    frames = []
    for i in range(attention_matrices.shape[0]):
        frames.append(
            go.Frame(
                data=[go.Heatmap(
                    z=attention_matrices[i],
                    colorscale='Viridis'
                )],
                name=f'frame{i}'
            )
        )
    
    fig = go.Figure(
        data=[go.Heatmap(z=attention_matrices[0], colorscale='Viridis')],
        frames=frames
    )
    
    fig.update_layout(
        title='Attention Pattern Evolution',
        xaxis_title='Token Position',
        yaxis_title='Token Position',
        width=800,
        height=800,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {'label': 'Play',
                 'method': 'animate',
                 'args': [None, {'frame': {'duration': 500, 'redraw': True},
                                'fromcurrent': True}]},
                {'label': 'Pause',
                 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                  'mode': 'immediate', 'transition': {'duration': 0}}]}
            ]
        }]
    )
    
    return fig

def plot_fractal_dimension(dimensions):
    """Plot fractal dimensions across levels with confidence intervals"""
    if isinstance(dimensions, torch.Tensor):
        dimensions = dimensions.detach().numpy()
    
    x = list(range(len(dimensions)))
    
    fig = go.Figure()
    
    # Main dimension line
    fig.add_trace(go.Scatter(
        x=x,
        y=dimensions,
        mode='lines+markers',
        name='Fractal Dimension',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    # Add confidence intervals
    std_dev = np.std(dimensions) * 0.1
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=np.concatenate([dimensions + std_dev, (dimensions - std_dev)[::-1]]),
        fill='toself',
        fillcolor='rgba(0,0,255,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        title={
            'text': 'Fractal Dimension Evolution',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Level',
        yaxis_title='Fractal Dimension',
        width=800,
        height=600,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def plot_manifold_dimension(manifold_dims):
    """Plot manifold dimension evolution with theoretical bounds"""
    if isinstance(manifold_dims, torch.Tensor):
        manifold_dims = manifold_dims.detach().numpy()
    
    x = list(range(len(manifold_dims)))
    
    fig = go.Figure()
    
    # Main dimension line
    fig.add_trace(go.Scatter(
        x=x,
        y=manifold_dims,
        mode='lines+markers',
        name='Manifold Dimension',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))
    
    # Theoretical bounds
    upper_bound = np.ones_like(manifold_dims) * max(manifold_dims)
    lower_bound = np.ones_like(manifold_dims) * min(manifold_dims)
    
    fig.add_trace(go.Scatter(
        x=x,
        y=upper_bound,
        mode='lines',
        name='Upper Bound',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=x,
        y=lower_bound,
        mode='lines',
        name='Lower Bound',
        line=dict(color='gray', dash='dot')
    ))
    
    # Add trend line
    z = np.polyfit(x, manifold_dims, 2)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=x,
        y=p(x),
        mode='lines',
        name='Trend',
        line=dict(color='purple', dash='dashdot')
    ))
    
    fig.update_layout(
        title={
            'text': 'Manifold Dimension Evolution with Bounds',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Level',
        yaxis_title='Intrinsic Dimension',
        width=800,
        height=600,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig
