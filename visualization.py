import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import torch
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

def create_error_figure(error_message):
    return go.Figure().update_layout(
        annotations=[{
            'text': f"Error: {error_message}",
            'xref': "paper",
            'yref': "paper",
            'showarrow': False,
            'font': {'size': 14}
        }],
        height=400
    )

def safe_plot_wrapper(plot_func):
    def wrapper(*args, **kwargs):
        try:
            fig = plot_func(*args, **kwargs)
            # Ensure proper initialization
            if hasattr(fig, 'update_layout'):
                fig.update_layout(
                    modebar_remove=['sendDataToCloud'],
                    hovermode='closest',
                    uirevision=True
                )
            return fig
        except Exception as e:
            print(f"Error in plotting: {str(e)}")
            return create_error_figure(str(e))
    return wrapper

@safe_plot_wrapper
def plot_attention_patterns(attention_matrix):
    try:
        if isinstance(attention_matrix, torch.Tensor):
            attention_matrix = attention_matrix.detach().numpy()
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add enhanced heatmap with error handling
        fig.add_trace(go.Heatmap(
            z=attention_matrix,
            colorscale='Viridis',
            showscale=True,
            name='Attention Pattern'
        ))
        
        # Ensure proper cleanup
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
            template='plotly_white',
            uirevision=True,
            hovermode='closest'
        )
        
        return fig
    except Exception as e:
        print(f"Error in attention pattern plotting: {str(e)}")
        return create_error_figure(str(e))

@safe_plot_wrapper
def plot_3d_attention(attention_matrix):
    try:
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
            height=800,
            uirevision=True
        )
        
        return fig
    except Exception as e:
        print(f"Error in 3D attention plotting: {str(e)}")
        return create_error_figure(str(e))

@safe_plot_wrapper
def plot_hierarchical_attention(attention_matrix):
    try:
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
            name='Clustered Attention'
        ))
        
        fig.update_layout(
            title='Hierarchically Clustered Attention Patterns',
            xaxis_title='Clustered Token Position',
            yaxis_title='Clustered Token Position',
            width=800,
            height=800,
            template='plotly_white',
            uirevision=True
        )
        
        return fig
    except Exception as e:
        print(f"Error in hierarchical attention plotting: {str(e)}")
        return create_error_figure(str(e))

@safe_plot_wrapper
def plot_attention_evolution(attention_matrices):
    try:
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
            }],
            uirevision=True
        )
        
        return fig
    except Exception as e:
        print(f"Error in attention evolution plotting: {str(e)}")
        return create_error_figure(str(e))

@safe_plot_wrapper
def plot_fractal_dimension(dimensions):
    try:
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
            hovermode='x unified',
            uirevision=True
        )
        
        return fig
    except Exception as e:
        print(f"Error in fractal dimension plotting: {str(e)}")
        return create_error_figure(str(e))

@safe_plot_wrapper
def plot_manifold_dimension(manifold_dims):
    try:
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
        
        fig.update_layout(
            title={
                'text': 'Manifold Dimension Evolution',
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
            hovermode='x unified',
            uirevision=True
        )
        
        return fig
    except Exception as e:
        print(f"Error in manifold dimension plotting: {str(e)}")
        return create_error_figure(str(e))
