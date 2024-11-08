import plotly.graph_objects as go
import numpy as np

def plot_attention_patterns(attention):
    # Create heatmap of attention patterns
    fig = go.Figure()
    
    attention_matrix = attention(attention.X).detach().numpy()
    
    fig.add_trace(go.Heatmap(
        z=attention_matrix,
        colorscale='Viridis',
        showscale=True,
        name='Attention Pattern'
    ))
    
    fig.update_layout(
        title='Attention Pattern Visualization',
        xaxis_title='Token Position',
        yaxis_title='Token Position',
        width=600,
        height=600
    )
    
    return fig

def plot_fractal_dimension(dimensions):
    # Plot fractal dimensions across levels
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(dimensions))),
        y=dimensions,
        mode='lines+markers',
        name='Fractal Dimension'
    ))
    
    fig.update_layout(
        title='Fractal Dimension vs Level',
        xaxis_title='Level',
        yaxis_title='Fractal Dimension',
        width=600,
        height=400
    )
    
    return fig
