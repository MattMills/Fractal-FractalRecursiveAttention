import streamlit as st
import numpy as np
import plotly.graph_objects as go
import torch
import json
from enhanced_fractal_attention import EnhancedFractalAttention
from visualization import plot_attention_patterns, plot_fractal_dimension, plot_manifold_dimension
from utils import generate_sample_data, render_latex_equations

st.set_page_config(page_title="Fractal-Fractal Recompressive Attention Visualizer", layout="wide")

st.title("Fractal-Fractal Recompressive Attention Visualizer")

@st.cache_data
def compute_attention_patterns(input_data, max_depth):
    try:
        attention = EnhancedFractalAttention(input_data, max_depth=max_depth)
        output = attention(attention.X)
        metrics = attention.get_metrics()
        return {
            'output': output.detach().numpy(),
            'metrics': {
                'information_content': float(metrics.information_content),
                'attention_conservation': float(metrics.attention_conservation),
                'fractal_dimension': float(metrics.fractal_dimension),
                'manifold_dimension': float(metrics.manifold_dimension)
            }
        }
    except Exception as e:
        st.error(f"Error computing attention patterns: {str(e)}")
        return None

@st.cache_data
def analyze_fractal_dimensions(input_data, max_depth):
    try:
        dimensions = []
        manifold_dims = []
        for level in range(max_depth):
            attention = EnhancedFractalAttention(input_data, level=level, max_depth=max_depth)
            metrics = attention.get_metrics()
            dimensions.append(metrics.fractal_dimension)
            manifold_dims.append(metrics.manifold_dimension)
        return dimensions, manifold_dims
    except Exception as e:
        st.error(f"Error analyzing dimensions: {str(e)}")
        return None, None

# Sidebar controls
st.sidebar.header("Parameters")
max_depth = st.sidebar.slider("Maximum Depth", 1, 10, 3)
input_dim = st.sidebar.slider("Input Dimension", 8, 64, 32)
attention_threshold = st.sidebar.slider("Attention Threshold", 0.0, 1.0, 0.5)

# Generate sample data
input_data = generate_sample_data(input_dim)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Mathematical Formalization")
    render_latex_equations()
    
    st.subheader("Attention Patterns")
    if st.button("Compute Attention"):
        with st.spinner("Computing attention patterns..."):
            result = compute_attention_patterns(input_data, max_depth)
            if result:
                fig = plot_attention_patterns(result['output'])
                st.plotly_chart(fig, use_container_width=True)
                
                st.json(result['metrics'])

with col2:
    st.subheader("Fractal Properties")
    if st.button("Analyze Fractal Dimensions"):
        with st.spinner("Analyzing fractal dimensions..."):
            dimensions, manifold_dims = analyze_fractal_dimensions(input_data, max_depth)
            if dimensions and manifold_dims:
                fig_fractal = plot_fractal_dimension(dimensions)
                st.plotly_chart(fig_fractal, use_container_width=True)
                
                fig_manifold = plot_manifold_dimension(manifold_dims)
                st.plotly_chart(fig_manifold, use_container_width=True)

    st.subheader("Information Conservation")
    if st.button("Verify Properties"):
        with st.spinner("Verifying mathematical properties..."):
            try:
                attention = EnhancedFractalAttention(input_data, max_depth=max_depth)
                metrics = attention.get_metrics()
                
                st.write("Enhanced Metrics:")
                st.json({
                    'Information Content': float(metrics.information_content),
                    'Attention Conservation': float(metrics.attention_conservation),
                    'Fractal Dimension': float(metrics.fractal_dimension),
                    'Manifold Dimension': float(metrics.manifold_dimension)
                })
            except Exception as e:
                st.error(f"Error verifying properties: {str(e)}")
