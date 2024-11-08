import streamlit as st
import numpy as np
import plotly.graph_objects as go
from fractal_attention import FractalAttention
from visualization import plot_attention_patterns, plot_fractal_dimension
from utils import generate_sample_data, render_latex_equations

st.set_page_config(page_title="Fractal-Fractal Recompressive Attention Visualizer", layout="wide")

st.title("Fractal-Fractal Recompressive Attention Visualizer")

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
            attention = FractalAttention(input_data, max_depth=max_depth)
            fig = plot_attention_patterns(attention)
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Fractal Properties")
    if st.button("Analyze Fractal Dimensions"):
        with st.spinner("Analyzing fractal dimensions..."):
            dimensions = []
            for level in range(max_depth):
                attention = FractalAttention(input_data, level=level, max_depth=max_depth)
                dim = attention.fractal_dimension(input_data, level)
                dimensions.append(dim)
            
            fig = plot_fractal_dimension(dimensions)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Information Conservation")
    if st.button("Verify Properties"):
        with st.spinner("Verifying mathematical properties..."):
            attention = FractalAttention(input_data, max_depth=max_depth)
            st.write("Information Content:", attention.get_information_content())
            st.write("Attention Conservation:", attention.verify_attention_conservation())
