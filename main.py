import streamlit as st
import numpy as np
import plotly.graph_objects as go
import torch
import logging
from enhanced_fractal_attention import EnhancedFractalAttention
from visualization import (
    plot_attention_patterns, plot_3d_attention, 
    plot_hierarchical_attention, plot_attention_evolution,
    plot_fractal_dimension, plot_manifold_dimension
)
from comparative_analysis import (
    AttentionComparator, plot_comparison, 
    analyze_temporal_evolution, plot_evolution_metrics
)
from utils import generate_sample_data, render_latex_equations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    st.set_page_config(
        page_title="Fractal-Fractal Recompressive Attention Visualizer",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Fractal-Fractal Recompressive Attention Visualizer")

    def handle_error(e):
        logger.error(f"Error occurred: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        return None

    def safe_visualization_wrapper(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Visualization error: {str(e)}")
                st.error(f"Visualization error: {str(e)}")
                return None
        return wrapper

    @st.cache_data
    def compute_attention_patterns(input_data, max_depth):
        try:
            attention = EnhancedFractalAttention(input_data, max_depth=max_depth)
            output = attention(attention.X)
            metrics = attention.get_metrics()
            
            # Generate temporal evolution data
            evolution_data = []
            temp_attention = attention.X
            for _ in range(max_depth):
                temp_attention = attention._fractal_attention(temp_attention, 0, 1)
                evolution_data.append(temp_attention.detach().numpy())
            
            return {
                'output': output.detach().numpy(),
                'evolution': np.stack(evolution_data),
                'metrics': {
                    'information_content': float(metrics.information_content),
                    'attention_conservation': float(metrics.attention_conservation),
                    'fractal_dimension': float(metrics.fractal_dimension),
                    'manifold_dimension': float(metrics.manifold_dimension)
                }
            }
        except Exception as e:
            logger.error(f"Error in compute_attention_patterns: {str(e)}")
            return handle_error(e)

    @st.cache_data
    def compute_comparative_analysis(input_data1, input_data2, max_depth):
        try:
            # Initialize comparator
            comparator = AttentionComparator()
            
            # Compute attention patterns for both inputs
            attention1 = EnhancedFractalAttention(input_data1, max_depth=max_depth)
            attention2 = EnhancedFractalAttention(input_data2, max_depth=max_depth)
            
            output1 = attention1(attention1.X)
            output2 = attention2(attention2.X)
            
            metrics1 = attention1.get_metrics()
            metrics2 = attention2.get_metrics()
            
            # Compare patterns
            comparison = comparator.compare_attention_patterns(output1, output2)
            comparison.metric_differences = comparator.compare_metrics(metrics1, metrics2)
            
            return comparison
        except Exception as e:
            logger.error(f"Error in compute_comparative_analysis: {str(e)}")
            return handle_error(e)

    @st.cache_data
    @safe_visualization_wrapper
    def analyze_fractal_dimensions(input_data, max_depth):
        dimensions = []
        manifold_dims = []
        attention = EnhancedFractalAttention(input_data, max_depth=max_depth)
        
        for level in range(max_depth):
            metrics = attention.get_metrics()
            dimensions.append(float(metrics.fractal_dimension))
            manifold_dims.append(float(metrics.manifold_dimension))
            
        return dimensions, manifold_dims

    # Sidebar controls
    st.sidebar.header("Parameters")
    max_depth = st.sidebar.slider("Maximum Depth", 1, 10, 3)
    input_dim = st.sidebar.slider("Input Dimension", 8, 64, 32)
    attention_threshold = st.sidebar.slider("Attention Threshold", 0.0, 1.0, 0.5)

    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Single Pattern", "Comparative Analysis"]
    )

    visualization_type = st.sidebar.selectbox(
        "Visualization Type",
        ["Standard", "3D Surface", "Hierarchical", "Evolution"]
    )

    # Generate sample data
    input_data = generate_sample_data(input_dim)
    
    if analysis_mode == "Comparative Analysis":
        # Generate second sample with slight variation
        input_data2 = generate_sample_data(input_dim) * 0.9 + generate_sample_data(input_dim) * 0.1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pattern Comparison")
            if st.button("Compare Patterns"):
                with st.spinner("Computing comparison..."):
                    comparison = compute_comparative_analysis(input_data, input_data2, max_depth)
                    if comparison:
                        fig = plot_comparison(comparison)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("Comparison Metrics")
                        st.json(comparison.metric_differences)
        
        with col2:
            st.subheader("Temporal Evolution")
            if st.button("Analyze Evolution"):
                with st.spinner("Analyzing pattern evolution..."):
                    result1 = compute_attention_patterns(input_data, max_depth)
                    result2 = compute_attention_patterns(input_data2, max_depth)
                    
                    if result1 and result2:
                        similarities, convergence_rates = analyze_temporal_evolution(
                            [torch.tensor(x) for x in result1['evolution']]
                        )
                        fig = plot_evolution_metrics(similarities, convergence_rates)
                        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Single pattern analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Mathematical Formalization")
            render_latex_equations()
            
            st.subheader("Attention Patterns")
            if st.button("Compute Attention"):
                with st.spinner("Computing attention patterns..."):
                    result = compute_attention_patterns(input_data, max_depth)
                    if result:
                        if visualization_type == "Standard":
                            fig = plot_attention_patterns(result['output'])
                        elif visualization_type == "3D Surface":
                            fig = plot_3d_attention(result['output'])
                        elif visualization_type == "Hierarchical":
                            fig = plot_hierarchical_attention(result['output'])
                        else:  # Evolution
                            fig = plot_attention_evolution(result['evolution'])
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with st.expander("Attention Metrics"):
                            st.json(result['metrics'])
        
        with col2:
            st.subheader("Fractal Properties")
            if st.button("Analyze Fractal Dimensions"):
                with st.spinner("Analyzing fractal dimensions..."):
                    dimensions, manifold_dims = analyze_fractal_dimensions(input_data, max_depth)
                    if dimensions and manifold_dims:
                        fig_fractal = plot_fractal_dimension(dimensions)
                        if fig_fractal:
                            st.plotly_chart(fig_fractal, use_container_width=True)
                        
                        fig_manifold = plot_manifold_dimension(manifold_dims)
                        if fig_manifold:
                            st.plotly_chart(fig_manifold, use_container_width=True)

except Exception as e:
    logger.error(f"Main application error: {str(e)}")
    st.error(f"Application error: {str(e)}")
