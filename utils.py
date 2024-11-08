import numpy as np
import streamlit as st

def generate_sample_data(dim):
    # Generate synthetic data with fractal-like properties
    X = np.random.randn(dim, dim)
    # Add self-similar structure
    for i in range(1, dim):
        X[i] = X[i-1] * 0.9 + np.random.randn(dim) * 0.1
    return X

def render_latex_equations():
    st.markdown(r"""
    ### Key Properties:
    
    1. **Self-Similarity:**
    $$A_n \simeq A_{n+1} \text{ under } \phi_n$$
    
    2. **Recompressive Consistency:**
    $$R_n \circ R_{n+1} = R_{n+1} \circ R_n$$
    
    3. **Attention Conservation:**
    $$\text{tr}(A_n) = \text{tr}(A_{n+1}) \text{ under } \phi_n$$
    
    4. **Fractal Dimension:**
    $$D = \lim_{n\to\infty} \frac{\log(N_n)}{\log(1/r_n)}$$
    
    ### Information Flow:
    
    The gradient $\nabla A_n$ defines a vector field on the manifold:
    $$\nabla A_n: \mathcal{M} \to T\mathcal{M}$$
    
    ### Recompressive Equilibrium:
    
    $$\lim_{n\to\infty} (R_n \circ A_n)(X) \to X^*$$
    """)
