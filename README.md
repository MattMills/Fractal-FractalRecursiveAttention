# Fractal-Fractal Recompressive Attention Visualizer

An interactive visualization tool for exploring and analyzing Fractal-Fractal Recompressive Attention patterns. This project implements a novel attention mechanism that leverages fractal geometry and manifold learning for enhanced pattern recognition.

## Key Features

- Enhanced fractal attention mechanism with dimension conservation
- Interactive visualization of attention patterns
- Real-time computation of fractal dimensions
- Analysis of manifold properties
- Information content measurement
- Attention conservation verification

## Mathematical Foundation

The implementation is based on key mathematical properties:

1. **Self-Similarity**: A_n ≅ A_{n+1} under φ_n
2. **Recompressive Consistency**: R_n ∘ R_{n+1} = R_{n+1} ∘ R_n
3. **Attention Conservation**: tr(A_n) = tr(A_{n+1}) under φ_n
4. **Fractal Dimension**: D = lim_{n→∞} log(N_n)/log(1/r_n)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run main.py
```

## Components

- `enhanced_fractal_attention.py`: Core implementation of the attention mechanism
- `visualization.py`: Plotting and visualization utilities
- `utils.py`: Helper functions and mathematical rendering
- `main.py`: Streamlit interface and application logic

## Interactive Features

1. **Parameter Control**
   - Maximum recursion depth
   - Input dimension
   - Attention threshold

2. **Visualization Options**
   - Standard attention pattern heatmaps
   - 3D surface visualization
   - Hierarchical clustering view
   - Evolution analysis
   - Fractal dimension plots
   - Manifold dimension visualization

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

MIT License - Feel free to use and modify the code.
