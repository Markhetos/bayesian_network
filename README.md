# Bayesian Network Structure Learning

A Julia implementation of probabilistic graphical model structure learning using hybrid search algorithms. This project discovers causal relationships in observational data by learning optimal Bayesian network structures across datasets of varying complexity.

## Overview

This implementation combines the K2 greedy search algorithm with stochastic hill climbing to efficiently explore the space of directed acyclic graphs (DAGs). The approach maximizes the Bayesian score (using Dirichlet priors and log-gamma stabilization) to balance model complexity with data fit.

**Key Features:**
- Hybrid structure learning combining K2 initialization with hill climbing refinement
- Scalable to datasets with 50+ variables and 10K+ samples
- Custom Bayesian scoring function with numerical stability optimizations
- Multiple random restarts and early stopping for improved convergence
- Automatic acyclicity enforcement and parent cardinality constraints

## Repository Structure
```
├── data/               # Input datasets (CSV format)
├── results/            # Learned structures (.gph) and visualizations (.png)
├── project1.jl         # Core structure learning implementation
├── project1.py         # Visualization helper (NetworkX + Matplotlib)
└── CS238 - Project 1 - README.pdf   # Detailed algorithm write-up
```

## Installation

### Julia Dependencies
```julia
using Pkg
Pkg.add(["CSV", "DataFrames", "Graphs", "SpecialFunctions"])
```

Requires Julia 1.9+. Optional: enable multi-threading for faster score evaluation.

### Python Visualization (Optional)
```bash
pip install networkx matplotlib pandas
```

## Usage

Learn a Bayesian network structure from data:
```bash
julia project1.jl data/small.csv results/small.gph
```

Visualize the learned graph:
```bash
python project1.py
```

## Algorithm

The structure learning pipeline:

1. **Data preprocessing**: Load CSV data and compute sufficient statistics
2. **K2 initialization**: Generate multiple candidate structures using random variable orderings
3. **Hill climbing refinement**: Stochastically explore edge modifications (add/remove/reverse) to improve Bayesian score
4. **Output**: Export optimal DAG in `.gph` format

### Hyperparameters
- Number of K2 random restarts (5-10 depending on dataset size)
- Maximum parents per node (3-4)
- Hill climbing iterations (1500 with early stopping)

## Results

Successfully learned interpretable structures across three domains:

| Dataset | Variables | Samples | Edges | Runtime | Score Improvement |
|---------|-----------|---------|-------|---------|-------------------|
| Titanic | 8 | 889 | 14 | 1.8s | +3.0 |
| Wine | 13 | 6,497 | 30 | 5.3s | +0.8 |
| Large | 50 | 10,000 | 123 | 158s | +5,282 |

The learned networks capture meaningful causal relationships (e.g., passenger class → survival in Titanic data, chemical properties → wine quality).

## Technical Details

**Bayesian Scoring:** Implements the Bayesian Dirichlet equivalent uniform (BDeu) score with:
- Uniform Dirichlet prior (α = 1)
- Log-gamma functions for numerical stability
- Decomposable scoring for efficient local computations

**Search Strategy:**
- Greedy K2 for rapid convergence to local optima
- Stochastic hill climbing for refinement and exploration
- Acyclicity checks using graph traversal
- Early stopping based on convergence criteria

## References

- Mykel J. Kochenderfer, Tim A. Wheeler, and Kyle H. Wray. *Algorithms for Decision Making*. MIT Press, 2022. Chapter 5.1.
- Cooper, G.F. & Herskovits, E. "A Bayesian method for the induction of probabilistic networks from data." *Machine Learning* 9, 309–347 (1992).

## License

MIT License - see repository for details.
