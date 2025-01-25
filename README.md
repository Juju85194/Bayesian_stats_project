# Bayesian Clustering of the Iris Dataset

This repository applies a Bayesian Dirichlet cluster process to analyze the well-known Iris dataset. It investigates the number of clusters present in the data and compares the results to a frequentist approach using Gaussian Mixture Models (GMM) with BIC.

## Project Structure

-   `iris_clustering.ipynb`: Main Jupyter Notebook containing the analysis, visualizations, and explanations.
-   `utils/bayesian_clustering.py`: Python module with core functions for the Dirichlet cluster process.

## Methodology

The project implements a Gibbs sampler for a 4D Dirichlet cluster process, as described in McCullagh and Yang (2008), to infer the posterior distribution of the number of clusters in the Iris dataset. It explores the impact of different hyperparameters (θ and λ) on the clustering results.

## Comparison with Frequentist Approach

The Bayesian approach is compared to a frequentist approach using Gaussian Mixture Models (GMM) with the Bayesian Information Criterion (BIC) for model selection.

## References

-   McCullagh, P., & Yang, J. (2008). How many clusters? Bayesian Analysis, 3(1), 101-120.
-   Neal, R. M. (2000). Markov chain sampling methods for Dirichlet process mixture models. Journal of computational and graphical statistics, 9(2), 249-265.
-   Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and their applications. Biometrika, 57(1), 97-109.