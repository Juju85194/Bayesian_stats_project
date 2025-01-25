import numpy as np
from scipy.stats import multivariate_normal

def dirichlet_cluster_process_4d(data, theta=9.0, lambda_=1.0, n_iter=500, burn_in=100):
    """
    Gibbs sampler for a 4D Dirichlet cluster process.

    Args:
        data: NumPy array of shape (n_samples, n_features) representing the data.
        theta: Parameter controlling within-cluster vs. between-cluster variance.
        lambda_: Parameter controlling the propensity to create new clusters.
        n_iter: Total number of MCMC iterations.
        burn_in: Number of burn-in iterations to discard.

    Returns:
        n_clusters_history: List of the number of clusters in each iteration (after burn-in).
        cluster_assignments_history: List of cluster assignments for each data point in each iteration (after burn-in).
    """
    n_samples, n_features = data.shape
    cluster_assignments = np.arange(n_samples)
    clusters = {}
    
    for i in range(n_samples):
        clusters[i] = {
            'n': 1,
            'sum_y': data[i].copy(),
            'sum_yy': np.outer(data[i], data[i])
        }
    
    n_clusters_history = []
    cluster_assignments_history = []
    
    for iter in range(n_iter):
        indices = np.random.permutation(n_samples)
        
        for i in indices:
            current_cluster = cluster_assignments[i]
            
            clusters[current_cluster]['n'] -= 1
            clusters[current_cluster]['sum_y'] -= data[i]
            clusters[current_cluster]['sum_yy'] -= np.outer(data[i], data[i])
            
            if clusters[current_cluster]['n'] == 0:
                del clusters[current_cluster]
            
            log_probs = []
            cluster_ids = list(clusters.keys())
            
            for c in cluster_ids:
                n_c = clusters[c]['n']
                sum_y_c = clusters[c]['sum_y']
                
                prior = np.log(n_c)
                mean_pred = sum_y_c / (n_c + 1/theta)
                cov_pred = (1 + 1/(n_c * theta + 1)) * np.eye(n_features)
                
                try:
                    likelihood = multivariate_normal.logpdf(data[i], mean=mean_pred, cov=cov_pred)
                except np.linalg.LinAlgError:
                    likelihood = -np.inf
                log_probs.append(prior + likelihood)
            
            prior_new = np.log(lambda_)
            mean_new = np.zeros(n_features)
            cov_new = (1 + theta) * np.eye(n_features)
            likelihood_new = multivariate_normal.logpdf(data[i], mean=mean_new, cov=cov_new)
            log_probs.append(prior_new + likelihood_new)
            
            max_log = np.max(log_probs)
            probs = np.exp(log_probs - max_log)
            probs /= probs.sum()
            
            chosen = np.random.choice(len(probs), p=probs)
            
            if chosen < len(cluster_ids):
                new_cluster = cluster_ids[chosen]
            else:
                new_cluster = max(clusters.keys(), default=-1) + 1
                clusters[new_cluster] = {
                    'n': 0,
                    'sum_y': np.zeros(n_features),
                    'sum_yy': np.zeros((n_features, n_features))
                }
            
            cluster_assignments[i] = new_cluster
            clusters[new_cluster]['n'] += 1
            clusters[new_cluster]['sum_y'] += data[i]
            clusters[new_cluster]['sum_yy'] += np.outer(data[i], data[i])
        
        if iter >= burn_in:
            n_clusters_history.append(len(clusters))
            cluster_assignments_history.append(cluster_assignments.copy())
        
        if (iter + 1) % 100 == 0:
            print(f"Iteration {iter + 1}/{n_iter}: {len(clusters)} clusters")
    
    return n_clusters_history, cluster_assignments_history