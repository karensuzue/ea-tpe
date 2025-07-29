import numpy as np
from organism import Organism
from surrogate import Surrogate
from param_space import ModelParams
from collections import Counter
from scipy.stats import gaussian_kde
from typing import Tuple, Dict, List, Union, Iterable, Any
from typeguard import typechecked

@typechecked
class MultivariateKDE:
    """  
    Multivariate Gaussian KDE with floor to prevent zero likelihood. 
    This wrapper over 'scipy.stats.gaussian_kde' supports optional bandwidth scaling
    and enforces a minimum density 'eps' to avoid zero likelihood values.
    """
    def __init__(self, data: np.ndarray, bw_factor: float=1.0, eps: float=1e-12):
        """
        Parameters:
        - data : ndarray, shape=(d,n)
            'd' (row) is the number of dimensions or features 
            'n' (column) is the number of samples
            For example, np.array([[1, 2, 3],     # dimension 1 values
                                   [4, 5, 6]])    # dimension 2 values
        - bw_factor: float
            Scaling factor for the KDE bandwidth. 
            The base bandwidth is chosen using Silverman's rule, then scaled by this factor.
        - eps : float
            Minimum floor value for the estimated density. Returned densities will be at least this value.
        """
        # If there is only 1 sample (n=1), KDE will fail
        if data.shape[1] == 1:
            data = np.hstack([data, data + 1e-3]) # duplicate it slightly
        
        # TODO: If number of numeric hyperparameters = 1, otherwise dont run this
        # Check if covariance matrix is singular: must have >1 distinct samples
        if np.linalg.matrix_rank(np.cov(data)) < data.shape[0]:
            # Add small noise to escape colinearity
            data = data + np.random.normal(scale=1e-3, size=data.shape) 

        self.kde = gaussian_kde(data, bw_method='silverman')
        self.kde.set_bandwidth(self.kde.factor * bw_factor)
        self.eps = eps


    def pdf(self, vec: Union[np.ndarray, List]):
        """
        Evaluate the KDE probability density function at given points.

        Parameters:
        - vec : array-like, shape (d,) or (d, m)
            Points at which to evaluate the density. Can be a single d-dimensional point
            or multiple points as columns in a (d, m) array.
        """
        vec = np.asarray(vec) # reshape to (d, )
        if vec.ndim == 1:
            vec = vec[:, None] # reshape to (d, 1)
        # Returns an array of shape (m,), corresponding to 1 density value per point   
        return np.maximum(self.kde.pdf(vec), self.eps)  
    
    def sample(self, n_samples = 1) -> np.ndarray:
        """ 
        Sample n new points from the estimated distribution. 
        Returns a matrix of shape (dimensions, n_samples)
        """
        return self.kde.resample(size=n_samples) # shape (dimensions, n_samples)

class CategoricalPMF:
    """ 
    Categorical probability mass function with Laplace smoothing to avoid zero probabilities.
    Computes smoothed category probabilities based on observed frequencies,
    ensuring all categories have non-zero likelihood (with smoothing factor 'alpha').
    """

    def __init__(self, values: Iterable[str], all_categories: List[str] | List[bool] | Tuple[str] | Tuple[bool], 
                 alpha = 1.0):
        """
        Parameters:
        - values: Iterable[str]
            List or iterable of observed categorical values.
        - all_categories: List[str]
            The full list of possible categories to support in the distribution. 
        - alpha: float
            Laplace smoothing parameter. Higher values increase the uniformity of the distribution.
        """
        # Count the frequency of each category in 'values'
        counts = Counter(values) 
        total = sum(counts[c] + alpha for c in all_categories)
        self.prob: Dict = {c: (counts[c] + alpha) / total for c in all_categories}
        self.eps = 1e-12

    def pmf(self, x):
        """ 
        Evaluate the smoothed probability of a category 'x' if it was part of
        'all_categories'; otherwise, returns 'self.eps' to avoid zero likelihood. 

        Parameters:
        - x: Any
            The category to evaluate.
        """
        return self.prob.get(x, self.eps)

    def sample(self, n_samples = 1) -> List[str] | List[bool]:
        """
        Sample n categories from the categorical PMF.

        Parameters:
            n_samples (int): Number of samples to draw.

        Returns:
            List: List of sampled categories (length = n_samples) if n_samples > 1, 
            otherwise a single sampled value.
        """
        probabilities = list(self.prob.values())
        samples = np.random.choice(self.all_categories, size=n_samples, p=probabilities)
        return samples.tolist()

class TPE(Surrogate):
    """
    Tree-structured Parzen Estimator (TPE) Solver for hyperparameter optimization.
    This class supports both categorical and numeric parameters, 
    and uses kernel density estimation (KDE) and probability mass functions (PMFs) 
    to model the likelihood of good and bad configurations.
    """
    def __init__(self, gamma: float = 0.2):
        """
        Parameters:
            gamma (float): Fraction of samples considered "good".
        """
        self.gamma = gamma # splitting parameter

        # Fitted distributions
        self.multi_l: MultivariateKDE = None # good, numeric
        self.multi_g: MultivariateKDE = None  # bad, numeric
        self.cat_l: Dict[str, CategoricalPMF] = {} # good, categorical
        self.cat_g: Dict[str, CategoricalPMF] = {} # bad, categorical
    
    def sample(self, num_samples: int, param_space: ModelParams) -> List[Organism]:
        """
        Returns 'num_samples' Organisms. 
        For each Organism's genotype, sample from the MultivariateKDE and CategoricalPMFs separately, 
        then reassemble into a full set of hyperparameters.
        """
        numeric_params = {
            **param_space.get_params_by_type('int'),
            **param_space.get_params_by_type('float'),
        }

        numeric_params_names = list(numeric_params.keys())

        # params: Dict[str, Any] = {} # shape (dimensions, n_samples)

        # Sample from the good numeric distribution 
        multi_samples = self.multi_g.sample(num_samples) # shape (dimensions, n_samples)
        assert multi_samples.shape[0] == len(numeric_params_names)
        assert multi_samples.shape[1] == num_samples

        # Align numeric parameter names to each dimension
        params = {
            name: list(multi_samples[i])
            for i, name in enumerate(numeric_params_names)
        }

        # Sample from the good categorical distribution
        for name, dist in self.cat_l.items():
            params[name] = dist.sample(num_samples)
        
        assert all(len(v) == num_samples for v in params.values())

        samples: List[Organism] = [] 
        for i in range(num_samples):
            genotype = {name: params[name][i] for name in param_space}
            org = Organism(param_space)
            org.set_genotype(genotype)
            samples.append(org)

        assert(len(samples) == num_samples)
        return samples

    def split_samples(self, samples: List[Organism]) -> Tuple[List[Organism], List[Organism]]:
        """ 
        Splits a given sample set into 'good' and 'bad' groups based on
        the objective.
        
        Parameters: 
            samples (List[Organism]): The sample set to split.

        Returns:
            Tuple[List[Organism], List[Organism]]: a tuple containing the good and bad sample groups.
        """
        if len(samples) < 2:
            raise RuntimeError("Need at least 2 samples before TPE can fit.")
        
        # Sort population/samples set (lowest/best first)
        samples.sort(key=lambda o: o.get_fitness())
        split_idx = max(1, int(len(samples) * self.gamma))
        good_samples = samples[:split_idx]
        bad_samples = samples[split_idx:]
        return good_samples, bad_samples
    
    def fit(self, samples: List[Organism], param_space: ModelParams) -> None:
        """ 
        Fit probabilistic models (KDEs and PMFs) to the good and bad sample groups.
        
        Parameters:
        """
        good_samples, bad_samples = self.split_samples(samples)
        
        numeric_params = {
            **param_space.get_params_by_type('int'),
            **param_space.get_params_by_type('float'),
        }
        categorical_params = {
            **param_space.get_params_by_type('cat'),
            **param_space.get_params_by_type('bool')
        }

        # For each sample set, extract values of numeric hyperparameters
        # Format shape (n_params, n_samples): [[value11, value12,...], [value21, value22, ...], ...]
        # Each parameter has its own row 
        good_num_samples = np.array([[o.get_genotype()[param_name] for o in good_samples] 
                            for param_name in numeric_params])
        bad_num_samples = np.array([[o.get_genotype()[param_name] for o in bad_samples] 
                            for param_name in numeric_params])
        
        # Fit Multivariate KDEs
        self.multi_l = MultivariateKDE(good_num_samples)
        self.multi_g = MultivariateKDE(bad_num_samples)

        # Fit independent PMFs
        self.cat_l.clear() 
        self.cat_g.clear()
        # Construct 2 PMFs (good and bad) for each categorical parameter
        # Format: {param_name: CategoricalPMF}
        self.cat_l = {
            param_name: CategoricalPMF(
                # Extract categorical values from samples in (d, n) format
                values = [o.get_genotype()[param_name] for o in good_samples],
                all_categories = info["bounds"]
            )
            for param_name, info in categorical_params.items()
        }

        self.cat_g = {
            param_name : CategoricalPMF(
                values = [o.get_genotype()[param_name] for o in bad_samples],
                all_categories = info["bounds"]
            )
            for param_name, info in categorical_params.items()
        }


    def expected_improvement(self, param_space: ModelParams, candidates: List[Organism]) -> np.ndarray:
        """
        Compute the expected improvement (EI) for a list of candidate organisms.

        Parameters:
        - candidates: List[Organism]
            Candidate organisms to evaluate
        
        Returns an array of EI scores, one per candidate.
        """
        ei_scores = []

        numeric_params = {
            **param_space.get_params_by_type('int'),
            **param_space.get_params_by_type('float'),
        }

        for org in candidates:
            genotype = org.get_genotype()
            # Numeric contribution (multivariate)
            if numeric_params:
                num_vals = [genotype[param_name] for param_name in numeric_params] # (, d_num)
                # 'num_vals' gets reshaped into (d_num, 1) here
                l_num = float(self.multi_l.pdf(num_vals)) # a single density value
                g_num = float(self.multi_g.pdf(num_vals)) 
            else: # If no numeric parameters exist, no contribution
                l_num = g_num = 1.0 
            
            # Categorical contribution (product of per-dim PMFs)
            l_cat = g_cat = 1.0
            # PMFs are univariate
            for param_name, pmf_l in self.cat_l.items():
                lx = pmf_l.pmf(genotype[param_name]) # a single density
                gx = self.cat_g[param_name].pmf(genotype[param_name])
                l_cat *= lx
                g_cat *= gx
            
            # To avoid NaN
            if (g_num * g_cat) <= 1e-12:
                ei_scores.append(0.0)
            else:
                ei_scores.append((l_num * l_cat) / (g_num * g_cat))
            org.set_ei((l_num * l_cat) / (g_num * g_cat))
        return np.asarray(ei_scores)

    def suggest(self, param_space: ModelParams, candidates: List[Organism], num_top_cand: int = 1) -> Tuple[List[Organism], np.ndarray, int]:
        """ 
        Suggest the top-k candidates based on expected improvement.

        Parameters:
            candidates (List[Organism]): Candidate organisms to rank.
            num_top_cand (int): Number of top candidates to return. 

        Returns:
            Tuple[List[Organism], np.ndarray, int]: A tuple containing the top candidates, their EI scores, and
            the number of soft evaluations performed. 
        """
        scores = self.expected_improvement(param_space, candidates)

        soft_eval_count = len(scores)

        # 'np.argsort' gives indices that would sort scores in ascending order
        # '::-1' reverses that, higher EI is better
        # ':k' selects the top-k indices
        sorted_indices = np.argsort(scores)[::-1][:num_top_cand]
        top_candidates = [candidates[int(i)] for i in sorted_indices]
        top_scores = scores[sorted_indices]
        return top_candidates, top_scores, soft_eval_count
    
 

