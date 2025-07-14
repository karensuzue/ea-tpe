import numpy as np
import random
from config import Config
from logger import Logger
from organism import Organism
from collections import Counter
from typing import List, Tuple, Iterable, Dict, Union
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import gaussian_kde


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
        # KDE needs more than 2 points
        if data.shape[1] == 1:
            # Make another copy of the only point in 'data' 
            data = np.hstack([data, data + 1e-3]) 
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

class CategoricalPMF:
    """ 
    Categorical probability mass function with Laplace smoothing to avoid zero probabilities.
    Computes smoothed category probabilities based on observed frequencies,
    ensuring all categories have non-zero likelihood (with smoothing factor 'alpha').
    """

    def __init__(self, values: Iterable[str], all_categories: List[str], alpha = 1.0):
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


class TPE:
    """
    Tree-structured Parzen Estimator (TPE) Solver for hyperparameter optimization.
    This class supports both categorical and numeric parameters, 
    and uses kernel density estimation (KDE) and probability mass functions (PMFs) 
    to model the likelihood of good and bad configurations.
    """
    def __init__(self, config: Config, logger: Logger, gamma = 0.2):
        """
        Parameters:
        - config: Config
            Configuration object containing parameter space 
            and dataset (for training the Random Forest)
        - logger: Logger
            Logger object for tracking progress and results
        - gamma: float
            Fraction of samples considered "good".
        """
        self.config = config
        self.logger = logger
        self.gamma = gamma # splitting parameter

        self.param_space = self.config.get_param_space()
        self.num_names = self.config.get_num_param_names()
        self.cat_names = self.config.get_cat_param_names()

        self.samples: List[Organism] = [] # Essentially, the "population" (evaluated on the true objective)
        self.X_train, self.X_test, self.y_train, self.y_test = self.config.load_dataset()

        self.multi_l = None # stores fitted distributions
        self.multi_g = None 
        self.cat_l, self.cat_g = {}

    def init_samples(self) -> None:
        """
        Initialize the sample population with random genomes based on the parameter space.
        This method clears any existing samples and generates a new population of organisms with
        randomly sampled hyperparameters.
        """
        self.samples.clear() # just in case

        sample_size = self.config.get_pop_size()
        for _ in range(sample_size):
            genome = {}
            for param_name, spec in self.param_space.items():
                if spec["type"] == "int":
                    genome[param_name] = random.randint(*spec["bounds"])
                elif spec["type"] == "float":
                    genome[param_name] = random.uniform(*spec["bounds"])
                elif spec["type"] == "cat":
                    genome[param_name] = random.choice(spec["bounds"])
                else:
                    raise ValueError(f"Unknown parameter type: {spec['type']}")
            self.samples.append(Organism(genome))
        assert (len(self.samples) == sample_size)
    
    def random_samples(self, count: int = 10) -> List[Organism]:
        """
        Generate a list of random candidate organisms.

        Parameters:
        - count: int
            Number of random samples to generate.

        Returns a list of randomly generated organisms.
        """
        samples = []
        for _ in range(count):
            genome = {}
            for param_name, spec in self.param_space.items():
                if spec["type"] == "int":
                    genome[param_name] = random.randint(*spec["bounds"])
                elif spec["type"] == "float":
                    genome[param_name] = random.uniform(*spec["bounds"])
                elif spec["type"] == "cat":
                    genome[param_name] = random.choice(spec["bounds"])
                else:
                    raise ValueError(f"Unknown parameter type: {spec['type']}")
            samples.append(Organism(genome))
        return samples

    def evaluate_org(self, org: Organism) -> None:
        """
        Evaluate the fitness of an organism using cross-validated accuracy.

        Parameters:
        - org: Organism
            The organism to evaluate. Its fitness will be updated in-place.
        """
        # Must maintain the same seed/random_state across experiments 
        model = RandomForestClassifier(**org.get_genome(), random_state=0)
        # Inverted, as TPE expects minimization
        score = -1 * cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy').mean() 
        org.set_fitness(score)
    
    def split_samples(self) -> Tuple[List[Organism], List[Organism]]:
        """ 
        Split the current sample set into 'good' and 'bad' groups based on
        the objective.

        Returns a tuple containing the good and bad sample groups.
        """
        if len(self.samples) < 2:
            raise RuntimeError("Need at least 2 trials before TPE can fit.")
        
        # Sort population/samples set (lowest/best first)
        self.samples.sort(key=lambda o: o.get_fitness())
        split_idx = int(len(self.samples) * self.gamma)  
        good_samples = self.samples[:split_idx]
        bad_samples = self.samples[split_idx:]
        return good_samples, bad_samples
    
    def fit(self, good_samples: List[Organism], bad_samples: List[Organism]) -> None:
        """ 
        Fit probabilistic models (KDEs and PMFs) to the good and bad sample groups.
        
        Parameters:
        - good_samples: List[Organism]
            Organisms with high fitness (low objective value).
        - bad_samples: List[Organism]
            Organism with low fitness.
        """
        
        # Extract values of numeric dimensions/variables from the sample set
        good_num_samples = np.array([[o.get_genome()[param_name] for o in good_samples] 
                                     for param_name in self.num_names])
        bad_num_samples = np.array([[o.get_genome()[param_name] for o in bad_samples] 
                                     for param_name in self.num_names])
        
        # Multivariate KDEs
        self.multi_l = MultivariateKDE(good_num_samples)
        self.multi_g = MultivariateKDE(bad_num_samples)


        # Independent PMFs
        self.cat_l.clear()
        self.cat_g.clear()
        # Construct 2 PMFs (good and bad) for each parameter
        self.cat_l = {
            param_name: CategoricalPMF(
                # Extract categorical values from samples in (d, n) format
                values = [o.get_genome()[param_name] for o in good_samples],
                all_categories = self.param_space[param_name]["bounds"]
            )
            for param_name in self.cat_names
        }

        self.cat_g = {
            param_name : CategoricalPMF(
                values = [o.get_genome()[param_name] for o in bad_samples],
                all_categories = self.param_space[param_name]["bounds"]
            )
            for param_name in self.cat_names
        }


    def expected_improvement(self, candidates: List[Organism]) -> np.ndarray:
        """
        Compute the expected improvement (EI) for a list of candidate organisms.

        Parameters:
        - candidates: List[Organism]
            Candidate organisms to evaluate
        
        Returns an array of EI scores, one per candidate.
        """

        ei_scores = []
        for org in candidates:
            genome = org.get_genome()
            # Numeric contribution (multivariate)
            if self.num_names:
                num_vals = [genome[param_name] for param_name in self.num_names] # (, d_num)
                # 'num_vals' gets reshaped into (d_num, 1) here
                l_num = self.multi_l.pdf(num_vals) # a single density value
                g_num = self.multi_g.pdf(num_vals)
            else: # If no numeric parameters exist, no contribution
                l_num = g_num = 1.0 
            
            # Categorical contribution (product of per-dim PMFs)
            l_cat = g_cat = 1.0
            # PMFs are univariate
            for param_name, pmf_l in self.cat_l.items():
                lx = pmf_l.pmf(genome[param_name]) # a single density
                gx = self.cat_g[param_name].pmf(genome[param_name])
                l_cat *= lx
                g_cat *= gx
            ei_scores.append((l_num * l_cat) / (g_num * g_cat))
        
        return np.asarray(ei_scores)

    def suggest(self, candidates: List[Organism], k: int = 1) -> Tuple[List[Organism], np.ndarray]:
        """ 
        Suggest the top-k candidates based on expected improvement.

        Parameters:
        - candidates: List[Organism]
            Candidate organisms to rank.
        - k: int
            Number of top candidates to return. 

        Returns a tuple containing the top-k organisms and their EI scores. 
        """
        scores = self.expected_improvement(candidates)
        # 'np.argsort' gives indices that would sort scores in ascending order
        # '::-1' reverses that, higher EI is better
        # ':k' selects the top-k indices
        indices = np.argsort(scores)[::-1][:k]
        return [candidates[i] for i in indices], scores[indices]

    def optimize(self):
        """
        Run the full TPE optimization loop.
        """
        # Initialize sample set
        self.init_samples()
        for org in self.samples:
            self.evaluate_org(org)

        eval_count = self.config.get_evaluations()
        for _ in range(eval_count - len(self.samples)):
            # Split sample set into 'good' and 'bad' groups
            good_samples, bad_samples = self.split_samples()
        
            self.fit(good_samples, bad_samples)

            # Randomly generate new candidates
            candidates = self.random_samples()
            ei = self.expected_improvement(candidates)
            best_org, best_score = self.suggest(candidates)

            self.samples.append(best_org)





        


