from param_space import ModelParams
from typeguard import typechecked
from typing import Dict, Any

@typechecked
class Organism:
    """
    This class encapsulates a set of hyperparameters (the "genotype") and stores additional information, 
    such as the objective value ("fitness") and expected improvement.
    """
    def __init__(self, param_space: ModelParams, fitness: float | None = None, ei: float | None = None):
        """
        Parameters:
            param_space (ModelParams): This object encapsulates the parameter space and allows you to sample from it.
            fitness (float): Objective function value.
            ei (float): Expected improvement score. 
        """
        self.param_space = param_space
        self.fitness = fitness
        self.ei = ei

        # Initialize genotype, a set of random hyperparameters ({parameter_name: value})
        self.genotype: Dict[str, Any] = self.param_space.generate_random_parameters()
        # Make sure parameter values align with scikit-learn's requirements
        self.param_space.fix_parameters(self.genotype)

    def __repr__(self):
        return f"Organism(genotype={self.genotype}, fitness={self.fitness})"
    
    def set_fitness(self, f: float) -> None:
        self.fitness = f
    
    def get_fitness(self) -> float:
        if self.fitness is None:
            raise ValueError("Organism hasn't been evaluated.")
        return self.fitness

    def set_ei(self, ei: float) -> None:
        self.ei = ei
    
    def get_ei(self) -> float:
        if self.ei is None:
            raise ValueError("Organism's Expected Improvement score has not been computed.")
        return self.ei

    def set_genotype(self, g: Dict[str, Any]) -> None:
        self.genotype = g
    
    def get_genotype(self) -> Dict[str, Any]:
        return self.genotype
    
