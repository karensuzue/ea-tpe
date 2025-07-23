from config import ParamSpace
from genotype import ModelParams
from typeguard import typechecked

@typechecked
class Organism:
    """
    This class encapsulates a ModelParams object and stores additional information, 
    such as the objective value (fitness) and expected improvement.
    """
    def __init__(self, genotype: ModelParams, fitness: float | None = None, ei: float | None = None):
        self.fitness = fitness
        self.genotype = genotype
        self.ei = ei

    def __repr__(self):
        return f"Organism(genotype={self.genotype}, fitness={self.fitness})"
    
    def set_fitness(self, f: float) -> None:
        self.fitness = f
    
    def get_fitness(self) -> float:
        if self.fitness is None:
            raise ValueError("Organism hasn't been evaluated.")
        return self.fitness

    def set_genotype(self, g: ModelParams) -> None:
        self.genotype = g
    
    def get_genotype(self) -> ModelParams:
        return self.genotype
    
    def set_ei(self, ei: float) -> None:
        self.ei = ei
    
    def get_ei(self) -> float:
        if self.ei is None:
            raise ValueError("Organism's Expected Improvement score has not been computed.")
        return self.ei