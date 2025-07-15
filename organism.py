from config import ParamSpace

class Organism:
    """
    A set of hyperparameters to be evolved.
    """
    def __init__(self, genome: ParamSpace):
        self.fitness: float | None = None
        self.genome = genome
        self.ei : float | None = None

    def __repr__(self):
        return f"Organism(genome={self.genome}, fitness={self.fitness})"
    
    def set_fitness(self, f: float) -> None:
        self.fitness = f
    
    def get_fitness(self) -> float:
        if self.fitness is None:
            raise ValueError("Organism hasn't been evaluated.")
        return self.fitness

    def set_genome(self, g: ParamSpace) -> None:
        self.genome = g
    
    def get_genome(self) -> ParamSpace:
        return self.genome
    
    def set_ei(self, ei: float) -> None:
        self.ei = ei
    
    def get_ei(self) -> float:
        if self.ei is None:
            raise ValueError("Organism's Expected Improvement score has not been computed.")
        return self.ei