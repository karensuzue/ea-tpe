from param_space import ModelParams
from typeguard import typechecked
from typing import Dict, Any

@typechecked
class Individual:
    """
    This class encapsulates a set of hyperparameters (the "params") and stores additional information, 
    such as the objective value ("performance") and expected improvement.
    """
    def __init__(self, params: Dict[str, Any], performance: float | None = None, ei: float | None = None,
                 train_score: float | None = None, test_score: float | None = None):
        """
        Parameters:
            param_space (ModelParams): This object encapsulates the parameter space and allows you to sample from it.
            performance (float): Objective function value, or the cross-validation score.
            ei (float): Expected improvement score. 
            final_train_score (float): Final accuracy on the training set.
            final_test_score (float): Final accuracy on the testing set.
        """
        self.performance = performance 
        self.ei = ei
        self.final_train_score = train_score
        self.final_test_score = test_score

        # Initialize params, a set of random hyperparameters ({parameter_name: value})
        self.params: Dict[str, Any] = params

    def __repr__(self):
        return f"Individual(params={self.params}, performance={self.performance})"
    
    def set_performance(self, f: float) -> None:
        self.performance = f
    
    def get_performance(self) -> float:
        if self.performance is None:
            raise ValueError("Individual hasn't been evaluated.")
        return self.performance

    def set_ei(self, ei: float) -> None:
        self.ei = ei
    
    def get_ei(self) -> float:
        if self.ei is None:
            raise ValueError("Individual's Expected Improvement score has not been computed.")
        return self.ei
    
    def set_train_score(self, score: float) -> None:
        self.final_train_score = score
    
    def get_train_score(self) -> float:
        if self.final_train_score is None:
            raise ValueError("Individual's final training accuracy score has not been computed.")
        return self.final_train_score
    
    def set_test_score(self, score: float) -> None:
        self.final_test_score = score
    
    def get_test_score(self) -> float:
        if self.final_test_score is None:
            raise ValueError("Individual's final test accuracy score has not been computed.")
        return self.final_test_score

    def set_params(self, g: Dict[str, Any]) -> None:
        self.params = g
    
    def get_params(self) -> Dict[str, Any]:
        return self.params
    
