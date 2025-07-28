import numpy as np
from dataclasses import dataclass

@dataclass
class Config:
    seed: int
    evaluations: int
    pop_size: int
    num_candidates: int
    tour_size: int
    mut_rate: float
    task_id: int
    model: str
    debug: bool
    rng: np.random.default_rng

# class Config:
#     def __init__(self, 
#                  seed: int = 0,
#                  evaluations: int = 500,
#                  pop_size: int = 50, 
#                  num_candidates: int = 10,
#                  tour_size: int = 2,
#                  mut_rate: float = 0.1,
#                  task_id: int = 359959, # Current OpenML dataset ID
#                  # logdir: str = 'results',
#                  model: str = 'RF',
#                  debug = False):
        
#         self.seed = seed
#         self.evaluations = evaluations
#         self.pop_size = pop_size
#         self.num_candidates = num_candidates
#         self.tour_size = tour_size
#         self.mut_rate = mut_rate
#         self.task_id = task_id # May not be necessary
#         # self.logdir = logdir
#         self.model = model
#         self.debug = debug

#         # self.logger = Logger(self.logdir)

#         # This should affect the entire system
#         random.seed(seed)
#         np.random.seed(seed)
    
#     def get_seed(self) -> int:
#         """ Returns the current seed. """
#         return self.seed
    
#     def get_evaluations(self) -> int:
#         """ Returns the number of evaluations. """
#         return self.evaluations
    
#     def get_pop_size(self) -> int:
#         """ Returns population size. """
#         return self.pop_size
    
#     def get_num_candidates(self) -> int:
#         """ Returns the number of candidate "offspring" For TPE and EA+TPE. """
#         return self.num_candidates
    
#     def get_tour_size(self) -> int:
#         """ Returns tournament size for EA selection. """
#         return self.tour_size
    
#     def get_mut_rate(self) -> float:
#         """ Returns mutation rate. """
#         return self.mut_rate
    
#     def get_task_id(self) -> int:
#         """ Returns current OpenML task ID. """
#         return self.task_id

#     # def get_logdir(self) -> str:
#     #     """ Returns the log directory path. """
#     #     return self.logdir
    
#     # def get_logger(self) -> Logger:
#     #     """ Returns the Logger object. """
#     #     return self.logger
    
#     def get_model(self) -> str:
#         return self.model
    
#     def get_debug(self) -> bool:
#         """ Returns 'debug' state. """
#         return self.debug