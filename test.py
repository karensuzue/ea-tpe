from ea import EA
from tpe import TPE, MultivariateKDE, CategoricalPMF
import numpy as np
from config import Config
from collections import Counter


# # config = Config(dataset_idx=0)
# # ea_solver = EA(config)
# # ea_solver.init_population()
# # # print(ea_solver.population)
# # ea_solver.evaluate_population()
# # print(ea_solver.population)

# # parents = ea_solver.select_parents()
# # print(parents)
# # ea_solver.mate_population(parents)
# # print(ea_solver.population)
# # print(len(ea_solver.population))

# # test_list = [1,2,3,4,5,6,7,8,9,10,11,12,13]
# # test_list2 = test_list[2:]
# # test_list3 = test_list[:3]
# # test_list4 = test_list[: len(test_list) - 3]
# # print(test_list2)
# # print(test_list3)
# # print(test_list4)

# counts = Counter(['cat', 'cat', 'dog'])
# print(counts)
# all_categories = ['cat', 'dog']
# total = sum(counts[c] for c in all_categories)
# print(total)
# prob = {c: (counts[c]) / total for c in all_categories}
# print(prob)


# Univariate, 1D array
MultivariateKDE(np.array([1, 2, 3, 4]))

# Univariate, shape (1, n)
MultivariateKDE(np.array([[1, 2, 3, 4]]))

# Multivariate, 2D
MultivariateKDE(np.array([[1, 2, 3], [4, 5, 6]]))  # shape (2, 3)

# Degenerate, 1 sample
MultivariateKDE(np.array([[1], [2]]))  # (2, 1)

# Constant data (singular covariance)
MultivariateKDE(np.array([[1, 1, 1], [2, 2, 2]]))
