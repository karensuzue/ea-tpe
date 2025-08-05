import os
import pickle
import numpy as np
from individual import Individual
from param_space import ModelParams, RandomForestParams
from tpe import MultivariateKDE, CategoricalPMF, TPE
from utils import eval_parameters_RF, load_task

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


# # Univariate, 1D array
# MultivariateKDE(np.array([1, 2, 3, 4]))

# # Univariate, shape (1, n)
# MultivariateKDE(np.array([[1, 2, 3, 4]]))

# # Multivariate, 2D
# MultivariateKDE(np.array([[1, 2, 3], [4, 5, 6]]))  # shape (2, 3)

# # Degenerate, 1 sample
# MultivariateKDE(np.array([[1], [2]]))  # (2, 1)

# # Constant data (singular covariance)
# MultivariateKDE(np.array([[1, 1, 1], [2, 2, 2]]))


# #https://github.com/automl/ASKL2.0_experiments/blob/84a9c0b3af8f7ac6e2a003d4dea5e6dce97d4315/experiment_scripts/utils.py
# def load_task(task_id: int, data_dir: str, preprocess=True):
#     """ Loads and splits the chosen task. """
#     cached_data_path = f"{data_dir}/{task_id}_{preprocess}.pkl"
#     if os.path.exists(cached_data_path):
#         d = pickle.load(open(cached_data_path, "rb"))
#         X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']
#     else:
#         print(f'Task {task_id} not found')
#         exit(0)

#     return X_train, y_train, X_test, y_test

# X_train, y_train, X_test, y_test = load_task(task_id=359959, data_dir='data')

# print(X_train)
# print(y_train)

# rng_ = np.random.default_rng(0)
# test_rf = RandomForestParams(rng_)
# params = test_rf.generate_random_parameters()
# print(params)
# test_rf.mutate_parameters(params)
# print(params)

# print(test_rf.get_params_by_type("cat"))


rng_ = np.random.default_rng(0)
# # Multivariate, 2D (2 parameters, 3 samples)
multi = MultivariateKDE(np.array([[1, 2, 3], [4, 5, 6]]), rng_)  # shape (2, 3)
print(multi.sample())
print(multi.pdf([[2.5, 3.5],[5.5, 6.5]]))

cat = CategoricalPMF(['cat', 'dog', 'cat', 'cat'], ('cat', 'dog'), rng_)
print(cat.sample())
print(cat.pmf('dog'))

test_rf = RandomForestParams(rng_)
# params = test_rf.generate_random_parameters()
# print("OG: ", params)
# test_rf.mutate_parameters(params)
# print("Mutated: ", params)


X_train, y_train, X_test, y_test = load_task(task_id = 359959, data_dir = "data")

tpe = TPE(rng = rng_)
population = [Individual(test_rf.generate_random_parameters()) for _ in range(50)]
# print("Population: ", population)
for ind in population:
    ind.set_performance(eval_parameters_RF(ind.get_params(), X_train, y_train, seed = rng_.integers(0, 2**32 - 1)))
tpe.fit(population, test_rf)

samples = tpe.sample(2, test_rf)
for ind in samples:
    ind.set_performance(eval_parameters_RF(ind.get_params(), X_train, y_train, seed = rng_.integers(0, 2**32 - 1)))
print(samples)

print("EXPECTED IMPROVEMENT")
print(tpe.expected_improvement(test_rf, samples))

print("SUGGEST")
print(tpe.suggest(test_rf, samples))