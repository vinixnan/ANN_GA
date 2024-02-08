from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.algorithms.soo.nonconvex.ga import GA
from problem import ANNProblem


def create_algorithm(pop_size, eliminate_duplicates):
    sampling = IntegerRandomSampling()
    crossover = TwoPointCrossover(prob=0.9)
    mutation_probability = 0.1
    mutation = BitflipMutation(prob=mutation_probability)

    algorithm = GA(
        pop_size=pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=eliminate_duplicates,
    )

    return algorithm


def run(
    X,
    y,
    n_gen,
    pop_size,
    max_number_of_layers,
    max_number_of_nodes_per_layer,
    max_iter_in_ann,
    cv=2,
    size=-1,
    random_state=None,
    eliminate_duplicates=True,
    verbose=True,
):
    problem = ANNProblem(
        max_number_of_layers,
        max_number_of_nodes_per_layer,
        X,
        y,
        max_iter_in_ann,
        size,
        cv,
        random_state,
    )
    algorithm = create_algorithm(pop_size, eliminate_duplicates)
    res = minimize(
        problem, algorithm, ("n_gen", n_gen), seed=random_state, verbose=verbose
    )
    return res
