# GA_MLP
## A very simple example of using Genetic Algorithms to generate a Multi-Layer Perceptron

An example of running a genetic algorithm is to find how many hidden layers and nodes in each layer an MLP has to have to better solve a particular classification problem. Problems are imported from the UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/) using their library.

## How to use

Just type python main.py after defining your problem in the .env file.

There is also a Dockerfile available for those who prefer it.

## .env description:

A succinct description of each parameter at .env file.

| Parameter  | Description  | Data type
|---|---|---|
| n_gen | Max number of generations for the genetic algorithm | int |
| pop_size | Max number of population size for the genetic algorithm | int |
| max_number_of_layers | Max number of hidden layers allowed | int |
| max_iter_in_ann | Max number of iterations allowed during each MLP training | int |
| eliminate_duplicates | Param for the genetic algorithm to eliminate duplicates or not | boolean |
| cv | Cross-validation k value | int |
| verbose | Print info as it executes | boolean |
| id_dataset | Id of the dataset for ucimlrepo lib | int |
| size | Size of the dataset to be considered. It takes a sample using pandas.sample(n=size) | int 
| max_number_of_nodes_per_layer  | Max number of nodes allowed in each hidden layer | int |


## Libraries

| Library | Version | Description
|---|---|---|
| Python | 3.11 |
| scikit-learn | 1.4.0 | ML library which contains MLPClassifier used in this example |
| pymoo| 0.6.1.1 | Evolutionary computation library for Python where we find the Genetic Algorithm |
| ucimlrepo | 0.0.3 | Library to get datasets from UC Irvine Machine Learning Repository |
| python-dotenv | 1.0.0 | To read .env files |
| numba | 0.58.1 | Necessary for pymoo |
| pyarrow | 15.0.0 | Necessary to work with parquets |
