import pandas as pd
from algorithm import run
from dataset import load_and_prepare
from dotenv import load_dotenv
import os
import time

load_dotenv()


n_gen = int(os.environ["n_gen"])
pop_size = int(os.environ["pop_size"])
max_number_of_layers = int(os.environ["max_number_of_layers"])
max_iter_in_ann = int(os.environ["max_iter_in_ann"])
eliminate_duplicates = bool(os.environ.get("eliminate_duplicates", True))
cv = int(os.environ.get("cv", 2))
verbose = os.environ.get("verbose", False)
random_state = os.environ.get("random_state")
id_dataset = int(os.environ["id_dataset"])
max_number_of_nodes_per_layer = int(os.environ["max_number_of_nodes_per_layer"])
size = int(os.environ.get("size", -1))


X, y, encoders, dataset_name = load_and_prepare(id_dataset)
print("Find for ", dataset_name)
start_time = time.time()
res = run(
    X,
    y,
    n_gen,
    pop_size,
    max_number_of_layers,
    max_number_of_nodes_per_layer,
    max_iter_in_ann,
    size=size,
    cv=cv,
    random_state=random_state,
    eliminate_duplicates=eliminate_duplicates,
    verbose=verbose,
)
print(
    "Best solution found: \nX = %s\nF = %s"
    % ([el for el in res.X if el > 0], res.F * -1)
)
print("--- %s seconds ---" % (time.time() - start_time))
