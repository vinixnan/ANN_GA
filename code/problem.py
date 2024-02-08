from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from pymoo.core.problem import ElementwiseProblem
import numpy as np
from dataset import join_df, split_df


class ANNProblem(ElementwiseProblem):
    def __init__(
        self,
        max_number_of_layers,
        max_number_of_nodes_per_layer,
        X,
        y,
        max_iter,
        size,
        cv=2,
        random_state=None,
    ):
        self.X = X
        self.y = y
        self.cv = cv
        self.size = size
        self.random_state = random_state
        self.max_iter = max_iter
        xl = np.zeros(max_number_of_layers)
        xu = np.ones(max_number_of_layers) * max_number_of_nodes_per_layer
        super().__init__(
            n_var=max_number_of_layers,
            n_obj=1,
            n_constr=1,
            xl=xl,
            xu=xu,
            vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        test_accuracy, _, violations = self.calculate_fitness(x)
        out["F"] = [test_accuracy * -1]
        out["G"] = [violations]

    def calculate_fitness(self, decision_variables):
        valid_decision_variables = [dv for dv in decision_variables if dv > 0]

        if len(valid_decision_variables) == 0:
            return None, None, 1
        else:
            classifier = MLPClassifier(
                max_iter=self.max_iter,
                hidden_layer_sizes=tuple(valid_decision_variables),
                random_state=self.random_state,
            )
            X, y = self.X, self.y
            if self.size != -1:
                df = join_df(X, y)
                df = df.sample(n=self.size)
                X, y = split_df(df)
            scores = cross_validate(
                classifier,
                self.X,
                self.y,
                cv=self.cv,
                scoring=("accuracy", "precision_micro"),
                return_train_score=False,
                error_score="raise",
            )
            test_accuracy = sum(scores["test_accuracy"]) / len(scores["test_accuracy"])
            test_precision = sum(scores["test_precision_micro"]) / len(
                scores["test_precision_micro"]
            )
            # print(valid_decision_variables, test_accuracy)
            return test_accuracy, test_precision, 0
