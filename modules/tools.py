from typing import Sequence, Mapping, Any
from modules.matrix import Matrix
import csv

class Datas:
    @staticmethod
    def load_csv(csv_filename: str, model: str) -> "Datas":
        with open(csv_filename, newline="") as csvfile:
            content = list(csv.reader(csvfile, delimiter=","))
        names = content[0]
        values_str = content[1:]
        values = [[int(value) for value in line] for line in values_str]

        return Datas(names, values, model)

    def __init__(self, names: Sequence[Any], values: Sequence[Sequence[int]], model: str):
        if not len(names) == len(values[0]):
            raise ValueError("names does not fit values")
        if model not in ["linear_regression"]:
            raise ValueError("unknown model")
        self._names = names
        self._values = values
        self._model = model
        matrix = Matrix(values)
        datas_values = matrix.T
        # number of sample in dataset
        self._m = len(datas_values.matrix[0])
        self._datas_table = {}
        for (column_name, column_values) in zip(names, datas_values.matrix):
            self.datas_table[column_name] = column_values

    @property
    def names(self):
        return self._names

    @property
    def values(self):
        return self._values

    @property
    def model(self):
        return self._model

    @property
    def datas_table(self):
        return self._datas_table

    @property
    def m(self):
        """
        number of sample in dataset
        """
        return self._m

    @property
    def n(self):
        """
        number of features in dataset
        """
        return self._n
    
    @property
    def features(self):
        """
        dataset features
        """
        return self._features

    @property
    def target(self):
        """
        dataset target
        """
        return self._target
    
    def set_features(self, column_names: Sequence[str]):
        self._features = {}
        for name in column_names:
            self.features[name] = self._datas_table[name]
        self._n = len(self.features)

    def set_target(self, column_name: str):
        self._target = {column_name: self._datas_table[column_name]}

    def get_polynomial_expression_result(self, y: float, x: float, coef: Sequence[float]):
        """
        Formula to calculate polynomial expression like y = ax + b or y = ax² + bx + c
        """
        for b in coef:
            y = y * x + b
        return y

    def MSE(self, target_y: Sequence[float], calculate_y: Sequence[float]) -> float:
        """
        Mean Squared Error
        """
        cost = 0
        for i in range(0, self._m - 1) :
            cost = cost + (calculate_y[i] - target_y[i]) * (calculate_y[i] - target_y[i])
        cost = cost / (2*self._m)
        return cost

    def RMSE(self):
        pass

    def gradient_descent(self, a: float, b: float, target_y: Sequence[float], alpha: float):
        """
        Use gradient descent algorithm to get new parameters that will bring us closer to the
        minimum Mean Squared Error possible
        """
        gradient_a = 0
        gradient_b = 0
        for i in range(0, self._m - 1) :
            gradient_a = gradient_a + self._features["x1"][i] * (a * self._features["x1"][i] + b - target_y[i])
            gradient_b = gradient_b + (a * self._features["x1"][i] + b - target_y[i])
        gradient_a = gradient_a / self._m
        gradient_b = gradient_b / self._m

        a = a - alpha * gradient_a
        b = b - alpha * gradient_b
        return a, b

