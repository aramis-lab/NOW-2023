import pickle
import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from typing import List


class Model:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X =X
        self.y = y
        self.min_x = np.min(self.X[:, 0])
        self.max_x = np.max(self.X[:, 0])
        self.min_y = np.min(self.X[:, 1])
        self.max_y = np.max(self.X[:, 1])
        self.estimator = svm.NuSVC(gamma="auto")

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def bbox(self) -> List[float]:
        return [self.min_x, self.max_x, self.min_y, self.max_y]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        X = np.array(df[["HC_left_volume", "HC_right_volume"]])
        y = np.array([1 if x == "AD" else 0 for x in df["group"].values])
        return cls(X, y)

    def fit(self):
        self.estimator.fit(self.X, self.y)

    def save(self):
        with open("model.pkl", "wb") as fp:
            pickle.dump(self.estimator, fp)

    def plot(self):
        xx, yy = np.meshgrid(
            np.linspace(self.min_x, self.max_x, 500),
            np.linspace(self.min_y, self.max_y, 500),
        )
        Z = self.estimator.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            aspect="auto",
            origin="lower",
            cmap=plt.cm.PuOr_r,
        )
        contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles="dashed")
        plt.scatter(self.X[:, 0], self.X[:, 1], s=30, c=self.y, cmap=plt.cm.Paired, edgecolors="k")
        plt.xticks(())
        plt.yticks(())
        plt.title(f"Number of sample = {self.n_samples}")
        plt.axis(self.bbox)
        plt.show()
