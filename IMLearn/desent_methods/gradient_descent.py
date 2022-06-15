from __future__ import annotations
from typing import Callable, NoReturn
import numpy as np

from IMLearn.base import BaseModule, BaseLR
from .learning_rate import FixedLR
from numpy.linalg import norm

OUTPUT_VECTOR_TYPE = ["last", "best", "average"]


def default_callback(model: GradientDescent, **kwargs) -> NoReturn:
    pass


class GradientDescent:
    """
    Gradient Descent algorithm

    Attributes:
    -----------
    learning_rate_: BaseLR
        Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

    tol_: float
        The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
        specified tolerance

    max_iter_: int
        The maximum number of GD iterations to be performed before stopping training

    out_type_: str
        Type of returned solution:
            - `last`: returns the point reached at the last GD iteration
            - `best`: returns the point achieving the lowest objective
            - `average`: returns the average point over the GD iterations

    callback_: Callable[[GradientDescent, ...], None]
        A callable function to be called after each update of the model while fitting to given data
        Callable function should receive as input a GradientDescent instance, and any additional
        arguments specified in the `GradientDescent.fit` function
    """
    def __init__(self,
                 learning_rate: BaseLR = FixedLR(1e-3),
                 tol: float = 1e-5,
                 max_iter: int = 1000,
                 out_type: str = "last",
                 callback: Callable[[GradientDescent, ...], None] = default_callback):
        """
        Instantiate a new instance of the GradientDescent class

        Parameters
        ----------
        learning_rate: BaseLR, default=FixedLR(1e-3)
            Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

        tol: float, default=1e-5
            The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
            specified tolerance

        max_iter: int, default=1000
            The maximum number of GD iterations to be performed before stopping training

        out_type: str, default="last"
            Type of returned solution. Supported types are specified in class attributes

        callback: Callable[[GradientDescent, ...], None], default=default_callback
            A callable function to be called after each update of the model while fitting to given data
            Callable function should receive as input a GradientDescent instance, and any additional
            arguments specified in the `GradientDescent.fit` function
        """
        self.learning_rate_ = learning_rate
        if out_type not in OUTPUT_VECTOR_TYPE:
            raise ValueError("output_type not supported")
        self.out_type_ = out_type
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.callback_ = callback

    def fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray):
        """
        Optimize module using Gradient Descent iterations over given input samples and responses

        Parameters
        ----------
        f : BaseModule
            Module of objective to optimize using GD iterations
        X : ndarray of shape (n_samples, n_features)
            Input data to optimize module over
        y : ndarray of shape (n_samples, )
            Responses of input data to optimize module over

        Returns
        -------
        solution: ndarray of shape (n_features)
            Obtained solution for module optimization, according to the specified self.out_type_

        Notes
        -----
        - Optimization is performed as long as self.max_iter_ has not been reached and that
        Euclidean norm of w^(t)-w^(t-1) is more than the specified self.tol_

        - At each iteration the learning rate is specified according to self.learning_rate_.lr_step

        - At the end of each iteration the self.callback_ function is called passing self and the
        following named arguments:
            - solver: GradientDescent
                self, the current instance of GradientDescent
            - weights: ndarray of shape specified by module's weights
                Current weights of objective
            - val: ndarray of shape specified by module's compute_output function
                Value of objective function at current point, over given data X, y
            - grad:  ndarray of shape specified by module's compute_jacobian function
                Module's jacobian with respect to the weights and at current point, over given data X,y
            - t: int
                Current GD iteration
            - eta: float
                Learning rate used at current iteration
            - delta: float
                Euclidean norm of w^(t)-w^(t-1)
        """

        sum_sol = np.zeros(f.weights.shape[0])
        best_sol = f.weights
        best_norm_sol = f.compute_output(X=X, y=y)
        num_inter = 0

        for t in range(self.max_iter_):
            sum_sol += f.weights
            old_weights = f.weights
            cur_jacob = f.compute_jacobian(X=X, y=y)
            step = self.learning_rate_.lr_step(t=t)
            f.weights = f.weights - step * cur_jacob
            cur_sol = f.compute_output(X=X, y=y)

            if cur_sol < best_norm_sol:
                best_sol = f.weights
                best_norm_sol = cur_sol
            delta = np.linalg.norm(f.weights - old_weights)
            num_inter += 1

            self.callback_(solver=self, weights=f.weights, val=cur_sol, grad= cur_jacob, t=t,
                           eta=self.learning_rate_.lr_step(t=t), delta=delta)
            if delta < self.tol_:
                break

        if self.out_type_ == OUTPUT_VECTOR_TYPE[0]:
            return f.weights
        if self.out_type_ == OUTPUT_VECTOR_TYPE[1]:
            return best_sol
        if self.out_type_ == OUTPUT_VECTOR_TYPE[2]:
            return sum_sol * (1/num_inter)

        # x_t_arr = []
        # w = f.weights
        # x_t_arr.append(w)
        # best_w = w
        # best_obj = f.compute_output(X=X, y=y)
        #
        # for t in range(self.max_iter_):
        #     etha = self.learning_rate_.lr_step(t=t)
        #     cur_grad = f.compute_jacobian(X=X, y=y)
        #
        #     new_w = w - etha*cur_grad
        #     dist = np.linalg.norm(w-new_w)
        #
        #     w = new_w
        #     f.weights = new_w
        #
        #     cur_obj = f.compute_output(X=X, y=y)
        #     if cur_obj < best_obj:
        #         best_obj = cur_obj
        #         best_w = w
        #     x_t_arr.append(w)
        #     self.callback_(self, w, cur_obj, cur_grad, t, etha, dist)
        #
        #     if dist < self.tol_:
        #         break
        #
        # if self.out_type_ == "last":
        #     return x_t_arr[-1]
        # if self.out_type_ == "best":
        #     return best_w
        # if self.out_type_ == "average":
        #     return np.mean(np.ndarray(x_t_arr))
        #

