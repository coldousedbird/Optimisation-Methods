import math
import scipy.optimize as optimize
import numpy as np
from typing import Callable, List


need_to_see_debug_notes = 0


# JUST DIFFERENT USEFUL FUNCTIONS
def debug_note(string):
    if need_to_see_debug_notes:
        print(string)


def error(prompt):
    print(prompt)
    exit()


def input_value(type_of_data, prompt, restrictions, error_prompt):
    i = 0
    while True:
        value = 0
        i += 1
        if i > 20:
            error("too much attempts")
        if type_of_data == "int" or type_of_data == "integer":
            try:
                value = int(input(prompt))
            except KeyboardInterrupt:
                error("Ok...")
            except:
                print("Incorrect data (must be int), try again")
                continue
        elif type_of_data == "float":
            try:
                value = float(input(prompt))
            except KeyboardInterrupt:
                error("Ok...")
            except:
                print("Incorrect data (must be float), try again")
                continue
        elif type_of_data == "bool" or type_of_data == "boolean":
            try:
                value = bool(input(prompt))
            except KeyboardInterrupt:
                error("Ok...")
            except:
                print("Incorrect data (must be boolean), try again")
                continue
        elif type_of_data == "str" or type_of_data == "string":
            try:
                value = str(input(prompt))
            except KeyboardInterrupt:
                error("Ok...")
            except:
                print("Incorrect data (must be string), try again")
                continue
        else:
            error("Incorrect data type")
        try:
            if restrictions(value):
                return value
            else:
                print(error_prompt)
                continue
        except KeyboardInterrupt:
            error("Ok...")
        except:
            error("Incorrect restrictions")


# UNCONDITIONAL METHOD OF MANY-DIMENSIONAL OPTIMIZATION
def unconditional_optimization(func: Callable[..., float], x_start: List[float], epsilon: float = 0.001):
    return optimize.minimize(fun=func, x0 = x_start, method="BFGS", options={'eps': epsilon}).x

# CONDITIONAL METHODS OF MANY-DIMENSIONAL OPTIMIZATION
def answer(x0, objective_func, rest_eq, rest_not_eq):
    # INPUT penalty parameter, Beta, Eps
    alpha_start = 1
    beta = 2
    eps = 0.001
    # print("alpha_start = 1, beta = 2, nu = 1/2, eps = 0.0001")
    alpha = alpha_start

    def getAuxilitaryFunctionResult(objective_func, alpha, rest_eq, rest_not_eq, x):
        H = 0
        for i in rest_eq:
            H += pow(abs(i(x)), 2)
        for i in rest_not_eq:
            H += pow(max(0, i(x)), 2)
        return objective_func(x) + alpha * H
    xcur = np.array(x0)
    xnew = unconditional_optimization(lambda x: getAuxilitaryFunctionResult(objective_func, alpha,
                                                                            rest_eq, rest_not_eq, x), xcur, eps)
    while ((xcur - xnew)**2).sum() > eps:
        alpha *= beta
        xcur = xnew
        xnew = unconditional_optimization(lambda x: getAuxilitaryFunctionResult(objective_func, alpha,
                                                                                rest_eq, rest_not_eq, x), xcur, eps)
        if alpha > 100000:
            break
    print("alpha in the end = ", alpha)
    return xnew