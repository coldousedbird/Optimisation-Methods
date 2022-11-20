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
def penalty_method(x0, objective_func, alpha_start, beta, eps, rest_eq, rest_not_eq):
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


def barrier_method(x0, objective_func, alpha_start, beta, eps, rest_not_eq):
    alpha = alpha_start

    def getAuxilitaryFunctionResult(objective_func, alpha, rest_not_eq, x):
        H = sum(1 / (0.000000001 + max(0, -i(x))**2) for i in rest_not_eq)
        return objective_func(x) + alpha * H
    xcur = np.array(x0)
    xnew = None
    atLeastOnePointFound = False
    while not (atLeastOnePointFound and (((xcur - xnew) ** 2).sum() < eps ** 2)):
        xtemp = unconditional_optimization(lambda x: getAuxilitaryFunctionResult(objective_func, alpha,
                                                                                 rest_not_eq, x), xcur)
        isInside = not any(neq(xtemp) > eps for neq in rest_not_eq)
        if (isInside):
            if not atLeastOnePointFound:
                atLeastOnePointFound = True
            else:
                xcur = xnew
            xnew = xtemp

        alpha *= beta
        if alpha < 0.00001:
            break
    print("alpha in the end = ", alpha)
    return xnew


# TASK ITSELF
def function_choise(number):
    # 1 argument - objective function
    # 2 argument - number of dimensions used in function
    # 3 argument - restrictions of non equality
    # 4 argument - restrictions of equality

    if number == 1:
        return [lambda x: x[0] ** 2 + x[1] ** 2, 2,
                [lambda x: x[0] + x[1] - 2], []]
    elif number == 2:
        return [lambda x: x[0] ** 2 + x[1] ** 2, 2,
                [lambda x: x[0] - 1], [lambda x: x[0] + x[1] - 2]]
    elif number == 3:
        return [lambda x: x[0] ** 2 + x[1] ** 2, 2,
                [], [lambda x: 1 - x[0], lambda x: x[0] + x[1] - 2]]
    elif number == 4:
        return [lambda x: (x[0] + 1) ** 3 / 3 + x[1], 2,
                [], [lambda x: 1 - x[0], lambda x: -x[1]]]
    elif number == 5:
        return [lambda x: x[0] + x[1], 2,
                [], [lambda x: x[0] ** 2 - x[1], lambda x: -x[0]]]
    elif number == 6:
        return [lambda x: 4 * (x[0] ** 2) + 8 * x[0] - x[1] - 3, 2,
                [lambda x: x[0] + x[1] + 2], []]
    elif number == 7:
        return [lambda x: (x[0] + 4) ** 2 + (x[1] - 4) ** 2, 2,
                [],
                [lambda x: 2 * x[0] - x[1] - 2, lambda x: -x[0], lambda x: -x[1]]]
    elif number == 8:
        return [lambda x: -x[0] * x[1] * x[2], 3,
                [],
                [lambda x: -x[0], lambda x: x[0] - 42, lambda x: -x[1], lambda x: x[1] - 42,
                 lambda x: -x[2], lambda x: x[2] - 42, lambda x: x[0] + 2 * x[1] + 2 * x[2] - 72]]
    elif number == 9:
        return [lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2, 2,
                [],
                [lambda x: (x[0] - 1) ** 3 - x[1] + 1, lambda x: x[0] + x[1] - 2]]


def main():
    print("""Choose task:
    1. (1.1) - x1**2 + x2**2                    | Restrictions: x1 + x2 -2 = 0   
    2. (1.2) - x1**2 + x2**2                    | Restrictions: 2 − x1 − x2 >= 0 ; x1 −1 = 0
    3. (1.3) - x1**2 + x2**2                    | Restrictions: x1 −1 >= 0 ; 2 − x1 − x2 >= 0
    4. (2.1) - (x1+1)**3 /3 +x2                 | Restrictions: x1 − 1 >= 0 ; x2 >= 0
    5. (2.2) - x1 + x2                          | Restrictions: x1**2 - x2 <= 0 ; x1>=0
    6. (2.5) - 4*(x1**2) + 8*x1-x2-3            | Restrictions: x1 + x2 = -2
    7. (2.9) - (x1+4)**2 + (x2-4)**2            | Restrictions: 2*x1 - x2 <=2 ; x1 >= 0 ; x2>=0 
    8. (2.13)- -x1 * x2 * x3                    | Restrictions: 0<=x1<=42 ; 0<=x2<=42 ; 0<=x3<=42 ; x1 + 2*x2 + 2*x3 <=72
    9. (hard)- (1 - x1)**2 + 100*(x2-x1**2)**2  | Restrictions: (x1-1)**3 - x2 + 1 < 0 ; x1 + x2 - 2 < 0
    """)

    function_number = input_value("int", "Input number of task: ", lambda x: 1 <= x <= 9, "There is no such function")
    function, dimensions_num, restrictions_of_equality, restrictions_of_non_equality = function_choise(function_number)

    # If there is no restrictions of equality, you can choose one of two methods
    if len(restrictions_of_equality) == 0:
        print('Methods:\n1. Penalty\n2. Barrier')
        method = input_value("int", "Choose method number: ", lambda x: 1 <= x <= 2, "There is no such method")
    else:
        method = 1

    # INPUT penalty parameter, Beta, Eps
    alpha_start = 1
    beta = 2
    eps = 0.001
    print("alpha_start = 1, beta = 2, nu = 1/2, eps = 0.0001")

    # INPUT X0
    print("Input start point:")
    start_point = [input_value("float", "x0[" + str(i) + "] = ", lambda x: True, "Incorrect data")
                   for i in range(dimensions_num)]

    while any(i(start_point) > eps for i in restrictions_of_non_equality) and method == 2:
        print("Start_point doesn't fit for barrier method. Try again:")
        start_point = [input_value("float", "x0[" + str(i) + "] = ", lambda x: True, "Incorrect data")
                       for i in range(dimensions_num)]

    # CALL METHODS
    result = []
    if method == 1:
        result = penalty_method(start_point, function, alpha_start, beta, eps,
                                restrictions_of_equality, restrictions_of_non_equality)
    else:
        result = barrier_method(start_point, function, alpha_start, 1 / beta, eps, restrictions_of_non_equality)

    # RESULT FORMATTED OUTPUT
    result_output = [round(i / eps) * eps for i in result]
    print("x*: ", result_output, "f(x*) = ", function(result))
    return 0


# INTRODUCTION
print("LABORATORY WORK #4")
print("Methods of barrier and penalty functions")
while True:
    main()  # Main function call
    continuation = 0
    continuation = input_value("int", "\nWould you like to try again? (1 for Yes, 0 for No): ",
                               lambda x: x == 1 or x == 0, "Only 1 and 0 are permitted, try again")
    if continuation == 0:
        print("Till the next time!")
        break
