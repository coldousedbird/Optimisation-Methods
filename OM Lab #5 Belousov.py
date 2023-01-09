import other as o
import scipy.optimize as optimize
import numpy as np
from typing import Callable, List




def projected_gradient_method(x0, objective_func, rest_eq, rest_not_eq):
    return o.answer(x0, objective_func, rest_eq, rest_not_eq)


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
    2. (1.2) - x1**2 + x2**2                    | Restrictions: 2 − x1 − x2 >= 0 ; x1 − 1 = 0
    3. (1.3) - x1**2 + x2**2                    | Restrictions: 2 − x1 − x2 >= 0 ; x1 − 1 >= 0 
    4. (2.1) - (x1+1)**3 /3 +x2                 | Restrictions: x1 − 1 >= 0 ; x2 >= 0
    5. (2.2) - x1 + x2                          | Restrictions: x1**2 - x2 <= 0 ; x1>=0
    6. (2.5) - 4*(x1**2) + 8*x1-x2-3            | Restrictions: x1 + x2 = -2
    7. (2.9) - (x1+4)**2 + (x2-4)**2            | Restrictions: 2*x1 - x2 <=2 ; x1 >= 0 ; x2>=0 
    8. (2.13)- -x1 * x2 * x3                    | Restrictions: 0<=x1<=42 ; 0<=x2<=42 ; 0<=x3<=42 ; x1 + 2*x2 + 2*x3 <=72
    9. (hard)- (1 - x1)**2 + 100*(x2-x1**2)**2  | Restrictions: (x1-1)**3 - x2 + 1 < 0 ; x1 + x2 - 2 < 0
    """)

    function_number = o.input_value("int", "Input number of task: ", lambda x: 1 <= x <= 9, "There is no such function")
    function, dimensions_num, restrictions_of_equality, restrictions_of_non_equality = function_choise(function_number)

    # INPUT X0
    print("Input start point:")
    start_point = [o.input_value("float", "x0[" + str(i) + "] = ", lambda x: True, "Incorrect data")
                   for i in range(dimensions_num)]
    eps = 0.001

    # CALL METHODS
    result = projected_gradient_method(start_point, function, restrictions_of_equality, restrictions_of_non_equality)

    # RESULT FORMATTED OUTPUT
    result_output = [round(i / eps) * eps for i in result]
    print("x*: ", result_output, "f(x*) = ", function(result))
    return 0


# INTRODUCTION
print("LABORATORY WORK #5")
print("Methods of Projected Gradient")
while True:
    main()  # Main function call
    continuation = 0
    continuation = o.input_value("int", "\nWould you like to try again? (1 for Yes, 0 for No): ",
                               lambda x: x == 1 or x == 0, "Only 1 and 0 are permitted, try again")
    if continuation == 0:
        print("Till the next time!")
        break
