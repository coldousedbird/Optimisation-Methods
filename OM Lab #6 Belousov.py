import numpy as np
from typing import Callable, List
import other as o
import math as mt

dimensionsInFunctions = [2 for i in range(7)]


def function_selection(function_number):
    if function_number == 1:
        return lambda x: 4 * (x[0] - 5) ** 2 + (x[1] - 6) ** 2
    if function_number == 2:
        if dimensionsInFunctions[1] == 2:
            return lambda x: 10 * 2 + (
                        (x[0] ** 2 - 10 * mt.cos(2 * 3.14 * x[0])) + (x[1] ** 2 - 10 * mt.cos(2 * 3.14 * x[1])))
        if dimensionsInFunctions[1] == 3:
            return lambda x: 10 * 3 + (
                        (x[0] ** 2 - 10 * mt.cos(2 * 3.14 * x[0])) + (x[1] ** 2 - 10 * mt.cos(2 * 3.14 * x[1])) + (
                            x[2] ** 2 - 10 * mt.cos(2 * 3.14 * x[2])))
        if dimensionsInFunctions[1] == 4:
            return lambda x: 10 * 4 + ((x[0] ** 2 - 10 * mt.cos(2 * 3.14 * x[0])) +
                                       (x[1] ** 2 - 10 * mt.cos(2 * 3.14 * x[1])) +
                                       (x[2] ** 2 - 10 * mt.cos(2 * 3.14 * x[2])) +
                                       (x[3] ** 2 - 10 * mt.cos(2 * 3.14 * x[3])))
    if function_number == 3:
        return lambda x: -20 * mt.exp(-0.2 * mt.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))) - mt.exp(
            0.5 * (mt.cos(2 * 3.14 * x[0]) + mt.cos(2 * 3.14 * x[1]))) + 2.71828 + 20
    if function_number == 4:
        return lambda x: (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (
                    2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    if function_number == 5:
        return lambda x: 100 * mt.sqrt(abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * abs(x[0] + 10)
    if function_number == 6:
        return lambda x: 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]
    if function_number == 7:
        return lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (x[0] - 1) ** 2


def best_samples(func: Callable[[np.array], float], x_start: List[float], eps: float, N: int = 100, M: int = 300,
                 b: float = 0.5):
    x = x_start
    k = 0
    h = 1
    while k < N:
        y_path = []
        f = []
        for _ in range(M):
            e = np.random.uniform(-1, 1, len(x))
            y = x + h * e / np.linalg.norm(e)
            y_path.append(y)
            f.append(func(y))

        min_index = np.argmin(f)
        f_min = f[min_index]

        if f_min < func(x):
            x = y_path[min_index]
            k += 1
        else:
            if h <= eps:
                return x
            elif h > eps:
                h *= b
    return x


def adaptive_method(func: Callable[[np.array], float], x_start: List[float], eps: float, N: int = 100, M: int = 300,
                    a: float = 1.5, b: float = 0.5):
    x = x_start
    h = 1
    k = 0
    j = 1
    while k < N:
        e = np.random.uniform(-1, 1, len(x))
        y = x + h * e / np.linalg.norm(e)

        if func(y) < func(x):
            z = x + a * (y - x)
            if func(z) < func(x):
                x = z
                h *= a
                k += 1
                j = 1
                continue

        if j < M:
            j += 1
        else:
            if h <= eps:
                break

            h *= b
            j = 1

    return x


def main():
    print('''Choose method:
    1. Adaptive search method
    2. Best samples method''')

    method = o.input_value('int', 'Method number: ', lambda x: 0 < x < 3, 'There is no such method')

    print('''Choose function:
    1. 4*(x1-5)**2 + (x2-6)**2
    2. 10*n + sum[x*i**2-10*cos(2*π*xi)] 
    3. -20 * exp(-0.2 * sqrt(0.5 * (x1**2 + x2**2)) - exp(0.5 * (cos(2*3.14*x1) + cos(2*3.14*x1))) + e + 20
    4. (1.5-x1+x1*x2) ** 2 + (2.25-x1+x1*x2**2) ** 2 + (2.625-x1+x1*x2**3) ** 2
    5. 100*sqrt(x2-0.01*x1**2) + 0.01*(x1+10)
    6. 0.26*(x1**2+x2**2) - 0.48*x1*x2
    7. 100*(x2-x1**2)**2 + (x1-1)**2''')

    function_num = o.input_value("int", 'Function number: ', lambda x: 0 < x < 8, 'There is no such function')
    if function_num == 2:
        dimensionsInFunctions[function_num - 1] = o.input_value("int", 'Dimensions number: ', lambda x: 1 < x < 5,
                                                                'Dimensions must be at least >1 and <5')

    function = function_selection(function_num)
    print('Input start point: ')
    x0 = [o.input_value('float', 'x0[' + str(i) + ']: ', lambda x: True, '')
          for i in range(dimensionsInFunctions[function_num - 1])]
    x_min = [0. in range(dimensionsInFunctions[function_num - 1])]

    eps = 0.001

    if method == 1:
        x_min = adaptive_method(function, x0, eps)

    if method == 2:
        x_min = best_samples(function, x0, eps)

    print('x*: ', x_min, ' func(x*): ', function(x_min))


# INTRODUCTION
print("LABORATORY WORK #6")
print("Methods of statistic and stochastic optimization")
while True:
    main()  # Main function call
    continuation = o.input_value("int", "\nWould you like to try again? (1 for Yes, 0 for No): ",
                                 lambda x: x == 1 or x == 0, "Only 1 and 0 are permitted, try again")
    if continuation == 0:
        print("Till the next time!")
        break
