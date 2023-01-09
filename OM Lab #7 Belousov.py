import numpy as np
from typing import Callable
import math
import random
import other as o


def boltzmann_annealing(func: Callable[[], float], x0: float, n_max: int = 1000, start_temperature: int = 10):
    def neighbour(x, t):            # Q
        # return [x[i] + t * np.random.standard_cauchy() for i in range(2)]
        return [x[i] + t * np.random.standard_normal() for i in range(2)]

    def temperature_func(n):        # T
        # print(math.log(abs(1.00001 + n)) + 0.0001)
        return t0 / (math.log(abs(1 + n)) + 1)

    def passage(f_new, f_old, t):   # h
        # print('passage', math.exp(-1 * (f_new - f_old) / t))
        try:
            result = math.exp(-1 * (f_new - f_old) / t)
        except:
            print('ERROR: -1 * (f_new - f_old) / t = ', -1 * (f_new - f_old) / t)
            result = 1
        return result
    t0 = start_temperature
    x_current = x0      # x - current point
    n = 1      # n - max number of iteration
    smallest_x = x0
    while n <= n_max:
        temperature = temperature_func(n / n_max)
        x_new = neighbour(x_current, temperature)
        # print('x_current = ', x_current, 'x_new = ', x_new)
        f_old = func(x_current)
        f_new = func(x_new)

        if n_max-n < 10:
            print('x_cur = ', x_current, 'x_new = ', x_new, 't = ', temperature)
            print('passage = ', passage(f_new, f_old, temperature))
        if passage(f_new, f_old, temperature) >= random.random(): # f_new < f_current
            if f_new < func(smallest_x):
                smallest_x = x_current
            x_current = x_new

        n += 1
    if func(smallest_x) < func(x_current):
        print('smallest_x activated')
        print('smallest_x = ', smallest_x, 'func(s_X) = ', func(smallest_x))
        return smallest_x
    else:
        print('smallest_x = ', smallest_x, 'func(s_X) = ', func(smallest_x))
        return x_current


def xing_yao_algorithm(func: Callable[[float], float], x0: float, n_max: int = 10000, start_temperature: int = 10):
    def neighbour(x, t):            # Q
        # return [x[i] + t * np.random.standard_cauchy() for i in range(2)]
        return [x[i] + t * np.random.standard_normal() for i in range(2)]

    def temperature_func(n):        # T
        # print(math.log(abs(1.00001 + n)) + 0.0001)
        return t0 / math.pow(n, 1. / len(x0))

    def passage(f_new, f_old, t):   # h
        # print('passage', math.exp(-1 * (f_new - f_old) / t))
        try:
            result = math.exp(-1 * (f_new - f_old) / t)
        except:
            print('ERROR: -1 * (f_new - f_old) / t = ', -1 * (f_new - f_old) / t)
            result = 1
        return result
    t0 = start_temperature
    x_current = x0      # x - current point
    n = 1      # n - max number of iteration
    smallest_x = x0
    while n <= n_max:
        temperature = temperature_func(n / n_max)
        x_new = neighbour(x_current, temperature)
        # print('x_current = ', x_current, 'x_new = ', x_new)
        f_old = func(x_current)
        f_new = func(x_new)

        if n_max-n < 10:
            print('x_cur = ', x_current, 'x_new = ', x_new, 't = ', temperature)
            print('passage = ', passage(f_new, f_old, temperature))
        if passage(f_new, f_old, temperature) >= random.random(): # f_new < f_current
            if f_new < func(smallest_x):
                smallest_x = x_current
            x_current = x_new

        n += 1
    if func(smallest_x) < func(x_current):
        print('smallest_x activated')
        return smallest_x
    else:
        print('smallest_x = ', smallest_x, 'func(s_X) = ', func(smallest_x))
        return x_current


def function_choice(number):
    # 1 argument - objective function
    # 2 argument - number of dimensions used in function
    # 3 argument - restrictions of non equality
    # 4 argument - restrictions of equality
    if number == 1:     # Функция Стыбинского-Танга (n=2)
        return lambda x: (x[0]**4-16*x[0]**2+5*x[0] + x[1]**4 - 16*x[1]**2 + 5*x[1]) / 2
    elif number == 2:
        return lambda x: math.sin(x[0]+x[1]) + (x[0]-x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1
    elif number == 3:
        return lambda x: -0.0001 * abs(math.sin(x[0]) * math.sin(x[1]) *
                        math.exp(abs(100 - ( math.sqrt(x[0]**2 + x[1]**2)) / 3.14) + 1)) ** 0.1
    elif number == 4:
        return lambda x: -math.cos(x[0])*math.cos(x[1])*math.exp( -(x[0] - 3.14)**2 - (x[1] - 3.14)**2)
    elif number == 5:
        return lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0]+x[1]**2 - 7)**2
    else:
        o.error('sorry, this function is absent (somehow). We will fix this error later')


def main():
    print('''Choose task:
    1. (x**4 - 16*x**2 + 5*x + y**4 - 16*y**2 + 5*y)/2  Функция Стыбинского-Танга (n=2)
    2. sin(x+y) + (x-y)**2 - 1.5*x + 2.5*y + 1  Функция МакКормика
    3.  -0.0001 * [| sin x * sin y * exp(|100 - sqrt(x**2+y**2) / pi|) |+1] ** 0.1  Функция \"крест на подносе\"
    4. Функция Изома
    5. Функция Химмельблау''')

    function_number = o.input_value("int", "Input number of task: ", lambda x: 1 <= x <= 5, "There is no such function")
    function = function_choice(function_number)

    print('''Choose method:
    1. Boltzmann annealing
    2. Xing Yao algorithm''')

    method_number = o.input_value("int", "Input number of method: ", lambda x: 1 <= x <= 2, "There is no such function")

    print("Input start point:")
    start_point = [o.input_value("float", "x0[" + str(i) + "] = ", lambda x: True, "Incorrect data")
                   for i in range(2)]

    max_iteration_number = o.input_value("int", "Input max number of iteration: ", lambda x: x > 0, "Incorrect data")

    if method_number == 1:
        result = boltzmann_annealing(function, start_point, max_iteration_number)
    if method_number == 2:
        result = xing_yao_algorithm(function, start_point, max_iteration_number)



    eps = 0.0001
    # RESULT FORMATTED OUTPUT
    #result_output = [round(i / eps) * eps for i in result]

    print("x*: ", result, "f(x*) = ", function(result))

# INTRODUCTION
print("LABORATORY WORK #7")
print("Methods of Boltzmann annealing and Xing Yao")
while True:
    main()  # Main function call
    continuation = 0
    continuation = o.input_value("int", "\nWould you like to try again? (1 for Yes, 0 for No): ",
                                 lambda x: x == 1 or x == 0, "Only 1 and 0 are permitted, try again")
    if continuation == 0:
        print("Till the next time!")
        break



