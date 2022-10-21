# LIBRARIES SETUP
import numpy as np
from typing import Callable, List
from scipy import optimize
import numdifftools as nd
import OneDimensionOptimization as ODO

# needed for NR and FR, don't know why
Path = []
# Number of dimensions in functions
dimensionsInFunctions = [2, 2, 4, 4]


def function_selection(function_number):
    if function_number == 1:
        return lambda x: 4*(x[0]-5)**2 + (x[1]-6)**2
    if function_number == 2:
        return lambda x: (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
    if function_number == 3:
        return lambda x: 100*(x[1]-x[0]**2)**2+(1-x[0])**2+90*(x[3]-x[2]**2)**2+(1-x[2])**2+10.1*((x[1]-1)**2+(x[3]-1)**2)+19.8*(x[1]-1)*(x[3]-1)
    if function_number == 4:
        return lambda x: (x[0]+10*x[1])**2+5*(x[2]-x[3])**2+(x[1]-2*x[2])**4+10*(x[0]-x[3])**4


# Gauss-Seidel method
def gauss_seidel(func: Callable[..., float], x_start: List[float], epsilon: float):
    step_crushing_ratio = 0.01
    dimensions_num = len(x_start)
    steps = np.array([1.0] * dimensions_num)
    point = x_start
    while steps[0] > epsilon:
        old_point = point.copy()

        for i in range(dimensions_num):             # cycle for dimensions
            def one_dimension_func(x):              # create function for only one plane
                args = point.copy()                  # copy last point in args
                args[i] = x
                return func(args)
            point[i] = ODO.fibonacci_minimization(one_dimension_func, point[i], steps[i], epsilon)

        full_old_point = old_point.copy()
        full_old_point.append(func(old_point))

        full_point = point.copy()
        full_point.append(func(point))

        if np.linalg.norm(np.array(full_old_point) - np.array(full_point)) <= epsilon:
            break

        steps -= step_crushing_ratio
    return point


# Fletcher-Reeves method
def fletcher_reeves(func: Callable[..., float], x_start: List[float], epsilon: float):
    xcur = np.array(x_start)
    Path.append(xcur)               # path - story of points
    dimensions_num = len(x_start)
    iteration = 0   # step1
    grad = optimize.approx_fprime(xcur, func, epsilon**4)  # step2
    prev_grad = 1
    pk = -1*grad
    while any([abs(grad[i]) > epsilon**2 for i in range(dimensions_num)]):   # step3
        if iteration % dimensions_num == 0:  # step4
            pk = -1*grad
        else:
            bk = (np.linalg.norm(grad)**2)/(np.linalg.norm(prev_grad)**2)    # step5
            prev_pk = pk
            pk = -1*grad + bk*prev_pk  # step6
        a = optimize.minimize_scalar(lambda x: func(xcur+pk*x), bounds=(0,)).x
        xcur = xcur + a*pk  # step8
        Path.append(xcur)
        iteration = iteration+1     # step8
        prev_grad = grad
        grad = optimize.approx_fprime(xcur, func, eps**4)     # step2
    return xcur  # step10


# Newton-Raphson method
def newton_raphson(func: Callable[..., float], x_start: List[float], epsilon: float):
    xcur = np.array(x_start)
    Path.append(xcur)
    hess_f = nd.Hessian(func)   # Hessian
    dimensions_num = len(x0)
    grad = optimize.approx_fprime(xcur, func, epsilon ** 4)  # step2
    y = 0
    while any([pow(abs(grad[i]), 1.5) > epsilon for i in range(dimensions_num)]):  # step3
        y = y + 1
        step = np.linalg.inv(hess_f(xcur))  # step 4 & 5
        pk = (-1 * step).dot(grad)  # step 6
        a = optimize.minimize_scalar(lambda a: func(xcur + pk * a), bounds=(0,)).x  # step7
        xcur = xcur + a * pk  # step8
        Path.append(xcur)
        grad = optimize.approx_fprime(xcur, func, epsilon * epsilon)  # step2
        if y > 100:
            break
    return xcur  # step10


while True:
    # METHOD PRINT
    print("Choose method: ")
    print('1. Gauss-Seidel method')
    print('2. Fletcher-Reeves method')
    print('3. Newton-Raphson method')

    # METHOD CHOICE
    method = -1
    while True:
        try:
            method = int(input('Input method number: '))
        except:
            print("Data is incorrect")
            continue
        if 0 < method < 4:
            break
        print('There is no such number')

    # FUNCTION PRINT
    print('Choose function:')
    print('1. 4*(x1-5)**2 + (x2-6)**2')
    print('2. (x1**2+x2-11)**2+(x1+x2**2-7)**2')
    print("3. 100*(x2-x1**2)**2+(1-x1)**2+90*(x4-x3**2)**2+(1-x3)**2+10.1((x2-1)**2+(x4-1)**2)+19.8*(x2-1)*(x4-1)")
    print("4. (x1+10*x2)**2+5*(x3-x4)**2+(x2-2*x3)**4+10*(x1-x4)**4")

    # FUNCTION CHOICE
    function_num = -1
    while True:
        try:
            function_num = int(input('Input function number: '))
        except:
            print("Data is incorrect")
            continue
        if 0 < function_num < 5:
            break
        print('There is no such number')

    # GET FUNCTION
    function = function_selection(function_num)

    # MAIN DATA INPUT (start point, steps lengths and precision)
    # variables initialize
    x0 = [0. for i in range(dimensionsInFunctions[function_num-1])]         # START POINT
    h = [0.0001 for i in range(dimensionsInFunctions[function_num-1])]      # STEP LENGTHS (increments)
    x_min = [0. for i in range(dimensionsInFunctions[function_num-1])]      # MIN POINT (answer)

    # start point input (necessary for all methods)
    print('Input coordinates of start:')
    i = 0
    while i < dimensionsInFunctions[function_num-1]:
        try:
            x0[i] = float(input(str(i+1)+': '))
            i = i+1
        except:
            print('Incorrect input')

    # step length input (useless)
    if method == -1:
        i = 0
        print('Input increments:')
        while i < dimensionsInFunctions[function_num-1]:
            try:
                h[i] = float(input(str(i+1)+': '))
                i = i+1
            except:
                print('Incorrect input')

    # precision (eps) input
    while True:
        try:
            eps = float(input("Input precision (1 > eps > 0): "))
            if 1 > eps > 0:
                break
            elif eps <= 0:
                print('eps must be positive number, try again')
            elif eps > 1:
                print('Your eps is too big, try again')
        except:
            print('Incorrect input')

    # METHOD CALL
    # Gauss-Seidel method
    if method == 1:
        x_min = gauss_seidel(function, x0, eps)

    # Fletcher-Reeves
    if method == 2:
        x_min = fletcher_reeves(function, x0, eps)

    # Newton-Rafson
    if method == 3:
        x_min = newton_raphson(function, x0, eps)

    # ANSWER OUTPUT
    print("x_min:", x_min)

    # CHOICE, DOES USER NEED TO TRY AGAIN
    cont = -1
    while True:
        try:
            cont = int(input('Do you want to try again? (0 for No, 1 for Yes): '))
        except:
            print('Incorrect data, try again')
            continue
        if cont != 0 and cont != 1:
            print('There is no such number, try again!')
            continue
        if cont == 0:
            print('Till the next time!')
            break
        if cont == 1:
            print()
            break

    # exit from program
    if cont == 0:
        break
