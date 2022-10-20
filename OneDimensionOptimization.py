from typing import Callable, List

# x0 - start point
# h - step length
def davis_swenn_kampy(func: Callable[..., float], x0: float, h: float):
    a = 0   # left border
    b = 0   # right border
    k = 0   # number of steps in one of
    wrong_x0_flag = 0
    # step right
    if func(x0 + k*h) >= func(x0 + (k+1)*h):
        #print('PATH 1')
        # minimum is to the right side of the begining point
        a = x0  # x0 is left border
        k += 1  # take step right
        while True:
            if k > 1000:
                wrong_x0_flag = 1
                break
            if func(x0 + k*h) < func(x0 + (k+1)*h):
                # right border was found! Success!
                b = x0 + (k+1)*h
                break
            else:
                # right border wasn't found, take another step right
                a = x0 + k*h
                k += 1
    elif func(x0 - k*h) >= func(x0 - (k+1)*h):
        #print('PATH 2')
        # minimum is to the left side of the begining point
        b = x0  # x_0 is right border
        k += 1  # take step left
        while True:
            if k > 1000:
                wrong_x0_flag = 1
                break
            if func(x0 - k * h) < func(x0 - (k+1)*h):
                # левый край найден! Отлично! Задача выполнена!
                a = x0 - (k + 1) * h
                break
            else:
                # right border wasn't found, take another step right
                b = x0 - k * h
                k += 1

    else:
        #print('PATH 3')
        # minimum is in between one step left and one step right of the start
        a = x0 - h
        b = x0 + h

    if wrong_x0_flag == 0:
        #print('[a,b] = [',a,', ',b,']')
        return [a,b]
    else:
        print('ERROR: The function is decreasing too long. Try to choose another x0 or bigger step length.')
        quit()





# Function finds member of fibonacci sequence under index N
def fibonacci(N):
    if N < 0:
        return 0
    if N == 0:
        return 1
    else: return fibonacci(N-1) + fibonacci (N-2)


# Fibbonacci method
def fibonacci_minimization(func: Callable[..., float], x0: float, h:float, eps: float):
    a, b=davis_swenn_kampy(func, x0, h)
    if eps <= 0:
        print('Choose positive eps')
        quit()
    if (b-a)/(2*eps) <= 2 and eps > 0:
        print('Try to choose lesser eps')
        quit()

    delta = eps/10
    # Step 1
    N=0
    while fibonacci(N)<(b-a)/(2*eps):
        N+=1
    # Step 2
    k=0
    # Step 3
    x1 = a+fibonacci(N-2)/fibonacci(N)*(b-a)
    x2 = a+fibonacci(N-1)/fibonacci(N)*(b-a)

    while True:
        # Steps 4 & 5
        if func(x1)<=func(x2):
            b=x2
            x2=x1
            x1=a+(b-a)*fibonacci(N-k-3)/fibonacci(N-k-1)
        else:
            a=x1
            x1=x2
            x2=a+(b-a)*fibonacci(N-k-2)/fibonacci(N-k-1)
        # Step 6
        if k==N-3:
            x2=x1+delta
            if func(x1)<=func(x2):
                b=x2
            else:
                a=x1
            break
        else:
            k=k+1
    # Step 7
    min_x=(a+b)/2

    return round(min_x/eps)*eps


# x_points = [0.0, 0.0]
# steps = [0.5, 1.0]
# epsilon = 0.0001
# i= 0
#
#
# def one_dimension_func(x):  # create function for only one plane
#     i=0
#     func = lambda x: 4*(x[0]-5)**2+(x[1]-6)**2
#     args=x_points.copy()  # copy last point in args
#     args[i]=x
#     return func(args)
#
#
#
# # minimizing that one function
# new_point = fibonacci_minimization(one_dimension_func, x_points[i], steps[i], epsilon)
# print('new point = ', new_point)

# function = lambda x: x**2 - 12*x + 40
# a = fibonacci_minimization(function, 4, 1, 0.001)
# print('min_x = ', a)