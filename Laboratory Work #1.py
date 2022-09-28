#import numpy as np
import math

# functions, been analyzed, from book
def function(x,n):
    if n == 1: return pow((x-1),2)
    if n == 2: return 4*pow(x,3) - 8*pow(x,2) - 11*x + 5
    if n == 3:
        if x == 0:
            print('Please, move x0 sligtly (about 0.001), program falls at non existing point.')
            quit()
        else:
            return x+(3/(x**2))
    if n == 4:
        if x == 2.0:
            # return (x + 2.0001) / (4 - pow(x, 2.0001))
            print('Please, move x0 sligtly (about 0.001), program falls at non existing point.')
            quit()
        elif x == -2.0:
            # return (x - 2.0001) / (4 - pow(x, -2.0001))
            print('Please, move x0 sligtly (about 0.001), program falls at non existing point.')
            quit()
        else:
            return (x+2.5) / (4-pow(x,2))
    if n == 5: return -math.sin(x)-(math.sin(3*x)/3)
    if n == 6: return -2*math.sin(x) - math.sin(2*x) - 2*math.sin(3*x)/3




# x0 - beginning point
# h - step
# n - number of analyzing function
def DSK(x0, h, n):
    a = 0   # левая граница
    b = 0   # правая граница
    k = 0   # количество шагов в одну из сторон
    wrong_x0_flag = 0
    #  отходим шаг от начала вправо и смотрим по краям
    if function(x0 + k*h, n) >= function(x0 + (k+1)*h, n):
        print('PATH 1')
        # точка минимума находится справа от начальной точки
        a = x0  # принимаем x0 за левый край
        k += 1  # делаем ещё шаг вправо
        while True:
            if k>100:
                wrong_x0_flag = 1
                break
            if function(x0 + k*h, n) < function(x0 + (k+1)*h, n):
                # правый край найден! Отлично! Задача выполнена!
                b = x0 + (k+1)*h
                break
            else:
                # правый край не найден, сдвигаем левый край и делаем ещё шаг вправо
                a = x0 + k*h
                k += 1
    elif function(x0 - k*h, n) >= function(x0 - (k+1)*h, n):
        print('PATH 2')
        # точка минимума находится слева от начальной точки
        b = x0  # принимаем x0 за правый край
        k += 1  # делаем ещё шаг влево
        while True:
            if k > 100:
                wrong_x0_flag = 1
                break
            if function(x0 - k * h, n) < function(x0 - (k+1)*h, n):
                # левый край найден! Отлично! Задача выполнена!
                a = x0 - (k + 1) * h
                break
            else:
                # левый край не найден, сдвигаем правый край влево и делаем ещё шаг влево
                b = x0 - k * h
                k += 1
        #print('[a,b] = [', a, ', ', b, ']')
        #print('wrong_x0_flag = ', wrong_x0_flag)
    else:
        print('PATH 3')
        # точка минимума находится между одним шагом влево и одним шагом вправо
        a = x0 - h
        b = x0 + h

    if wrong_x0_flag == 0:
        print('[a,b] = [',a,', ',b,']')
        return [a,b]
    else: return 'Incorrect' # вот здесь я заменил i на I и всё заработало

# Function finds member of fibonacci sequence under index N
def fibonacci(N):
    if N < 0:
        return 0
    if N == 0:
        return 1
    else: return fibonacci(N-1) + fibonacci (N-2)

# Choose method
print('Methods:')
print('    0. Davis-Swenn-Kampy method')
print('    1. Passive search method')
print('    2. Fibonacci method')
# print('    3. Gauss-Seidel method')
# print('    4. Fletcher-Reeves method')
method = int(input('Input method number: '))
if method > 2 or method < 0: print('Wrong number')

# Input
f_N = int(input('Input number of function: '))
while (f_N > 6 or f_N < 1):
    print('There is no such function, bro. Choose another one, from 1 to 5.')
    f_N = int(input('Input number of function: '))

x0 = float(input('x0: '))
h = float(input('h>0: '))
while (h <= 0):
    print('Step number must be bigger then 0')
    f_N = int(input('Input number of function: '))
segment = DSK(x0, h, f_N)
if segment == 'Incorrect':
    print('You have chosen incorrect x0 or h, do another try.')
    method = -1
else:
    a = segment[0]
    b = segment[1]

# Passive Search method
if method == 1:
    x = a

    N = int(input('N = '))
    length = (b-x)/N
    # Search
    min_f_x = function(x, f_N)
    min_x = x
    for i in range(N):
        f_x = function(x+i*length, f_N)
        if f_x < min_f_x:
            min_f_x = f_x
            min_x = x+i*length
    # Output
    print('f(x*) = min f(x) = ', round(min_f_x*1000)/1000)
    print('x* = ', round(min_x*1000)/1000)

# Fibonacci method
if method == 2:
    while True:
        eps = float(input('eps(>0) = '))
        if eps <= 0:
            print('Choose positive eps')
            continue
        if (b - a) / (2 * eps) <= 2 and eps > 0:
            print('Try to choose lesser eps')
            continue
        if eps > 0 and (b - a) / (2 * eps) > 2: break
    delta = eps / 10
    # Step 1
    N = 0
    while fibonacci(N) < (b-a)/(2*eps):
        N += 1
    # Step 2
    k = 0
    # Step 3
    x1 = a + fibonacci(N-2)/fibonacci(N)*(b-a)
    x2 = a + fibonacci(N-1)/fibonacci(N)*(b-a)

    while True:
        # Steps 4 & 5
        if function(x1, f_N) <= function(x2, f_N):
            b = x2
            x2 = x1
            x1 = a + (b-a)*fibonacci(N-k-3)/fibonacci(N-k-1)
        else:
            a = x1
            x1 = x2
            x2 = a + (b-a)*fibonacci(N-k-2)/fibonacci(N-k-1)
        # Step 6
        if k == N-3:
            x2 = x1 + delta
            if function(x1, f_N) <= function(x2, f_N):
                b = x2
            else:
                a = x1
            break
        else:
            k = k + 1
    # Step 7
    min_x = (a+b) / 2
    # Output
    print('f(x*) = min f(x) = ', round(function(min_x, f_N)*10000)/10000)
    print('x* = ', round(min_x*10000)/10000)




















