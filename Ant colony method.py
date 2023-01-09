# муравьиный алгоритм решения задачи коммивояжера
import numpy as np
import random
import matplotlib.pyplot as plt
import other as o

print('''Варианты ввода данных:
1. Ввести координаты узлов вручную 
2. Ввести матрицу смежности
3. Сгенерировать матрицу смежности случайным образом
4. Заготовленная матрица смежности на 5 городов
5. Заготовленная матрица смежности на 20 городов''')
input_variant = o.input_value('int', 'Ваш выбор: ', lambda iv: 0 < iv < 6, 'Такого варианта нет, попробуйте ещё.')
CITIES = o.input_value('int', 'Количество городов: ', lambda num: num > 2, 'Необходимо хотя бы 3 города')

# =============================================================================
if input_variant == 1:      # координаты вручную
    COORDINATES = np.zeros((CITIES, 2))
    L = np.zeros((CITIES, CITIES))
    for i in range(len(CITIES)):
        print('Узел ', i, ' ')
        COORDINATES[i][0] = o.input_value('float', 'x: ', lambda cord: cord >= 0, 'Координата должна быть положительной')
        COORDINATES[i][1] = o.input_value('float', 'y: ', lambda cord: cord >= 0, 'Координата должна быть положительной')
    print('Координаты городов:')
    print(COORDINATES)
    for i in range(len(L)):
        for y in range(len(L)):
            L[i][y] = np.sqrt((COORDINATES[i][0] - COORDINATES[y][0])**2 + (COORDINATES[i][1] - COORDINATES[y][1])**2)

if input_variant == 2:      # матрица смежности вручную
    # ввод матрицы смежности графа
    L = np.zeros(len(CITIES), len(CITIES))
    for i in range(len(CITIES)):
        for y in range(len(CITIES)):
            L[i][y] = o.input_value('float', 'Cities '+i+'-'+y+': ', lambda l: l >= 0)

if input_variant == 3:      # генерация случайной матрицы
    min_length = o.input_value('float', 'Минимальная длина между городами: ',
                             lambda min_l: min_l > 0, 'Длина должна быть положительным числом')
    max_length = o.input_value('float', 'Максимальная длина между городами: ',
                             lambda min_l: min_l > min_length, 'Максимальная длина должна быть больше минимальной')
    L = np.random.randint(min_length, max_length, size=(CITIES, CITIES))
    np.fill_diagonal(L, 0)

if input_variant == 4:      # матрица на 5 городов
    L = np.array([[0, 2, 30, 9, 1],
                  [4, 0, 47, 7, 7],
                  [31, 33, 0, 33, 36],
                  [20, 13, 16, 0, 28],
                  [9, 36, 22, 22, 0]])

if input_variant == 4:      # матрица на 20 городов
    L = np.array([[0, 48, 16, 11, 28, 42, 49, 1, 24, 40, 14, 43, 12, 32, 39, 6, 42, 11, 39, 9],
                  [29, 0, 42, 19, 29, 41, 34, 29, 24, 2, 4, 15, 10, 17, 20, 37, 21, 15, 1, 19],
                  [29, 40, 0, 23, 29, 6, 18, 37, 36, 27, 2, 40, 42, 46, 22, 48, 41, 44, 15, 48],
                  [8, 23, 36, 0, 45, 8, 42, 49, 8, 25, 48, 2, 35, 9, 32, 46, 26, 12, 27, 31],
                  [32, 5, 26, 9, 0, 13, 16, 11, 34, 2, 10, 6, 6, 39, 45, 4, 19, 19, 26, 41],
                  [43, 6, 12, 18, 37, 0, 35, 43, 10, 5, 31, 22, 45, 49, 13, 38, 49, 35, 11, 14],
                  [39, 13, 29, 10, 19, 5, 0, 38, 44, 48, 18, 48, 49, 34, 26, 45, 11, 31, 33, 12],
                  [30, 29, 48, 30, 1, 12, 2, 0, 19, 28, 13, 38, 13, 4, 25, 5, 49, 38, 22, 42],
                  [5, 48, 9, 44, 34, 13, 9, 9, 0, 46, 1, 36, 4, 19, 20, 26, 25, 38, 48, 47],
                  [49, 21, 8, 48, 9, 34, 2, 20, 9, 0, 33, 38, 12, 9, 7, 46, 34, 16, 41, 1],
                  [40, 2, 20, 14, 35, 8, 6, 20, 33, 34, 0, 23, 8, 15, 14, 31, 28, 7, 8, 46],
                  [2, 8, 48, 40, 31, 31, 36, 24, 11, 45, 10, 0, 49, 40, 2, 27, 49, 42, 2, 4],
                  [40, 40, 12, 1, 11, 6, 46, 36, 36, 45, 47, 23, 0, 49, 4, 31, 9, 7, 23, 11],
                  [35, 18, 18, 7, 16, 37, 46, 36, 34, 45, 4, 12, 6, 0, 15, 30, 39, 42, 5, 47],
                  [46, 46, 38, 43, 40, 49, 37, 31, 16, 18, 15, 9, 27, 49, 0, 13, 27, 22, 19, 27],
                  [13, 44, 18, 44, 40, 23, 25, 9, 23, 13, 32, 22, 7, 28, 28, 0, 21, 26, 42, 2],
                  [3, 10, 46, 6, 23, 35, 47, 6, 40, 7, 25, 37, 26, 6, 6, 15, 0, 34, 15, 28],
                  [25, 9, 28, 12, 31, 46, 47, 45, 13, 46, 35, 35, 36, 20, 28, 13, 37, 0, 16, 24],
                  [31, 29, 32, 6, 29, 2, 6, 23, 14, 45, 43, 33, 32, 7, 22, 31, 35, 28, 0, 48],
                  [19, 42, 46, 15, 19, 19, 14, 32, 32, 19, 35, 39, 44, 7, 48, 20, 14, 14, 28, 0]])

print('Матрица смежности:')
print(L)

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

# объявление коэффициентов

CITIES = len(L[0])
AGES = 50 * CITIES
ANTS = 20

a = 0.7  # коэффициент запаха
b = 1.5  # коэффициент расстояния
rho = 0.45  # коэффициент высыхания
Q = 120  # количество выпускаемого феромона
e = 5  # количество элитных муравьев

ph = Q / (CITIES)  # начальное значение феромона

# инициализация матрицы "краткости" дуг графа
rev_L = np.zeros((CITIES, CITIES))
for i in range(CITIES):
    for j in range(CITIES):
        if i != j:
            rev_L[i, j] = 1 / L[i, j]

# инициализация матрицы феромонов
tao = np.ones((CITIES, CITIES)) * ph

BEST_DIST = float("inf")  # лучшая длина маршрута
BEST_ROUTE = None  # лучший маршрут
antROUTE = np.zeros((ANTS, CITIES))  # матрица маршрутов муравьев в одном поколении (номера узлов графа)
antDIST = np.zeros(ANTS)  # вектор длины маршрута муравьев в одном поколении
antBEST_DIST = np.zeros(AGES)  # вектор лучших длин маршрутов в каждом поколении
antAVERAGE_DIST = np.zeros(AGES)

# основной цикл алгоритма
# ---------- начало освновного цикла ----------
for age in range(AGES):
    antROUTE.fill(0)
    antDIST.fill(0)

    # ---------- начало цикла обхода графа муравьями ----------
    for k in range(ANTS):

        # =============================================================================
        #         # начальное расположение муравья в графе (случайное)
        #         antROUTE[k, 0] = random.randint(0, CITIES-1)
        # =============================================================================

        # начальное расположение муравья в графе (равномерное)
        antROUTE[k, 0] = k % CITIES

        # =============================================================================
        #         # начальное расположение муравья в графе (все с одного)
        #         antROUTE[k, 0] = 1
        # =============================================================================

        # ---------- начало обхода графа k-ым муравьем ----------
        for s in range(1, CITIES):
            from_city = int(antROUTE[k, s - 1])  # текущее положение муравья
            P = (tao[from_city] ** a) * (rev_L[from_city] ** b)
            # вероятность посещения уже посещенных городов = 0
            for i in range(s):
                P[int(antROUTE[k, i])] = 0

            # вероятность выбора направления, сумма всех P = 1
            assert (np.sum(P) > 0), "Division by zero. P = %s, \n tao = %s \n rev_L = %s" % (
            P, tao[from_city], rev_L[from_city])
            P = P / np.sum(P)
            # выбираем направление
            isNotChosen = True
            while isNotChosen:
                rand = random.random()
                for p, to in zip(P, list(range(CITIES))):
                    if p >= rand:
                        antROUTE[k, s] = to  # записываем город №s в вектор k-ого муравья
                        isNotChosen = False
                        break
        # =============================================================================
        #             # локальное обновление феромона
        #             for s in range(CITIES):
        #                 city_to = int(antROUTE[k, s])
        #                 city_from = int(antROUTE[k, s-1])
        # #               tao[city_from, city_to] = tao[city_from, city_to] + (Q / antDIST[k]) # ant-cycle AntSystem
        #                 tao[city_from, city_to] = tao[city_from, city_to] + t # Ant-density AS
        #                 tao[city_to, city_from] = tao[city_from, city_to]
        # =============================================================================
        # ---------- конец цила обхода графа ----------

        # вычисляем длину маршрута k-ого муравья
        for i in range(CITIES):
            city_from = int(antROUTE[k, i - 1])
            city_to = int(antROUTE[k, i])
            antDIST[k] += L[city_from, city_to]

        # сравниваем длину маршрута с лучшим показателем
        if antDIST[k] < BEST_DIST:
            BEST_DIST = antDIST[k]
            BEST_ROUTE = antROUTE[k]
    # ---------- конец цикла обхода графа муравьями ----------

    # ---------- обновление феромонов----------
    # высыхание по всем маршрутам (дугам графа)
    tao *= (1 - rho)

    # цикл обновления феромона
    for k in range(ANTS):
        for s in range(CITIES):
            city_to = int(antROUTE[k, s])
            city_from = int(antROUTE[k, s - 1])
            tao[city_from, city_to] = tao[city_from, city_to] + (Q / antDIST[k])  # ant-cycle AntSystem
            #            tao[city_from, city_to] = tao[city_from, city_to] + Q # Ant-density AS
            tao[city_to, city_from] = tao[city_from, city_to]

    # проход элитных е-муравьев по лучшему маршруту
    for s in range(CITIES):
        city_to = int(BEST_ROUTE[s])
        city_from = int(BEST_ROUTE[s - 1])
        tao[city_from, city_to] = tao[city_from, city_to] + (e * Q / BEST_DIST)  # ant-cycle AntSystem
        tao[city_to, city_from] = tao[city_from, city_to]

    # ---------- конец обновления феромона ----------

    # конец поколения муравьев

    # сбор информации для графиков
    antBEST_DIST[age] = BEST_DIST
    antAVERAGE_DIST[age] = np.average(antDIST)

# ---------- конец основного цикла ----------

# выдача веса лучшего маршрута на выход
print(int(BEST_DIST))

x = list(range(AGES))
y = antBEST_DIST

fig, ax = plt.subplots()
plt.plot(x, y)
# =============================================================================
# for k in range(ANTS):
#     plt.plot(x, antALL_DIST[k])
# =============================================================================
plt.plot(x, antAVERAGE_DIST)
plt.grid(True)
plt.show()