import numpy as np


def func_acivate(x):
    if x < 0.5:
        return 0
    else:
        return 1


def input_signal(house, rock, beaty):
    x = np.array([house, rock, beaty])
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1] # Веса для 2 нейрона скрытого слоя
    weight1 = np.array([w11, w12])  # Матрица 2 на 3
    weight2 = np.array([-1, 1])  # Вектор 1 на 3

    sum_hidden = np.dot(weight1, x)
    print(f'Значение сумм на нейронах скрытого слоя: {str(sum_hidden)}')

    out_hidden = np.array([func_acivate(x) for x in sum_hidden])
    print(f'Значение на выходах нейронах скрытого слоя: {str(out_hidden)}')

    sum_end = np.dot(weight2, out_hidden)
    y = func_acivate(sum_end)
    print(f'Выходное значение НС: {str(y)}')

    return y


a, b, c = list(map(int, input().split()))
res = input_signal(a, b, c)
if res == 1:
    print('Ты мне нравишься')
else:
    print('Созвонимся')