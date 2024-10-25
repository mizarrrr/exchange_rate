import pandas as pd
import scipy.linalg as linalg
import scipy.stats as stats
import numpy as np
import pylab

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error # MSE

def isscalar(x): # проверка, является ли x числом
    return not isinstance(x, (list, tuple, dict, np.ndarray))

def nans(dims): # создание многомерного массива, заполненного NaN
    return np.nan * np.ones(dims)

def ssa(y, dim) -> tuple: # сингулярный спектральный анализ для временного ряда
    n = len(y)
    t = n - dim + 1

    # создаём траекторную матрицу, dim размерность матрицы, после нормализуем на sqrt(t) для компенсации увеличения дисперсии значений ряда
    yy = linalg.hankel(y, np.zeros(dim))
    yy = yy[:-dim + 1, :] / np.sqrt(t)

    # сингулярное разложение с ненулевыми сингулярными значениями
    _, s, v = linalg.svd(yy, full_matrices=False, lapack_driver='gesvd')
    vt = np.matrix(v).T # матрица правых сингулярных векторов
    pc = np.matrix(yy) * vt # матрциа главных компонент

    return np.asarray(pc), s, np.asarray(vt) # возвращается: 1) pc - матрица главных компонент, 2) s - вектор сингулярных значений (корни собств.зн), 3) v - матрица правых сингулрных векторов

def inv_ssa(pc: np.ndarray, v: np.ndarray, k) -> np.ndarray: # обратное преобразование SSA для восстановления исходного временного ряда из его компонент
    if isscalar(k): k = [k]
    if pc.ndim != 2:
        raise ValueError('pc must be a 2-dimensional matrix')
    if v.ndim != 2:
        raise ValueError('v must be a 2-dimensional matrix')

    t, dim = pc.shape
    n_points = t + dim - 1

    pc_comp = np.asarray(np.matrix(pc[:, k]) * np.matrix(v[:, k]).T) # выбираются только те главные компоненты, которые указаны в k, и вычисляется произведение pc и v.T

    xr = np.zeros(n_points)
    times = np.zeros(n_points)

    # цикл по компонентам, в котором происходит постепенное накопление реконструированного ряда в массиве xr с учетом количества наложений
    for i in range(dim):
        xr[i : t + i] = xr[i : t + i] + pc_comp[:, i]
        times[i : t + i] = times[i : t + i] + 1

    # итоговый реконструированный ряд xr нормализуется на sqrt(t)
    xr = (xr / times) * np.sqrt(t)
    return xr

def ssa_predict(x, dim, k, n_forecast, e=None, max_iter=10000) -> np.ndarray: # прогнозирование временного ряда
    if not e:
        e = 0.0001 * (np.max(x) - np.min(x))
    
    mean_x = x.mean()
    x = x - mean_x # центровка ряда
    xf = nans(n_forecast)

    # выполняется цикл по количеству точек для прогнозирования n_forecast
    for i in range(n_forecast):
        # здесь мы используем предыдущее значение в качестве начальной оценки
        x = np.append(x, x[-1])
        yq = x[-1]
        y = yq + 2 * e
        n_iter = max_iter
        while abs(y - yq) > e: # если разница между текущим и предыдущим значением прогноза меньше e, то итерационный процесс останавливается
            yq = x[-1]
            pc, _, v = ssa(x, dim)
            xr = inv_ssa(pc, v, k)
            y = xr[-1]
            x[-1] = y
            n_iter -= 1
            if n_iter <= 0:
                print('ssa_predict> number of iterations exceeded')
                break
        xf[i] = x[-1]
        
    xf = xf + mean_x # прогнозируемый ряд xf центрируется обратно, добавляя mean_x
    return xf 

def ssa_cutoff_order(x: np.ndarray, dim=200, cutoff_pctl=75, show_plot=False): # определение оптимального количество компонент
    _, s, _ = ssa(x, dim) # получаем вектор сингулярных значений s
    curve = -s/s.sum() * np.log(s/s.sum()) # вычисляется кривая приращения информационной энтропии
    pctl = np.percentile(curve, cutoff_pctl) # находится процентиль pctl кривой приращения информационной энтропии на уровне cutoff_pctl
    n_cutoff = sum(curve > pctl) # определяется порядок n_cutoff, соответствующий точке, где кривая приращения информационной энтропии пересекает линию процентиля pctl
    
    # если флаг show_plot = True, то строится график кривой приращения информационной энтропии с вертикальной линией, обозначающей порядок n_cutoff
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(curve, color="blue")
        plt.ylabel('Приращение сингулярной энтропии')
        plt.xlabel('Количество компонент')
        plt.vlines(n_cutoff, 0, max(curve), 'g', linestyles='dotted')
        plt.title('Оптимальное количество компонент: %d' % n_cutoff)
        plt.grid(True)
        plt.show()
    
    return n_cutoff # функция возвращает оптимальный порядок (количество компонент) n_cutoff для SSA-разложения

# функция принимает на вход список и возвращает один "плоский" список, состоящий из всех элементов вложенных списков
# если l = [[1, 2], [3, 4], [5]], то flatten(l) вернет [1, 2, 3, 4, 5].
def flatten(l):
    return [item for sublist in l for item in sublist]

excel_curs = pd.read_excel('curs.xlsx')
curs = pd.DataFrame(excel_curs)
data_curs = np.array(curs[0])

# 1) аппроксимация ряда

mer_interval_2 = 20
L_2 = 7
r_2 = 6

pc, _, v = ssa(data_curs[-mer_interval_2:], L_2) # вычисляется SSA-разложение текущего ряда x с размерностью окна dim
xr = inv_ssa(pc, v, list(range(r_2))) # выполняется реконструкция ряда xr с использованием выбранных компонент k

a = list(range(0, 365))

plt.figure(figsize=(10, 6))
plt.plot(a[-mer_interval_2:], data_curs[-mer_interval_2:], label='Исходные данные', color='blue')
plt.plot(a[-mer_interval_2:], xr, label='Аппроксимация', color='red')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Аппроксимация курса валют')
plt.legend()
plt.grid(True)
plt.show()
    
std_dev = mean_squared_error(data_curs[-mer_interval_2:], xr)
print("Среднеквадратическое отклонение -> ", round(std_dev, 6)) # MSE

# 2) прогнозирование ряда

prognoz = 10

mer_interval = 300
L = 100
r = 50

# разделяем данные на обучающую и прогнозируемую части
train_data = data_curs[-prognoz-mer_interval:-prognoz]
test_data = data_curs[-prognoz:]

# Выполняем прогнозирование
test_data_predict = ssa_predict(train_data, L, list(range(r)), prognoz, 1e-4)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(a[-prognoz-mer_interval:-prognoz+1], data_curs[-prognoz-mer_interval:-prognoz+1], label='Исходные данные', color='blue')
plt.plot(a[-prognoz:], data_curs[-prognoz:], label='Прогнозируемые данные', color='red')
plt.plot(a[-prognoz:], test_data_predict, label='Прогноз', color='green')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Прогнозирование курса валют на 10 дней')
plt.legend()
plt.grid(True)
plt.show()
    
std_dev = mean_squared_error(np.array(data_curs[-prognoz:]), test_data_predict)
print("Среднеквадратическое отклонение -> ", round(std_dev, 6)) # MSE

# 3) визуализоация основных компонент временного ряда

def plot_ssa_components(y, dim):
    pc, s, v = ssa(y, dim)
    
    # Определение порядка компонент, не связанных с шумом
    ssa_cutoff = ssa_cutoff_order(y, dim, show_plot=True)
    print('Оптимальное количестко компонент -> ', ssa_cutoff)
    
    # Реконструкция временного ряда из выбранных компонент
    k = list(range(ssa_cutoff))
    reconstructed = inv_ssa(pc, v, k)
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(y, label='Исходный ряд', color='blue')
    plt.plot(reconstructed, label='Реконструированный ряд', color='red')
    plt.xlabel('Индекс')
    plt.ylabel('Значение')
    plt.title('Временной ряд и его реконструкция')
    plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(ssa_cutoff):
        plt.figure(figsize=(5, 3))
        plt.plot(pc[:, i], label=f'Компонента {i+1}', color='blue')
        plt.xlabel('Индекс')
        plt.ylabel('Значение')
        plt.title(f'Основные компоненты временного ряда')
        plt.legend()
        plt.grid(True)
        plt.show()

mre_int = 365
plot_ssa_components(data_curs[-mre_int:], dim=14)