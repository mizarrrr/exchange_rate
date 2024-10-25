import pandas as pd

excel_curs = pd.read_excel('curs.xlsx')
curs = pd.DataFrame(excel_curs)

m = curs.shape[0]

curs

import matplotlib.pyplot as plt

# Построение графика
plt.plot(curs.index, curs.iloc[:, 0], label='Исходные данные', color='blue')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('График 2 временного ряда')
plt.legend()
plt.grid(True)
plt.show()

# 1. Удаление выбросов
# 2) Z-оценка

import scipy.stats as stats

stats.zscore(curs)

for i in range(n):
    if abs(stats.zscore(curs)['Курс'][i]) > 3: # если оцнека больше 3, то он считается выбросом
        print('Выброс на ', i, ' позиции с Z-оценкой -> ', stats.zscore(curs)['Курс'][i])
        
# 7) Local Outlier Factor (LOF)

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

estimator_1 = LocalOutlierFactor() # ovelty=False, модель будет использоваться для обнаружения выбросов, а не новизны
estimator_1.fit_predict(curs)

outlier_scores_1 = estimator_1.negative_outlier_factor_ # выбросы имеют более высокий по модулю
outlier_scores_1

x_LOF_1 = []
y_LOF_1 = []
for i in range(m):
    if outlier_scores_1[i] < -3.3:
        x_LOF_1.append(i)
        y_LOF_1.append(curs['Курс'][i])
        print('Выброс на ', i, ' позиции')
        
# 2. Фильтрация шума
# 1) Статистика Бокса-Пирса

import scipy.stats as sps

alfa = 0.05
print('Критическое значение x2-критерия: ', sps.chi2.isf(alfa, 1))

from statsmodels.stats.diagnostic import acorr_ljungbox

acorr_ljungbox(curs, boxpierce=True)

# 2) Фильтр Чебышева 2-го рода

import scipy.signal as signal

# Определение параметров фильтра
N = 1  # порядок фильтра
rp = 1  # максимальная пульсация в полосе пропускания (дБ)
Wc = 0.5  # частота среза (отн. частоты дискретизации)

# Проектирование фильтра Чебышева 2-го рода
b, a = signal.cheby2(N, rp, Wc, 'low')

# Применение фильтра к данным
filtered_curs = signal.filtfilt(b, a, curs.values.flatten())

# Построение сравнительного графика
plt.figure(figsize=(10, 6))
plt.plot(curs.index, curs.iloc[:, 0], label='Исходные данные', color='blue')
plt.plot(curs.index, filtered_curs, label='Отфильтрованные данные', color='red')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Сравнение исходных и отфильтрованных данных во 2 временном ряду')
plt.legend()
plt.grid(True)
plt.show()

# Построение сравнительного графика
plt.figure(figsize=(10, 6))
plt.plot(curs.index, curs.iloc[:, 0] - filtered_curs, label='Остатки', color='red')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('График остатков')
plt.legend()
plt.grid(True)
plt.show()

# 3. Сглаживание значений ряда
# 3) Алгоритм экспоненциального скользящего среднего

from sklearn.metrics import mean_squared_error

stockValues_1 = pd.DataFrame(filtered_curs)

mse_zn_1 = 1000
alfaa_1 = 0
for i in range(1, 10):
    emaa_1 = stockValues_1.ewm(alpha = i/10).mean()
    print(mean_squared_error(filtered_curs, emaa_1))
    if mean_squared_error(filtered_curs, emaa_1) < mse_zn_1:
        mse_zn_1 = mean_squared_error(filtered_curs, emaa_1)
        alfaa_1 = i/10

print('\nВыбранное альфа: ', alfaa_1) 

filtered_curs_1 = np.append(filtered_curs, filtered_curs[-1])

stockValues_1 = pd.DataFrame(filtered_curs_1)
em_curs = stockValues_1.ewm(alpha = 0.5).mean()
em_curs_1 = stockValues_1.ewm(alpha = 0.9).mean()
        
# Построение сравнительного графика
plt.figure(figsize=(10, 6))
plt.plot(curs.index, filtered_curs, label='Исходные данные', color='blue')
plt.plot(curs.index, em_curs[0][:365], label='Сглаженные данные 0.5', color='red')
plt.plot(curs.index, em_curs_1[0][:365], label='Сглаженные данные 0.9', color='green')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Сравнение исходных и сглаженных данных в 1 временном ряду')
plt.legend()
plt.grid(True)
plt.show()

# Построение сравнительного графика
plt.figure(figsize=(10, 6))
plt.plot(curs.index, em_curs[0][:365] - filtered_curs, label='Остатки 0.5', color='red')
plt.plot(curs.index, em_curs_1[0][:365] - filtered_curs, label='Остатки 0.9', color='green')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('График остатков')
plt.legend()
plt.grid(True)
plt.show()

ema_curs = em_curs[0][:365]

# 4. Проверка рядов на стационарность
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

import warnings
warnings.filterwarnings("ignore")

# До предварительной обработки - 2 набор
print('До предварительной обработки - 2 набор')

print('\nТест Дикки-Фуллера') # Тест Дики-Фуллера
print('P-значение: ', adfuller(curs)[1])
if adfuller(curs)[1] < 0.05:
    print('Ряд стационарный')
else:
    print('Ряд НЕ стационарный')

print('\nТест Квятского-Филлипса-Шмидта-Шина') # Тест Квятского-Филлипса-Шмидта-Шина
print('P-значение: ', kpss(curs, regression = 'ct')[1])
if kpss(curs, regression = 'ct')[1] > 0.05:
    print('Ряд стационарный')
else:
    print('Ряд НЕ стационарный')
    
# После пунктов 1-3 - 2 набор
print('После предварительной обработки - 2 набор')

print('\nТест Дикки-Фуллера') # Тест Дики-Фуллера
print('P-значение: ', adfuller(ema_curs)[1])
if adfuller(ema_curs)[1] < 0.05:
    print('Ряд стационарный')
else:
    print('Ряд НЕ стационарный')

print('\nТест Квятского-Филлипса-Шмидта-Шина') # Тест Квятского-Филлипса-Шмидта-Шина
print('P-значение: ', kpss(ema_curs, regression = 'ct')[1])
if kpss(ema_curs, regression = 'ct')[1] > 0.05:
    print('Ряд стационарный')
else:
    print('Ряд НЕ стационарный')
    
# Диффиринцирование для избавления от нестационарности

curs_diff = ema_curs.diff()

# Построение сравнительного графика
plt.figure(figsize=(10, 6))
plt.plot(curs.index[1:], curs_diff[1:], label='Продиференцированные данные', color='blue')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Продифференцированные данные во 2 временном ряду')
plt.legend()
plt.grid(True)
plt.show()

print('После дифференцирования - 2 набор')

print('\nТест Дикки-Фуллера') # Тест Дики-Фуллера
print('P-значение: ', adfuller(curs_diff[1:])[1])
if adfuller(curs_diff[1:])[1] < 0.05:
    print('Ряд стационарный')
else:
    print('Ряд НЕ стационарный')

print('\nТест Квятского-Филлипса-Шмидта-Шина') # Тест Квятского-Филлипса-Шмидта-Шина
print('P-значение: ', kpss(curs_diff[1:], regression = 'ct')[1])
if kpss(curs_diff[1:], regression = 'ct')[1] > 0.05:
    print('Ряд стационарный')
else:
    print('Ряд НЕ стационарный')
    
# 4. Проверка рядов на наличие тренда
# 1) Метод сравнения средних

ema_curs_1 = ema_curs[:183] # всего 365 значений
ema_curs_2 = ema_curs[183:]

y1_sr = np.average(ema_curs_1)
y2_sr = np.average(ema_curs_2)

print('Средние: ', y1_sr, ' и ', y2_sr)

s1_sqw = np.var(ema_curs_1)
s2_sqw = np.var(ema_curs_2)

print('Дисперсии: ', s1_sqw, ' и ', s2_sqw)

if s1_sqw != s2_sqw:
    t_B = abs(y1_sr - y2_sr) / (np.sqrt(s1_sqw/len(ema_curs_1)) + np.sqrt(s2_sqw/len(ema_curs_2)))
else:
    t_B = (abs(y1_sr - y2_sr) / s1_sqw) * np.sqrt((len(ema_curs_1)*len(ema_curs_2)) / (len(ema_curs_1)+len(ema_curs_2)))

print('\nЗначение критерия Стьюдента: ', t_B)
print('Критическое значение t-критерия Стюдента: ', sps.t.ppf(q = 1 - alfa, df = len(ema_curs_1) + len(ema_curs_2) - 2))

if t_B > sps.t.ppf(q = 1 - alfa, df = len(ema_curs_1) + len(ema_curs_2) - 2):
    print('Нулевая гипотеза о равенстве средних (об отсутсвии тенденции) отвергается')
else:
    print('Нулевая гипотеза о равенстве средних (об отсутсвии тенденции) НЕ отвергается')
    
# Итоговые графики

# Построение сравнительного графика
plt.figure(figsize=(10, 6))
plt.plot(curs.index, curs, label='Исходные данные', color='blue')
plt.plot(curs.index, emа_curs, label='Полученные данные', color='red')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Сравнение исходных данных и данных после анализа и перед дифференцированием во 2 временном ряду')
plt.legend()
plt.grid(True)
plt.show()