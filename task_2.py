# продолжение 1 задания

curs_itog1 = ewm_curs
curs_itog1.to_excel('curs.xlsx', index=False)
print(curs_itog1)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(curs_itog1.index, curs_itog1, label='Исходные данные', color='blue')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('График 2 временного ряда')
plt.legend()
plt.grid(True)
plt.show()

# 1) Модель ARIMA

import itertools
from statsmodels.tsa.arima.model import ARIMA

def aic_(m, d_, n):
    print('Длина мерного интервала = ', m)
    p_mas, d_mas, q_mas, AIC_mas = [], [], [], []
    
    if n != 1:
        train_data = curs_diff_itog1[-m - 10:365 - 10]

    p = range(1, 4)
    d = [d_]
    q = range(1, 4)
    pdq = list(itertools.product(p, d, q))
    
    best_pdq = (0, 0, 0)
    best_aic = np.inf
    
    best_pdq_2 = (0, 0, 0)
    best_aic_2 = np.inf
    
    best_pdq_3 = (0, 0, 0)
    best_aic_3 = np.inf
    
    best_pdq_4 = (0, 0, 0)
    best_aic_4 = np.inf

    for params in pdq:
        p_mas.append(params[0])
        d_mas.append(params[1])
        q_mas.append(params[2])
        
        model_test = ARIMA(train_data, order = params)
        result_test = model_test.fit()
        AIC_mas.append(result_test.aic)
        
        if result_test.aic < best_aic:
            best_pdq_4 = best_pdq_3
            best_aic_4 = best_aic_3
            
            best_pdq_3 = best_pdq_2
            best_aic_3 = best_aic_2
            
            best_pdq_2 = best_pdq
            best_aic_2 = best_aic
            
            best_pdq = params
            best_aic = result_test.aic
        
        elif result_test.aic < best_aic_2:
            best_pdq_4 = best_pdq_3
            best_aic_4 = best_aic_3
            
            best_pdq_3 = best_pdq_2
            best_aic_3 = best_aic_2
            
            best_pdq_2 = params
            best_aic_2 = result_test.aic
            
        elif result_test.aic < best_aic_3:
            best_pdq_4 = best_pdq_3
            best_aic_4 = best_aic_3
            
            best_pdq_3 = params
            best_aic_3 = result_test.aic
            
        elif result_test.aic < best_aic_4:
            best_pdq_4 = params
            best_aic_4 = result_test.aic
        

    dff = pd.DataFrame(list(zip(p_mas, d_mas, q_mas, AIC_mas)), columns=['p', 'd', 'q', 'AIC'])
    print(dff)
    print('\nОптимальные параметры -> ', best_pdq, ' с минимальным AIC = ', best_aic)
    print('Вторые по оптимальности параметры -> ', best_pdq_2, ' с AIC = ', best_aic_2)
    print('Третие по оптимальности параметры -> ', best_pdq_3, ' с AIC = ', best_aic_3)
    print('Чертвёртые по оптимальности параметры -> ', best_pdq_4, ' с AIC = ', best_aic_4, '\n')


def aic_pr(d_, n):
    p_mas, d_mas, q_mas, AIC_mas = [], [], [], []
    
    if n != 1:
        train_data = curs_diff_itog1

    p = range(1, 4)
    d = [d_]
    q = range(1, 4)
    pdq = list(itertools.product(p, d, q))
    
    best_pdq = (0, 0, 0)
    best_aic = np.inf

    for params in pdq:
        p_mas.append(params[0])
        d_mas.append(params[1])
        q_mas.append(params[2])
        
        model_test = ARIMA(train_data, order = params)
        result_test = model_test.fit()
        AIC_mas.append(result_test.aic)
        
        if result_test.aic < best_aic:
            best_pdq = params
            best_aic = result_test.aic
        

    dff = pd.DataFrame(list(zip(p_mas, d_mas, q_mas, AIC_mas)), columns=['p', 'd', 'q', 'AIC'])
    print(dff)
    print('\nОптимальные параметры -> ', best_pdq, ' с минимальным AIC = ', best_aic)

aic_pr(1, 2)

best_pdq_1 = (2, 1, 3)

model_1 = ARIMA(curs_itog1, order = best_pdq_1)
result_1 = model_1.fit()

forecast = result_1.predict(start = 0,  
                            end = 364,  
                            typ = 'levels')

index_dat = []
for i in range(0, 365, 1):
    index_dat.append(i)

ci = 1.96 * np.std(curs_itog1) / np.sqrt(len(curs_itog1))

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(curs_itog1.index, curs_itog1, label='Исходные данные', color='blue')
plt.plot(index_dat[1:], np.array(forecast[1:]), label='Прогноз', color='red')
plt.fill_between(curs_itog1.index, curs_itog1 - ci, curs_itog1 + ci, color='gray', alpha=0.3, label='Доверительный интервал')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Прогнозирование 1 временного ряда')
plt.legend()
plt.grid(True)
plt.show()

std_dev = mean_squared_error(curs_itog1, forecast)
print("Среднеквадратическое отклонение:", round(std_dev, 9)) # MSE

mas_1 = [20, 50, 100, 200, 300, 355]
for i in mas_1:
    aic_(i, 1, 2) # т.к. ряд НЕстационарный и его было решено продифференцировать (взять 1 разности)
    
best_pdq_2 = (2, 1, 2)
best_m = 355

model_2 = ARIMA(curs_itog1[365 - best_m - 10:355], order = best_pdq_2)
result_2 = model_2.fit()

forecast_2 = result_2.predict(start = 355,  
                              end = 364,  
                              typ = 'levels')

index_dat_2 = []
for i in range(355, 365, 1):
    index_dat_2.append(i)

ci_2 = 1.96 * np.std(curs_itog1) / np.sqrt(len(curs_itog1))

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(curs_itog1.index, curs_itog1, label='Исходные данные', color='blue')
plt.plot(curs_itog1.index[-10:], curs_itog1[-10:], label='Прогнозируемые данные', color='green')
plt.plot(index_dat_2, np.array(forecast_2), label='Прогноз', color='red')
plt.fill_between(curs_itog1.index, curs_itog1 - ci_2, curs_itog1 + ci_2, color='gray', alpha=0.3, label='Доверительный интервал')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Прогнозирование 2 временного ряда')
plt.legend()
plt.grid(True)
plt.show()

std_dev_2 = mean_squared_error(curs_itog1[-10:], forecast_2)
print("Среднеквадратическое отклонение:", round(std_dev_2, 6)) # MSE

# модель BSTS в отдельном модуле

# 3) Разложение ряда на компоненты
# 3.1) Выделение тренда, сезонной составляющей и остаточной состовляющей

# выделение тренда
def trend(itog):
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(itog.index, itog, label='Исходные данные', color='blue')
    
    z_2 = np.polyfit(itog.index, np.log(itog), 1)
    p_2 = np.exp(z_2[0] * itog.index + z_2[1])
    plt.plot(itog.index, p_2, label= 'Экспоненциальный тренд')
    print('Функция, описывающая тренд -> y = e^(', z_2[0], ' * x + ', z_2[1], ')')
    print('MSE с экспоненциальным трендом -> ', mean_squared_error(itog, p_2), '\n')
    
    z = np.polyfit(itog.index, itog, 1) # аппроксимация полинома методом наименьших квадратов
    p = np.poly1d(z)
    plt.plot(itog.index, p(itog.index), label= 'Линейный тренд')
    print('Функция, описывающая тренд -> y = ', z[0], ' * x + ', z[1])
    print('MSE с линейным трендом -> ', mean_squared_error(itog, p(itog.index)), '\n')
    
    z_1 = np.polyfit(itog.index, itog, 2)
    p_1 = np.poly1d(z_1)
    plt.plot(itog.index, p_1(itog.index), label= 'Квадратичный тренд')
    print('Функция, описывающая тренд -> y = ', z_1[0], ' * x**2 + ', z_1[1], ' * x + ', z_1[2])
    print('MSE с квадратичным трендом -> ', mean_squared_error(itog, p_1(itog.index)), '\n')
    
    z_3 = np.polyfit(itog.index, itog, 3)
    p_3 = np.poly1d(z_3)
    plt.plot(itog.index, p_3(itog.index), label= 'Полиномиальный тренд степени 3')
    print('Функция, описывающая тренд -> y = ', z_3[0], ' * x**3 + ', z_3[1], ' * x**2 + ', z_3[2], ' * x + ', z_3[3])
    print('MSE с полиномиальным трендом степени 3 -> ', mean_squared_error(itog, p_3(itog.index)), '\n')

    z_4 = np.polyfit(itog.index, itog, 4)
    p_4 = np.poly1d(z_4)
    plt.plot(itog.index, p_4(itog.index), label= 'Полиномиальный тренд степени 4')
    print('Функция, описывающая тренд -> y = ', z_4[0], ' * x**4 + ', z_4[1], ' * x**3 + ', z_4[2], ' * x**2 + ', z_4[3], ' * x + ', z_4[4])
    print('MSE с полиномиальным трендом степени 4 -> ', mean_squared_error(itog, p_4(itog.index)), '\n')

    z_5 = np.polyfit(itog.index, itog, 5)
    p_5 = np.poly1d(z_5)
    plt.plot(itog.index, p_5(itog.index), label= 'Полиномиальный тренд степени 5')
    print('Функция, описывающая тренд -> y = ', z_5[0], ' * x**5 + ', z_5[1], ' * x**4 + ', z_5[2], ' * x**3 + ', z_5[3], ' * x**2 + ', z_5[4], ' * x +  ', z_5[5])
    print('MSE с польномиальным трендом степени 5 -> ', mean_squared_error(itog, p_5(itog.index)), '\n')

    z_6 = np.polyfit(itog.index, itog, 6)
    p_6 = np.poly1d(z_6)
    plt.plot(itog.index, p_6(itog.index), label= 'Полиномиальный тренд степени 6')
    print('Функция, описывающая тренд -> y = ', z_6[0], ' * x**6 + ', z_6[1], ' * x**5 + ', z_6[2], ' * x**4 + ', z_6[3], ' * x**3 + ', z_6[4], ' * x**2 +  ', z_6[5], ' * x +  ', z_6[6])
    print('MSE с польномиальным трендом степени 6 -> ', mean_squared_error(itog, p_6(itog.index)), '\n')
    
    plt.xlabel('Индекс')
    plt.ylabel('Значение')
    plt.title('График временного ряда')
    plt.legend()
    plt.grid(True)
    plt.show()
    
trend(curs_itog1)

z_c = np.polyfit(curs_itog1.index, curs_itog1, 6)
p_c = np.poly1d(z_c)
curs_trend = p_c(curs_itog1.index)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(curs_itog1.index, curs_itog1, label='Исходные данные', color='blue')
plt.plot(curs_itog1.index, curs_trend, label='Тренд', color='red')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('График 2 временного ряда')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# подсчёт коэффициентов Фурье
def fourier_coefficients(x, n_harmonics):
    n = len(x)
    t = np.arange(0, n)
    T = n
    coefficients = []
    for k in range(1, n_harmonics + 1):
        a_k = 2 * np.sum(x * np.cos(2 * np.pi * k * t / T)) / n
        b_k = 2 * np.sum(x * np.sin(2 * np.pi * k * t / T)) / n
        coefficients.append((a_k, b_k))
    return coefficients

# вычисляем сезонную компоненту по Фурье-формуле
def fourier_series(t, coefficients):
    n_harmonics = len(coefficients)
    T = len(t)
    f_series = np.zeros(T)
    for i in range(n_harmonics):
        a_k, b_k = coefficients[i]
        f_series += a_k * np.cos(2 * np.pi * (i+1) * t / T) + b_k * np.sin(2 * np.pi * (i+1) * t / T)
    return f_series


# исходный временной ряд без тренда
seasonality_and_noise_1 = curs_itog1 - curs_trend

# вычисляем Фурье-коэффициенты
n_harmonics_1 = 4 # задаем количество гармоник
coefficients_1 = fourier_coefficients(seasonality_and_noise_1, n_harmonics_1)

# построение сезонной компоненты
t_1 = np.arange(len(seasonality_and_noise_1))
seasonality_1 = fourier_series(t_1, coefficients_1)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(seasonality_and_noise_1, label='Сезонность + шум', color='blue')
plt.plot(seasonality_1, label='Сезонность', color='red')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('График 1 временного ряда')
plt.legend()
plt.grid(True)
plt.show()

# выделение остатков и проверка на нормальность
import math
from scipy.stats import shapiro, lognorm

ostatk_curs = curs_itog1 - curs_trend - seasonality_1

# проверка на нормальность с помощью теста Шапиро-Уилка, 
# если p-значение меньше 0,05, то распределение не является нормальным

def normal(ostatk):
    k2, p = stats.normaltest(ostatk)
    alpha = 0.05
    if p < alpha:
        print("Распределение остатков не является нормальным, p = ", p)
    else:
        print("Распределение остатков является нормальным")
        
normal(ostatk_curs)

curs_x = curs_trend + seasonality_1

print('MSE исходного ряда и тренда+сезонности -> ', round(mean_squared_error(curs_itog1, curs_x), 6))

# Построение графика
plt.figure(figsize=(10, 3))
plt.plot(curs_itog1.index, curs_itog1, color='blue')
plt.plot(curs_itog1.index, curs_x, color='red')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Тренд + сезонность 2 временного ряда')
plt.grid(True)
plt.show()

# Итоговое разложение
plt.figure(figsize=(10, 3))
plt.plot(curs_itog1.index, curs_itog1, color='blue')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Исходные данные 2 временного ряда')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(curs_itog1.index, curs_trend, color='red')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Тренд 2 временного ряда')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(curs_itog1.index, seasonality_1, label='Сезонность', color='green')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Сезонность 2 временного ряда')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(curs_itog1.index, ostatk_curs, color='orange')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Остатки 2 временного ряда')
plt.grid(True)
plt.show()

print(np.sum(ostatk_curs))

# исследование точности на разных мерных интервалах

def normal_os(ostatk):
    k2, p = stats.normaltest(ostatk)
    alpha = 0.05
    if p < alpha:
        # print("Распределение остатков не является нормальным, p = ", p)
        return 0
    else:
        # print("Распределение остатков является нормальным")
        return 1

def toch_tssh_c(n_toch, curs_itog1):
    print('Длина мерного интервала -> ', n_toch)
    # # проверяем какой тренд лучше
    # trend(curs_itog1[355 - n_toch:355])
    
    trend = []
    for x in range(355 - n_toch, 365):
        if n_toch == 20:
            trend.append(4.541364089624726e-06  * x**6 +  -0.009405523492456365  * x**5 +  8.115875368743193  * x**4 +  -3734.6792261047867  * x**3 +  966630.3109878586  * x**2 +   -133423617.49129844  * x +   7672909539.051811)
        elif n_toch == 50:
            trend.append(2.6104400463538223e-08  * x**6 +  -5.1160524005343136e-05  * x**5 +  0.041749546342978956  * x**4 +  -18.158286282408994  * x**3 +  4439.450446272466  * x**2 +   -578485.4803969769  * x +   31387500.097665615)
        elif n_toch == 100:
            trend.append(2.717433400809302e-10  * x**6 +  -5.107499928392287e-07  * x**5 +  0.00039846224194812157  * x**4 +  -0.16514059420707417  * x**3 +  38.34307919724223  * x**2 +   -4728.543325332965  * x +   241984.9615010949)
        elif n_toch == 200:
            trend.append(3.709586540233709e-10  * x**5 +  -4.425222021958549e-07  * x**4 +  0.00020735448589764507  * x**3 +  -0.0480947971016508  * x**2 +  5.594835958191495  * x +   -239.8190349738432)
        elif n_toch == 300:
            trend.append(1.5140164425200503e-10  * x**5 +  -1.518922267522905e-07  * x**4 +  5.587927045122719e-05  * x**3 +  -0.009292483739170548  * x**2 +  0.7167051371791153  * x +   0.6026808225943535)
        else:
            trend.append(6.655657520383483e-13  * x**6 +  -6.674999332216042e-10  * x**5 +  2.4705500990624896e-07  * x**4 +  -4.175303885607696e-05  * x**3 +  0.003235530138132651  * x**2 +   -0.07365424234116971  * x +   19.57820905626513)
        
    seasonality_and_noise = curs_itog1[355 - n_toch:355] - trend[:n_toch]
    if n_toch == 20:
        n_harmonics = 1
        # for n_harmonics in range(1, 150):
        #     coefficients = fourier_coefficients(seasonality_and_noise, n_harmonics)
        #     t = np.arange(n_toch)
        #     seasonality = fourier_series(t, coefficients)
        #     ostatk_data = seasonality_and_noise - seasonality
        #     if normal_os(ostatk_data):
        #         print(n_harmonics)  
    elif n_toch == 50:
        n_harmonics = 1 
    elif n_toch == 100:
        n_harmonics = 3      
    elif n_toch == 200:
        n_harmonics = 4
    elif n_toch == 300:
        n_harmonics = 5
    else:
        n_harmonics = 145
        
    coefficients = fourier_coefficients(seasonality_and_noise, n_harmonics)
    t = np.arange(n_toch + 10)
    seasonality = fourier_series(t, coefficients)
    # ostatk_data = seasonality_and_noise - seasonality
    # normal(ostatk_data)

    # Построение графика
    plt.figure(figsize=(10, 3))
    plt.plot(curs_itog1[355 - n_toch:].index, curs_itog1[355 - n_toch:], color='blue', label='Исходный ряд')
    plt.plot(curs_itog1[355 - n_toch:355].index, trend[:n_toch] + seasonality[:n_toch], color='red', label='Тренд + сезонность')
    plt.plot(curs_itog1[354:].index, trend[n_toch - 1:] + seasonality[n_toch - 1:], color='green', label='Прогноз')
    plt.xlabel('Индекс')
    plt.ylabel('Значение')
    plt.title('Тренд + сезонность 2 временного ряда')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print('MSE прогноза -> ', round(mean_squared_error(curs_itog1[355:], trend[n_toch:] + seasonality[n_toch:]), 6), '\n')

n_toch_c = [20, 50, 100, 200, 300, 355]
for i in n_toch_c:
    toch_tssh_c(i, curs_itog1)
    
# 3.2) Аддитивная нелинейная регрессионная модель (пакет Prophet)

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from datetime import date, timedelta


start_dt = date(2023, 1, 1)
end_dt = date(2023, 12, 31)

delta = timedelta(days=1)

dates = []
while start_dt <= end_dt:
    dates.append(start_dt.isoformat())
    start_dt += delta

    
dff_curs = pd.DataFrame(list(zip(dates, curs_itog1)), columns=['ds', 'y'])
print(dff_curs)

m = Prophet()
m.fit(dff_curs)

future = m.make_future_dataframe(periods=400)
forecast = m.predict(future)

# fig1 = m.plot(forecast)
# print(forecast)



print('MSE исходного ряда и тренда+сезонности -> ', round(mean_squared_error(curs_itog1, forecast['trend'][:365] + forecast['weekly'][:365]), 6))

# Построение графика
plt.figure(figsize=(10, 3))
plt.plot(curs_itog1.index, curs_itog1, color='blue')
plt.plot(curs_itog1.index, forecast['trend'][:365] + forecast['weekly'][:365], color='red')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Исходные данные 2 временного ряда и тренд+сезонность')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(forecast['trend'][:365].index, forecast['trend'][:365], color='red')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Тренд 2 временного ряда')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(forecast['weekly'][:365].index, forecast['weekly'][:365], label='Сезонность', color='green')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Сезонность 2 временного ряда')
plt.grid(True)
plt.show()

ostatk_2_curs = curs_itog1 - forecast['trend'][:365] - forecast['weekly'][:365]

plt.figure(figsize=(10, 3))
plt.plot(ostatk_2_curs.index, ostatk_2_curs, color='orange')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('Остатки 2 временного ряда')
plt.grid(True)
plt.show()

normal(ostatk_2_curs)
print(np.sum(ostatk_2_curs))

# исследование точности на разных мерных интервалах

def toch_tssh_c(n_toch, curs_itog1):
    print('Длина мерного интервала -> ', n_toch)
    m_3 = Prophet()
    m_3.fit(dff_curs[355 - n_toch:355])
    
    future_3 = m_3.make_future_dataframe(periods=10)
    forecast_3 = m_3.predict(future_3)
    
    fig3 = m_3.plot(forecast_3)
    
    # Построение графика
    plt.figure(figsize=(10, 3))
    plt.plot(curs_itog1.index[:356], curs_itog1[:356], label='Исходные данные', color='blue')
    plt.plot(curs_itog1.index[355:], curs_itog1[355:], label='Прогнозируемые данные', color='red')
    plt.plot(curs_itog1.index[355:], forecast_3['yhat'][-10:], label='Прогноз', color='green')
    plt.xlabel('Индекс')
    plt.ylabel('Значение')
    plt.title('Прогнозирование 2 временного ряда')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    std_dev = mean_squared_error(curs_itog1[355:], forecast_3['yhat'][-10:])
    print("Среднеквадратическое отклонение:", round(std_dev, 6)) # MSE    
    


n_toch_с = [20, 50, 100, 200, 300, 355]
for i in n_toch_с:
    toch_tssh_c(i, curs_itog1)