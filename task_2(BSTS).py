from typing import Optional
import matplotlib.pyplot as plt

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
from numpyro.infer import MCMC, NUTS, Predictive

from sklearn.metrics import mean_squared_error # MSE

def horseshoe_prior(p):
    r_local = numpyro.sample('r_local', dist.Normal(jnp.zeros(p)))
    half = jnp.ones(p) * 0.5
    rho_local = numpyro.sample('rho_local', dist.InverseGamma(half, half))
    r_global = numpyro.sample('r_global', dist.Normal(0.0))
    rho_global = numpyro.sample('rho_global', dist.InverseGamma(0.5, 0.5))
    z = numpyro.sample('z', dist.Normal(1.0, 100.0))
    lam = r_local * jnp.sqrt(rho_local)
    tau = r_global * jnp.sqrt(rho_global)
    beta = numpyro.primitives.deterministic('beta', z * lam * tau)
    return beta


class MinMaxScaler(object):
    def __init__(self):
        self.start_val = None
        self.max_val = None

    def _check_type(self, series):
        pass

    def fit(self, series):
        self._check_type(series)
        self.start_val = series[0]
        self.max_val = np.max(np.abs(series - self.start_val))
        return self

    def transform(self, series):
        return (series - self.start_val) / self.max_val

    def inv_transform(self, series):
        return series * self.max_val + self.start_val

    def fit_transform(self, series):
        self.fit(series)
        return self.transform(series)


class BSTS(object):
    """Байесовский структурный временной ряд
    """

    def __init__(self, seasonality: Optional[int] = None):
        """

        Параметры
        ----------
        сезонность: Необязательно[int]
            Сезонность временного ряда
        """
        self.seasonality = seasonality
        self.samples = None
        self.mcmc = None
        self.y_train = None
        self.scaler = MinMaxScaler()

    def _model_fn(self, y, X=None, future=0):
        N = len(y)
        freedom = numpyro.sample('freedom', dist.Uniform(2, 20))
        season = self.seasonality
        if self.seasonality is None:
            season = 1
        scale_delta = numpyro.sample(
            'scale_delta', dist.HalfNormal(0.5 / season)
        )
        scale_mu = numpyro.sample('scale_mu', dist.HalfNormal(5.0))
        scale_y = numpyro.sample('scale_y', dist.HalfCauchy(5.0))

        init_mu = numpyro.sample(
            'init_mu', dist.Normal(0.0, 5.0)
        )
        init_delta = numpyro.sample(
            'init_delta', dist.Normal(0.0, 10.0)
        )
        if self.seasonality is not None:
            scale_tau = numpyro.sample('scale_tau', dist.HalfNormal(5.0))
            init_tau = numpyro.sample(
                'init_tau', dist.Normal(jnp.zeros(self.seasonality - 1), 5.0)
            )
        else:
            init_tau = jnp.zeros(1)

        regression_term = 0.0
        if X is not None:
            beta = horseshoe_prior(X.shape[1])
            reg_constant = numpyro.sample(
                'reg_constant', dist.Normal(0.0, 10.0)
            )
            regression_term = jnp.dot(X, beta) + reg_constant

        def transition_fn(carry, t):
            tau, delta, mu = carry

            if self.seasonality is not None:
                exp_tau = -tau.sum()
                new_tau = numpyro.sample(
                    'tau', dist.Normal(exp_tau, scale_tau)
                )
                new_tau = jnp.where(t < N, new_tau, exp_tau)
                tau = jnp.concatenate([tau, new_tau[None]])[1:]

            new_delta = numpyro.sample(
                'delta', dist.Laplace(delta, scale_delta)
            )
            new_delta = jnp.where(t < N, new_delta, delta)

            exp_mu = mu + delta
            new_mu = numpyro.sample(
                'mu', dist.Normal(loc=exp_mu, scale=scale_mu)
            )
            new_mu = jnp.where(t < N, new_mu, exp_mu)

            expectation = new_mu + tau[-1]
            if X is not None:
                expectation += regression_term[t]
            y_model = numpyro.sample(
                'y', dist.StudentT(df=freedom, loc=expectation, scale=scale_y)
            )
            return (tau, new_delta, new_mu), y_model

        with numpyro.handlers.condition(data={'y': y}):
            _, ys = scan(
                transition_fn,
                (init_tau, init_delta, init_mu),
                jnp.arange(N + future)
            )
        if future > 0:
            numpyro.deterministic("y_forecast", ys[-future:])

    def fit(self,
            y: np.ndarray,
            X: Optional[np.ndarray] = None,
            num_warmup: int = 2000,
            num_samples: int = 2000,
            num_chains: int = 4,
            rng_key: jax.Array = random.PRNGKey(0)):
        """Выборки из задней части НАИЛУЧШЕЙ модели с учетом данных

        Параметры
        ----------
        y : np.ndarray
            Массив временных рядов
        X: Необязательный[np.ndarray]
            Необязательные ковариаты для временных рядов. Должен иметь ту же длину, что и y
        num_warmup : int
            Количество шагов прогрева для HMC
        num_samples : int
            Количество выборок для последующего
        num_chains : into
            Количество запускаемых цепочек HMC
        rng_key : джакс.Массив
            Случайный ключ Jax для использования
        """
        self.y_train = jnp.array(self.scaler.fit_transform(y))
        self.X_train = None
        if X is not None:
            assert X.shape[0] == len(y)
            self.X_train = jnp.array(self.scaler.transform(X))
        kernel = NUTS(self._model_fn)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains
        )
        self.mcmc.run(rng_key, self.y_train, self.X_train)
        self.samples = self.mcmc.get_samples()

    def predict(self,
                future: int,
                X: Optional[np.ndarray] = None,
                rng_key: jax.Array = random.PRNGKey(1)
                ) -> jax.Array:
        """Возвращает выборки из апостериорного прогноза будущего         траектория

        Параметры
        ----------
        future : int
            Количество прогнозируемых шагов в будущее
        X : Необязательный[np.ndarray]
            Необязательные ковариаты. Если модель была снабжена ковариатами, это будет необходимо.
        rng_key : jax.Массив
            Jax случайный ключ для использования

        Возвращается
        -------
        jax.Array
            Образцы из апостериорного прогноза формы
            `(num_samples, будущее)`
        """
        if self.samples is None:
            raise ValueError(
                'Model must be fit before prediction.'
            )
        predictive = Predictive(self._model_fn,
                                self.samples,
                                return_sites=["y_forecast"])
        if X is not None:
            X = self.scaler.transform(X)
            X = jnp.concatenate([self.X_train, X], axis=0)
        preds = predictive(rng_key, self.y_train, X, future=future)[
            "y_forecast"
        ]
        return self.scaler.inv_transform(preds)

    def plot(self):
        """Построение графиков подгонки по образцу и компонентов модели
        """
        if self.samples is None:
            raise ValueError(
                'Model must be fit before prediction.'
            )
        y_train = self.scaler.inv_transform(self.y_train)
        nrows = 3
        if self.seasonality is not None:
            nrows += 1
        fig, axes = plt.subplots(figsize=(12, 14), nrows=nrows)
        axes = axes.flatten()
        axes[0].plot(y_train, label='actual')
        preds = self.samples['mu'].mean(axis=0)
        if self.seasonality is not None:
            preds += self.samples['tau'].mean(axis=0)
        if self.X_train is not None:
            preds += self.samples['reg_constant'].mean()
            preds += self.X_train @ self.samples['beta'].mean(axis=0)
        preds = self.scaler.inv_transform(preds)
        axes[0].plot(preds, label='predicted')
        axes[0].set_title('Time series')
        axes[0].legend()
        axes[1].plot(
            self.scaler.inv_transform(self.samples['mu'].mean(axis=0))
        )
        axes[1].set_title('Trend')
        axes[2].plot(
            self.scaler.inv_transform(self.samples['delta'].mean(axis=0))
        )
        axes[2].set_title('Change in trend')
        if self.seasonality is not None:
            axes[3].plot(
                self.scaler.inv_transform(self.samples['tau'].mean(axis=0))
            )
            axes[3].set_title('Seasonality')
        return fig, axes

    def plot_future(self,
                    y_future: np.ndarray,
                    X_future: Optional[np.ndarray] = None):
        """Отображает будущую траекторию в сравнении с прогнозируемой траекторией

        Параметры
        ----------
        y_future : np.ndarray
            Будущая траектория y
        X_future : Необязательно[np.ndarray]
            Необязательные ковариаты. Должны быть указаны, если модель соответствовала ковариаты
        """
        forecast = self.predict(len(y_future), X_future)
        mean_forecast = forecast.mean(axis=0)
        std_forecast = forecast.std(axis=0)

        fig, ax = plt.subplots(figsize=(10, 4))
        x_train = np.arange(len(self.y_train))
        x_test = np.arange(
            len(self.y_train),
            len(self.y_train) + len(mean_forecast)
        )
        y_train = self.scaler.inv_transform(self.y_train)
        ax.plot(x_train, y_train, label='training')
        ax.plot(x_test, mean_forecast, label='prediction', linestyle='--')
        print('Прогноз:')
        print(mean_forecast)
        print('MSE -> ', round(mean_squared_error(mean_forecast, y_future), 6))
        ax.plot(x_test, y_future, label='actual')
        ax.fill_between(
            x_test,
            mean_forecast - std_forecast,
            mean_forecast + std_forecast,
            color='black',
            alpha=0.2
        )
        ax.legend()
        return ax

    def _fill_plot(self, x, y, mean_forecast, std_forecast, ax):
        ax.plot(x, y, label='actual')
        ax.plot(x, mean_forecast, label='prediction', linestyle='--')
        ax.fill_between(
            x,
            mean_forecast - std_forecast,
            mean_forecast + std_forecast,
            color='black',
            alpha=0.2
        )
        ax.legend()
        return ax

    def plot_impact(self,
                    y_future: np.ndarray,
                    X_future: Optional[np.ndarray] = None):
        """Отображает влияние будущей траектории

        Параметры
        ----------
        y_future : np.ndarray
            Будущая траектория y
        X_future : Необязательно[np.ndarray]
            Необязательные ковариаты. Должны быть указаны, если модель соответствовала         ковариаты
        """
        forecast = self.predict(len(y_future), X_future)
        mean_forecast = forecast.mean(axis=0)
        std_forecast = forecast.std(axis=0)
        cumulative_std = np.cumsum(forecast, axis=1).std(axis=0)

        y_train = self.scaler.inv_transform(self.y_train)
        mean_forecast = np.concatenate(
            [y_train, mean_forecast]
        )
        std_forecast = np.concatenate(
            [np.zeros_like(y_train), std_forecast]
        )
        cumulative_std = np.concatenate(
            [np.zeros_like(y_train), cumulative_std]
        )
        y = np.concatenate([y_train, y_future])
        x = np.arange(len(y))

        fig, axes = plt.subplots(figsize=(10, 15), nrows=3)
        axes = axes.flatten()
        self._fill_plot(x, y, mean_forecast, std_forecast, axes[0])
        axes[0].set_title('Original')

        self._fill_plot(
            x,
            np.zeros_like(y),
            y - mean_forecast,
            std_forecast,
            axes[1]
        )
        axes[1].set_title('Pointwise')

        self._fill_plot(
            x,
            np.zeros_like(y),
            np.cumsum(y - mean_forecast),
            cumulative_std,
            axes[2]
        )
        axes[2].set_title('Cumulative')
        return axes
    
import pandas as pd

excel_curs = pd.read_excel('curs.xlsx')
curs = pd.DataFrame(excel_curs) # начальные данные

curs_itog1 = curs[0]
print(curs_itog1)

# 2) Модель структурного временного ряда (BSTS)

import time

from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf

# проверка сезонности

plt.figure(figsize=(12, 6))
plot_acf(curs_itog1, lags=364)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF) Plot')
plt.show()

# сезонность = None

def bsts_models_с(train_len, x1, x2, x3):
    test_len = 10
    
    y_train = np.array(curs_itog1[-train_len-test_len:-test_len])

    model = BSTS(seasonality=None)
    model.fit(y = y_train, num_warmup = x1, num_samples = x2, num_chains = x3)

    # model.plot()
    # model.predict(10).shape # выведет матрицу для следующих k шагов
    
    frame_sub = curs_itog1.iloc[-train_len-test_len:].values
    y_train_test = frame_sub[:]

    y_test = y_train_test[-test_len:]
    model.plot_future(y_test)
    
bsts_models_с(100, 500, 500, 3)