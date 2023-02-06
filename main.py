from matplotlib import pyplot
from scipy.constants import *
from math import e
from numba import njit
from pandas import DataFrame
from japanize_matplotlib import japanize

japanize()
print(g)  # 重力加速度
print(e)  # 自然対数の底
重力加速度 = g
K: float = 1.0  # 空気抵抗の係数
時間の変位: float = 0.001  # 分割時間
初期時刻: float = 0.0  # 初期時刻
質量: float = 1.0  # 質量
初速度: float = 0  # 初期速度
初期位置: float = 0  # 初期位置
steps: int = int(10 / 時間の変位)  # 計算ステップ数

fig_size = (10, 10)


@njit
def オイラー法(x0: float, v0: float) -> list[float]:
    v1 = v0 + (-(K / 質量) * v0 - 重力加速度) * 時間の変位
    x1 = x0 + v0 * 時間の変位
    return [x1, v1]


@njit
def ホイン法(x0: float, v0: float) -> list[float]:
    v_k_1 = (-(K / 質量) * v0 - 重力加速度)
    v_k_2 = (-(K / 質量) * (v_k_1 + v0) - 重力加速度)
    v1 = v0 + (v_k_1 + v_k_2) * 時間の変位 / 2
    x_k_1 = v0
    x_k_2 = v0 + v_k_2
    x1 = x0 + (x_k_1 + x_k_2) * 時間の変位 / 2
    return [x1, v1]


@njit
def ルンゲクッタ法(x0: float, v0: float) -> list[float]:
    v_k_1 = (-(K / 質量) * v0 - 重力加速度)
    v_k_2 = (-(K / 質量) * (v0 + v_k_1 / 2) - 重力加速度)
    v_k_3 = (-(K / 質量) * (v0 + v_k_2 / 2) - 重力加速度)
    v_k_4 = (-(K / 質量) * (v0 + v_k_3) - 重力加速度)
    v1 = v0 + (v_k_1 + 2 * v_k_2 + 2 * v_k_3 + v_k_4) * 時間の変位 / 6

    x_k_1 = v0
    x_k_2 = v0 + v_k_1 / 2
    x_k_3 = v0 + v_k_2 / 2
    x_k_4 = v0 + v_k_3
    x1 = x0 + (x_k_1 + x_k_2 * 2 + x_k_3 * 2 + x_k_4) * 時間の変位 / 6

    return [x1, v1]


@njit
def 解析解(t: float) -> list[float]:
    x = -質量 * 重力加速度 * t / K + (初速度 + 質量 * 重力加速度 / K) * (-質量 / K) * e ** (-K * t / 質量) - (
            初速度 + 質量 * 重力加速度 / K) * (-質量 / K) + 初期位置
    v = -質量 * g / K + (初速度 + 質量 * 重力加速度 / K) * e ** (-K * t / 質量)
    return [x, v]


speed = DataFrame()
position = DataFrame()

オイラー法_時刻: list[float] = list()
オイラー法_速度: list[float] = list()
オイラー法_位置: list[float] = list()

x0 = 初期位置
v0 = 初速度
for i in range(steps):
    現在時刻 = 初期時刻 + 時間の変位 * i
    x1, v1 = オイラー法(x0, v0)
    オイラー法_時刻.append(現在時刻)
    オイラー法_位置.append(x1)
    オイラー法_速度.append(v1)
    x0 = x1
    v0 = v1

ホイン法_時刻: list[float] = list()
ホイン法_速度: list[float] = list()
ホイン法_位置: list[float] = list()

x0 = 初期位置
v0 = 初速度
for i in range(steps):
    現在時刻 = 初期時刻 + 時間の変位 * i
    x1, v1 = ホイン法(x0, v0)
    ホイン法_時刻.append(現在時刻)
    ホイン法_位置.append(x1)
    ホイン法_速度.append(v1)
    x0 = x1
    v0 = v1

ルンゲクッタ法_時刻: list[float] = list()
ルンゲクッタ法_速度: list[float] = list()
ルンゲクッタ法_位置: list[float] = list()

x0 = 初期位置
v0 = 初速度
for i in range(steps):
    現在時刻 = 初期時刻 + 時間の変位 * i
    x1, v1 = ルンゲクッタ法(x0, v0)
    ルンゲクッタ法_時刻.append(現在時刻)
    ルンゲクッタ法_位置.append(x1)
    ルンゲクッタ法_速度.append(v1)
    x0 = x1
    v0 = v1

解析解_時刻: list[float] = list()
解析解_速度: list[float] = list()
解析解_位置: list[float] = list()

x0 = 初期位置
v0 = 初速度
print(初期位置)
for i in range(10000):
    現在時刻 = 初期時刻 + 1e-3 * i
    x1, v1 = 解析解(現在時刻)
    解析解_時刻.append(現在時刻)
    解析解_位置.append(x1)
    解析解_速度.append(v1)

pyplot.figure(figsize=fig_size, dpi=300)
pyplot.title('位置')
pyplot.plot(オイラー法_時刻, オイラー法_位置, label='オイラー法')
pyplot.plot(ホイン法_時刻, ホイン法_位置, label='ホイン法')
pyplot.plot(ルンゲクッタ法_時刻, ルンゲクッタ法_位置, label='ルンゲクッタ法')
pyplot.plot(解析解_時刻, 解析解_位置, label='解析解')
pyplot.legend()
pyplot.show()
pyplot.close()

pyplot.figure(figsize=fig_size, dpi=300)
pyplot.title('速度')
pyplot.plot(オイラー法_時刻, オイラー法_速度, label='オイラー法')
pyplot.plot(ホイン法_時刻, ホイン法_速度, label='ホイン法')
pyplot.plot(ルンゲクッタ法_時刻, ルンゲクッタ法_速度, label='ルンゲクッタ法')
pyplot.plot(解析解_時刻, 解析解_速度, label='解析解')
pyplot.legend()
pyplot.show()
pyplot.close()
