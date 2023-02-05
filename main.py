from matplotlib import pyplot
from scipy.constants import *
from math import e
from numba import njit
from pandas import DataFrame
print(g)  # 重力加速度
print(e)  # 自然対数の底
重力加速度 = g
K: float = 1.0  # 空気抵抗の係数
時間の変位: float = 1e-4  # 分割時間
時刻: float = 0.0  # 初期時刻
質量: float = 1.0  # 質量
初速度: float = 0  # 初期速度
初期位置: float = 0  # 初期位置
steps: int = 100000  # 計算ステップ数


@njit
def オイラー法(x0: float, v0: float) -> list[float]:
    v1 = v0 + (-(K / 質量) * v0 - 重力加速度) * 時間の変位
    x1 = x0 + v0 * 時間の変位
    return [x1, v1]


@njit
def 解析解(t: float) -> list[float]:
    x = -質量 * 重力加速度 * t / K + (初速度 + 質量 * 重力加速度 / K) * (-質量 / K) * e ** (-K * t / 質量) - (
            初速度 + 質量 * 重力加速度 / K) * (-質量 / K) + 初期位置
    v = -質量 * g / K + (初速度 + 質量 * 重力加速度 / K) * e ** (-K * t / 質量)
    return [x, v]


time_arr: list[float] = list()
speed_arr: list[float] = list()
pos_arr: list[float] = list()

x0 = 初期位置
v0 = 初速度
for i in range(steps):
    now = 時刻 + 時間の変位 * i
    x1, v1 = オイラー法(x0, v0)
    time_arr.append(now)
    pos_arr.append(x1)
    speed_arr.append(v1)
    x0 = x1
    v0 = v1

a_time_arr: list[float] = list()
a_speed_arr: list[float] = list()
a_pos_arr: list[float] = list()

x0 = 初期位置
v0 = 初速度
print(初期位置)
for i in range(steps):
    now = 時刻 + 時間の変位 * i
    x1, v1 = 解析解(now)
    a_time_arr.append(now)
    a_pos_arr.append(x1)
    a_speed_arr.append(v1)

pyplot.figure(figsize=(10, 10))
pyplot.plot(time_arr, pos_arr)
pyplot.plot(a_time_arr, a_pos_arr)
pyplot.show()
pyplot.close()

diff = [a - b for a, b in zip(a_pos_arr, pos_arr)]
pyplot.figure(figsize=(10, 10))
pyplot.plot(time_arr, diff)
pyplot.show()

pyplot.figure(figsize=(10, 10))
pyplot.plot(time_arr, speed_arr)
pyplot.plot(a_time_arr, a_speed_arr)
pyplot.show()
pyplot.close()
