from matplotlib import pyplot
from scipy.constants import g
from math import e
from numba import njit
from japanize_matplotlib import japanize

japanize()
print(g)  # 重力加速度
print(e)  # 自然対数の底
重力加速度 = g
K: float = 1.0  # 空気抵抗の係数
時間の変位: float = 0.2  # 分割時間
初期時刻: float = 0.0  # 初期時刻
質量: float = 1.0  # 質量
初速度: float = 20  # 初期速度
初期位置: float = 0  # 初期位置
steps: int = int(10 / 時間の変位)  # 計算ステップ数

fig_size = (10, 10)


@njit
def 導関数(y) -> float:
    return -(K / 質量) * y - 重力加速度


@njit
def オイラー法(x0: float, v0: float) -> list[float]:
    v0 += 導関数(v0) * 時間の変位
    x0 += v0 * 時間の変位
    return [x0, v0]


@njit
def ホイン法(x0: float, v0: float) -> list[float]:
    v_1 = v0 + 導関数(v0) * 時間の変位
    v_2 = v0 + (導関数(v0) + 導関数(v_1)) * 時間の変位 / 2
    x0 += (v0 + v_1) * 時間の変位 / 2
    return [x0, v_2]


@njit
def ルンゲクッタ法(x0: float, v0: float) -> list[float]:
    v_k_1 = 導関数(v0) * 時間の変位
    v_k_2 = 導関数(v0 + v_k_1 / 2.0) * 時間の変位
    v_k_3 = 導関数(v0 + v_k_2 / 2.0) * 時間の変位
    v_k_4 = 導関数(v0 + v_k_3) * 時間の変位
    v1 = v0 + (v_k_1 + 2.0 * v_k_2 + 2.0 * v_k_3 + v_k_4) / 6
    x_k_1 = v0 * 時間の変位
    x_k_2 = (v0 + v_k_1 / 2.0) * 時間の変位
    x_k_3 = (v0 + v_k_2 / 2.0) * 時間の変位
    x_k_4 = (v0 + v_k_3) * 時間の変位
    x1 = x0 + (x_k_1 + x_k_2 * 2.0 + x_k_3 * 2.0 + x_k_4) / 6
    # x1 = x0 + (v0 * 6 + v_k_1 + v_k_2 * 2.0 + v_k_3 * 2.0 + v_k_4) * 時間の変位 / 6
    # x1 = x0 + v1 * 時間の変位
    # x1 = x0 + (x_k_1 + x_k_4) / 2
    return [x1, v1]


@njit
def 解析解(t: float) -> list[float]:
    x = -質量 * 重力加速度 * t / K + (初速度 + 質量 * 重力加速度 / K) * (-質量 / K) * e ** (-K * t / 質量) - (
            初速度 + 質量 * 重力加速度 / K) * (-質量 / K) + 初期位置
    v = -質量 * g / K + (初速度 + 質量 * 重力加速度 / K) * e ** (-K * t / 質量)
    return [x, v]


オイラー法_時刻: list[float] = list()
オイラー法_速度: list[float] = list()
オイラー法_位置: list[float] = list()
オイラー法_位置_誤差: list[float] = list()
オイラー法_速度_誤差: list[float] = list()

x0 = 初期位置
v0 = 初速度
for i in range(steps):
    現在時刻 = 初期時刻 + 時間の変位 * i
    x1, v1 = オイラー法(x0, v0)
    オイラー法_時刻.append(現在時刻)
    オイラー法_位置.append(x1)
    オイラー法_速度.append(v1)
    _x, _v = 解析解(現在時刻)
    オイラー法_位置_誤差.append(x1 - _x)
    オイラー法_速度_誤差.append(v1 - _v)
    x0 = x1
    v0 = v1

ホイン法_時刻: list[float] = list()
ホイン法_速度: list[float] = list()
ホイン法_位置: list[float] = list()
ホイン法_位置_誤差: list[float] = list()
ホイン法_速度_誤差: list[float] = list()

x0 = 初期位置
v0 = 初速度
for i in range(steps):
    現在時刻 = 初期時刻 + 時間の変位 * i
    x1, v1 = ホイン法(x0, v0)
    ホイン法_時刻.append(現在時刻)
    ホイン法_位置.append(x1)
    ホイン法_速度.append(v1)
    _x, _v = 解析解(現在時刻)
    ホイン法_位置_誤差.append(x1 - _x)
    ホイン法_速度_誤差.append(v1 - _v)
    x0 = x1
    v0 = v1

ルンゲクッタ法_時刻: list[float] = list()
ルンゲクッタ法_速度: list[float] = list()
ルンゲクッタ法_位置: list[float] = list()
ルンゲクッタ法_位置_誤差: list[float] = list()
ルンゲクッタ法_速度_誤差: list[float] = list()

x0 = 初期位置
v0 = 初速度
for i in range(steps):
    現在時刻 = 初期時刻 + 時間の変位 * i
    x1, v1 = ルンゲクッタ法(x0, v0)
    ルンゲクッタ法_時刻.append(現在時刻)
    ルンゲクッタ法_位置.append(x1)
    ルンゲクッタ法_速度.append(v1)
    _x, _v = 解析解(現在時刻)
    ルンゲクッタ法_位置_誤差.append(x1 - _x)
    ルンゲクッタ法_速度_誤差.append(v1 - _v)
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
# pyplot.show()
pyplot.savefig('位置.png')
pyplot.close()

pyplot.figure(figsize=fig_size, dpi=300)
pyplot.title('速度')
pyplot.plot(オイラー法_時刻, オイラー法_速度, label='オイラー法')
pyplot.plot(ホイン法_時刻, ホイン法_速度, label='ホイン法')
pyplot.plot(ルンゲクッタ法_時刻, ルンゲクッタ法_速度, label='ルンゲクッタ法')
pyplot.plot(解析解_時刻, 解析解_速度, label='解析解')
pyplot.legend()
# pyplot.show()
pyplot.savefig('速度.png')
pyplot.close()

pyplot.figure(figsize=fig_size, dpi=300)
pyplot.title('位置の誤差')
pyplot.plot(オイラー法_時刻, オイラー法_位置_誤差, label='オイラー法')
pyplot.plot(ホイン法_時刻, ホイン法_位置_誤差, label='ホイン法')
pyplot.plot(ルンゲクッタ法_時刻, ルンゲクッタ法_位置_誤差, label='ルンゲクッタ法')
pyplot.legend()
# pyplot.show()
pyplot.savefig('位置の誤差.png')
pyplot.close()

pyplot.figure(figsize=fig_size, dpi=300)
pyplot.title('速度の誤差')
pyplot.plot(オイラー法_時刻, オイラー法_速度_誤差, label='オイラー法')
pyplot.plot(ホイン法_時刻, ホイン法_速度_誤差, label='ホイン法')
pyplot.plot(ルンゲクッタ法_時刻, ルンゲクッタ法_速度_誤差, label='ルンゲクッタ法')
pyplot.legend()
# pyplot.show()
pyplot.savefig('速度の誤差.png')
pyplot.close()
