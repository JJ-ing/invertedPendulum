import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def v_train():
    # 准备数据
    num = 500
    x = np.linspace(1, num, num)
    r1 = np.load("SARSA_r.npz")
    r2 = np.load("QLearning_r.npz")
    y1 = r1["arr_0"]
    y2 = r2["arr_0"]

    # # 三次样条数据插值
    # x_smooth = np.linspace(min(x), max(x), 500)
    # y_smooth = make_interp_spline(x, y)(x_smooth)

    # 画图
    ax = plt.subplot()
    plt.plot(x, y1, color='red', label='SARSA')
    plt.plot(x, y2, color='blue', label='QLearning')
    plt.legend()

    plt.title('R-T', fontsize=20)
    plt.xlabel('episode', fontsize=15, labelpad=10)
    plt.ylabel('reward', fontsize=15, labelpad=10)


    plt.savefig('./r-t.png', dpi=120, bbox_inches='tight')
    plt.show()


def v_test_SARSA():
    # 准备数据
    num = 1000
    x = np.linspace(1, num, num)
    r1 = np.load("SARSA_test_end_nums.npz")
    # r2 = np.load("QLearning_r.npz")
    y1 = r1["arr_0"]
    # y2 = r2["arr_0"]
    for i in range(len(y1)):
        if y1[i] > 2500:
            y1[i] = 2000

    # # 三次样条数据插值
    # x_smooth = np.linspace(min(x), max(x), 500)
    # y_smooth = make_interp_spline(x, y)(x_smooth)

    # 画图
    ax = plt.subplot()
    plt.plot(x, y1, color='green')
    # plt.plot(x, y2, color='blue')

    plt.title('steps of convergence of SARSA', fontsize=20)
    plt.xlabel('episode', fontsize=15, labelpad=10)
    plt.ylabel('step', fontsize=15, labelpad=10)

    plt.savefig('./SARSA_test_end_nums.png', dpi=120, bbox_inches='tight')
    plt.show()


def v_test_QLearning():
    # 准备数据
    data = np.load("QLearning_test_result.npz")
    r = data["arr_0"]

    # 读取决策序列
    # actions = data["arr_1"]
    # for i in range(len(actions)):
    #     if actions[i] == -3:
    #         actions[i] = 0
    #     elif actions[i] == 3:
    #         actions[i] = 2
    #     else:
    #         actions[i] = 1
    # print(list(actions))

    y_a = []
    y_a1 = []
    for i in range(len(r)):
        y_a.append(abs(r[i][0]))
        y_a1.append(r[i][1])
    last_a = y_a[-1]
    for i in range(1*len(r)):
        y_a.append(last_a)
    num = 2 * len(r)
    x = np.linspace(1, num, num)

    # last_a1 = y_a1[-1]
    # for i in range(1*len(r)):
    #     y_a1.append(last_a1)
    # num = 2 * len(r)
    # x = np.linspace(1, num, num)

    # 画图
    ax = plt.subplot()
    plt.plot(x, y_a, color='teal')
    plt.title('convergence of alpha of QLearning', fontsize=20)
    plt.xlabel('step', fontsize=15, labelpad=10)
    plt.ylabel('|alpha|', fontsize=15, labelpad=10)

    plt.savefig('./QLearning_test_result_a.png', dpi=120, bbox_inches='tight')
    plt.show()


v_train()
# v_test_SARSA()
# v_test_QLearning()