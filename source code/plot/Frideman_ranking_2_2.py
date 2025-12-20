# -*- ecoding: utf-8 -*-
# @Project : Paper3
# @Author: Hangyu Lou
# @Time: 2024/9/9 11:30 

import numpy as np
import matplotlib.pyplot as plt



# 创建 2x2 子图
fig, axs = plt.subplots(2, 2, figsize=(20, 10.5))
axs = axs.flatten()  # 将 2x2 数组展平为一维

# 绘图函数
def plot_results1(ax, title=None, y_limit=None):
    # 数据定义
    Friedman_IGD = {0: 2.31, 1: 4.85, 2: 7.00, 3: 4.54,
                    4: 5.46, 5: 2.85, 6: 1.00}

    Friedman_HV = {0: 2.38, 1: 5.08, 2: 7.00, 3: 4.54,
                   4: 5.38, 5: 2.77, 6: 1.00}
    x = np.arange(len(Friedman_IGD))
    x1, width1 = np.arange(len(Friedman_IGD)) - 0.22, 0.42
    x2, width2 = np.arange(len(Friedman_IGD)) + 0.22, 0.42

    IGD = list(Friedman_IGD.values())
    HV = list(Friedman_HV.values())

    ax.bar(
        x=x1,
        height=IGD,
        width=width1,
        color="C1",
        linestyle="-.",
    )
    ax.bar(
        x=x2,
        height=HV,
        width=width2,
        color="#0099fd",
    )

    ax.set(
        xticks=x,
        xticklabels=['PDA', 'HLS-EA', 'DRL-MOA', 'PMOCO', 'ML-AM', 'EMNH', 'CUMNM'],
        ylim=y_limit,
    )

    for x, y in zip(x1, IGD):
        ax.text(x, y + 0.06, '%.2f' % y, ha='center', va='bottom', fontsize=11)

    for x_2, y_2 in zip(x2, HV):
        ax.text(x_2, y_2 + 0.06, '%.2f' % y_2, ha='center', va='bottom', fontsize=11)

    ax.set_ylabel(title, fontsize=16, labelpad=5)
    ax.tick_params(labelsize=15, pad=10)
    ax.grid(axis='y', linestyle="-.")
    # 旋转 x 轴刻度标签以防重叠
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=20, horizontalalignment='right')
    ax.legend(["IGD metric", "HV metric"], fontsize=13,loc='upper right')

def plot_results2(ax, title=None, y_limit=None):
    # 数据定义
    Friedman_IGD = {0: 5.14, 1: 2.29, 2: 7.00, 3: 3.86,
                    4: 5.14, 5: 3.71, 6: 1.00}

    Friedman_HV = {0: 5.14, 1: 2.29, 2: 7.00, 3: 3.86,
                    4: 5.14, 5: 3.71, 6: 1.00}

    x = np.arange(len(Friedman_IGD))
    x1, width1 = np.arange(len(Friedman_IGD)) - 0.22, 0.42
    x2, width2 = np.arange(len(Friedman_IGD)) + 0.22, 0.42

    IGD = list(Friedman_IGD.values())
    HV = list(Friedman_HV.values())

    ax.bar(
        x=x1,
        height=IGD,
        width=width1,
        color="C1",
        linestyle="-.",
    )
    ax.bar(
        x=x2,
        height=HV,
        width=width2,
        color="#0099fd",
    )

    ax.set(
        xticks=x,
        xticklabels=['IMOLEM', 'CoEA-DAE','DRL-MOA', 'PMOCO', 'ML-AM', 'EMNH', 'CUMNM'],
        ylim=y_limit,
    )

    for x, y in zip(x1, IGD):
        ax.text(x, y + 0.06, '%.2f' % y, ha='center', va='bottom', fontsize=11)

    for x_2, y_2 in zip(x2, HV):
        ax.text(x_2, y_2 + 0.06, '%.2f' % y_2, ha='center', va='bottom', fontsize=11)

    ax.set_ylabel(title, fontsize=16, labelpad=5)
    ax.tick_params(labelsize=15, pad=10)
    ax.grid(axis='y', linestyle="-.")
    # 旋转 x 轴刻度标签以防重叠
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=20, horizontalalignment='right')
    ax.legend(["IGD metric", "HV metric"], fontsize=13,loc='upper right')

def plot_results3(ax, title=None, y_limit=None):
    # 数据定义
    Friedman_IGD = {0: 4.83, 1: 3.33, 2: 2.00, 3: 3.83,
                    4: 1.00}

    Friedman_HV = {0: 4.83, 1: 3.33, 2: 2.00, 3: 3.83,
                    4: 1.00}

    x = np.arange(len(Friedman_IGD))
    x1, width1 = np.arange(len(Friedman_IGD)) - 0.22, 0.42
    x2, width2 = np.arange(len(Friedman_IGD)) + 0.22, 0.42

    IGD = list(Friedman_IGD.values())
    HV = list(Friedman_HV.values())

    ax.bar(
        x=x1,
        height=IGD,
        width=width1,
        color="C1",
        linestyle="-.",
    )
    ax.bar(
        x=x2,
        height=HV,
        width=width2,
        color="#0099fd",
    )

    ax.set(
        xticks=x,
        xticklabels=['CMMO', 'INSGA-II', 'M-MOEA/D', 'ML-AM', 'CUMNM'],
        ylim=y_limit,
    )

    for x, y in zip(x1, IGD):
        ax.text(x, y + 0.06, '%.2f' % y, ha='center', va='bottom', fontsize=11)

    for x_2, y_2 in zip(x2, HV):
        ax.text(x_2, y_2 + 0.06, '%.2f' % y_2, ha='center', va='bottom', fontsize=11)

    ax.set_ylabel(title, fontsize=16, labelpad=5)
    ax.tick_params(labelsize=15, pad=10)
    ax.grid(axis='y', linestyle="-.")
    # 旋转 x 轴刻度标签以防重叠
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=20, horizontalalignment='right')
    ax.legend(["IGD metric", "HV metric"], fontsize=13,loc='upper right')

def plot_results4(ax, title=None, y_limit=None):
    # 数据定义
    Friedman_IGD = {0: 3.33, 1: 5.67, 2: 3.00, 3: 7.00,
                    4: 5.33, 5: 6.33, 6: 3.67, 7: 1.67}

    Friedman_HV = {0: 3.33, 1: 5.67, 2: 3.00, 3: 7.33,
                   4: 5.00, 5: 6.33, 6: 3.67, 7: 1.67}

    x = np.arange(len(Friedman_IGD))
    x1, width1 = np.arange(len(Friedman_IGD)) - 0.22, 0.42
    x2, width2 = np.arange(len(Friedman_IGD)) + 0.22, 0.42

    IGD = list(Friedman_IGD.values())
    HV = list(Friedman_HV.values())

    ax.bar(
        x=x1,
        height=IGD,
        width=width1,
        color="C1",
        linestyle="-.",
    )
    ax.bar(
        x=x2,
        height=HV,
        width=width2,
        color="#0099fd",
    )

    ax.set(
        xticks=x,
        xticklabels=['Direct', 'MOFPA', 'FHCo','DRL-MOA', 'PMOCO', 'ML-AM', 'EMNH', 'CUMNM'],
        ylim=y_limit,
    )

    for x, y in zip(x1, IGD):
        ax.text(x, y + 0.06, '%.2f' % y, ha='center', va='bottom', fontsize=11)

    for x_2, y_2 in zip(x2, HV):
        ax.text(x_2, y_2 + 0.06, '%.2f' % y_2, ha='center', va='bottom', fontsize=11)

    ax.set_ylabel(title, fontsize=16, labelpad=5)
    ax.tick_params(labelsize=15, pad=10)
    ax.grid(axis='y', linestyle="-.")
    # 旋转 x 轴刻度标签以防重叠
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=20, horizontalalignment='right')
    ax.legend(["IGD metric", "HV metric"], fontsize=13,loc='upper right')


# 在每个子图中绘制相同的内容
titles = ["Average Friedman Ranking on MOTSP",
          'Average Friedman Ranking on MOCVRP',
          'Average Friedman Ranking on MOVRPTW',
          'Average Friedman Ranking on MOKP']  # 可以根据需要自定义标题

y_limits = [(0, 8), (0, 8), (0, 6), (0, 8)]  # 可以根据需要自定义 y 轴限制

plot_results1(axs[0], title=titles[0], y_limit=y_limits[0])
plot_results2(axs[1], title=titles[1], y_limit=y_limits[1])
plot_results3(axs[2], title=titles[2], y_limit=y_limits[2])
plot_results4(axs[3], title=titles[3], y_limit=y_limits[3])

plt.tight_layout()
plt.show()

