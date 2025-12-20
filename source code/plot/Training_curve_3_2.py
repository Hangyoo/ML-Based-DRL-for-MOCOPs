# -*- ecoding: utf-8 -*-
# @Project : Paper3 
# @Author: Hangyu Lou

import random

import numpy as np
import matplotlib.pyplot as plt



# 创建 3x2 子图
fig, axs = plt.subplots(2, 3, figsize=(20, 7.3)) # 6.3
axs = axs.flatten()  # 将 2x2 数组展平为一维

def plot_results1(ax, title=None, y_limit=None):
    # 数据定义
    x = np.arange(100)

    ax.plot(x, sorted([random.uniform(0, 1) for i in range(100)]), color='#82B0D2', linewidth=2.2)
    ax.plot(range(20), sorted([random.uniform(0, 1) for i in range(20)]),color='#96C37D', linewidth=2.2)
    ax.plot(range(20), sorted([random.uniform(0, 1) for i in range(20)]), color='darkorange', linewidth=2.2)
    ax.set(
        xticks=[0,20,40,60,80,100],
        xticklabels=["0",'400','800','1200','1600','2000'],
        ylim=y_limit,
    )
    ax.set_ylabel(title, fontsize=16, labelpad=0)
    ax.tick_params(labelsize=15, pad=10)
    ax.grid(axis='y', linestyle="-.")
    ax.legend(["Training from scratch", "CUMNM + Fine-tuning Decoder", "CUMNM + Fine-tuning Entire"], fontsize=13, loc=4)

def plot_results2(ax, title=None, y_limit=None):
    # 数据定义
    x = np.arange(100)

    ax.plot(x, sorted([random.uniform(0, 1) for i in range(100)]), color='#82B0D2', linewidth=2.2)
    ax.plot(range(20), sorted([random.uniform(0, 1) for i in range(20)]),color='#96C37D', linewidth=2.2)
    ax.plot(range(20), sorted([random.uniform(0, 1) for i in range(20)]), color='darkorange', linewidth=2.2)
    ax.set(
        xticks=[0,20,40,60,80,100],
        xticklabels=["0",'400','800','1200','1600','2000'],
        ylim=y_limit,
    )
    ax.set_ylabel(title, fontsize=16, labelpad=0)
    ax.tick_params(labelsize=15, pad=10)
    ax.grid(axis='y', linestyle="-.")
    ax.legend(["Training from scratch", "CUMNM + Fine-tuning Decoder", "CUMNM + Fine-tuning Entire"], fontsize=13, loc=4)

def plot_results3(ax, title=None, y_limit=None):
    # 数据定义
    x = np.arange(100)

    ax.plot(x, sorted([random.uniform(0, 1) for i in range(100)]), color='#82B0D2', linewidth=2.2)
    ax.plot(range(20), sorted([random.uniform(0, 1) for i in range(20)]),color='#96C37D', linewidth=2.2)
    ax.plot(range(20), sorted([random.uniform(0, 1) for i in range(20)]), color='darkorange', linewidth=2.2)
    ax.set(
        xticks=[0,20,40,60,80,100],
        xticklabels=["0",'400','800','1200','1600','2000'],
        ylim=y_limit,
    )
    ax.set_ylabel(title, fontsize=16, labelpad=0)
    ax.tick_params(labelsize=15, pad=10)
    ax.grid(axis='y', linestyle="-.")
    ax.legend(["Training from scratch", "CUMNM + Fine-tuning Decoder", "CUMNM + Fine-tuning Entire"], fontsize=13, loc=4)

def plot_results4(ax, title=None, y_limit=None):
    # 数据定义
    x = np.arange(100)

    ax.plot(x, sorted([random.uniform(0, 1) for i in range(100)]), color='#82B0D2', linewidth=2.2)
    ax.plot(range(20), sorted([random.uniform(0, 1) for i in range(20)]),color='#96C37D', linewidth=2.2)
    ax.plot(range(20), sorted([random.uniform(0, 1) for i in range(20)]), color='darkorange', linewidth=2.2)
    ax.set(
        xticks=[0,20,40,60,80,100],
        xticklabels=["0",'400','800','1200','1600','2000'],
        ylim=y_limit,
    )
    ax.set_xlabel('Number of Epochs', fontsize=16, labelpad=0)
    ax.set_ylabel(title, fontsize=16, labelpad=0)
    ax.tick_params(labelsize=15, pad=10)
    ax.grid(axis='y', linestyle="-.")
    ax.legend(["Training from scratch", "CUMNM + Fine-tuning Decoder", "CUMNM + Fine-tuning Entire"], fontsize=13, loc=4)

def plot_results5(ax, title=None, y_limit=None):
    # 数据定义
    x = np.arange(100)

    ax.plot(x, sorted([random.uniform(0, 1) for i in range(100)]), color='#82B0D2', linewidth=2.2)
    ax.plot(range(20), sorted([random.uniform(0, 1) for i in range(20)]),color='#96C37D', linewidth=2.2)
    ax.plot(range(20), sorted([random.uniform(0, 1) for i in range(20)]), color='darkorange', linewidth=2.2)
    ax.set(
        xticks=[0,20,40,60,80,100],
        xticklabels=["0",'400','800','1200','1600','2000'],
        ylim=y_limit,
    )
    ax.set_xlabel('Number of Epochs', fontsize=16, labelpad=0)
    ax.set_ylabel(title, fontsize=16, labelpad=0)
    ax.tick_params(labelsize=15, pad=10)
    ax.grid(axis='y', linestyle="-.")
    ax.legend(["Training from scratch", "CUMNM + Fine-tuning Decoder", "CUMNM + Fine-tuning Entire"], fontsize=13, loc=4)

def plot_results6(ax, title=None, y_limit=None):
    # 数据定义
    x = np.arange(100)

    ax.plot(x, sorted([random.uniform(0, 1) for i in range(100)]), color='#82B0D2', linewidth=2.2)
    ax.plot(range(20), sorted([random.uniform(0, 1) for i in range(20)]),color='#96C37D', linewidth=2.2)
    ax.plot(range(20), sorted([random.uniform(0, 1) for i in range(20)]), color='darkorange', linewidth=2.2)
    ax.set(
        xticks=[0,20,40,60,80,100],
        xticklabels=["0",'400','800','1200','1600','2000'],
        ylim=y_limit,
    )
    ax.set_xlabel('Number of Epochs', fontsize=16, labelpad=0)
    ax.set_ylabel(title, fontsize=16, labelpad=0)
    ax.tick_params(labelsize=15, pad=10)
    ax.grid(axis='y', linestyle="-.")
    ax.legend(["Training from scratch", "CUMNM + Fine-tuning Decoder", "CUMNM + Fine-tuning Entire"], fontsize=13, loc=4)

# 在每个子图中绘制相同的内容
titles = ["HV Metric on MOTSP",
          "HV Metric on MOMIXTSP",
          'HV Metric on MOCVRP',
          'HV Metric on MOVRPTW',
          'HV Metric on MOKP',
          'HV Metric on MOOVRP']  # 可以根据需要自定义标题

y_limits = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]  # 可以根据需要自定义 y 轴限制

plot_results1(axs[0], title=titles[0], y_limit=y_limits[0])
plot_results2(axs[1], title=titles[1], y_limit=y_limits[1])
plot_results3(axs[2], title=titles[2], y_limit=y_limits[2])
plot_results4(axs[3], title=titles[3], y_limit=y_limits[3])
plot_results5(axs[4], title=titles[4], y_limit=y_limits[4])
plot_results6(axs[5], title=titles[5], y_limit=y_limits[5])

plt.tight_layout()
plt.show()
