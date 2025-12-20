import numpy as np
import matplotlib.pyplot as plt
 
Friedman_IGD = {0: 9.33, 1: 7.00, 2: 4.33, 3: 3.33,
                4: 8.33, 5: 9.00, 6: 5.83, 7: 2.83, 8: 4.00, 9: 1.00}

Friedman_HV = {0: 9.83, 1: 7.00, 2: 3.83, 3: 3.33,
               4: 8.00, 5: 8.83, 6: 5.83, 7: 2.83, 8: 4.50, 9: 1.00}

def plot_results():
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))

    plot_info = ("fit_time", "Fit times (s)", ax1, None)

    x = np.arange(len(Friedman_IGD))
    x1, width1 = np.arange(len(Friedman_IGD))-0.22, 0.42
    x2, width2 = np.arange(len(Friedman_IGD))+0.22, 0.42
    key, title, ax1, y_limit = plot_info[0],plot_info[1],plot_info[2],plot_info[3]

    IGD = Friedman_IGD.values()
    HV = Friedman_HV.values()

    ax1.bar(
        x=x1,
        height=IGD,
        width=width1,
        color="C1",
        linestyle = "-.",
        # hatch="."
    )
    ax1.bar(
        x=x2,
        height=HV,
        width=width2,
        color="#0099fd",
        # hatch = "xxx"
    )

    ax1.set(
        title= None, # title,
        xticks=x,
        xticklabels=['DRL-MOA',"DRL-MOA-T", "AM-MOCO", "MLMA",
                     "NSGA-II","MOEA/D","MOGLS","PDA","HLS-EA","ML-NMCO"],
        ylim=y_limit,
    )

    for x, y in zip(x1, IGD):
        plt.text(x, y + 0.06, '%.2f' % y, ha='center', va='bottom', fontsize=13)

    for x_2, y_2 in zip(x2, HV):
        plt.text(x_2, y_2 + 0.06, '%.2f' % y_2, ha='center', va='bottom', fontsize=13)

    # ax1.set_xlabel("State-of-the-art algorithms", fontsize=18, labelpad=15)
    ax1.set_ylabel("Average Friedman Ranking on MOTSP", fontsize=18, labelpad=15)
    # 调整坐标轴刻度大小
    plt.tick_params(labelsize=15, pad=10)
    labels = ax1.get_xticklabels()
    ax1.grid(axis='y', linestyle="-.")  # 以虚线形式显示横轴刻度线
    plt.setp(labels, rotation=20, horizontalalignment='right')
    plt.legend(["IGD metric","HV metric"], fontsize=16)
    plt.show()


plot_results()