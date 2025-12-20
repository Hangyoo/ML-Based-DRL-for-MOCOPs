# -*- ecoding: utf-8 -*-
# @Project : 
# @Author: Hangyu Lou


import math 
import random
import numpy as np
import matplotlib.pyplot as plt

# 生成 x 轴的数据点，范围从 -2π 到 2π，步长为 0.1
x = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 400)
# 创建一个 1 行 3 列的子图
plt.figure(figsize=(12, 4))

def my_function(x):
    a = random.uniform(-1.5,1.5)
    b = random.uniform(-5,5)
    return a*np.sin(x+b)

obj = []
for i in range(10):
    # 创建图形并绘制 sin 曲线
    obj.append(my_function(x))
obj = np.array(obj)

for i in range(10):
    plt.subplot(1, 3, 1)
    plt.plot(x, obj[i], color="blue")
plt.subplot(1, 3, 1)
plt.plot(x, np.mean(obj,axis=0), color = 'red', linewidth=3, label="Average value")
plt.legend(loc=1)
plt.title('10 functions')
plt.xlabel("x")
plt.ylabel("asin(x+b)")

obj = []
for i in range(100):
    # 创建图形并绘制 sin 曲线
    obj.append(my_function(x))
obj = np.array(obj)

for i in range(100):
    plt.subplot(1, 3, 2)
    plt.plot(x, obj[i], color="blue")
plt.subplot(1, 3, 2)
plt.plot(x, np.mean(obj,axis=0), color = 'red', linewidth=3, label="Average value")
plt.legend(loc=1)
plt.title('100 functions')
plt.xlabel("x")
# plt.ylabel("asin(x+b)")

obj = []
for i in range(1000):
    # 创建图形并绘制 sin 曲线
    obj.append(my_function(x))
obj = np.array(obj)

for i in range(1000):
    plt.subplot(1, 3, 3)
    plt.plot(x, obj[i], color="blue")
plt.subplot(1, 3, 3)
plt.plot(x, np.mean(obj,axis=0), color = 'red', linewidth=3, label="Average value")
plt.legend(loc=1)
plt.title('1000 functions')
# 添加标题和标签
plt.xlabel("x")
# plt.ylabel("asin(x+b)")


# 显示网格
plt.grid(True)
# 显示图形
plt.show()