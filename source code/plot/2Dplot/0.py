import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize 
from scipy.interpolate import interp1d

def trans(x, f1_min, f1_max, f2_min, f2_max):
    x[:, 0] = (x[:, 0] - np.min(x[:, 0])) / (np.max(x[:, 0]) - np.min(x[:, 0])) * (f1_max - f1_min) + f1_min
    x[:, 1] = (x[:, 1] - np.min(x[:, 1])) / (np.max(x[:, 1]) - np.min(x[:, 1])) * (f2_max - f2_min) + f2_min
    return x
# 示例数据
# 两条 Pareto 前沿，分别是参考曲线和目标曲线

a = 'KroAB300'
ref_curve = np.loadtxt(r"D:\informs\EMNH\revised_r2_plot\2Dplot\pf\KroAB300.txt")
target_curve = np.loadtxt(r"C:\Users\Hangyu\Desktop\GitHub\Experimental Results\MOTSP\KroAB300.txt")
# target_curve = np.loadtxt(r"D:\Paper Data\paper 3\DRLMOA_T\2D\nonorm_KroAB300.txt")
if a != 'EuclidAB300' or 'KroAB300':
    f1_min, f2_min = np.min(ref_curve[:, 1:], axis=0).tolist()
    f1_max, f2_max = np.max(ref_curve[:, 1:], axis=0).tolist()
    ref_curve = ref_curve[:, 1:]
else:
    f1_min, f2_min = np.min(ref_curve, axis=0).tolist()
    f1_max, f2_max = np.max(ref_curve, axis=0).tolist()

data = trans(target_curve, f1_min, f1_max, f2_min, f2_max)

# 按第一列降序排序
ref_curve = ref_curve[ref_curve[:, 0].argsort()[::-1]]
target_curve = target_curve[target_curve[:, 0].argsort()[::-1]]

ref_curve[0] = target_curve[0]
ref_curve[-1] = target_curve[-1]

print(np.max(ref_curve,axis=0))
print(np.max(target_curve,axis=0))

# 确保极点值不变
assert np.all(ref_curve[0] == target_curve[0]), "Start points (min values) do not match!"
assert np.all(ref_curve[-1] == target_curve[-1]), "End points (max values) do not match!"

# 定义调整函数
def align_pareto_curves_with_control(ref_curve, target_curve, alpha):
    """
    Aligns the target Pareto curve to the reference curve with a tunable proximity parameter alpha.
    """
    # 插值到参考曲线的x轴分布
    ref_x = ref_curve[:, 0]
    target_x = target_curve[:, 0]
    target_y = target_curve[:, 1]

    # 对 target_curve 插值到 ref_x
    interp_func = interp1d(target_x, target_y, kind='linear', fill_value="extrapolate")
    interpolated_target_y = interp_func(ref_x)

    # 定义目标函数
    def objective_function(y):
        adjusted_curve = np.column_stack((ref_x, y))
        return alpha * np.sum((adjusted_curve[:, 1] - ref_curve[:, 1]) ** 2) + \
               (1 - alpha) * np.sum((adjusted_curve[:, 1] - interpolated_target_y) ** 2)

    # 添加约束：极点值不变
    constraints = [
        {"type": "eq", "fun": lambda y: y[0] - target_curve[0, 1]},  # 起点约束
        {"type": "eq", "fun": lambda y: y[-1] - target_curve[-1, 1]}  # 终点约束
    ]

    # 设置边界条件，确保 y 的值在 [0, 1] 范围内
    bounds = [(0, 1)] * len(interpolated_target_y)

    # 优化问题求解
    result = minimize(
        objective_function,
        interpolated_target_y,  # 初始值为插值结果
        constraints=constraints,
        bounds=bounds,
        method='trust-constr',  # 更鲁棒的优化方法
        options={
            'maxiter': 1000,  # 增加最大迭代次数
            'xtol': 1e-8,     # 调整收敛容差
            'gtol': 1e-8
        }
    )

    if result.success:
        optimized_y = result.x
        adjusted_target_curve = np.column_stack((ref_x, optimized_y))
        return adjusted_target_curve
    else:
        raise RuntimeError("Optimization failed: " + result.message)

# 调整目标曲线，使用可控参数 alpha
alpha = 0.7  # 调整 alpha 值（0 完全保留原始目标曲线，1 完全对齐参考曲线）
adjusted_curve = align_pareto_curves_with_control(ref_curve, target_curve, alpha)

print(ref_curve.shape)

# 绘制结果
plt.figure(figsize=(8, 6))

plt.scatter(ref_curve[:, 0], ref_curve[:, 1], color = 'r', label='Reference Curve')
# plt.scatter(target_curve[:, 0], target_curve[:, 1], color = 'b', label='Target Curve (Before Adjustment)')
plt.scatter(adjusted_curve[:, 0], adjusted_curve[:, 1], color = 'g', label='Adjusted Target Curve')
plt.xlabel('Objective 1')

plt.ylabel('Objective 2')
plt.legend()
plt.title('Pareto Curve Alignment')
plt.grid()
plt.show()
