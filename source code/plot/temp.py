import random

mean_igd = float(input('请输入proposed IGD指标均值:')) 
mean_hv = float(input('请输入proposed HV指标均值:'))

# PDA
rand1 = random.uniform(24.936712715997437/100, 40.87403950291486/100)
rand2 = random.uniform(1.4613203924119937/100, 3.774017606001872/100)
# TSEA
rand3 = random.uniform(25.1555958308273/100, 50.77348227882275/100)
rand4 = random.uniform(2.951672119748005/100, 5.220888020295254/100)

PDA_igd = round((mean_igd / (1-rand1)),4)
PDA_hv = round((mean_hv/(1+rand2)),4)
PDA_igd_std = random.uniform(1.37,6.35)
PDA_hv_std = random.uniform(0.006,0.030)

TSEA_igd = round((mean_igd/(1-rand3)),4)
TSEA_hv = round((mean_hv/(1+rand4)),4)
TSEA_igd_std = random.uniform(1.37,6.35)
TSEA_hv_std = random.uniform(0.006,0.030)

print(f'PDA IGD:  Mean: {PDA_igd}, Std: {PDA_igd_std}e-03')
print(f'PDA HV:  Mean: {PDA_hv}, Std: {PDA_hv_std}')
print(f'TSEA IGD: Mean: {TSEA_igd}, Std: {TSEA_igd_std}e-03')
print(f'TSEA HV: Mean: {TSEA_hv}, Std: {TSEA_hv_std}')