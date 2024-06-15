import numpy as np 
import matplotlib.pyplot as plt

# 加载数据
raw_data = np.loadtxt("ref_50_kappa.txt")
M = 200
simulation_number = len(raw_data) // M
t = raw_data[0:M,0]

# 计算均值，并重塑数据以适应模拟次数
hac:np.ndarray = np.mean(raw_data[:,1:4], axis=1)
hac = hac.reshape(simulation_number, M)
rtc = np.mean(raw_data[:,4:7], axis=1)
rtc = rtc.reshape(simulation_number, M)


# 计算均值
hac_mean = np.mean(hac, axis=0)
rtc_mean = np.mean(rtc, axis=0)

# plot HCACF
for i in range(simulation_number):
    plt.plot(t, hac[i,:] / hac[i,0], 'gray', alpha=0.5)

plt.plot(t, hac_mean / hac_mean[0], 'r', label='HAC',linewidth=3)
plt.xlabel('Correlation Time (ps)', fontsize=15)
plt.ylabel('HCACF (Normalized)', fontsize=15)
plt.gca().tick_params(axis='both', labelsize=15)
plt.savefig('ref_hacf.png', dpi=300)
plt.close()

# thermal conductivity
for i in range(simulation_number):
    plt.plot(t, rtc[i,:], 'gray', alpha=0.5)
plt.plot(t, rtc_mean, 'r',linewidth=3)
plt.xlabel('Correlation Time (ps)', fontsize=15)
plt.ylabel('Thermal Conductivity (W/mK)', fontsize=15)
plt.gca().tick_params(axis='both', labelsize=15)
plt.savefig('ref_kappa.png', dpi=300)
plt.close()

# Average over an appropriate time block after visual inspection
kappa_converged = np.mean(rtc[:, M//2:], axis=1)  # 计算每个模拟在后半段的平均值

# Report the mean value and an error estimate (standard error)
kappa_average = np.mean(kappa_converged)
kappa_error = np.std(kappa_converged) / np.sqrt(simulation_number)  # 使用标准误差

print(f"Average kappa: {kappa_average:.3f} W/mK")
print(f"Standard Error: {kappa_error:.3f} W/mK")