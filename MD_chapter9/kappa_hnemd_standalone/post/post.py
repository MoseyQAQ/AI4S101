import numpy as np
import matplotlib.pyplot as plt

def plot_ref():
    # 加载数据
    kappa = np.loadtxt("ref_kappa.txt")

    N = 200  # 每次运行的数据数量
    Ns = kappa.shape[0] // N  # 独立运行的数量
    t = kappa[:N, 0]  # 时间

    # 计算热导率的累积平均
    kxx = np.cumsum(kappa[:, 1].reshape(Ns, N), axis=1) / np.arange(1, N + 1)
    kxy = np.cumsum(kappa[:, 2].reshape(Ns, N), axis=1) / np.arange(1, N + 1)

    # 计算并显示平均热导率及其标准误差
    kxx_mean = np.mean(kxx[-1, :])
    kxx_error = np.std(kxx[-1, :]) / np.sqrt(Ns)
    print(f'k_xx = ({kxx_mean} +- {kxx_error}) W/mK')

    # 绘制 kxx
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    for run_data in kxx:
        plt.plot(t, run_data, '-')
    plt.plot(t, np.mean(kxx, axis=0), '--', linewidth=3)
    plt.xlabel('Time (ps)', fontsize=12)
    plt.ylabel('$\kappa_{xx}$ (W/mK)', fontsize=12)
    plt.title('LJ argon @ T = 20 K from HNEMD')
    plt.gca().tick_params(axis='both', labelsize=12)



    # 绘制 kyx
    plt.subplot(1,2,2)
    for run_data in kxy:
        plt.plot(t, run_data, '-')
    plt.plot(t, np.mean(kxy, axis=0), '--', linewidth=3)
    plt.xlabel('Time (ps)', fontsize=12)
    plt.ylabel('$\kappa_{yx}$ (W/mK)', fontsize=12)
    plt.title('LJ argon @ T = 20 K from HNEMD')
    plt.gca().tick_params(axis='both', labelsize=12)

    plt.savefig("ref.pdf")

def plot_my():
    from glob import glob 
    data_list = glob("my_kappa*.txt")
    data = []
    for i in data_list:
        data.append(np.loadtxt(i))
    kappa = np.concatenate(data, axis=0)

    N = 200  # 每次运行的数据数量
    Ns = kappa.shape[0] // N  # 独立运行的数量
    t = kappa[:N, 0]  # 时间
    kxx = np.cumsum(kappa[:, 1].reshape(Ns, N), axis=1) / np.arange(1, N + 1)
    kxy = np.cumsum(kappa[:, 2].reshape(Ns, N), axis=1) / np.arange(1, N + 1)

    # 计算并显示平均热导率及其标准误差
    kxx_mean = np.mean(kxx[-1, :])
    kxx_error = np.std(kxx[-1, :]) / np.sqrt(Ns)
    print(f'k_xx = ({kxx_mean} +- {kxx_error}) W/mK')
    print(np.mean(kxy[-1, :]))

    # 绘制 kxx
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    for run_data in kxx:
        plt.plot(t, run_data, '-')
    plt.plot(t, np.mean(kxx, axis=0), '--', linewidth=3)
    plt.xlabel('Time (ps)', fontsize=12)
    plt.ylabel('$\kappa_{xx}$ (W/mK)', fontsize=12)
    plt.title('LJ argon @ T = 20 K from HNEMD')
    plt.gca().tick_params(axis='both', labelsize=12)



    # 绘制 kyx
    plt.subplot(1,2,2)
    for run_data in kxy:
        plt.plot(t, run_data, '-')
    plt.plot(t, np.mean(kxy, axis=0), '--', linewidth=3)
    plt.xlabel('Time (ps)', fontsize=12)
    plt.ylabel('$\kappa_{yx}$ (W/mK)', fontsize=12)
    plt.title('LJ argon @ T = 20 K from HNEMD')
    plt.gca().tick_params(axis='both', labelsize=12)

    plt.savefig("my.pdf")

    
#plot_ref()
plot_my()