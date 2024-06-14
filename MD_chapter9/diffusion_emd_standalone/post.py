import numpy as np
from tqdm import tqdm

def find_msd(file_name: str, Nd: int, Nc:int, N: int,
             delta_tau: float, outfile: str):
    msd_raw_data = np.load(file_name)
    M = Nd - Nc 
    f = open(outfile, 'w')
    for nc in tqdm(range(Nc)):
        msd = np.zeros(3)
        for m in range(M):
            msd += np.sum((msd_raw_data[m,:,:] - msd_raw_data[m+nc+1,:,:])**2, axis=0)
    
        msd /= N * M
        f.write(f'{nc * delta_tau} {msd[0]} {msd[1]} {msd[2]}\n')
    
    f.close()


def find_vac(file_name: str, Nd: int, Nc:int, N: int, delta_tau: float,
             outfile: str):
    vac_raw_data = np.load(file_name)
    M = Nd - Nc
    print(Nc,Nd,M)
    f = open(outfile, 'w')
    for nc in tqdm(range(Nc)):
        vac = np.zeros(3)
        for m in range(M):
            vac += np.sum(vac_raw_data[m,:,:] * vac_raw_data[m+nc,:,:], axis=0)
        
        vac /= N * M
        f.write(f'{nc * delta_tau} {vac[0]} {vac[1]} {vac[2]}\n')
    
    f.close()

def plot_vac(file_name: str, reffile):
    data = np.loadtxt(file_name)
    import matplotlib.pyplot as plt
    t = data[:, 0] * 10 * 10.18 / 1000
    vac = np.mean(data[:, 2:4], axis=1) * (1.6/1.66*100)
    plt.plot(t, vac)

    refdata = np.loadtxt(reffile)
    t = data[:, 0] * 10 * 10.18 / 1000
    refvac = np.mean(refdata[:, 2:4], axis=1) * (1.6/1.66*100)
    plt.plot(t, refvac, 'r')
    plt.xlabel('Time (ps)')
    plt.ylabel('VACF')
    plt.legend(['VACF', 'Reference'])

    plt.savefig("my_vac.png")

def plot_msd(file_name: str, reffile):
    data = np.loadtxt(file_name)
    import matplotlib.pyplot as plt
    t = data[:, 0] * 10 * 10.18 / 1000
    time_step =  t[1]-t[0]
    msd = np.mean(data[:, 2:4], axis=1)/100
    plt.plot(t+time_step, msd)
    
    refdata = np.loadtxt(reffile)
    t = data[:, 0] * 10 * 10.18 / 1000
    refmsd = np.mean(refdata[:, 2:4], axis=1)/100
    plt.plot(t, refmsd, 'r')

    plt.xlabel('Time (ps)')
    plt.ylabel('MSD')
    plt.legend(['MSD', 'Reference'])
    plt.savefig("my_msd.png")

def main():
    Np = 100000
    Ns = 10
    Nd = Np / Ns 
    Nc = Nd / 10
    time_step = 10.0 / 1.018051e+1
    n = 256
    find_vac('vel_new.npy', int(Nd), int(Nc), N=n, delta_tau = Ns*time_step, outfile='my_vac.dat')
    plot_vac('my_vac.dat','ref_vac.dat')
    find_msd('coord.npy',Nd=int(Nd),Nc=int(Nc),N=n,delta_tau=Ns*time_step,outfile='my_msd.dat')
    plot_msd('my_msd.dat','ref_msd.dat')
    pass 

main()