import numpy as np
from glob import glob 
from tqdm import tqdm
def find_hac_kappa(Nd, Nc, dt, T_0, V, h, outfile):
    dt_in_ps = dt * 1.018051e+1 / 1000.0
    M = Nd - Nc

    hac = find_hac(Nc, M, h)
    factor = dt * 0.5 * 1.573769e+5 / (8.617343e-5 * T_0 * T_0 * V)
    rtc = find_rtc(Nc, factor, hac)

    with open(outfile, 'a') as f:
        for i in range(Nc):
            f.write(f"{i*dt_in_ps:.5e} {hac[i][0]:.5e} {hac[i][1]:.5e} {hac[i][2]:.5e} {rtc[i][0]:.5e} {rtc[i][1]:.5e} {rtc[i][2]:.5e}\n")

def find_hac(Nc, M, h):
    hac = np.zeros((Nc, 3))
    for nc in range(Nc):
        for m in range(M):
            hac[nc] += h[m] * h[m + nc]
    
    hac /= M
    return hac

def find_rtc(Nc, factor, hac):

    rtc = np.zeros((Nc, 3))
    for nc in range(1, Nc):
        rtc[nc] = rtc[nc-1] + (hac[nc-1] + hac[nc]) * factor
    
    return rtc

if __name__ == "__main__":
    Ne, Np = 20000, 20000
    Ns = 10
    Nd = Np / Ns
    Nc = Nd/ 10
    time_step = 10.0 / 1.018051e+1
    T_0 = 60

    for i in tqdm(glob("./my_data/*.npy")):
        data = np.load(i)
        find_hac_kappa(int(Nd), int(Nc),time_step * Ns, T_0, 9993.948264,data, 'my_50_kappa.txt')