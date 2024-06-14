import numpy as np
from time import time
from tqdm import tqdm
from joblib import dump

def timer(func):
    def func_wrapper(*args, **kwargs):
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s cost time: %.3f s' % (func.__name__, time_spend))
        return result
    return func_wrapper

class Units:
    k_B: float = 8.617343e-5
    TIME_UNIT_CONVERSION: float = 1.018051e+1

class LJParameters:
    def __init__(self,epsilon: float = 1.032e-2,sigma: float = 3.405,cutoff: float = 9.0):

        # LJ势参数初始化
        self.epsilon: float = epsilon
        self.sigma: float = sigma
        self.cutoff: float = cutoff
        self.cutoffSquare: float = cutoff**2
        self.sigma3: float = sigma**3
        self.sigma6: float = sigma**6
        self.sigma12: float = sigma**12
        self.e24s6: float = 24.0 * epsilon * self.sigma6
        self.e48s12: float = 48.0 * epsilon * self.sigma12
        self.e4s6: float = 4.0 * epsilon * self.sigma6
        self.e4s12: float = 4.0 * epsilon * self.sigma12

class Atom:
    def __init__(self, filename: str='xyz.in',
                 cutoffNeighbor: float=10.0, MaxNeighbor: int=1000) -> None:
        '''
        读取xyz文件，初始化原子的坐标，质量，速度，势能，动能

        :param filename: xyz文件名
        :returns: None
        '''

        self.filename = filename
        
        # 读取xyz文件
        self.number, self.box, self.coords, self.mass = self.parseXyzFile(self.filename)
        self.halfBox = 0.5 * self.box

        # 初始化原子的力、速度、势能、动能
        self.forces = np.zeros((self.number, 3))
        self.velocities = np.zeros((self.number, 3))
        self.pe = 0.0
        self.ke = self.getKineticEnergy()

        # Neighbor list parameters
        self.MaxNeighbor: int = MaxNeighbor
        self.cutoffNeighbor: float = cutoffNeighbor
        self.NeighborNumber: np.ndarray = np.zeros(self.number, dtype=int)
        self.NeighborList: np.ndarray = np.zeros((self.number, self.MaxNeighbor), dtype=int)

        # msd and vacf calculation
        self.coord_msd: np.ndarray = np.zeros((self.number, 3))
        
    
    def parseXyzFile(self, filename: str) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        '''
        读取xyz文件，返回原子数，盒子大小，原子坐标，原子质量

        :param filename: xyz文件名
        :returns: 原子数，盒子大小，原子坐标，原子质量, 盒子逆矩阵
        '''

        with open(filename, 'r') as f:
            # 读取第一行的原子数
            number = int(f.readline().strip()) 

            # 初始化坐标和质量
            coords = np.zeros((number, 3))    
            mass = np.zeros(number)           

            # 读取box的大小
            box = [float(x) for x in f.readline().split()]
            box = np.array(box)

            # 遍历读取原子坐标和质量
            for i in range(number):
                line = f.readline().split()
                coords[i] = [float(line[1]), float(line[2]), float(line[3])]
                mass[i] = float(line[4])

        return number, box, coords, mass
    
    def getKineticEnergy(self) -> float:
        '''
        返回体系的动能：K = 0.5 * sum(m_i * v_i^2)

        :returns: 动能
        '''
        return 0.5 * np.sum(np.sum(self.velocities**2, axis=1) * self.mass)
    
    def initializeVelocities(self, temperature: float) -> None:
        '''
        初始化给定温度下的原子速度

        :param temperature: 目标温度
        :returns: None
        '''

        # 计算总质量
        totalMass = np.sum(self.mass)

        # 初始化速度，速度大小在-1到1之间均匀分布
        self.velocities = np.random.uniform(-1, 1, (self.number, 3))

        # 计算质心速度
        centerOfMassVelocity = np.sum(self.velocities * self.mass[:, np.newaxis], axis=0) / totalMass
        
        # 去除质心速度，防止整体运动
        self.velocities -= centerOfMassVelocity

        # 调整速度，使得温度符合目标温度
        self.scaleVelocities(temperature)
    
    def scaleVelocities(self, temperature: float) -> None:
        '''
        调整速度，使得温度符合目标温度

        :param temperature: 目标温度
        :returns: None
        '''

        # 计算当前动能，并计算当前温度
        self.ke = self.getKineticEnergy()
        currentTemperature = 2.0 * self.ke / (3.0 * Units.k_B * self.number)

        # 计算调整因子
        scalingFactor = np.sqrt(temperature / currentTemperature)

        # 调整速度
        self.velocities *= scalingFactor
    def applyPbc(self) -> None:
        '''
        应用周期性边界条件

        :returns: None
        '''
        self.coords %= self.box

    def getForce(self, lj: LJParameters) -> None:
        self.pe =0.0
        self.forces = np.zeros((self.number, 3))

        for i in range(self.number-1):
            neighbors = self.NeighborList[i, :self.NeighborNumber[i]]
            neighbors = neighbors[neighbors > i]
            rij = self.coords[neighbors] - self.coords[i]
            rij = rij - self.box * np.where(rij > self.halfBox, 1, 0) + self.box * np.where(rij < -self.halfBox, 1, 0)

            r2 = np.sum(rij**2, axis=1)
            mask = r2 < lj.cutoffSquare
            rij = rij[mask]
            r2 = r2[mask]

            r2i = 1.0 / r2
            r6i = r2i**3
            r8i = r6i * r2i
            r12i = r6i**2
            r14i = r12i * r2i

            f_ij = lj.e24s6 * r8i - lj.e48s12*r14i

            self.pe += np.sum(lj.e4s12*r12i - lj.e4s6*r6i)
            force = f_ij[:, np.newaxis] * rij

            np.add.at(self.forces, [i], np.sum(force, axis=0))
            np.subtract.at(self.forces, neighbors[mask], force)

    def findNeighbor(self):

        # reset neighbor list
        self.NeighborList = np.zeros((self.number, self.MaxNeighbor), dtype=int)
        self.NeighborNumber = np.zeros(self.number, dtype=int)

        # calculate cutoff square
        cutoffSquare = self.cutoffNeighbor**2

        index = np.triu_indices(self.number, 1)
        rij = self.coords[index[1]] - self.coords[index[0]]
        rij = rij - self.box * np.where(rij > self.halfBox, 1, 0) + self.box * np.where(rij < -self.halfBox, 1, 0)

        r2 = np.sum(rij**2, axis=1)
        mask = r2 < cutoffSquare
        pairs = np.column_stack((index[0][mask], index[1][mask]))

        # build neighbor list
        for i, j in pairs:
            self.NeighborList[i, self.NeighborNumber[i]] = j
            self.NeighborNumber[i] += 1
            self.NeighborList[j, self.NeighborNumber[j]] = i
            self.NeighborNumber[j] += 1
        
        if np.any(self.NeighborNumber > self.MaxNeighbor):
            raise ValueError(f'Error: number of neighbors exceeds the maximum value {self.MaxNeighbor}')
                
    def update(self, isStepOne: bool , dt: float) -> None:
        '''
        基于Verlet算法更新原子的坐标和速度

        :param isStepOne: 是否是第一步
        :param dt: 时间步长
        :returns: None
        '''

        # 如果是第一步，就只更新速度的一半
        half_dt = 0.5 * dt
        self.velocities += half_dt * self.forces / self.mass[:, np.newaxis]

        # 完全更新坐标
        if isStepOne:
            self.coords += dt * self.velocities
            self.coord_msd += self.velocities * dt

def main():

    # constants
    Ne, Np = 100000, 100000
    Ns = 10
    Nd = Np / Ns
    Nc = Nd/ 10

    time_step = 10.0 / Units.TIME_UNIT_CONVERSION
    T_0 = 1.475 * 119.8

    r_neighbor = 15.0
    neighbor_update_freq = 10
    r_lj = 10.0

    # initialize atom and LJ
    lj = LJParameters(cutoff=r_lj)
    atom = Atom('1.xyz',cutoffNeighbor=r_neighbor,MaxNeighbor=100)

    # initialize velocities
    atom.initializeVelocities(T_0)

    # equilibration
    t_start = time()
    atom.findNeighbor()
    atom.getForce(lj)
    for i in tqdm(range(Ne)):
        if i % neighbor_update_freq == 0:
            atom.findNeighbor()
        atom.update(True, time_step)
        atom.getForce(lj)
        atom.update(False, time_step)
        atom.scaleVelocities(T_0)
        atom.applyPbc()
    t_end = time()
    print('Equilibration time: %.3f s' % (t_end - t_start))
    
    # production
    coord_all = np.zeros((int(Nd), atom.number, 3))
    vel_all = np.zeros((int(Nd), atom.number, 3))
    t_start = time()
    for i in tqdm(range(Np)):
        if i % neighbor_update_freq == 0:
            atom.findNeighbor()
        atom.update(True, time_step)
        atom.getForce(lj)
        atom.update(False, time_step)
        atom.applyPbc()

        if i % Ns == 0:
            step = i // Ns
            coord_all[step] = atom.coord_msd
            vel_all[step] = atom.velocities
    t_end = time()
    print('Production time: %.3f s' % (t_end - t_start))

    # save data
    np.save('coord.npy', coord_all)
    np.save('vel.npy', vel_all)

main()