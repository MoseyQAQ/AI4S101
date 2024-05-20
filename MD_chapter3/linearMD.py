import numpy as np
from time import time

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
    def __init__(self, filename: str='xyz.in') -> None:
        '''
        读取xyz文件，初始化原子的坐标，质量，速度，势能，动能

        :param filename: xyz文件名
        :returns: None
        '''

        self.filename = filename
        
        # 读取xyz文件
        self.number, self.box, self.coords, self.mass, self.boxInv = self.parseXyzFile(self.filename)

        # 初始化原子的力、速度、势能、动能
        self.forces = np.zeros((self.number, 3))
        self.velocities = np.zeros((self.number, 3))
        self.pe = 0.0
        self.ke = self.getKineticEnergy()

        # Neighbor list parameters
        self.MaxNeighbor: int = 1000
        self.cutoffNeighbor: float = 10.0
        self.NeighborNumber: np.ndarray = np.zeros(self.number, dtype=int)
        self.NeighborList: np.ndarray = np.zeros((self.number, self.MaxNeighbor), dtype=int)
        
    
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
            box = np.array(box).reshape((3,3))

            # 遍历读取原子坐标和质量
            for i in range(number):
                line = f.readline().split()
                coords[i] = [float(line[1]), float(line[2]), float(line[3])]
                mass[i] = float(line[4])

            # 求box的逆矩阵
            boxInv = np.linalg.inv(box)

        return number, box, coords, mass, boxInv
    
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
    
    def applyMic(self, rij: np.ndarray) -> np.ndarray:
        '''
        对于给定的两个原子间的距离，应用最小镜像约定

        :param rij: 两个原子间的距离
        :returns: None
        '''

        # rij转换为分数坐标
        rijFractional = np.dot(rij, self.boxInv)

        # 对于每一个维度，如果分数坐标小于-0.5，就加1，如果分数坐标大于0.5，就减1
        for i in range(3):
            rijFractional[i] -= np.floor(rijFractional[i] + 0.5)
        
        # 转换为笛卡尔坐标
        rij = np.dot(rijFractional, self.box)
        
        return rij

    def getForce(self, lj: LJParameters) -> None:
        '''
        计算原子间的力和势能

        :param lj: LJ势参数
        :returns: None
        '''

        # 初始化势能和力
        self.pe = 0.0
        self.forces = np.zeros((self.number, 3))

        # 遍历所有原子对
        for i in range(self.number-1):
            for j in range(i+1, self.number):

                # 计算两个原子间的距离,并应用最小镜像约定
                rij = self.coords[j] - self.coords[i]
                self.applyMic(rij)
                r2 = rij[0]**2 + rij[1]**2 + rij[2]**2

                # 如果距离大于截断距离，就跳过
                if r2 > lj.cutoffSquare:
                    continue
                
                # 计算一些常量
                r2_inv = 1.0 / r2
                r4_inv = r2_inv * r2_inv
                r6_inv = r2_inv * r4_inv
                r8_inv = r4_inv * r4_inv
                r12_inv = r4_inv * r8_inv
                r14_inv = r6_inv * r8_inv
                force_ij = lj.e24s6 * r8_inv - lj.e48s12 * r14_inv

                # 更新势能和力
                self.pe += lj.e4s12 * r12_inv - lj.e4s6 * r6_inv
                self.forces[i] += force_ij * rij
                self.forces[j] -= force_ij * rij

    def applyPbc(self) -> None:
        '''
        对原子坐标应用周期性边界条件，支持三斜盒子

        :returns: None
        '''

        # 转换为分数坐标
        coordsFractional = np.dot(self.coords, self.boxInv)

        # 对于每一个维度，如果分数坐标小于0，就加1，如果分数坐标大于1，就减1
        for i in range(3):
            coordsFractional[:, i] -= np.floor(coordsFractional[:, i])
        
        # 转换为笛卡尔坐标
        self.coords = np.dot(coordsFractional, self.box)
    
    @timer
    def findNeighborON2(self):
        cutoffSquare = self.cutoffNeighbor**2

        for i in range(self.number-1):
            for j in range(i+1, self.number):
                rij = self.coords[j] - self.coords[i]
                rij = self.applyMic(rij)
                r2 = np.sum(rij**2)
                if r2 < cutoffSquare:
                    self.NeighborList[i, self.NeighborNumber[i]] = j
                    self.NeighborNumber[i] += 1

                    if self.NeighborNumber[i] >= self.MaxNeighbor:
                        raise ValueError(f'Error: number of neighbors for atom {i} exceeds the maximum value {self.MaxNeighbor}')
    
    def findNeighborON1(self):
        raise NotImplementedError

    def update(self, isStepOne: bool , dt: float) -> None:
        '''
        基于Verlet算法更新原子的坐标和速度

        :param isStepOne: 是否是第一步
        :param dt: 时间步长
        :returns: None
        '''

        # 如果是第一步，就只更新速度的一半
        halfdt = 0.5 * dt
        self.velocities += halfdt * self.forces / self.mass[:, np.newaxis]

        # 完全更新坐标
        if isStepOne:
            self.coords += dt * self.velocities

def readRun(filename: str='run.in') -> tuple[float, float, int, int]:
    '''
    读取run文件，返回速度，时间步长，总步数

    :param filename: run文件名
    :returns: 速度（即温度），时间步长，总步数, neighbor_flag
    '''
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('velocity'):
                velocity = float(line.split()[1])
            if line.startswith('time_step'):
                time_step = float(line.split()[1])
            if line.startswith('run'):
                run = int(line.split()[1])
            if line.startswith('neighbor_flag'):
                neighbor_flag = int(line.split()[1])
    return velocity, time_step, run, neighbor_flag

def main():
    timer_start = time.time()
    
    # 读取run文件
    velocity, time_step, run, neighbor_flag = readRun()

    # 初始化原子和LJ势参数
    atom = Atom()
    lj = LJParameters()

    # 输出热力学量的频率
    thermo_file = 'thermo.out'
    f = open(thermo_file, 'w')
    thermo_freq = 100

    # 初始化速度
    atom.initializeVelocities(velocity)

    # 开始模拟
    for i in range(run):
        atom.applyPbc()
        atom.update(True, time_step)
        atom.getForce(lj)
        atom.update(False, time_step)

        # 输出热力学量
        if i % thermo_freq == 0:
            ke = atom.getKineticEnergy()
            temp = 2.0 * ke / (3.0 * Units.k_B * atom.number)
            print(f"{temp:16.16f} {ke:16.16f} {atom.pe:16.16f}")
            f.write(f"{temp:16.16f} {ke:16.16f} {atom.pe:16.16f}\n")
    
    f.close()
    timer_end = time.time()
    print(f"Total time: {timer_end - timer_start:.1f} s")
        

if __name__ == '__main__':
    #main()
    atom = Atom(filename='xyz.in')
    velocity, time_step, run, neighbor_flag = readRun('run.in')
    print(atom.number)
    print(atom.box)
    print(atom.boxInv)