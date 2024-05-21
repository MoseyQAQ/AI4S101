import numpy as np
import time

def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
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
        self.number, self.box, self.coords, self.mass = self.parseXyzFile(self.filename)

        # 初始化原子的力、速度、势能、动能
        self.forces = np.zeros((self.number, 3))
        self.velocities = np.zeros((self.number, 3))
        self.pe = 0.0
        self.ke = self.getKineticEnergy()
    
    def parseXyzFile(self, filename: str) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        '''
        读取xyz文件，返回原子数，盒子大小，原子坐标，原子质量

        :param filename: xyz文件名
        :returns: 原子数，盒子大小，原子坐标，原子质量
        '''

        with open(filename, 'r') as f:
            # 读取第一行的原子数
            number = int(f.readline().strip()) 

            # 初始化坐标和质量
            coords = np.zeros((number, 3))    
            mass = np.zeros(number)           

            # 读取box的大小
            abc = [float(x) for x in f.readline().split()]   
            box = [abc[0], abc[1], abc[2], abc[0]/2, abc[1]/2, abc[2]/2] 
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

    def applyMic(self, rij: np.ndarray) -> None:
        '''
        对于给定的两个原子间的距离，应用最小镜像约定

        :param rij: 两个原子间的距离
        :returns: None
        '''

        # 对于每一个维度，如果距离大于盒子的一半，就减去盒子的大小，如果距离小于盒子的一半，就加上盒子的大小
        for i in range(3):
            if rij[i] > self.box[i+3]:
                rij[i] -= self.box[i]
            elif rij[i] < -self.box[i+3]:
                rij[i] += self.box[i]
    @timer
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
        对原子坐标应用周期性边界条件

        :returns: None
        '''

        # 对于每一个维度，如果坐标小于0，就加上盒子大小，如果坐标大于盒子大小，就减去盒子大小
        for i in range(3):
            self.coords[:,i] = np.where(self.coords[:,i] < 0, self.coords[:,i] + self.box[i], self.coords[:,i])
            self.coords[:,i] = np.where(self.coords[:,i] > self.box[i], self.coords[:,i] - self.box[i], self.coords[:,i])

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

def readRun(filename: str='run.in') -> tuple[float, float, int]:
    '''
    读取run文件，返回速度，时间步长，总步数

    :param filename: run文件名
    :returns: 速度（即温度），时间步长，总步数
    '''
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('velocity'):
                velocity = float(line.split()[1])
            if line.startswith('time_step'):
                time_step = float(line.split()[1]) / Units.TIME_UNIT_CONVERSION
            if line.startswith('run'):
                run = int(line.split()[1])
    return velocity, time_step, run

def main():
    timer_start = time.time()
    
    # 读取run文件
    velocity, time_step, run = readRun()

    # 初始化原子和LJ势参数
    atom = Atom()
    lj = LJParameters()

    # 输出热力学量的频率
    thermo_file = 'thermo.out'
    f = open(thermo_file, 'w')
    thermo_freq = 1

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
            #print(f'{i:04d} {temp:16.16f} {atom.pe:16.16f} {ke:16.16f} {atom.pe + ke:16.16f}')
            #f.write(f'{i:04d} {temp:16.16f} {atom.pe:16.16f} {ke:16.16f} {atom.pe + ke:16.16f}\n')
            print(f"{temp:16.16f} {ke:16.16f} {atom.pe:16.16f}")
            f.write(f"{temp:16.16f} {ke:16.16f} {atom.pe:16.16f}\n")
    
    f.close()
    timer_end = time.time()
    print(f"Total time: {timer_end - timer_start:.1f} s")
        

if __name__ == '__main__':
    main()