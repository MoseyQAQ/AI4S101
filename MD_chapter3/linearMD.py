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
    def __init__(self, filename: str='xyz.in',
                 cutoffNeighbor: float=10.0, MaxNeighbor: int=1000,
                 neighborFlag: int=0) -> None:
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
        self.MaxNeighbor: int = MaxNeighbor
        self.cutoffNeighbor: float = cutoffNeighbor
        self.NeighborNumber: np.ndarray = np.zeros(self.number, dtype=int)
        self.NeighborList: np.ndarray = np.zeros((self.number, self.MaxNeighbor), dtype=int)
        self.NeighborFlag: int = neighborFlag
        self.numUpdates: int = 0
        self.coord_ref: np.ndarray = np.zeros((self.number, 3))
        
    
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
            if rijFractional[i] < -0.5:
                rijFractional[i] += 1.0
            elif rijFractional[i] > 0.5:
                rijFractional[i] -= 1.0
        
        # 转换为笛卡尔坐标
        rij = np.dot(rijFractional, self.box)
        
        return rij
    
    def getForce(self, lj: LJParameters) -> None:
        self.pe =0.0
        self.forces = np.zeros((self.number, 3))

        if self.NeighborFlag == 0:
            index = np.triu_indices(self.number, 1)

            rij = self.coords[index[1]] - self.coords[index[0]]
            rij_frac = np.dot(rij, self.boxInv)
            rij_frac = np.where(rij_frac < -0.5, rij_frac + 1.0, rij_frac)
            rij_frac = np.where(rij_frac > 0.5, rij_frac - 1.0, rij_frac)
            rij = np.dot(rij_frac, self.box)

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

            np.add.at(self.forces, index[0][mask], force)
            np.subtract.at(self.forces, index[1][mask], force)

        else:
            for i in range(self.number-1):
                neighbors = self.NeighborList[i, :self.NeighborNumber[i]]
                rij = self.coords[neighbors] - self.coords[i]
                rij_frac = np.dot(rij, self.boxInv)
                rij_frac = np.where(rij_frac < -0.5, rij_frac + 1.0, rij_frac)
                rij_frac = np.where(rij_frac > 0.5, rij_frac - 1.0, rij_frac)
                rij = np.dot(rij_frac, self.box)

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

        index = np.triu_indices(self.number, 1)
        rij = self.coords[index[1]] - self.coords[index[0]]
        rij_frac = np.dot(rij, self.boxInv)
        rij_frac = np.where(rij_frac < -0.5, rij_frac + 1.0, rij_frac)
        rij_frac = np.where(rij_frac > 0.5, rij_frac - 1.0, rij_frac)
        rij = np.dot(rij_frac, self.box)

        r2 = np.sum(rij**2, axis=1)
        mask = r2 < cutoffSquare
        pairs = np.column_stack((index[0][mask], index[1][mask]))

        # build neighbor list
        for i, j in pairs:
            self.NeighborList[i, self.NeighborNumber[i]] = j
            self.NeighborNumber[i] += 1
        
        if np.any(self.NeighborNumber > self.MaxNeighbor):
            raise ValueError(f'Error: number of neighbors exceeds the maximum value {self.MaxNeighbor}')
    
    def getThickness(self) -> np.ndarray:
        '''
        计算盒子的厚度

        :returns: 盒子的厚度
        '''
        volume = np.abs(np.linalg.det(self.box))
        area1 = np.linalg.norm(np.cross(self.box[0], self.box[1]))
        area2 = np.linalg.norm(np.cross(self.box[1], self.box[2]))
        area3 = np.linalg.norm(np.cross(self.box[2], self.box[0]))
        return volume / np.array([area1, area2, area3])
    

    def getCell(self, thickness: np.ndarray, cutoffInv: float, numCells: np.ndarray) -> np.ndarray:
        '''
        得到每个盒子中原子数目

        :param thickness: 盒子的厚度
        :param cutoffInv: 截断距离的倒数
        :param numCells: 每个方向盒子的个数
        :returns: 每个原子所在的盒子索引
        '''
        coordFractional = np.dot(self.coords, self.boxInv)
        cellIndex = np.floor(coordFractional * thickness * cutoffInv).astype(int)
        cellIndex = np.mod(cellIndex, numCells)
        return cellIndex

    def findNeighborON1(self):
        cutoffInv = 1.0 / self.cutoffNeighbor
        cutoffSquare = self.cutoffNeighbor**2

        # 计算盒子的厚度
        thickness = self.getThickness()

        # 计算每个方向盒子的个数
        numCells = np.floor(thickness * cutoffInv).astype(int)
        totalNumCells = numCells[0] * numCells[1] * numCells[2]

        # 获得每个原子所在的盒子索引
        cellIndex = self.getCell(thickness, cutoffInv, numCells)

        # 计算每个盒子中有哪些原子
        cellAtoms = {tuple(idx): [] for idx in np.ndindex(*numCells)}
        for atom_idx, cell_idx in enumerate(cellIndex):
            cellAtoms[tuple(cell_idx)].append(atom_idx)

        # 计算近邻盒子
        offsets = np.array([[-1,0,1]]*3)
        neighbor_offsets = np.array(np.meshgrid(*offsets)).T.reshape(-1, 3)
        
        # walk through each atom
        for n in range(self.number):
            currentCell = tuple(cellIndex[n])

            neighbor_cells = (currentCell + neighbor_offsets) % numCells
            neighbor_cells = [tuple(cell) for cell in neighbor_cells]

            all_neighbors = np.array([m for cell in neighbor_cells if cell in cellAtoms for m in cellAtoms[cell] if n < m])
            if all_neighbors.size == 0:
                continue

            rij = self.coords[all_neighbors] - self.coords[n]
            rij_frac = np.dot(rij, self.boxInv)
            rij_frac = np.where(rij_frac < -0.5, rij_frac + 1.0, rij_frac)
            rij_frac = np.where(rij_frac > 0.5, rij_frac - 1.0, rij_frac)
            rij = np.dot(rij_frac, self.box)

            r2 = np.sum(rij**2, axis=1)
            valid_neighbors = all_neighbors[r2 < cutoffSquare]

            self.NeighborList[n, :len(valid_neighbors)] = valid_neighbors
            self.NeighborNumber[n] = len(valid_neighbors)

            if len(valid_neighbors) > self.MaxNeighbor:
                raise ValueError(f'Error: number of neighbors exceeds the maximum value {self.MaxNeighbor}')
            
    def checkIfNeedUpdate(self) -> bool:
        '''
        检查是否需要更新NeighborList

        :returns: 是否需要更新
        '''

        diff = self.coords - self.coord_ref
        diff_frac = np.dot(diff, self.boxInv)
        diff_frac = np.where(diff_frac < -0.5, diff_frac + 1.0, diff_frac)
        diff_frac = np.where(diff_frac > 0.5, diff_frac - 1.0, diff_frac)
        diff = np.dot(diff_frac, self.box)
        displacement_squared = np.sum(diff**2, axis=1)

        if np.any(displacement_squared > 0.25):
            print(np.max(displacement_squared))
            self.coord_ref = self.coords.copy()
            return True

        return False
    
    def findNeighbor(self):
        '''
        更新NeighborList
        
        '''
        if self.checkIfNeedUpdate():
            self.numUpdates += 1
            self.applyPbc()

            # 重置NeighborList
            self.NeighborNumber: np.ndarray = np.zeros(self.number, dtype=int)
            self.NeighborList: np.ndarray = np.zeros((self.number, self.MaxNeighbor), dtype=int)
            
            if self.NeighborFlag == 1:
                self.findNeighborON1()
            elif self.NeighborFlag == 2:
                self.findNeighborON2()
                
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
                time_step = float(line.split()[1]) / Units.TIME_UNIT_CONVERSION
            if line.startswith('run'):
                run = int(line.split()[1])
            if line.startswith('neighbor_flag'):
                neighbor_flag = int(line.split()[1])
    return velocity, time_step, run, neighbor_flag

def main():
    timer_start = time()
    
    # 读取run文件
    velocity, time_step, run, neighbor_flag = readRun()

    # 初始化原子和LJ势参数
    atom = Atom(MaxNeighbor=1000, cutoffNeighbor=10.0, neighborFlag=neighbor_flag)
    lj = LJParameters()

    # 输出热力学量的频率
    thermo_file = 'thermo.out'
    f = open(thermo_file, 'w')
    thermo_freq = 100

    # 初始化速度
    atom.initializeVelocities(velocity)

    # 开始模拟
    for i in range(run):
        if atom.NeighborFlag != 0:
            atom.findNeighbor()
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
    timer_end = time()
    print(f"Total time: {timer_end - timer_start:.1f} s")
    print(f'Neighbor update times: {atom.numUpdates}')
        

if __name__ == '__main__':
    main()