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

class TersoffParameters:
    def __init__(self, A: float=1830.8, B: float=471.18, lamb: float=2.4799, mu: float=1.7322, beta: float=1.1e-6, 
                 n: float=0.78734, c: float=1.0039e5, d: float=16.217, h: float=-0.59825,
                 R: float=2.7, S: float=3.0) -> None:
        self.A = A
        self.B = B
        self.lamb = lamb
        self.mu = mu
        self.beta = beta
        self.n = n
        self.c = c 
        self.d = d
        self.h = h
        self.R = R
        self.S = S
        self.minus_half_over_n = -0.5 / n
        self.pi = np.pi 
        self.pi_factor = 1.0 / (R-S)
        self.c2 = c**2
        self.d2 = d**2
        self.c2overd2 = self.c2 / self.d2
        pass

class Atom:
    def __init__(self, filename: str='xyz.in',
                 cutoffNeighbor: float=10.0, MaxNeighbor: int=1000,
                 neighborFlag: int=0, tersoff: TersoffParameters=None) -> None:
        '''
        读取xyz文件，初始化原子的坐标，质量，速度，势能，动能

        :param filename: xyz文件名
        :returns: None
        '''

        self.filename = filename
        
        # 读取xyz文件
        self.number, self.box, self.coords, self.boxInv = self.parseXyzFile(self.filename)
        self.mass = np.full(self.number, 28)

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

        # Tersoff potential 
        self.b = np.zeros((self.number, self.MaxNeighbor))
        self.bp = np.zeros((self.number, self.MaxNeighbor))
        if tersoff is not None:
            self.tersoff = tersoff
        
    
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

            # 读取box的大小
            box = [float(x) for x in f.readline().split()]
            box = np.array(box).reshape((3,3))

            # 遍历读取原子坐标和质量
            for i in range(number):
                line = f.readline().split()
                coords[i] = [float(line[1]), float(line[2]), float(line[3])]

            # 求box的逆矩阵
            boxInv = np.linalg.inv(box)

        return number, box, coords, boxInv
    
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
    
    def getForce_lj(self, lj: LJParameters) -> None:
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
        displacement_squared = np.sum(diff**2, axis=1)

        if np.any(displacement_squared > 0.25):
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
            
            self.coord_ref = np.copy(self.coords)
                
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
    def getForce_tersoff(self) -> None:

        # reset forces and potential energy
        self.pe = 0.0
        self.forces = np.zeros((self.number, 3))

        for n1 in range(self.number):
            neighbors = self.NeighborList[n1, :self.NeighborNumber[n1]]
            for idx2, n2 in enumerate(neighbors):
                r12 = self.coords[n2] - self.coords[n1]
                r12_frac = np.dot(r12, self.boxInv)
                r12_frac = np.where(r12_frac < -0.5, r12_frac + 1.0, r12_frac)
                r12_frac = np.where(r12_frac > 0.5, r12_frac - 1.0, r12_frac)
                r12 = np.dot(r12_frac, self.box)
                d12 = np.linalg.norm(r12)
                d12_inv = 1.0 / d12

                fc12, fcp12, fa12, fap12, fr12, frp12 = self.find_all(d12)
                b12 = self.b[n1, idx2]
                bp12 = self.bp[n1, idx2]
                f12 = np.zeros(3)
                factor1 = -b12 * fa12 + fr12 
                factor2 = -b12 * fap12 + frp12
                factor3 = (fcp12 * factor1 + fc12 * factor2) / d12 
                f12 += 0.5 * factor3 * r12

                for idx3, n3 in enumerate(neighbors):
                    if n3 == n2:
                        continue
                    r13 = self.coords[n3] - self.coords[n1]
                    r13_frac = np.dot(r13, self.boxInv)
                    r13_frac = np.where(r13_frac < -0.5, r13_frac + 1.0, r13_frac)
                    r13_frac = np.where(r13_frac > 0.5, r13_frac - 1.0, r13_frac)
                    r13 = np.dot(r13_frac, self.box)
                    d13 = np.linalg.norm(r13)
                    fc13 = self.find_fc(d13)
                    fa13 = self.find_fa(d13)
                    bp13 = self.bp[n1, idx3]

                    cos123 = np.dot(r12, r13) / (d12 * d13)
                    g123, gp123 = self.find_g_and_gp(cos123)
                    
                    cos_x = r13[0]/(d12*d13) - r12[0]*cos123/(d12**2)
                    cos_y = r13[1]/(d12*d13) - r12[1]*cos123/(d12**2)
                    cos_z = r13[2]/(d12*d13) - r12[2]*cos123/(d12**2)
                    factor123a = (-bp12 * fc12 * fa12 * fc13 - bp13 * fc13 * fa13 * fc12) * gp123
                    factor123b = -bp13 * fc13 * fa13 * fcp12 * g123 * d12_inv
                    f12[0] += 0.5 * (r12[0] * factor123b + factor123a * cos_x)
                    f12[1] += 0.5 * (r12[1] * factor123b + factor123a * cos_y)
                    f12[2] += 0.5 * (r12[2] * factor123b + factor123a * cos_z)
                
                self.pe += factor1 * fc12 * 0.5
                self.forces[n1] += f12
                self.forces[n2] -= f12

    def find_b_and_bp(self) -> None:
        for n1 in range(self.number):
            neighbors = self.NeighborList[n1, :self.NeighborNumber[n1]]
            for idx2, n2 in enumerate(neighbors):
                coord_12 = self.coords[n2] - self.coords[n1]
                coord_12_frac = np.dot(coord_12, self.boxInv)
                coord_12_frac = np.where(coord_12_frac < -0.5, coord_12_frac + 1.0, coord_12_frac)
                coord_12_frac = np.where(coord_12_frac > 0.5, coord_12_frac - 1.0, coord_12_frac)
                coord_12 = np.dot(coord_12_frac, self.box)
                d_12 = np.linalg.norm(coord_12)

                zeta = 0.0

                for n3 in neighbors:
                    if n3 == n2: 
                        continue
                    coord_13 = self.coords[n3] - self.coords[n1]
                    coord_13_frac = np.dot(coord_13, self.boxInv)
                    coord_13_frac = np.where(coord_13_frac < -0.5, coord_13_frac + 1.0, coord_13_frac)
                    coord_13_frac = np.where(coord_13_frac > 0.5, coord_13_frac - 1.0, coord_13_frac)
                    coord_13 = np.dot(coord_13_frac, self.box)
                    d_13 = np.linalg.norm(coord_13)

                    cos = np.dot(coord_12, coord_13) / (d_12 * d_13)
                    zeta += self.find_g(cos) * self.find_fc(d_12)
                
                bzn = np.power(self.tersoff.beta * zeta, self.tersoff.n)
                b12 = np.power(1.0 + bzn, self.tersoff.minus_half_over_n)
                self.b[n1, idx2] = b12
                self.bp[n1, idx2] = -b12 * bzn * 0.5 / ((1.0 + bzn) * zeta)
    
    def find_fc(self, d_12: float) -> float:
        if d_12 < self.tersoff.S:
            return 1.0
        elif d_12 < self.tersoff.R:
            return 0.5 * np.cos((self.tersoff.pi_factor * (d_12 - self.tersoff.S))) + 0.5 
        else:
            return 0.0
    def find_fa(self, d12: float) -> float:
        return self.tersoff.B * np.exp(-self.tersoff.mu * d12)
    
    def find_g_and_gp(self, cos: float) -> tuple[float, float]:
        temp = self.tersoff.d2 + (cos - self.tersoff.h)**2
        g = 1.0 + self.tersoff.c2overd2 - self.tersoff.c2 / temp 
        gp = 2.0 * self.tersoff.c2 * (cos - self.tersoff.h) / temp**2
        return g, gp
    
    def find_g(self, cos: float) -> float:
        temp = self.tersoff.d2 + (cos - self.tersoff.h)**2
        g = 1.0 + self.tersoff.c2overd2 - self.tersoff.c2 / temp 
        return g
    def find_all(self, d12: float) -> tuple[float, float, float, float, float, float]:
        fr = self.tersoff.A * np.exp(-self.tersoff.lamb * d12)
        frp = -self.tersoff.lamb * fr
        fa = self.tersoff.B * np.exp(-self.tersoff.mu * d12)
        fap = -self.tersoff.mu * fa

        if d12 < self.tersoff.S:
            fc = 1.0
            fcp = 0.0
        elif d12 < self.tersoff.R:
            fc = 0.5 * np.cos(self.tersoff.pi_factor * (d12 - self.tersoff.S)) + 0.5
            fcp = -0.5 * self.tersoff.pi_factor * np.sin(self.tersoff.pi_factor * (d12 - self.tersoff.S))
        else:
            fc = 0.0
            fcp = 0.0
        return fc, fcp, fa, fap, fr, frp

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
    atom =Atom(filename='model.xyz', cutoffNeighbor=3.1, MaxNeighbor=10, neighborFlag=2, tersoff=TersoffParameters())
    
    atom.findNeighbor()
    atom.getForce_tersoff()
    np.save('a.npy', atom.forces)
    print(np.allclose(atom.coord_ref, atom.coords))
    delta = 2.0e-5
    f = np.zeros((atom.number, 3))
    for n in range(atom.number):
        atom.coords[n, 0] = atom.coord_ref[n, 0] + delta
        atom.getForce_tersoff()
        pePostive = atom.pe
        atom.coords[n, 0] = atom.coord_ref[n, 0] - delta
        atom.getForce_tersoff()
        peNegative = atom.pe
        atom.coords[n, 0] = atom.coord_ref[n, 0]
        fx = (peNegative - pePostive) / (2.0 * delta)

        atom.coords[n, 1] = atom.coord_ref[n, 1] + delta
        atom.getForce_tersoff()
        pePostive = atom.pe
        atom.coords[n, 1] = atom.coord_ref[n, 1] - delta
        atom.getForce_tersoff()
        peNegative = atom.pe
        atom.coords[n, 1] = atom.coord_ref[n, 1]
        fy = (peNegative - pePostive) / (2.0 * delta)

        atom.coords[n, 2] = atom.coord_ref[n, 2] + delta
        atom.getForce_tersoff()
        pePostive = atom.pe
        atom.coords[n, 2] = atom.coord_ref[n, 2] - delta
        atom.getForce_tersoff()
        peNegative = atom.pe
        atom.coords[n, 2] = atom.coord_ref[n, 2]
        fz = (peNegative - pePostive) / (2.0 * delta)

        f[n] = np.array([fx, fy, fz])
    
    np.save('n.npy', f)
        

if __name__ == '__main__':
    main()