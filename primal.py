import time
import numpy as np
import simplex

"""
Algoritmo primal-dual
"""

EPSILON = 1e-7

class Primal:

    def __init__(self, A, b, Z, verbose=False):
        self.__A = A
        self.__b = b
        self.__Z = Z
        self.__verbose = verbose
        self.__preBvars = None

        self.__rspA = None
        self.__rspb = None
        self.__rspZ = None

        self.__rspColumns = None
        self.__rspRows = None

        self.__J = []
        self.__y = []

        self.__rows = len(self.__A)
        self.__columns = len(self.__A[0])

        self.__status = ""
        self.__executionTime = 0
        
        self.__artificialVarsSum = None
        
        self.__xb = None
        self.__optimalZ = None

    @property
    def status(self):
        return self.__status

    @property
    def executionTime(self):
        return self.__executionTime

    def __Setup(self):
        self.__A = np.array(self.__A, dtype=float)
        self.__b = np.array(self.__b, dtype=float)
        self.__Z = np.array(self.__Z, dtype=float)

        minValue = np.min(self.__Z)
        self.__y = [minValue] * self.__rows

    def __DefActiveSet(self):
        oldJ = self.__J
        self.__J = []

        for j in range(self.__columns):
            cost = self.__Z[j] - np.array(self.__y) @ self.__A[:, j]
            if cost <= EPSILON:
                self.__J.append(j)

        return not oldJ == self.__J

    def __BuildRSP(self):
        m = self.__rows
        nJ = len(self.__J)
        sign = np.where(self.__b > 0, 1.0, -1.0)

        self.__rspColumns = nJ + m
        self.__rspZ = np.concatenate([np.zeros(nJ), np.ones(m)])
        self.__rspb = self.__b.copy()
        self.__rspA = np.hstack([self.__A[:, self.__J], np.diag(sign)])
        self.__rspInitialBvars = [nJ + i for i in range(m)]

    def __FindMultiplier(self, simplexRSP):
        minMultiplier = np.inf

        for j in range(self.__columns):
            if j in self.__J:
                continue

            Aj = self.__A[:, j]
            mulNum = self.__Z[j] - self.__y @ Aj
            mulDen = simplexRSP.dualSol[: self.__rows] @ Aj

            if mulNum < EPSILON or mulDen < EPSILON:
                continue

            if mulNum / mulDen < minMultiplier:
                minMultiplier = mulNum / mulDen

        return minMultiplier


    def __CheckOptimality (self, optimal):
        
        self.__artificialVarsSum = 0.0
        for i in range(len(self.__J), self.__rspColumns):
            self.__artificialVarsSum += abs(optimal[i])
        optimalFound = self.__artificialVarsSum <= EPSILON

        if optimalFound:
            self.__xb = [0] * self.__columns
            for j in range(len(self.__J)):
                self.__xb[self.__J[j]] = optimal[j]
            self.__optimalZ = self.__Z @ self.__xb

            return True
        
        return False


    def __SaveBvars(self, simplexRSP):
        lenJ = len(self.__J)
        # Guardando as variaveis da base anterior e classificando entre f e a
        # a: artificial do RSP
        # f: fluxo do problema original
        self.__preBvars = [("f", self.__J[i]) if i < lenJ else ("a", i - lenJ) for i in simplexRSP.Bvars]

    def __GetPrevBvars(self):
        # Pegando a base anterior, levando em consideracao a mudanca da variavel self.__J atual
        if self.__preBvars is None:
            return None
        posIndex = {j: p for p, j in enumerate(self.__J)}
        lenJ = len(self.__J)
        bVars = []
        for varType, idx in self.__preBvars:
            if varType == "f":
                # Fluxo
                if idx not in posIndex:
                    return None
                bVars.append(posIndex[idx])
            else:
                # Artificial do RSP
                bVars.append(lenJ + idx)
        return bVars


    def Solver(self):
        self.__Setup()

        iterations = 1
        timeBegin = time.perf_counter()

        if self.__verbose:
            print(f"  [Primal-Dual] Inicializado: {self.__rows} restrições, {self.__columns} variáveis.")

        while True:
            
            self.__DefActiveSet()
            self.__BuildRSP()

            bVars = self.__GetPrevBvars()
            if bVars is None:
                bVars = self.__rspInitialBvars
                
            simplexRSP = simplex.Simplex(self.__rspA, self.__rspb, self.__rspZ,
                                         verbose=self.__verbose, initialBvars=bVars)
            optimal, zOptimal = simplexRSP.Solver()

            if optimal is None:
                self.__status = "RSP ilimitado"
                return None, None

            optimalFound = self.__CheckOptimality(optimal)

            if optimalFound:
                self.__status = f"Solucao otima encontrada em {iterations} iteracoes"
                self.__executionTime = time.perf_counter() - timeBegin
                return np.array(self.__xb), self.__optimalZ

            multiplier = self.__FindMultiplier(simplexRSP)

            if multiplier == np.inf:
                self.__status = "O problema e inviavel (dual ilimitado)"
                return None, None

            self.__y += multiplier * np.array(simplexRSP.dualSol[: self.__rows])

            if self.__verbose:
                print(f"\n  [Primal-Dual It. {iterations}] |J| = {len(self.__J)}")
                print(f"    -> RSP: FO = {zOptimal:.4f}, θ = {multiplier:.4e}, artificiais = {self.__artificialVarsSum:.6f}")

            self.__SaveBvars(simplexRSP)
            iterations += 1