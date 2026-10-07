import time
import numpy as np
import simplex

"""
Algoritmo primal-dual
"""


class Primal:

    def __init__(self, A, b, Z, verbose=False):
        self.__A = A
        self.__b = b
        self.__Z = Z
        self.__verbose = verbose

        self.__dualA = None
        self.__dualb = None
        self.__dualZ = None

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

        self.__dualA = np.transpose(self.__A)
        self.__dualb = self.__Z.copy()
        self.__dualZ = self.__b

        minValue = np.min(self.__Z)
        self.__y = [minValue] * self.__rows

    def __DefActiveSet(self):
        oldJ = self.__J
        self.__J = []

        for j in range(self.__columns):
            cost = self.__Z[j] - np.array(self.__y) @ self.__A[:, j]
            if cost <= 1e-4:
                self.__J.append(j)

        return not oldJ == self.__J

    def __BuildRSP(self):
        self.__rspColumns = len(self.__J) + self.__rows
        self.__rspRows = self.__rows

        self.__rspZ = [0] * len(self.__J) + [1] * self.__rows
        self.__rspb = self.__b
        self.__rspA = []

        for i in range(self.__rspRows):
            lineI = self.__A[i, self.__J]
            lineI = np.concatenate([lineI, np.zeros(self.__rows)])
            lineI[i + len(self.__J)] = 1
            self.__rspA.append(lineI)

    def __FindMultiplier(self, simplexRSP):
        minMultiplier = np.inf

        for j in range(self.__columns):
            if j in self.__J:
                continue

            Aj = self.__A[:, j]
            mulNum = self.__Z[j] - self.__y @ Aj
            mulDen = simplexRSP.dualSol[: self.__rows] @ Aj

            if mulNum < 1e-4:
                continue

            if mulDen > 1e-4 and mulNum / mulDen < minMultiplier:
                minMultiplier = mulNum / mulDen

        return minMultiplier

    def Solver(self):
        self.__Setup()

        iterations = 1
        timeBegin = time.perf_counter()

        if self.__verbose:
            print(
                f"  [Primal-Dual] Inicializado: {self.__rows} restrições, {self.__columns} variáveis."
            )

        while True:
            newJ = self.__DefActiveSet()

            if newJ == False:
                self.__status = "O problema tem solucao ilimitada"
                if self.__verbose:
                    print(
                        f"  [Primal-Dual] Conjunto ativo inalterado. Problema ilimitado."
                    )
                return None, None

            if self.__verbose:
                print(
                    f"\n  [Primal-Dual It. {iterations}] Conjunto ativo |J| = {len(self.__J)} variáveis."
                )

            self.__BuildRSP()

            if self.__verbose:
                print(
                    f"    -> Resolvendo RSP ({self.__rspRows} linhas x {self.__rspColumns} colunas)..."
                )

            simplexRSP = simplex.Simplex(
                self.__rspA, self.__rspb, self.__rspZ, verbose=self.__verbose
            )
            optimal, zOptimal = simplexRSP.Solver()

            if optimal is None:
                self.__status = "RSP ilimitado"
                return None, None

            multiplier = self.__FindMultiplier(simplexRSP)

            if self.__verbose:
                print(
                    f"    -> RSP finalizado (FO = {zOptimal:.4f}). Multiplicador θ = {multiplier:.4e}"
                )

            if multiplier != np.inf:
                self.__y = self.__y + multiplier * np.array(
                    simplexRSP.dualSol[: self.__rows]
                )

            optimalFound = True
            artificiais_soma = 0.0
            for i in range(len(self.__J), self.__rspColumns):
                artificiais_soma += abs(optimal[i])
                if optimal[i] > 1e-5 or optimal[i] < -1e-4:
                    optimalFound = False

            if self.__verbose:
                print(
                    f"    -> Soma das variáveis artificiais: {artificiais_soma:.6f} "
                    f"({'= 0 -> ÓTIMO ATINGIDO!' if optimalFound else '> 0 -> Próxima iteração'})"
                )

            if optimalFound:
                xb = [0] * self.__columns
                for j in range(len(self.__J)):
                    xb[self.__J[j]] = optimal[j]
                optimalZ = self.__Z @ xb

                self.__status = (
                    f"Solucao otima encontrada em {iterations} iteracoes"
                )
                self.__executionTime = time.perf_counter() - timeBegin

                if self.__verbose:
                    print(
                        f"  [Primal-Dual] Concluído em {iterations} iterações. Tempo: {self.__executionTime:.4f}s"
                    )

                return np.array(xb), optimalZ

            iterations += 1