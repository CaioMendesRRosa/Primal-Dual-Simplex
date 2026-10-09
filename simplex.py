import numpy as np

"""
Algoritmo simplex que utiliza o tableau
"""

EPSILON = 1e-7

class Simplex:

    def __init__(self, A, b, Z, verbose=False, initialBvars=None):
        self.__A = np.array(A, dtype=float)
        self.__b = np.array(b, dtype=float)
        self.__Z = np.array(Z, dtype=float)
        self.__verbose = verbose
        self.__initialBvars = initialBvars

        self.__Bvars = []
        self.__dualSol = None
        self.__tableau = None

        self.__rows, self.__columns = self.__A.shape
        self.__status = ""

    @property
    def status(self):
        return self.__status

    @property
    def dualSol(self):
        return self.__dualSol

    @property
    def Bvars(self):
        return self.__Bvars

    def __Setup(self):
        if self.__initialBvars != None:
            # Warm Start
            B = self.__A[:, self.__initialBvars]
            self.__Bvars = self.__initialBvars
            self.__tableau = np.linalg.solve(B, np.hstack((self.__A, self.__b.reshape(-1, 1))))
            return
        
        self.__Bvars = [ self.__columns - self.__rows + i for i in range(self.__rows)  ]
        self.__tableau = np.hstack(([self.__A, self.__b.reshape(-1, 1)]))

    def __FindDualSol(self):
        # Encontrada a solucao dual depois de terminar o algoritmo
        B = self.__A[:, self.__Bvars]
        Cb = self.__Z[self.__Bvars]
        self.__dualSol = np.linalg.solve(B.T, Cb)

    def __FindPivotColumn(self, c):
        # Encotrando variavel que entra
        pivotColumn = np.argmin(c)
        optimalFound = False
        if c[pivotColumn] >= -EPSILON:
            self.__FindDualSol()
            optimalFound = True
        return optimalFound, pivotColumn

    def __FindPivotRow(self, pivotColumn):
        # Encontrando variavel que sai
        bestPivotValue = np.inf
        pivotRow = -1
        for i in range(self.__rows):
            if self.__tableau[i][pivotColumn] <= EPSILON:
                continue

            minTest = (self.__tableau[i][self.__columns] / self.__tableau[i][pivotColumn])
            if minTest < bestPivotValue:
                bestPivotValue = minTest
                pivotRow = i
        return pivotRow

    def __Pivoting(self, pivotColumn, pivotRow):
        # Atualizando o tableau
        mul = self.__tableau[pivotRow, pivotColumn]
        self.__tableau[pivotRow] /= mul

        for i in range(self.__rows):
            if i != pivotRow:
                self.__tableau[i] -= (self.__tableau[i][pivotColumn] * self.__tableau[pivotRow])

    def Solver(self):
        self.__Setup()
        iterations = 0

        while True:
            Cb = self.__Z[self.__Bvars]
            
            # Custo reduzido das colunas
            c = self.__Z - Cb @ self.__tableau[:, : self.__columns]

            optimal, pivotColumn = self.__FindPivotColumn(c)

            if optimal:
                self.__status = ( f"Solucao otima encontrada em {iterations} iteracoes")
                
                optimalValues = np.zeros(self.__columns)
                optimalValues[self.__Bvars] = self.__tableau[:, self.__columns]
                zOptimal = sum(self.__Z * optimalValues)
                
                if self.__verbose:
                    print(f"      [Simplex RSP] -> Ótimo encontrado em {iterations} iterações. FO = {zOptimal:.4f}")
                    
                return optimalValues, zOptimal

            pivotRow = self.__FindPivotRow(pivotColumn)
            if pivotRow == -1:
                self.__status = "O problema tem solucao ilimitada"
                if self.__verbose:
                    print(f"      [Simplex RSP] -> Ilimitado detectado na iteração {iterations}.")
                    
                return None, None

            self.__Pivoting(pivotColumn, pivotRow)
            self.__Bvars[pivotRow] = pivotColumn
            iterations += 1