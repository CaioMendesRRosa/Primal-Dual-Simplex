import numpy as np
from scipy import optimize

'''
Algoritmo simplex que inverte a matriz toda i
'''

class Simplex:

    def __init__ (self, A, b, Z):

        self.__A = A # Restricoes
        self.__b = b # Lado direito das restricoes 
        self.__Z = Z # Funcao objetivo

        self.__B = [] # base
        self.__Binv = [] # Inversa da Base
        self.__Bvars = [] # variaveis da base
        self.__dualSol = None

        self.__rows = len(self.__A)
        self.__columns = len(self.__A[0])

        self.__status = ""

    @property
    def status (self):
        return self.__status

    @property
    def dualSol (self):
        return self.__dualSol

    def __Setup (self):
        # Inicializando a base
        self.__Bvars = [self.__columns - self.__rows + i for i in range(self.__rows)]
        self.__B = [[] for i in range(self.__rows)]

        columnsNum = len(self.__Bvars)
        for i in range(columnsNum):
            for j in range(self.__rows):
                self.__B[j].append(self.__A[j][self.__Bvars[i]])

        return

    @property
    def Bvars (self):
        return self.__Bvars

    def __FindPivotColumn (self, Cb, Binv):

        # Fazendo Cb * Binv * A[j]
        aux = Cb @ Binv
        Zj = []
        for j in range(self.__columns):
            Aj = [self.__A[i][j] for i in range(self.__rows)]
            Zj.append(aux @ Aj)

        # Calculando os custos
        Cj = [self.__Z[i] - Zj[i] for i in range(self.__columns)]

        # Encontrando o menor custo
        optimal = True
        pivotColumn = -1
        a = []
        for i in range(self.__columns):
            if Cj[i] < 0 and (pivotColumn != -1 or Cj[i] < Cj[pivotColumn]):
                optimal = False
                pivotColumn = i

        if optimal:
            self.__dualSol = aux

        return optimal, pivotColumn

    def __FindPivotRow (self, xb, pivotColumn):

        # Fazendo invB * Aj
        newA = self.__Binv @ [self.__A[i][pivotColumn] for i in range(self.__rows)]

        # Teste da razao minima
        bestPivotValue = np.inf
        pivotRow = -1
        for i in range (self.__rows):

            if newA[i] <= 1e-6:
                continue
            
            minTest = xb[i] / newA[i]
            if (minTest < bestPivotValue):
                bestPivotValue = minTest
                pivotRow = i

        return pivotRow

    def Solver(self):

        self.__Setup()

        iterations = 0

        while True:

            self.__Binv = np.linalg.inv(self.__B)
            Cb = [self.__Z[self.__Bvars[i]] for i in range (self.__rows)]

            # Encontrando variavel que entra
            optimal, pivotColumn = self.__FindPivotColumn(Cb, self.__Binv)

            # Solucao atual
            xb = self.__Binv @ self.__b

            # Verificando se nao ja encontro a solucao otima
            if optimal:
                self.__status = f"Optimal Found in {iterations} iterations"

                optimalValues = []
                for i in range(self.__columns):
                    if i in self.__Bvars:
                        optimalValues.append(xb[self.__Bvars.index(i)])
                    else:
                        optimalValues.append(0)

                zOptimal = sum([self.__Z[i] * optimalValues[i] for i in range(self.__columns)])
                return np.array(optimalValues), zOptimal

            # Encontrando variavel que sai
            pivotRow = self.__FindPivotRow(xb, pivotColumn)

            # Verificando se o problema tem solucao infinita
            if pivotRow == -1:
                self.__status = f"Problem has infinite solution"
                return None
                
            # Substituindo a coluna pivo
            self.__Bvars[pivotRow] = pivotColumn
            for i in range (self.__rows):
                self.__B[i][pivotRow] = self.__A[i][pivotColumn]

            iterations += 1

if __name__ == "__main__":

    A = [
        [2, 2, 1, 0, 0],
        [6, 1, 0, 1, 0],
        [4, 2, 0, 0, 1]
        ]
    b = [5, 7, 7]
    Z = [-30, -40, 0, 0, 0]

    res = optimize.linprog(Z, A_eq=A, b_eq=b, method="highs")

    # Exibindo os resultados
    print("Solucao Scipy")
    print(f"Valor otimo da funcao objetivo (Z): {res.fun}")
    print(f"Solucao otima: {res.x}\n")

    simplex = Simplex(A, b, Z)
    optimal, zOptimal = simplex.Solver()

    print("Solucao Simplex")
    print (simplex.status)
    print(f"Solucao Otima: {optimal}")
    print(f"F* = {zOptimal}")