import numpy as np
from scipy import optimize

'''
Algoritmo simplex que utiliza o tableau

PROVAVELMENTE ERRADO NO MOMENTO
'''


class Simplex:

    def __init__ (self, A, b, Z):
        self.__A = np.array(A, dtype=float)
        self.__b = np.array(b, dtype=float)
        self.__Z = np.array(Z, dtype=float)

        self.__Bvars = []  # variáveis da base
        self.__dualSol = None
        self.__tableau = None

        self.__rows, self.__columns = self.__A.shape
        self.__status = ""

    @property
    def status (self):
        return self.__status

    @property
    def dualSol (self):
        return self.__dualSol

    @property
    def Bvars (self):
        return self.__Bvars


    def __Setup (self):
        # Inicializando a base
        self.__Bvars = [self.__columns - self.__rows + i for i in range(self.__rows)]
        
        # Adiciona as restricoes no tableau e o b no lado direito
        self.__tableau = np.hstack(([self.__A, self.__b.reshape(-1, 1)]))


    def __FindDualSol(self, c):
        B = self.__A[:, self.__Bvars]
        Cb = self.__Z[self.__Bvars]
        Binv = np.linalg.inv(B)
        self.__dualSol = Cb @ Binv
        return


    def __FindPivotColumn(self, c):
        # Encontra coluna com menor custo
        pivotColumn = np.argmin(c)
        
        optimalFound = False
        # Verifica se atingiu o estado otimo
        if c[pivotColumn] >= -1e-6:
            self.__FindDualSol(c)
            optimalFound = True

        return optimalFound, pivotColumn


    def __FindPivotRow(self, pivotColumn):
        # Teste da razao minima
        bestPivotValue = np.inf
        pivotRow = -1
        for i in range (self.__rows):

            if self.__tableau[i][pivotColumn] <= 1e-6:
                continue
            
            minTest = self.__tableau[i][self.__columns] / self.__tableau[i][pivotColumn]
            if (minTest < bestPivotValue):
                bestPivotValue = minTest
                pivotRow = i

        return pivotRow


    def __Pivoting(self, pivotColumn, pivotRow):
        # Dividindo a linha da variavel que sai pelo pivo
        mul = self.__tableau[pivotRow, pivotColumn]
        self.__tableau[pivotRow] /= mul

        # Atualizada as outras linhas
        for i in range(self.__rows):
            if i != pivotRow:
                self.__tableau[i] -= self.__tableau[i][pivotColumn] * self.__tableau[pivotRow]

        return


    def Solver(self):
        self.__Setup()
        
        iterations = 0

        while True:

            # Calculando custo
            Cb = self.__Z[self.__Bvars] # coeficientes das variaveis basicas
            c = self.__Z - Cb @ self.__tableau[:, :self.__columns] # multiplicando os coeficientes por todas as colunas do tableau tirando a ultima

            # Encontrando variavel que entra
            optimal, pivotColumn = self.__FindPivotColumn(c)

            if optimal:
                self.__status = f"Optimal Found in {iterations} iterations"

                # Encontra os valore otimos (coluna mais a direita)
                optimalValues = np.zeros(self.__columns)
                optimalValues[self.__Bvars] = self.__tableau[:, self.__columns]

                zOptimal = sum(self.__Z * optimalValues)

                return optimalValues, zOptimal

            # Encontrando variavel que sai
            pivotRow = self.__FindPivotRow(pivotColumn)

            # Verificando se o problema tem solucao infinita
            if pivotRow == -1:
                self.__status = "Problem has infinite solution"
                return None, None

            # Redefinindo o tableau
            self.__Pivoting(pivotColumn, pivotRow)

            # Atualiza a base
            self.__Bvars[pivotRow] = pivotColumn

            iterations += 1

            if iterations % 3000 == 0:
                full_tableau = np.vstack((self.__tableau, np.append(c, 0)))
                print(self.__Bvars)
                print("-----TABLEAU-----")
                print(full_tableau)
                print("------------------\n")

                


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