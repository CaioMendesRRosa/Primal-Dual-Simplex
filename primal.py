import numpy as np
from scipy import optimize
import simplex

'''
Algoritmo primal-dual
'''

class Primal:

    def __init__ (self, A, b, Z):

        self.__A = A # Restricoes
        self.__b = b # Lado direito das restricoes 
        self.__Z = Z # Funcao objetivo

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

    @property
    def status (self):
        return self.__status

    def __Setup (self):
        # Inicializando o problema Dual

        self.__A = np.asarray(self.__A, dtype=float)
        self.__b = np.asarray(self.__b, dtype=float)
        self.__Z = np.asarray(self.__Z, dtype=float)

        self.__dualA = np.transpose(self.__A)
        self.__dualb = self.__Z.copy()
        self.__dualZ = self.__b

        # Comecando com uma solucao dual factivel
        minValue = np.min(self.__Z)
        self.__y = [minValue] * self.__rows

        # Interferencia para tentar impedir degeneracao
        #for i in range (self.__rows):
        #    self.__b[i] += 1e-2

        return


    def __DefActiveSet (self):
        # Encontrando o conjunto das variaveis ativas

        oldJ = self.__J
        self.__J = []

        for j in range (self.__columns):
            cost = self.__Z[j] - np.array(self.__y) @ self.__A[:, j]
            if cost <= 1e-4:
                self.__J.append(j)

        return not oldJ == self.__J



    def __BuildRSP (self):
        # Construindo o problema subrestrito

        self.__rspColumns = len(self.__J) + self.__rows
        self.__rspRows = self.__rows

        # 0 para variaveis de descisao e 1 para as variaveis de folga
        self.__rspZ = [0] * len(self.__J) + [1] * self.__rows
        self.__rspb = self.__b
        self.__rspA = []

        for i in range (self.__rspRows):
            lineI = self.__A[i, self.__J] # Restricao de A apenas com as variaveis ativas
            lineI = np.concatenate([lineI, np.zeros(self.__rows)]) # Variaveis de folga
            lineI[i + len(self.__J)] = 1 # Variavel de folga ativa
            self.__rspA.append(lineI)

        return
        

    def __FindMultiplier (self, simplexRSP):
        # Encontrando o multiplicador teta

        minMultiplier = np.inf

        for j in range(self.__columns):
            if j in self.__J:
                continue
            
            Aj = self.__A[:, j]
            mulNum = self.__Z[j] - self.__y @ Aj
            mulDen = simplexRSP.dualSol[:self.__rows] @ Aj

            if mulNum < 1e-4:
                continue

            if mulDen > 1e-4 and mulNum / mulDen < minMultiplier:
                minMultiplier = mulNum / mulDen

        return minMultiplier


    def Solver(self):
        
        self.__Setup()

        iterations = 1

        while True:
            newJ = self.__DefActiveSet()

            if newJ == False:
                self.__status = "O problema tem solucao ilimitada"
                return

            self.__BuildRSP()
            
            simplexRSP = simplex.Simplex(self.__rspA, self.__rspb, self.__rspZ)
            optimal, zOptimal = simplexRSP.Solver()

            multiplier = self.__FindMultiplier(simplexRSP)

            if multiplier != np.inf:
                # Resetando o multiplicador
                # ATENCAO: faz o algoritmo converger errado em alguns casos
                #if multiplier <= 1e-5:
                #    multiplier = 1.2
                self.__y = self.__y + multiplier * np.array(simplexRSP.dualSol[:self.__rows])

            optimalFound = True
            for i in range(len(self.__J), self.__rspColumns):
                if optimal[i] > 1e-5 or optimal[i] < -1e-4:
                    optimalFound = False
                    break

            if optimalFound:
                xb = [0] * self.__columns
                for j in range (len(self.__J)):
                    xb[self.__J[j]] = optimal[j]
                optimalZ = self.__Z @ xb

                self.__status = f"Solucao otima encontrada em {iterations} iteracoes"
            
                return np.array(xb), optimalZ

            iterations += 1