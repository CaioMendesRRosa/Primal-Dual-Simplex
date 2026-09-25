import numpy as np
from scipy import optimize
import simplex
import simplexInv

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

        self.__dualA = np.transpose(self.__A)
        self.__dualb = [i for i in self.__Z]
        self.__dualZ = [i for i in self.__b]

        self.__y = [0 for i in self.__b]

        # Interferencia para tentar impedir degeneracao
        # for i in range (self.__rows):
        #     self.__b[i] += 1e-6

        return


    def __DefActiveSet (self):
        # Encontrando o conjunto das variaveis ativas

        oldJ = self.__J
        self.__J = []

        for j in range (self.__columns):
            cost = self.__Z[j] - (np.array(self.__y)) @ [self.__A[i][j] for i in range(self.__rows)]
            if cost <= 1e-4:
                self.__J.append(j)

        return not oldJ == self.__J



    def __BuildRSP (self):
        # Construindo o problema subrestrito

        self.__rspColumns = len(self.__J) + self.__rows
        self.__rspRows = self.__rows

        self.__rspZ = [0 for i in self.__J] + [1 for i in range(self.__rspRows)]
        self.__rspb = self.__b
        self.__rspA = []

        for i in range (self.__rspRows):
            lineI = [self.__A[i][j] for j in self.__J]
            lineI += [0 for j in range(self.__rows)]
            lineI[i + len(self.__J)] = 1
            self.__rspA.append(lineI)

        return
        

    def __FindMultiplier (self, simplexRSP):
        # Encontrando o multiplicador teta

        minMultiplier = np.inf

        for j in range(self.__columns):
            if j in self.__J:
                continue
            
            Aj = [self.__A[i][j] for i in range(self.__rows)]
            mulNum = (self.__Z[j] - ( self.__y @ np.array(Aj)))
            mulDen = (simplexRSP.dualSol[:self.__rows]) @ np.array(Aj)

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
                print ("eita")
                return

            self.__BuildRSP()
            
            simplexRSP = simplex.Simplex(self.__rspA, self.__rspb, self.__rspZ)
            optimal, zOptimal = simplexRSP.Solver()

            multiplier = self.__FindMultiplier(simplexRSP)

            if multiplier != np.inf:
                if iterations % 100 == 0:
                    multiplier = 1.5
                self.__y = self.__y + multiplier * np.array(simplexRSP.dualSol[:self.__rows])

            optimalFound = True
            for i in range(len(self.__J), self.__rspColumns):
                if optimal[i] > 1e-5 or optimal[i] < -1e-4:
                    optimalFound = False
                    break

            if optimalFound:
                xb = [0 for i in range (self.__columns)]
                for j in range (len(self.__J)):
                    xb[self.__J[j]] = optimal[j]
                optimalZ = sum([self.__Z[j] * xb[j] for j in range(self.__columns)])
            
                return np.array(xb), optimalZ

            iterations += 1