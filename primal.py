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
        # Inicializando a base

        self.__dualA = np.transpose(self.__A)
        self.__dualb = [i for i in self.__Z]
        self.__dualZ = [i for i in self.__b]

        self.__y = [0 for i in self.__b]

        return


    def __DefActiveSet (self):
        # Encontrando o conjunto das variaveis ativas

        oldJ = self.__J
        self.__J = []

        for j in range (self.__columns):
            if self.__Z[j] - (np.array(self.__y)) @ [self.__A[i][j] for i in range(self.__rows)] <= 1e-8:
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

            if mulNum < 1e-8:
                continue

            if mulDen > 1e-5 and mulNum / mulDen < minMultiplier:
                minMultiplier = mulNum / mulDen

        return minMultiplier


    def Solver(self):
        
        self.__Setup()

        iterations = 0

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
                self.__y = self.__y + multiplier * np.array(simplexRSP.dualSol[:self.__rows])


            optimalFound = True
            for i in range(len(self.__J), self.__rspColumns):
                if optimal[i] > 1e-5 or optimal[i] < -1e-5:
                    optimalFound = False
                    break

            if optimalFound:
                xb = [0 for i in range (self.__columns)]
                for j in range (len(self.__J)):
                    xb[self.__J[j]] = optimal[j]
                optimalZ = sum([self.__Z[j] * xb[j] for j in range(self.__columns)])
            
                return np.array(xb), optimalZ

            if iterations % 1000 == 0:
                print (multiplier)

            iterations += 1
            


def readInstance(dir):
    edges = []
    vertexNum = 0
    edgesNum = 0
    bestSol = 0

    with open(dir, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
                
            lineVal = line.split()
            
            # Descricao
            if lineVal[0] == 'p':
                vertexNum = int(lineVal[2])
                edgesNum = int(lineVal[3])
                
            # Aresta
            elif lineVal[0] == 'e':
                u = int(lineVal[1])
                v = int(lineVal[2])
                c = int(lineVal[3])
                edges.append((u, v, c))

            # Melhor solucao
            elif lineVal[0] == 's':
                bestSol = int(lineVal[1])

    return vertexNum, edgesNum, edges, bestSol


def InitPL (vertexNum, edgesNum, edges):
    
    # Nro de vertices, Nro de arestas, Nro de folgas
    variablesNum = (vertexNum + (edgesNum) + (edgesNum * 2))
    Z = [0] * variablesNum

    A = []
    edgeCurrent = vertexNum
    excessNum = vertexNum + edgesNum
    for i in edges:
        restriction1 = [0] * variablesNum # Du - Dv
        restriction2 = [0] * variablesNum # Dv - Du

        restriction1[edgeCurrent] = 1
        restriction2[edgeCurrent] = 1

        restriction1[i[0] - 1] = -1
        restriction1[i[1] - 1] = 1

        restriction2[i[0] - 1] = 1
        restriction2[i[1] - 1] = -1

        restriction1[excessNum] = -1
        restriction2[excessNum + 1] = -1

        excessNum += 2

        Z[edgeCurrent] = i[2]
        edgeCurrent += 1

        A.append(restriction1)
        A.append(restriction2)

    restrictionS = [0] * variablesNum # Xs = 0
    restrictionT = [0] * variablesNum # Xt = 0

    restrictionS[0] = 1
    restrictionT[vertexNum - 1] = 1

    A.append(restrictionS)
    A.append(restrictionT)

    b = [0] * ( (edgesNum * 2) + 2) # Restricoes das arestas + 2 restricoes Xs e Xt
    b[(edgesNum * 2) + 1] = 1

    return A, b, Z


if __name__ == "__main__":

    vertexNum, edgesNum, edges, bestSol = readInstance("instances/a.in")

    print (f"Vértices: {vertexNum}")
    print (f"Arestas: {edgesNum}")

    A, b, Z = InitPL(vertexNum, edgesNum, edges)
    
    primal = Primal(A, b, Z)
    optimal, optimalZ = primal.Solver()

    print (f"Arestas: {optimal}")
    print (f"Corte otimo: {optimalZ}")
    print (bestSol)
