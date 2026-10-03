import numpy as np
import time

'''
Algoritmo de Corte mínimo Stoer-Wagner
'''

class StoerWagner:

    def __init__ (self, edges, vertexNum, s, t):
        
        self.__edges = edges

        self.__edgesNum = len(edges)
        self.__vertexNum = vertexNum

        self.__s = s # fonte
        self.__t = t # sumidouro

        # Matriz de Adjacencia do grafo
        self.__adjMat = np.array([[0] * vertexNum for i in range(vertexNum)])

        self.__minCut = np.inf # Menor corte encontrado

        self.__inactiveVertex = [] # Lista com vertices que se juntaram a t (sumidouro)
        self.__activeVertex = vertexNum # Quantidade de vertices ativos

        self.__executionTime = 0

    @property
    def executionTime (self):
        return self.__executionTime

    def __Setup(self):

        # Inicializando a matriz
        for edge in self.__edges:
            self.__adjMat[edge[0] - 1][edge[1] - 1] = edge[2]
            self.__adjMat[edge[1] - 1][edge[0] - 1] = edge[2]

        return

    
    def __FindMaxCostVertex (self, costVertex):
        
        maxCost = 0
        maxVertex = -1

        # Encontrando a aresta de maior custo entre d e um vertice de fora
        for i in range(self.__vertexNum):

            if i + 1 == self.__t:
                continue

            if i + 1 in self.__d:
                continue 
            
            if i + 1 in self.__inactiveVertex:
                continue

            if costVertex[i] > maxCost:
                maxCost = costVertex[i]
                maxVertex = i + 1

        return maxVertex


    def __UpdateGraph (self, mergeVertex):

        # Colocando o mergeVertex que fundiu com t como inativo
        self.__inactiveVertex.append(mergeVertex)
        self.__activeVertex -= 1

        # Somando as arestas de mergeVertex com t
        self.__adjMat[:, self.__t - 1] += self.__adjMat[:, mergeVertex - 1]
        self.__adjMat[self.__t - 1] += self.__adjMat[mergeVertex - 1]

        return


    def __MinCutPhase(self):
        
        self.__d = [self.__s] # Vertices que foram fundidos no mesmo grupo de s(fonte)
        dVertexCount = 1

        costVertex = np.array(self.__adjMat[self.__s - 1]) # Peso das arestas do grupo de s(fonte)

        cut = 0 # Corte da ultima iteracao

        while dVertexCount < self.__activeVertex - 1:

            maxVertex = self.__FindMaxCostVertex(costVertex)

            self.__d.append(maxVertex)
            dVertexCount += 1

            # Somando os pesos das arestas do vertice que entrou em d
            costVertex = costVertex + self.__adjMat[maxVertex - 1]

        mergeVertex = self.__d[-1] # Vertice que ira fundir com t
        cut = costVertex[self.__t - 1] # Custo do corte

        self.__UpdateGraph(mergeVertex) # Atualizando o grafo

        return cut


    def Solver(self):
        self.__Setup()

        timeBegin = time.perf_counter()

        while self.__activeVertex > 1:
            minCutPhase = self.__MinCutPhase()
            self.__minCut = min(self.__minCut, minCutPhase)

        self.__executionTime = time.perf_counter() - timeBegin
        return self.__minCut