import time
import numpy as np

"""
Algoritmo de Corte mínimo Stoer-Wagner
"""

# 13
# 20
# 31

class StoerWagner:
    # Constrói a classe
    def __init__(self, edges, vertexNum, s=1, t=None, verbose=False):
        self.__edges = edges
        self.__edgesNum = len(edges)
        self.__vertexNum = vertexNum
        # vertice onde cada fase começa
        self.__s = s
        # sumidouro t fixo, onde os vertices são fundidos
        self.__t = t
        self.__verbose = verbose

        #matriz de adjacência
        self.__adjMat = np.array([[0] * vertexNum for i in range(vertexNum)])
        #menor corte começa como infinito 
        self.__minCut = np.inf

        self.__inactiveVertex = []
        self.__activeVertex = vertexNum
        self.__executionTime = 0

    @property
    def executionTime(self):
        return self.__executionTime
    # preenche a matriz
    def __Setup(self):
        # grava nos 2 sentidos
        for edge in self.__edges:
            self.__adjMat[edge[0] - 1][edge[1] - 1] = edge[2]
            self.__adjMat[edge[1] - 1][edge[0] - 1] = edge[2]
    # escolhe o proximo vértice
    def __FindMaxCostVertex(self, costVertex):
        maxCost = 0
        maxVertex = -1

        for i in range(self.__vertexNum):
            # pula o t
            if i + 1 == self.__t:
                continue
            # pula o grupo da fase atual
            if i + 1 in self.__d:
                continue
            # pulas os ja fundidos
            if i + 1 in self.__inactiveVertex:
                continue
            # escolhe o de maior ligação
            if costVertex[i] > maxCost:
                maxCost = costVertex[i]
                maxVertex = i + 1

        return maxVertex
    # função de fundir
    def __UpdateGraph(self, mergeVertex):
        self.__inactiveVertex.append(mergeVertex)
        self.__activeVertex -= 1

        #soma a coluna do vertice na coluna do t
        self.__adjMat[:, self.__t - 1] += self.__adjMat[:, mergeVertex - 1]
        #soma a linha na linha do t
        self.__adjMat[self.__t - 1] += self.__adjMat[mergeVertex - 1]

    # função principal
    def __MinCutPhase(self):
        #começa com o vertice s
        self.__d = [self.__s]
        dVertexCount = 1
        costVertex = np.zeros(self.__vertexNum)
        # guarda o ultimo vertice adicionado
        maxVertex = self.__s
        # repete enquanto tem menos V que o total de ativos
        while dVertexCount < self.__activeVertex:
            #soma a linha do ultimo vértice
            costVertex = costVertex + self.__adjMat[maxVertex - 1]
            #escolhe o proximo mais ligado
            maxVertex = self.__FindMaxCostVertex(costVertex)
            #guarda o vertice
            self.__d.append(maxVertex)
            dVertexCount += 1
        #garante que se o t não vier, ele  usa o ultimo da lista
        if self.__t == None:
            self.__t = self.__d[-1]
        # o ultimo vertice real é o penultimo
        mergeVertex = self.__d[-2]
        #o corte é o placar de t
        cut = costVertex[self.__t - 1]
        # funde
        self.__UpdateGraph(mergeVertex)

        return cut

    def Solver(self):
        # preenche a matriz
        self.__Setup()

        if self.__verbose:
            print(f"  [Stoer-Wagner] Grafo: {self.__vertexNum} vértices e {self.__edgesNum} arestas.")

        timeBegin = time.perf_counter()
        phase = 1
        # enquanto tem mais de  vertice ativo
        while self.__activeVertex > 1:
            minCutPhase = self.__MinCutPhase()
            #atualiza o menor corte
            self.__minCut = min(self.__minCut, minCutPhase)
            phase += 1

            if self.__verbose:
                print(f"  [Stoer-Wagner] Fase {phase:2d}: Vértices restantes = {self.__activeVertex:2d} | "
                      f"Corte da fase = {minCutPhase:8.2f} | Menor corte global = {self.__minCut:8.2f}")

        self.__executionTime = time.perf_counter() - timeBegin
        if self.__verbose:
            print(f"  [Stoer-Wagner] Concluído em {self.__executionTime:.4f}s. Corte ótimo = {self.__minCut}")
        # devolve o menor corte
        return self.__minCut
