import time
import numpy as np

"""
Algoritmo de Corte mínimo Stoer-Wagner
"""


class StoerWagner:

    def __init__(self, edges, vertexNum, s, t, verbose=False):
        self.__edges = edges
        self.__edgesNum = len(edges)
        self.__vertexNum = vertexNum
        self.__s = s
        self.__t = t
        self.__verbose = verbose

        self.__adjMat = np.array([[0] * vertexNum for i in range(vertexNum)])
        self.__minCut = np.inf

        self.__inactiveVertex = []
        self.__activeVertex = vertexNum
        self.__executionTime = 0

    @property
    def executionTime(self):
        return self.__executionTime

    def __Setup(self):
        for edge in self.__edges:
            self.__adjMat[edge[0] - 1][edge[1] - 1] = edge[2]
            self.__adjMat[edge[1] - 1][edge[0] - 1] = edge[2]

    def __FindMaxCostVertex(self, costVertex):
        maxCost = 0
        maxVertex = -1

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

    def __UpdateGraph(self, mergeVertex):
        self.__inactiveVertex.append(mergeVertex)
        self.__activeVertex -= 1

        self.__adjMat[:, self.__t - 1] += self.__adjMat[:, mergeVertex - 1]
        self.__adjMat[self.__t - 1] += self.__adjMat[mergeVertex - 1]

    def __MinCutPhase(self):
        self.__d = [self.__s]
        dVertexCount = 1
        costVertex = np.array(self.__adjMat[self.__s - 1])

        while dVertexCount < self.__activeVertex - 1:
            maxVertex = self.__FindMaxCostVertex(costVertex)
            self.__d.append(maxVertex)
            dVertexCount += 1
            costVertex = costVertex + self.__adjMat[maxVertex - 1]

        mergeVertex = self.__d[-1]
        cut = costVertex[self.__t - 1]
        self.__UpdateGraph(mergeVertex)

        return cut

    def Solver(self):
        self.__Setup()

        if self.__verbose:
            print(
                f"  [Stoer-Wagner] Grafo: {self.__vertexNum} vértices e {self.__edgesNum} arestas."
            )

        timeBegin = time.perf_counter()
        fase = 1

        while self.__activeVertex > 1:
            minCutPhase = self.__MinCutPhase()
            self.__minCut = min(self.__minCut, minCutPhase)
            if self.__verbose:
                print(
                    f"  [Stoer-Wagner] Fase {fase:2d}: Vértices restantes = {self.__activeVertex:2d} | "
                    f"Corte da fase = {minCutPhase:8.2f} | Menor corte global = {self.__minCut:8.2f}"
                )
            fase += 1

        self.__executionTime = time.perf_counter() - timeBegin
        if self.__verbose:
            print(
                f"  [Stoer-Wagner] Concluído em {self.__executionTime:.4f}s. Corte ótimo = {self.__minCut}"
            )
        return self.__minCut