import numpy as np
from scipy import optimize
from primal import Primal
from InstanceData import InstanceData


if __name__ == "__main__":

    instanceData = InstanceData()
    instanceData.readInstance("instances/instance4.min")

    print ("Problema de Corte Minimo" if instanceData.problemType == "min" else "Problema de Multiplas Mercadorias")
    print (f"Vértices: {instanceData.vertexNum}")
    print (f"Arestas: {instanceData.edgesNum}")

    A, b, Z = instanceData.InitPL()
    print (np.matrix(A))
    print(b)
    print(Z)

    primal = Primal(A, b, Z)
    optimal, optimalZ = primal.Solver()
    print(optimal)

    optimalEdges = []
    for i in range(instanceData.vertexNum, instanceData.vertexNum + instanceData.edgesNum):
        if optimal[i] >= 1:
            optimalEdges.append(instanceData.edges[i - instanceData.vertexNum])

    print (f"Arestas: {optimalEdges}")
    print (f"Corte otimo: {optimalZ}")
    print (instanceData.bestSol)
