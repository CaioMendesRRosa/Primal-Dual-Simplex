import numpy as np
from scipy import optimize
from primal import Primal
from InstanceData import InstanceData
from pulpSolver import *

if __name__ == "__main__":

    instanceData = InstanceData()
    instanceData.readInstance("instances/instance1.min")

    print ("Problema de Corte Minimo" if instanceData.problemType == "min" else "Problema de Multiplas Mercadorias")
    print (f"Vértices: {instanceData.vertexNum}")
    print (f"Arestas: {instanceData.edgesNum}")

    A, b, Z = instanceData.InitPL()

    primal = Primal(A, b, Z)
    optimal, optimalZ = primal.Solver()

    print ("\n-----Solucao Primal-Dual Simplex-----")
    if instanceData.problemType == "min":
        optimalEdges = []
        for i in range(instanceData.vertexNum, instanceData.vertexNum + instanceData.edgesNum):
            if optimal[i] >= 1:
                optimalEdges.append(instanceData.edges[i - instanceData.vertexNum])

        print (f"Arestas: {optimalEdges}")
    print (f"Solucao otima: {optimalZ}")

    prob, optimalEdges = buildModelPulp(instanceData)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Resultados do pulp
    print("\n-----Solucao Pulp-----")
    print(f"Status: {pulp.LpStatus[prob.status]}")
    print(f"Valor otimo: {pulp.value(prob.objective)}")

    selectedEdges = []
    for i in range(instanceData.edgesNum):
        if pulp.value(optimalEdges[i]) > 0.5:
            selectedEdges.append(instanceData.edges[i])
            
    print (f"Arestas selecionadas: {selectedEdges}")

