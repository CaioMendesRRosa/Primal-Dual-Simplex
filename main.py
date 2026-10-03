import numpy as np
from primal import Primal
from InstanceData import InstanceData
from pulpSolver import *
from stoerWagner import StoerWagner
import time


if __name__ == "__main__":

    # Altere as instancias aqui
    dir = "instances/mc_instance1.max"

    # Classe com as informacoes do problema (Qnt de vertices, arestas, etc)
    instanceData = InstanceData()
    instanceData.readInstance(dir)

    print ("Problema de Corte Minimo" if instanceData.problemType == "min" else "Problema de Multiplas Mercadorias")
    print (f"Vértices: {instanceData.vertexNum}")
    print (f"Arestas: {instanceData.edgesNum}")

    if instanceData.problemType == "min":
        stoerWagner = StoerWagner(instanceData.edges, instanceData.vertexNum, 1, instanceData.vertexNum)
        minCut = stoerWagner.Solver()
        print("\n-----Solucao Stoer-Wagner-----")
        print (f"Tempo de Execucao: {stoerWagner.executionTime:.3f}s")
        print (f"Corte otimo: {minCut}")

    # Incializando o problema de programacao linear
    A, b, Z = instanceData.InitPL()

    # Resolvendo com o primal-dual simplex
    primal = Primal(A, b, Z)
    optimal, optimalZ = primal.Solver()

    print ("\n-----Solucao Primal-Dual Simplex-----")
    print (primal.status)
    print (f"Tempo de Execucao: {primal.executionTime:.3f}s")

    if instanceData.problemType == "min":
        optimalEdges = []
        for i in range(instanceData.vertexNum, instanceData.vertexNum + instanceData.edgesNum):
            if optimal[i] >= 1:
                optimalEdges.append(instanceData.edges[i - instanceData.vertexNum])

        print (f"Arestas: {optimalEdges}")
    print (f"Solucao otima: {optimalZ:.3f}\n")

    # Resolvendo com o pulp
    prob, optimalEdges = buildModelPulp(instanceData)

    prob.solve(pulp.PULP_CBC_CMD(msg=False)) # Coloque True para imprimir as iteracoes do Pulp

    # Resultados do pulp
    print("\n-----Solucao Pulp-----")
    print(f"Status: {pulp.LpStatus[prob.status]}")
    print(f"Valor otimo: {pulp.value(prob.objective)}")

    if instanceData.problemType == "min":
        selectedEdges = []
        for i in range(instanceData.edgesNum):
            if pulp.value(optimalEdges[i]) > 0.5:
                selectedEdges.append(instanceData.edges[i])
            
        print (f"Arestas selecionadas: {selectedEdges}")

