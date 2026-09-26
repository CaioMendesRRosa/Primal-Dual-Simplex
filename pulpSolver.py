import pulp
from InstanceData import InstanceData

def buildModelPulp(instanceData):
    # Criar problema
    prob = pulp.LpProblem("Corte_Minimo", pulp.LpMinimize)

    # Vertices
    d = [pulp.LpVariable(f"d_{i}", cat=pulp.LpBinary) for i in range(instanceData.vertexNum)]

    # Arestas
    x = [pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(instanceData.edgesNum)]

    # Funcao objetivo
    prob += pulp.lpSum(instanceData.edges[i][2] * x[i] for i in range(instanceData.edgesNum))

    # Restricao das arestas
    for i in range (instanceData.edgesNum):
        edge = instanceData.edges[i]
        prob += x[i] >= d[edge[0] - 1] - d[edge[1] - 1]
        prob += x[i] >= d[edge[1] - 1] - d[edge[0] - 1]

    # Restricao dos vertices
    prob += d[0] == 0
    prob += d[instanceData.vertexNum - 1] == 1

    return prob, x


if __name__ == "__main__":
    instanceData = InstanceData()
    instanceData.readInstance("instances/instance1.min")

    prob, optimalEdges = buildModelPulp(instanceData)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Resultados do pulp
    print(f"Valor otimo: {pulp.value(prob.objective)}")
    for i in range(instanceData.edgesNum):
        if pulp.value(optimalEdges[i]) > 0.5:
            print(f"aresta: {instanceData.edges[i]}")