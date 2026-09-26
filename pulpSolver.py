import pulp
from InstanceData import InstanceData


def buildModelPulp(instanceData):

    if instanceData.problemType == "min":
        prob, x = buildModelPulpMinCut(instanceData)
        return prob, x

    if instanceData.problemType == "mcf":
        prob, x = buildModelPulpMaxFlow(instanceData)
        return prob, x

    return None



def buildModelPulpMinCut(instanceData):
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


def buildModelPulpMaxFlow(instanceData):
    # Criar problema
    prob = pulp.LpProblem("Fluxo_Maximo_Multiplas_Mercadorias", pulp.LpMaximize)

    # Fluxo das arestas
    f = [pulp.LpVariable(f"f_{i}", cat=pulp.LpContinuous, lowBound=0) for i in range(instanceData.edgesNum * 2 * instanceData.commoditiesNum)]

    # Funcao objetivo
    fo = 0
    for j in range (instanceData.commoditiesNum):
        commoditySum = 0
        for i in range (instanceData.edgesNum):

            if instanceData.edges[i][1] == instanceData.destiny[j]:
                commoditySum += f[i * 2 * instanceData.commoditiesNum + j]
            elif instanceData.edges[i][0] == instanceData.destiny[j]:
                commoditySum += f[i * 2 * instanceData.commoditiesNum + j + instanceData.commoditiesNum]

        fo += (j + 1) * commoditySum
    
    prob += fo
            
    for i in range (instanceData.edgesNum):

        commoditySum = 0
        for j in range(instanceData.commoditiesNum):

            currentEdge = i * 2 * instanceData.commoditiesNum + j
            commoditySum += f[currentEdge] + f[currentEdge + instanceData.commoditiesNum]

        prob += commoditySum <= instanceData.edges[i][2]

    for j in range (instanceData.commoditiesNum):

        for k in range (instanceData.vertexNum):

            if k + 1 == instanceData.origin[j] or k + 1 == instanceData.destiny[j]:
                continue
            
            commoditySum = 0
            for i in range (instanceData.edgesNum):

                currentEdge = i * 2 * instanceData.commoditiesNum + j
                
                if k + 1 == instanceData.edges[i][0]:
                    commoditySum += f[currentEdge] - f[currentEdge + instanceData.commoditiesNum]
                elif k + 1 == instanceData.edges[i][1]:
                    commoditySum += -f[currentEdge] + f[currentEdge + instanceData.commoditiesNum]

            prob += commoditySum == 0

    return prob, f


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