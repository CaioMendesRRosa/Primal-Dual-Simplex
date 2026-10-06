import pulp


def _new_variable(problem, name, low_bound=0, up_bound=None, category="Continuous"):
    """Create a variable using either the PuLP 4 or the PuLP 3 API."""
    if hasattr(problem, "add_variable"):
        return problem.add_variable(
            name,
            lowBound=low_bound,
            upBound=up_bound,
            cat=category,
        )
    return pulp.LpVariable(
        name,
        lowBound=low_bound,
        upBound=up_bound,
        cat=category,
    )


def buildModelPulp(instanceData):
    if instanceData.problemType == "min":
        return buildModelPulpMinCut(instanceData)
    if instanceData.problemType == "mcf":
        return buildModelPulpMaxFlow(instanceData)
    raise ValueError(f"Tipo de problema desconhecido: {instanceData.problemType}")


def buildModelPulpMinCut(instanceData):
    problem = pulp.LpProblem("Corte_Minimo_st", pulp.LpMinimize)
    d = [
        _new_variable(problem, f"d_{vertex}", up_bound=1, category="Continuous")
        for vertex in range(1, instanceData.vertexNum + 1)
    ]
    x = [
        _new_variable(problem, f"x_e{edgeIndex}", up_bound=1, category="Continuous")
        for edgeIndex in range(1, instanceData.edgesNum + 1)
    ]

    problem += pulp.lpSum(
        instanceData.edges[i][2] * x[i] for i in range(instanceData.edgesNum)
    ), "custo_do_corte"

    for i, (u, v, _) in enumerate(instanceData.edges):
        problem += x[i] >= d[u - 1] - d[v - 1], f"corte_e{i + 1}_uv"
        problem += x[i] >= d[v - 1] - d[u - 1], f"corte_e{i + 1}_vu"

    problem += d[instanceData.source - 1] == 0, "fonte_no_lado_zero"
    problem += d[instanceData.sink - 1] == 1, "sumidouro_no_lado_um"
    return problem, x


def buildModelPulpMaxFlow(instanceData):
    problem = pulp.LpProblem("Fluxo_Maximo_Multiplas_Mercadorias", pulp.LpMaximize)
    k = instanceData.commoditiesNum
    m = instanceData.edgesNum
    flows = [
        _new_variable(problem, f"f_{i}", category="Continuous")
        for i in range(2 * m * k)
    ]
    objectiveTerms = []

    for edgeIndex, (u, v, capacity) in enumerate(instanceData.edges):
        offset = edgeIndex * 2 * k
        capacityTerms = []
        for commodity in range(k):
            forward = offset + commodity
            reverse = offset + k + commodity
            capacityTerms.extend((flows[forward], flows[reverse]))
            prize = commodity + 1
            if v == instanceData.destiny[commodity]:
                objectiveTerms.append(prize * flows[forward])
            if u == instanceData.destiny[commodity]:
                objectiveTerms.append(prize * flows[reverse])

            if v == instanceData.origin[commodity] or u == instanceData.destiny[commodity]:
                problem += flows[forward] == 0, f"proibido_f{commodity + 1}_e{edgeIndex + 1}_uv"
            if u == instanceData.origin[commodity] or v == instanceData.destiny[commodity]:
                problem += flows[reverse] == 0, f"proibido_f{commodity + 1}_e{edgeIndex + 1}_vu"

        problem += pulp.lpSum(capacityTerms) <= capacity, f"capacidade_e{edgeIndex + 1}"

    problem += pulp.lpSum(objectiveTerms), "fluxo_total_com_premios"

    for commodity in range(k):
        source = instanceData.origin[commodity]
        sink = instanceData.destiny[commodity]
        for vertex in range(1, instanceData.vertexNum + 1):
            if vertex in (source, sink):
                continue

            balanceTerms = []
            for edgeIndex, (u, v, _) in enumerate(instanceData.edges):
                offset = edgeIndex * 2 * k
                forward = flows[offset + commodity]
                reverse = flows[offset + k + commodity]
                if u == vertex:
                    balanceTerms.extend((forward, -reverse))
                if v == vertex:
                    balanceTerms.extend((-forward, reverse))
            problem += (
                pulp.lpSum(balanceTerms) == 0,
                f"conservacao_f{commodity + 1}_v{vertex}",
            )

    return problem, flows
