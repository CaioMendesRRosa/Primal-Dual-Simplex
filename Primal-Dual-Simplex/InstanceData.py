class InstanceData:
    def __init__(self):
        self.edges = []
        self.problemType = None
        self.vertexNum = None
        self.edgesNum = None
        self.bestSol = None
        self.commoditiesNum = 0
        self.origin = []
        self.destiny = []
        self.source = None
        self.sink = None
        self.variableNames = []
        self.constraintNames = []

    def readInstance(self, path):
        # Clear previous contents so one reader can safely load several files.
        self.__init__()
        with open(path, "r", encoding="utf-8-sig") as instanceFile:
            for lineNumber, rawLine in enumerate(instanceFile, start=1):
                line = rawLine.strip()
                if not line or line.startswith("c"):
                    continue
                fields = line.split()
                descriptor = fields[0]

                if descriptor == "p":
                    if self.problemType is not None:
                        raise ValueError(f"Linha {lineNumber}: existe mais de uma descrição do problema")
                    self.problemType = fields[1]
                    self.vertexNum = int(fields[2])
                    self.edgesNum = int(fields[3])
                    if self.problemType == "mcf":
                        self.commoditiesNum = int(fields[4])
                        self.origin = [None] * self.commoditiesNum
                        self.destiny = [None] * self.commoditiesNum
                    elif self.problemType == "min":
                        self.source = 1
                        self.sink = self.vertexNum
                    else:
                        raise ValueError(f"Linha {lineNumber}: tipo de problema desconhecido: {self.problemType}")

                elif descriptor == "a":
                    if self.vertexNum is None:
                        raise ValueError(f"Linha {lineNumber}: a descrição p deve vir antes das arestas")
                    u, v, capacity = map(int, fields[1:4])
                    if not (1 <= u <= self.vertexNum and 1 <= v <= self.vertexNum):
                        raise ValueError(f"Linha {lineNumber}: vértice fora do intervalo 1..{self.vertexNum}")
                    if capacity < 0:
                        raise ValueError(f"Linha {lineNumber}: capacidade negativa")
                    self.edges.append((u, v, capacity))

                elif descriptor == "s":
                    self.bestSol = int(fields[1])

                elif descriptor == "n":
                    if self.problemType == "min":
                        node = int(fields[1])
                        if fields[2] == "s":
                            self.source = node
                        elif fields[2] == "t":
                            self.sink = node
                    elif self.problemType == "mcf":
                        node = int(fields[1])
                        commodity = int(fields[3]) - 1
                        if not 0 <= commodity < self.commoditiesNum:
                            raise ValueError(f"Linha {lineNumber}: índice de mercadoria inválido")
                        if fields[2] == "s":
                            self.origin[commodity] = node
                        elif fields[2] == "t":
                            self.destiny[commodity] = node

        if self.problemType is None:
            raise ValueError("A instância não contém uma linha p")
        if len(self.edges) != self.edgesNum:
            raise ValueError(
                f"A instância declara {self.edgesNum} arestas, mas contém {len(self.edges)}"
            )
        if self.problemType == "min":
            if not (1 <= self.source <= self.vertexNum and 1 <= self.sink <= self.vertexNum):
                raise ValueError("Fonte ou sumidouro fora do intervalo de vértices")
            if self.source == self.sink:
                raise ValueError("Fonte e sumidouro precisam ser distintos")
        else:
            if any(node is None for node in self.origin + self.destiny):
                raise ValueError("Toda mercadoria precisa declarar uma fonte e um sumidouro")
            if any(not 1 <= node <= self.vertexNum for node in self.origin + self.destiny):
                raise ValueError("Fonte ou sumidouro de mercadoria fora do intervalo de vértices")
            if any(s == t for s, t in zip(self.origin, self.destiny)):
                raise ValueError("A fonte e o sumidouro de cada mercadoria precisam ser distintos")
        return

    def InitPL(self):
        if self.problemType == "min":
            return self.InitPLMinCut()
        if self.problemType == "mcf":
            return self.InitPLMaxFlow()
        raise ValueError("Problema não definido")

    def InitialDualFeasible(self):
        """Construct a feasible dual start for the max-flow formulation."""
        if self.problemType != "mcf":
            return None

        rowCount = self.edgesNum + 1 + self.commoditiesNum * (self.vertexNum - 2)
        dual = [0.0] * rowCount
        for edgeIndex, (u, v, _) in enumerate(self.edges):
            destinationPrizes = [
                commodity + 1
                for commodity, destination in enumerate(self.destiny)
                if destination in (u, v)
            ]
            if destinationPrizes:
                # Every flow variable on this edge has capacity coefficient
                # +1. A price at least as negative as its largest sink reward
                # makes all corresponding min-form dual inequalities feasible.
                dual[edgeIndex] = -float(max(destinationPrizes))
        return dual

    def InitPLMinCut(self):
        n = self.vertexNum
        m = self.edgesNum
        variablesNum = n + m + 2 * m
        edgeStart = n
        slackStart = n + m
        Z = [0.0] * variablesNum
        A = []
        b = []
        self.variableNames = [f"d_{v}" for v in range(1, n + 1)]
        self.variableNames.extend(
            f"x_e{i}_{u}_{v}" for i, (u, v, _) in enumerate(self.edges, start=1)
        )
        self.constraintNames = []

        for i, (u, v, capacity) in enumerate(self.edges):
            xIndex = edgeStart + i
            firstSlack = slackStart + 2 * i
            Z[xIndex] = float(capacity)

            # x_uv >= d_u - d_v  <=>  x_uv - d_u + d_v - s = 0.
            forward = [0.0] * variablesNum
            forward[xIndex] = 1.0
            forward[u - 1] -= 1.0
            forward[v - 1] += 1.0
            forward[firstSlack] = -1.0

            reverse = [0.0] * variablesNum
            reverse[xIndex] = 1.0
            reverse[u - 1] += 1.0
            reverse[v - 1] -= 1.0
            reverse[firstSlack + 1] = -1.0

            A.extend((forward, reverse))
            b.extend((0.0, 0.0))
            self.variableNames.extend((f"s_e{i + 1}_uv", f"s_e{i + 1}_vu"))
            self.constraintNames.extend((f"cut_e{i + 1}_{u}_{v}_uv", f"cut_e{i + 1}_{u}_{v}_vu"))

        sourceRow = [0.0] * variablesNum
        sourceRow[self.source - 1] = 1.0
        sinkRow = [0.0] * variablesNum
        sinkRow[self.sink - 1] = 1.0
        A.extend((sourceRow, sinkRow))
        b.extend((0.0, 1.0))
        self.constraintNames.extend((f"d_{self.source}_fixado_0", f"d_{self.sink}_fixado_1"))
        return A, b, Z

    def InitPLMaxFlow(self):
        n = self.vertexNum
        m = self.edgesNum
        k = self.commoditiesNum
        flowVariableCount = 2 * m * k
        variablesNum = flowVariableCount + m
        Z = [0.0] * variablesNum
        A = []
        b = []
        self.variableNames = []
        self.constraintNames = []

        for edgeIndex, (u, v, _) in enumerate(self.edges, start=1):
            self.variableNames.extend(f"f_{commodity + 1}_{u}_{v}_e{edgeIndex}" for commodity in range(k))
            self.variableNames.extend(f"f_{commodity + 1}_{v}_{u}_e{edgeIndex}" for commodity in range(k))
        self.variableNames.extend(f"s_cap_e{i}" for i in range(1, m + 1))

        # Capacity constraints, with one nonnegative slack per edge.
        for edgeIndex, (u, v, capacity) in enumerate(self.edges):
            row = [0.0] * variablesNum
            edgeOffset = edgeIndex * 2 * k
            for commodity in range(k):
                forward = edgeOffset + commodity
                reverse = edgeOffset + k + commodity
                row[forward] = 1.0
                row[reverse] = 1.0
                prize = float(commodity + 1)
                if v == self.destiny[commodity]:
                    Z[forward] -= prize
                if u == self.destiny[commodity]:
                    Z[reverse] -= prize
            row[flowVariableCount + edgeIndex] = 1.0
            A.append(row)
            b.append(float(capacity))
            self.constraintNames.append(f"capacidade_e{edgeIndex + 1}_{u}_{v}")

        # All forbidden source-entry and sink-exit flows are nonnegative, so
        # their sum being zero is equivalent to fixing each one to zero.
        forbiddenFlows = [0.0] * variablesNum
        for edgeIndex, (u, v, _) in enumerate(self.edges):
            edgeOffset = edgeIndex * 2 * k
            for commodity in range(k):
                forward = edgeOffset + commodity
                reverse = edgeOffset + k + commodity
                if v == self.origin[commodity] or u == self.destiny[commodity]:
                    forbiddenFlows[forward] = 1.0
                if u == self.origin[commodity] or v == self.destiny[commodity]:
                    forbiddenFlows[reverse] = 1.0
        A.append(forbiddenFlows)
        b.append(0.0)
        self.constraintNames.append("fluxo_proibido_na_fonte_ou_sumidouro")

        # Flow conservation at every internal vertex of every commodity.
        for commodity in range(k):
            for vertex in range(1, n + 1):
                if vertex in (self.origin[commodity], self.destiny[commodity]):
                    continue
                row = [0.0] * variablesNum
                for edgeIndex, (u, v, _) in enumerate(self.edges):
                    edgeOffset = edgeIndex * 2 * k
                    forward = edgeOffset + commodity
                    reverse = edgeOffset + k + commodity
                    if u == vertex:
                        row[forward] += 1.0
                        row[reverse] -= 1.0
                    if v == vertex:
                        row[forward] -= 1.0
                        row[reverse] += 1.0
                A.append(row)
                b.append(0.0)
                self.constraintNames.append(f"conservacao_f{commodity + 1}_v{vertex}")

        return A, b, Z
