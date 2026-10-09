class InstanceData:

    def __init__(self):
        self.edges = []
        self.problemType = None
        self.vertexNum = None
        self.edgesNum = None
        self.bestSol = None
        self.commoditiesNum = None
        self.origin = []
        self.destiny = []

    def readInstance(self, dir, verbose=False):
        if verbose:
            print(f"  [Leitura] Carregando arquivo: {dir}")
        with open(dir, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                lineVal = line.split()

                if lineVal[0] == "p":
                    self.problemType = lineVal[1]
                    self.vertexNum = int(lineVal[2])
                    self.edgesNum = int(lineVal[3])

                    if self.problemType != "mcf":
                        continue

                    self.commoditiesNum = int(lineVal[4])
                    self.origin = [0] * self.commoditiesNum
                    self.destiny = [0] * self.commoditiesNum

                elif lineVal[0] == "a":
                    u = int(lineVal[1])
                    v = int(lineVal[2])
                    c = int(lineVal[3])
                    self.edges.append((u, v, c))

                elif lineVal[0] == "s":
                    self.bestSol = int(lineVal[1])

                elif lineVal[0] == "n":
                    if self.problemType == "min" and lineVal[2] == "s":
                        self.origin = int(lineVal[1])
                    elif self.problemType == "min" and lineVal[2] == "t":
                        self.destiny = int(lineVal[1])
                    elif lineVal[2] == "s":
                        self.origin[int(lineVal[3]) - 1] = int(lineVal[1])
                    else:
                        self.destiny[int(lineVal[3]) - 1] = int(lineVal[1])

                elif lineVal[0] == "c":
                    continue

        if verbose:
            info = f"  [Leitura] Concluída: {self.vertexNum} vértices, {self.edgesNum} arestas"
            if self.problemType == "mcf":
                info += f", {self.commoditiesNum} mercadorias"
            print(info)

    def InitPL(self, verbose=False):
        if verbose:
            print(f"  [InitPL] Montando formulação do PL ({self.problemType})...")

        if self.problemType == "min":
            A, b, Z = self.InitPLMinCut()
            if verbose:
                print(
                    f"  [InitPL] Matriz A: {len(A)} restrições x {len(Z)} variáveis."
                )
            return A, b, Z

        if self.problemType == "mcf":
            A, b, Z = self.InitPLMaxFlow()
            if verbose:
                print(
                    f"  [InitPL] Matriz A: {len(A)} restrições x {len(Z)} variáveis."
                )
            return A, b, Z

        print("Problema não definido.")
        return None

    def InitPLMinCut(self):
        variablesNum = self.vertexNum + self.edgesNum + (self.edgesNum * 2)
        Z = [0] * variablesNum
        A = []
        edgeCurrent = self.vertexNum
        excessNum = self.vertexNum + self.edgesNum

        for edge in self.edges:
            restriction1 = [0] * variablesNum
            restriction2 = [0] * variablesNum

            restriction1[edgeCurrent] = 1
            restriction2[edgeCurrent] = 1

            restriction1[edge[0] - 1] = -1
            restriction1[edge[1] - 1] = 1

            restriction2[edge[0] - 1] = 1
            restriction2[edge[1] - 1] = -1

            restriction1[excessNum] = -1
            restriction2[excessNum + 1] = -1

            excessNum += 2
            Z[edgeCurrent] = edge[2]
            edgeCurrent += 1

            A.append(restriction1)
            A.append(restriction2)

        restrictionS = [0] * variablesNum
        restrictionT = [0] * variablesNum

        restrictionS[self.origin - 1] = 1
        restrictionT[self.destiny - 1] = 1

        A.append(restrictionS)
        A.append(restrictionT)

        b = [0] * ((self.edgesNum * 2) + 2)
        b[self.edgesNum * 2 + 1] = 1

        return A, b, Z

    def InitPLMaxFlow(self):
        variablesNum = self.commoditiesNum * (self.edgesNum * 2) + self.edgesNum
        Z = [0] * variablesNum
        A = []
        b = (
            [0] * (self.edgesNum)
            + [0] * (self.commoditiesNum * (self.vertexNum - 2))
            + [0]
        )
        bIndex = 0

        edgeCurrent = 0
        slackNum = self.commoditiesNum * (self.edgesNum * 2)
        restrictionOD = [0] * variablesNum

        for edge in self.edges:
            restriction1 = [0] * variablesNum
            edgeCurrentRestriction = edgeCurrent * self.commoditiesNum * 2

            for j in range(self.commoditiesNum):
                forward = edgeCurrentRestriction + j
                reverse = edgeCurrentRestriction + self.commoditiesNum + j

                restriction1[forward] = 1
                restriction1[reverse] = 1

                if edge[1] == self.destiny[j]:
                    restrictionOD[reverse] = 1
                if edge[0] == self.destiny[j]:
                    restrictionOD[forward] = 1

                if edge[1] == self.origin[j]:
                    restrictionOD[forward] = 1
                if edge[0] == self.origin[j]:
                    restrictionOD[reverse] = 1

                weight = j + 1
                if self.destiny[j] == edge[1]:
                    Z[forward] -= weight
                if self.destiny[j] == edge[0]:
                    Z[reverse] -= weight

            restriction1[slackNum] = 1
            b[bIndex] = edge[2]
            bIndex += 1
            slackNum += 1
            edgeCurrent += 1
            A.append(restriction1)

        A.append(restrictionOD)

        for i in range(self.commoditiesNum):
            for k in range(self.vertexNum):
                if (
                    k + 1 == self.origin[i]
                    or k + 1 == self.destiny[i]
                    or (
                        self.edges[j][0] == self.edges[j][1]
                        and self.edges[j][0] == k + 1
                    )
                ):
                    continue

                restriction = [0] * variablesNum
                for j in range(self.edgesNum):
                    edgeCurrent = j * (self.commoditiesNum * 2) + i
                    if (
                        self.edges[j][0] == self.edges[j][1]
                        and self.edges[j][0] == k + 1
                    ):
                        continue

                    if self.edges[j][0] == k + 1:
                        restriction[edgeCurrent] = 1
                        restriction[edgeCurrent + self.commoditiesNum] = -1

                    if self.edges[j][1] == k + 1:
                        restriction[edgeCurrent] = -1
                        restriction[edgeCurrent + self.commoditiesNum] = 1

                A.append(restriction)

        return A, b, Z