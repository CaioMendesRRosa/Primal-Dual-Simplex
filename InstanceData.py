
class InstanceData:

    def __init__ (self):
        self.edges = []
        self.problemType = None
        self.vertexNum = None
        self.edgesNum = None
        self.bestSol = None
        self.commoditiesNum = None
        self.origin = []
        self.destiny = []


    def readInstance(self, dir):

        with open(dir, 'r') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                    
                lineVal = line.split()
                
                # Descricao
                if lineVal[0] == 'p':
                    self.problemType = lineVal[1]
                    self.vertexNum = int(lineVal[2])
                    self.edgesNum = int(lineVal[3])

                    if self.problemType != "mcf":
                        continue

                    self.commoditiesNum = int(lineVal[4])
                    self.origin = [0] * self.commoditiesNum
                    self.destiny = [0] * self.commoditiesNum

                # Aresta
                elif lineVal[0] == 'a':
                    u = int(lineVal[1])
                    v = int(lineVal[2])
                    c = int(lineVal[3])
                    self.edges.append((u, v, c))

                # Melhor solucao
                elif lineVal[0] == 's':
                    self.bestSol = int(lineVal[1])

                # Origem e Destino
                elif lineVal[0] == 'n' and self.problemType == 'mcf':
                    if lineVal[2] == 's':
                        self.origin[int(lineVal[3]) - 1] = int(lineVal[1])
                    else:
                        self.destiny[int(lineVal[3]) - 1] = int(lineVal[1])

                # Comentario
                elif lineVal[0] == 'c':
                    continue

        return


    def InitPL (self):
        if self.problemType == "min":
            A, b, Z = self.InitPLMinCut()
        else:
            A, b, Z = self.InitPLMaxCut()

        return A, b, Z


    def InitPLMinCut (self):
        
        # Nro de vertices, Nro de arestas, Nro de folgas
        variablesNum = (self.vertexNum + (self.edgesNum) + (self.edgesNum * 2))
        Z = [0] * variablesNum

        A = []
        edgeCurrent = self.vertexNum
        excessNum = self.vertexNum + self.edgesNum
        for edge in self.edges:
            restriction1 = [0] * variablesNum # Du - Dv
            restriction2 = [0] * variablesNum # Dv - Du

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

        restrictionS = [0] * variablesNum # Xs = 0
        restrictionT = [0] * variablesNum # Xt = 0

        restrictionS[0] = 1
        restrictionT[self.vertexNum - 1] = 1

        A.append(restrictionS)
        A.append(restrictionT)

        b = [0] * ( (self.edgesNum * 2) + 2) # Restricoes das arestas + 2 restricoes Xs e Xt
        b[self.edgesNum * 2 + 1] = 1

        return A, b, Z


    def InitPLMaxCut (self):
        
        # Nro de Commodities * (Nro de arestas * 2 ) + Nro de folgas ( Nro de arestas )
        variablesNum = self.commoditiesNum * (self.edgesNum * 2) + self.edgesNum 
        Z = [0] * variablesNum

        A = []

        b = [0] * ( self.edgesNum ) + [0] * ( self.commoditiesNum * (self.vertexNum - 2) )
        bIndex = 0

        edgeCurrent = 0
        slackNum = self.commoditiesNum * (self.edgesNum * 2)

        for edge in self.edges:
            restriction1 = [0] * variablesNum
            edgeCurrentRestriction = edgeCurrent * self.commoditiesNum * 2

            if edge[1] not in self.destiny:
                for j in range(self.commoditiesNum):
                    restriction1[edgeCurrentRestriction + j] = 1
                    restriction1[edgeCurrentRestriction + self.commoditiesNum + j] = 1
            else:
                for j in range(self.commoditiesNum):
                    restriction1[edgeCurrentRestriction + j] = 1
                    restriction1[edgeCurrentRestriction + self.commoditiesNum + j] = 1
                Z[edgeCurrentRestriction + self.destiny.index(edge[1])] = self.destiny.index(edge[1]) + 1

            restriction1[slackNum] = 1
            b[bIndex] = edge[2]
            bIndex += 1
            slackNum += 1

            edgeCurrent += 1

            A.append(restriction1)

        for i in range (self.commoditiesNum):
            for k in range (self.vertexNum):
                if k + 1 == self.origin[i] or k + 1 == self.destiny[i]:
                    continue
                restriction = [0] * variablesNum
                for j in range(self.edgesNum):
                    if self.edges[j][0] != k + 1 and self.edges[j][1] != k + 1:
                        continue
                    edgeCurrent = j * (self.commoditiesNum * 2) + i
                    restriction[edgeCurrent] = 1
                    restriction[edgeCurrent + self.commoditiesNum] = 1
                A.append(restriction)

        return A, b, Z