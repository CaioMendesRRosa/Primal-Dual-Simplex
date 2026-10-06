import numpy as np


class Simplex:
    """Tableau simplex for min c'x subject to Ax=b and x>=0."""

    PIVOT_TOL = 1e-12
    REDUCED_COST_TOL = 1e-10
    FEASIBILITY_TOL = 1e-8

    def __init__(
        self,
        A,
        b,
        Z,
        initialBasis=None,
        initialTableau=None,
        computeDual=True,
        pricingRule="bland",
    ):
        self.__A = np.asarray(A, dtype=float)
        self.__b = np.asarray(b, dtype=float)
        self.__Z = np.asarray(Z, dtype=float)
        if self.__A.ndim != 2:
            raise ValueError("A deve ser uma matriz bidimensional")
        if self.__b.ndim != 1 or self.__Z.ndim != 1:
            raise ValueError("b e Z devem ser vetores")
        self.__rows, self.__columns = self.__A.shape
        if self.__b.size != self.__rows or self.__Z.size != self.__columns:
            raise ValueError("Dimensões incompatíveis entre A, b e Z")
        if not all(np.all(np.isfinite(v)) for v in (self.__A, self.__b, self.__Z)):
            raise ValueError("A, b e Z devem conter apenas valores finitos")

        self.__initialBasis = initialBasis
        self.__initialTableau = initialTableau
        self.__computeDual = computeDual
        if pricingRule not in ("bland", "dantzig"):
            raise ValueError("A regra de entrada deve ser 'bland' ou 'dantzig'")
        self.__pricingRule = pricingRule
        self.__Bvars = []
        self.__dualSol = None
        self.__tableau = None
        self.__status = ""
        self.__iterations = 0

    @property
    def status(self):
        return self.__status

    @property
    def dualSol(self):
        return None if self.__dualSol is None else self.__dualSol.copy()

    @property
    def Bvars(self):
        return self.__Bvars.copy()

    @property
    def iterations(self):
        return self.__iterations

    def __setup(self):
        if self.__initialTableau is not None:
            tableau = np.asarray(self.__initialTableau, dtype=float)
            if tableau.shape != (self.__rows, self.__columns + 1):
                raise ValueError("Tableau inicial com dimensões incompatíveis")
            if self.__initialBasis is None or len(self.__initialBasis) != self.__rows:
                raise ValueError("O tableau inicial requer uma base completa")
            self.__Bvars = list(self.__initialBasis)
            if len(set(self.__Bvars)) != len(self.__Bvars) or any(
                index < 0 or index >= self.__columns for index in self.__Bvars
            ):
                raise ValueError("Base inválida para o tableau inicial")
            self.__tableau = tableau.copy()
            return

        if self.__initialBasis is None:
            if self.__rows > self.__columns:
                raise ValueError("A base identidade inicial não cabe em A")
            self.__Bvars = list(range(self.__columns - self.__rows, self.__columns))
            basisMatrix = self.__A[:, self.__Bvars]
            if not np.allclose(basisMatrix, np.eye(self.__rows), atol=1e-10, rtol=1e-10):
                raise ValueError("A base padrão exige uma matriz identidade nas últimas colunas")
            rhs = self.__b.copy()
            if np.min(rhs, initial=0.0) < -self.FEASIBILITY_TOL:
                raise ValueError("A base padrão exige lados direitos não negativos")
            rhs[np.abs(rhs) <= self.FEASIBILITY_TOL] = 0.0
            self.__tableau = np.column_stack((self.__A, rhs))
            return

        self.__Bvars = list(self.__initialBasis)
        if len(self.__Bvars) != self.__rows or len(set(self.__Bvars)) != len(self.__Bvars):
            raise ValueError("A base inicial deve conter uma variável distinta por restrição")
        if any(index < 0 or index >= self.__columns for index in self.__Bvars):
            raise ValueError("Índice inválido na base inicial")
        basisMatrix = self.__A[:, self.__Bvars]
        canonical = np.linalg.solve(basisMatrix, np.column_stack((self.__A, self.__b)))
        rhs = canonical[:, self.__columns]
        if np.min(rhs, initial=0.0) < -self.FEASIBILITY_TOL:
            raise ValueError("A base inicial não é primalmente factível")
        canonical[:, self.__columns][np.abs(rhs) <= self.FEASIBILITY_TOL] = 0.0
        self.__tableau = canonical

    def __findDualSolution(self):
        if self.__rows == 0:
            self.__dualSol = np.zeros(0)
            return
        basis = self.__A[:, self.__Bvars]
        basicCosts = self.__Z[self.__Bvars]
        self.__dualSol = np.linalg.solve(basis.T, basicCosts)

    def __findPivotColumn(self, reducedCosts):
        basic = set(self.__Bvars)
        scale = max(1.0, np.max(np.abs(self.__Z), initial=0.0))
        eligible = [
            j for j in range(self.__columns)
            if j not in basic and reducedCosts[j] < -self.REDUCED_COST_TOL * scale
        ]
        if not eligible:
            if self.__computeDual:
                self.__findDualSolution()
            return True, -1
        # Bland's rule selects the smallest eligible variable index.
        if self.__pricingRule == "dantzig":
            bestReducedCost = min(reducedCosts[j] for j in eligible)
            tieTolerance = self.REDUCED_COST_TOL * scale
            return False, next(j for j in eligible if reducedCosts[j] <= bestReducedCost + tieTolerance)
        return False, eligible[0]

    def __findPivotRow(self, pivotColumn):
        columnScale = max(1.0, np.max(np.abs(self.__tableau[:, pivotColumn]), initial=0.0))
        pivotTolerance = self.PIVOT_TOL * columnScale
        bestRatio = np.inf
        pivotRow = -1

        for row in range(self.__rows):
            coefficient = self.__tableau[row, pivotColumn]
            if coefficient <= pivotTolerance:
                continue
            rhs = self.__tableau[row, self.__columns]
            if rhs < -self.FEASIBILITY_TOL:
                raise ArithmeticError("A base deixou de ser primalmente factível")
            ratio = max(0.0, rhs) / coefficient
            tieTolerance = self.PIVOT_TOL * max(1.0, abs(ratio), abs(bestRatio) if np.isfinite(bestRatio) else 1.0)
            if ratio < bestRatio - tieTolerance:
                bestRatio = ratio
                pivotRow = row
            elif abs(ratio - bestRatio) <= tieTolerance and (
                pivotRow < 0 or self.__Bvars[row] < self.__Bvars[pivotRow]
            ):
                pivotRow = row
                bestRatio = ratio
        return pivotRow

    def __pivot(self, pivotColumn, pivotRow):
        pivotValue = self.__tableau[pivotRow, pivotColumn]
        pivotRowValues = self.__tableau[pivotRow].copy() / pivotValue
        factors = self.__tableau[:, pivotColumn].copy()
        factors[pivotRow] = 0.0
        self.__tableau -= factors[:, None] * pivotRowValues[None, :]
        self.__tableau[pivotRow] = pivotRowValues
        self.__tableau[:, pivotColumn] = 0.0
        self.__tableau[pivotRow, pivotColumn] = 1.0
        self.__tableau[pivotRow, np.abs(pivotRowValues) < 1e-14] = 0.0

    def Solver(self):
        self.__setup()
        self.__iterations = 0
        visitedBases = set()

        while True:
            basisKey = tuple(self.__Bvars)
            if basisKey in visitedBases and self.__pricingRule == "dantzig":
                # Bland's rule prevents a degenerate cycle if Dantzig pricing
                # revisits a basis in the step-optimization subproblem.
                self.__pricingRule = "bland"
            visitedBases.add(basisKey)

            basicCosts = self.__Z[self.__Bvars]
            reducedCosts = self.__Z - basicCosts @ self.__tableau[:, :self.__columns]
            optimal, pivotColumn = self.__findPivotColumn(reducedCosts)

            if optimal:
                values = np.zeros(self.__columns)
                if self.__rows:
                    values[self.__Bvars] = self.__tableau[:, self.__columns]
                values[np.abs(values) <= self.FEASIBILITY_TOL] = 0.0
                objective = float(self.__Z @ values)
                self.__status = f"Solução ótima encontrada em {self.__iterations} pivôs"
                return values, objective

            try:
                pivotRow = self.__findPivotRow(pivotColumn)
            except ArithmeticError as error:
                self.__status = f"Falha numérica: {error}"
                return None, None
            if pivotRow < 0:
                self.__status = "Problema ilimitado"
                return None, None

            self.__pivot(pivotColumn, pivotRow)
            self.__Bvars[pivotRow] = pivotColumn
            self.__iterations += 1

    @classmethod
    def SolveTwoPhase(cls, A, b, Z):
        """Solve an equality-form LP with an artificial-variable Phase I."""
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        Z = np.asarray(Z, dtype=float)
        if A.ndim != 2 or b.ndim != 1 or Z.ndim != 1:
            raise ValueError("A deve ser uma matriz e b e Z devem ser vetores")
        originalRows, originalColumns = A.shape
        if b.size != originalRows or Z.size != originalColumns:
            raise ValueError("Dimensões incompatíveis entre A, b e Z")
        if not all(np.all(np.isfinite(v)) for v in (A, b, Z)):
            raise ValueError("A, b e Z devem conter apenas valores finitos")

        if originalRows == 0:
            if np.any(Z < -cls.REDUCED_COST_TOL):
                return None, None, None, "Problema ilimitado", 0
            return np.zeros(originalColumns), 0.0, np.zeros(0), "Solução ótima", 0

        rowSigns = np.where(b < 0.0, -1.0, 1.0)
        normalizedA = rowSigns[:, None] * A
        normalizedB = rowSigns * b
        phaseOneA = np.column_stack((normalizedA, np.eye(originalRows)))
        phaseOneC = np.concatenate((np.zeros(originalColumns), np.ones(originalRows)))

        phaseOne = cls(phaseOneA, normalizedB, phaseOneC)
        phaseOneValues, phaseOneObjective = phaseOne.Solver()
        if phaseOneValues is None:
            return None, None, None, phaseOne.status, phaseOne.iterations
        if phaseOneObjective < -cls.FEASIBILITY_TOL:
            return None, None, None, "Falha numérica na Fase I", phaseOne.iterations
        if phaseOneObjective > cls.FEASIBILITY_TOL * max(1.0, np.sum(np.abs(normalizedB))):
            return None, None, None, "Problema inviável", phaseOne.iterations

        workingA = phaseOneA.copy()
        workingB = normalizedB.copy()
        rowTransform = np.eye(originalRows)
        basis = phaseOne.Bvars

        while any(variable >= originalColumns for variable in basis):
            basisMatrix = workingA[:, basis]
            try:
                canonicalTransform = np.linalg.solve(basisMatrix, np.eye(len(basis)))
            except np.linalg.LinAlgError:
                return None, None, None, "Base singular ao remover variáveis artificiais", phaseOne.iterations

            workingA = canonicalTransform @ workingA
            workingB = canonicalTransform @ workingB
            rowTransform = canonicalTransform @ rowTransform

            row = next(i for i, variable in enumerate(basis) if variable >= originalColumns)
            basicSet = set(basis)
            rowScale = max(1.0, np.max(np.abs(workingA[row, :originalColumns]), initial=0.0))
            entering = next(
                (
                    j for j in range(originalColumns)
                    if j not in basicSet and abs(workingA[row, j]) > cls.PIVOT_TOL * rowScale
                ),
                None,
            )

            if entering is not None:
                basis[row] = entering
                continue

            rhsTolerance = cls.FEASIBILITY_TOL * max(1.0, np.max(np.abs(normalizedB), initial=0.0))
            if abs(workingB[row]) > rhsTolerance:
                return None, None, None, "Problema inviável após a Fase I", phaseOne.iterations

            workingA = np.delete(workingA, row, axis=0)
            workingB = np.delete(workingB, row)
            rowTransform = np.delete(rowTransform, row, axis=0)
            basis.pop(row)

        phaseTwoA = workingA[:, :originalColumns]
        phaseTwo = cls(phaseTwoA, workingB, Z, initialBasis=basis)
        values, objective = phaseTwo.Solver()
        totalPivots = phaseOne.iterations + phaseTwo.iterations
        if values is None:
            return None, None, None, phaseTwo.status, totalPivots

        dual = np.zeros(originalRows)
        if phaseTwo.dualSol is not None:
            dual = rowSigns * (rowTransform.T @ phaseTwo.dualSol)
        return values, objective, dual, "Solução ótima", totalPivots
