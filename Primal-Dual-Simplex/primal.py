import time

import numpy as np

import simplex


class Primal:
    """Primal-dual simplex for min c'x subject to Ax=b and x>=0."""

    # Match the active-set tolerance used by the original primal-dual method;
    # treating near-tight variables as inactive causes repeated tiny steps.
    REDUCED_COST_TOL = 1e-4
    FEASIBILITY_TOL = 1e-7
    DIRECTION_TOL = 1e-10

    def __init__(self, A, b, Z, initialDual=None):
        self.__originalA = np.asarray(A, dtype=float)
        self.__originalb = np.asarray(b, dtype=float)
        self.__Z = np.asarray(Z, dtype=float)

        if self.__originalA.ndim != 2:
            raise ValueError("A deve ser uma matriz bidimensional")
        if self.__originalb.ndim != 1 or self.__Z.ndim != 1:
            raise ValueError("b e Z devem ser vetores")
        if self.__originalA.shape != (self.__originalb.size, self.__Z.size):
            raise ValueError("Dimensões incompatíveis entre A, b e Z")
        if not all(np.all(np.isfinite(v)) for v in (self.__originalA, self.__originalb, self.__Z)):
            raise ValueError("A, b e Z devem conter apenas valores finitos")

        self.__rows, self.__columns = self.__originalA.shape
        self.__providedInitialDual = (
            None if initialDual is None else np.asarray(initialDual, dtype=float)
        )
        if self.__providedInitialDual is not None and self.__providedInitialDual.shape != (self.__rows,):
            raise ValueError("A solução dual inicial deve ter um valor por restrição")
        self.__rowSigns = np.where(self.__originalb < 0, -1.0, 1.0)
        self.__A = self.__rowSigns[:, None] * self.__originalA
        self.__b = self.__rowSigns * self.__originalb

        self.__y = None
        self.__solution = None
        self.__dualSolution = None
        self.__objective = None
        self.__status = ""
        self.__executionTime = 0.0
        self.__outerIterations = 0
        self.__simplexPivots = 0

    @property
    def status(self):
        return self.__status

    @property
    def executionTime(self):
        return self.__executionTime

    @property
    def solution(self):
        return None if self.__solution is None else self.__solution.copy()

    @property
    def dualSolution(self):
        return None if self.__dualSolution is None else self.__dualSolution.copy()

    @property
    def objective(self):
        return self.__objective

    @property
    def outerIterations(self):
        return self.__outerIterations

    @property
    def simplexPivots(self):
        return self.__simplexPivots

    def __fail(self, message, timeBegin):
        self.__status = message
        self.__executionTime = time.perf_counter() - timeBegin
        return None, None

    def __initialDualFeasibleSolution(self):
        if self.__providedInitialDual is not None:
            self.__y = self.__rowSigns * self.__providedInitialDual
            reducedCosts = self.__Z - self.__A.T @ self.__y
            if np.min(reducedCosts, initial=0.0) < -self.REDUCED_COST_TOL:
                return False, "a solução dual inicial fornecida é inviável"
            return True, ""

        if np.all(self.__Z >= 0.0):
            # This is already a dual-feasible point for the minimization LP.
            self.__y = np.zeros(self.__rows)
            return True, ""

        # Represent free dual variables as y+ - y-. The added nonnegative
        # slack variables turn A'y <= c into an equality feasibility problem.
        dualFeasibilityA = np.hstack(
            (
                self.__A.T,
                -self.__A.T,
                np.eye(self.__columns),
            )
        )
        dualFeasibilityB = self.__Z.copy()
        dualFeasibilityC = np.zeros(2 * self.__rows + self.__columns)

        values, _, _, status, pivots = simplex.Simplex.SolveTwoPhase(
            dualFeasibilityA,
            dualFeasibilityB,
            dualFeasibilityC,
        )
        self.__simplexPivots += pivots
        if values is None:
            return False, f"Não foi possível obter uma solução dual factível: {status}"

        self.__y = values[:self.__rows] - values[self.__rows:2 * self.__rows]
        reducedCosts = self.__Z - self.__A.T @ self.__y
        tolerance = self.REDUCED_COST_TOL
        if np.min(reducedCosts, initial=0.0) < -tolerance:
            return False, "A inicialização produziu uma solução dual numericamente inviável"
        return True, ""

    def __activeSet(self):
        reducedCosts = self.__Z - self.__A.T @ self.__y
        tolerance = self.REDUCED_COST_TOL
        if np.min(reducedCosts, initial=0.0) < -10.0 * tolerance:
            raise ArithmeticError("A solução dual perdeu viabilidade numérica")
        return np.flatnonzero(reducedCosts <= tolerance)

    def __buildRestrictedProblem(self, active):
        identity = np.eye(self.__rows)
        # Two-sided artificials keep the Phase-I dual direction bounded in
        # both signs while preserving feasibility iff the restricted primal
        # itself is feasible.
        restrictedA = np.column_stack((self.__A[:, active], -identity, identity))
        restrictedC = np.concatenate((np.zeros(active.size), np.ones(2 * self.__rows)))
        return restrictedA, self.__b.copy(), restrictedC

    def __maximizeDualStep(self, active, reducedCosts, phaseOneValue):
        """Choose a long feasible dual step through a compact auxiliary LP."""
        activeMask = np.zeros(self.__columns, dtype=bool)
        activeMask[active] = True
        inactive = np.flatnonzero(~activeMask)

        # Let delta = theta * direction. The constraints on the direction
        # become homogeneous in delta, and theta is maximized directly.
        # Represent each free delta_i as u_i - theta. Then 0 <= u_i <= 2theta
        # is equivalent to -theta <= delta_i <= theta.
        decisionCount = self.__rows + 1
        thetaIndex = self.__rows
        inequalities = []
        rightHandSides = []

        for variable in active:
            column = self.__A[:, variable]
            inequalities.append(np.concatenate((column, [-np.sum(column)])))
            rightHandSides.append(0.0)

        for row in range(self.__rows):
            upperBound = np.zeros(decisionCount)
            upperBound[row] = 1.0
            upperBound[thetaIndex] = -2.0
            inequalities.append(upperBound)
            rightHandSides.append(0.0)

        for variable in inactive:
            column = self.__A[:, variable]
            inequalities.append(np.concatenate((column, [-np.sum(column)])))
            rightHandSides.append(max(0.0, float(reducedCosts[variable])))

        equality = np.concatenate((self.__b, [-phaseOneValue - np.sum(self.__b)]))
        inequalityCount = len(inequalities)
        stepA = np.zeros((inequalityCount + 1, decisionCount + inequalityCount))
        stepB = np.concatenate((np.asarray(rightHandSides), [0.0]))
        stepA[:inequalityCount, :decisionCount] = np.asarray(inequalities)
        stepA[:inequalityCount, decisionCount:] = np.eye(inequalityCount)
        stepA[inequalityCount, :decisionCount] = equality
        stepC = np.zeros(decisionCount + inequalityCount)
        stepC[thetaIndex] = -1.0

        # The slack variables plus theta form a known feasible basis. Build
        # its tableau directly to avoid factoring this large basis matrix.
        initialBasis = list(range(decisionCount, decisionCount + inequalityCount))
        initialBasis.append(thetaIndex)
        rawTableau = np.column_stack((stepA, stepB))
        initialTableau = rawTableau.copy()
        equalityRow = rawTableau[inequalityCount].copy()
        equalityPivot = -stepA[inequalityCount, thetaIndex]
        if equalityPivot <= self.DIRECTION_TOL:
            return None, None, "a normalização da direção dual é degenerada"
        thetaColumn = stepA[:inequalityCount, thetaIndex]
        initialTableau[:inequalityCount] += (
            (thetaColumn / equalityPivot)[:, None] * equalityRow[None, :]
        )
        initialTableau[inequalityCount] = -equalityRow / equalityPivot
        stepSimplex = simplex.Simplex(
            stepA,
            stepB,
            stepC,
            initialBasis=initialBasis,
            initialTableau=initialTableau,
            computeDual=False,
        )
        values, _ = stepSimplex.Solver()
        self.__simplexPivots += stepSimplex.iterations
        if values is None:
            return None, None, f"subproblema de passo dual: {stepSimplex.status}"

        theta = float(values[thetaIndex])
        if theta <= self.DIRECTION_TOL:
            return None, theta, "o subproblema não encontrou passo dual positivo"
        return values[:self.__rows] - theta, theta, ""

    def __maximumFeasibleDualStep(self, active, reducedCosts, direction):
        """Find the blocking ratio for a feasible primal-dual direction."""
        activeMask = np.zeros(self.__columns, dtype=bool)
        activeMask[active] = True
        inactive = np.flatnonzero(~activeMask)
        if inactive.size == 0:
            return None, "não há variáveis fora do conjunto ativo para limitar o passo"

        directionalCosts = self.__A[:, inactive].T @ direction
        scale = max(1.0, np.max(np.abs(directionalCosts), initial=0.0))
        blockers = directionalCosts > self.DIRECTION_TOL * scale
        if not np.any(blockers):
            return None, "a direção dual não encontra uma restrição limitante"

        ratios = np.maximum(reducedCosts[inactive][blockers], 0.0) / directionalCosts[blockers]
        step = float(np.min(ratios))
        if not np.isfinite(step) or step <= self.DIRECTION_TOL:
            return None, "o passo dual máximo é nulo ou numericamente inválido"
        return step, ""

    def __finish(self, active, restrictedValues, timeBegin):
        x = np.zeros(self.__columns, dtype=float)
        x[active] = restrictedValues[:active.size]
        artificials = restrictedValues[active.size:]

        scale = max(1.0, np.max(np.abs(self.__b), initial=0.0), np.max(np.abs(x), initial=0.0))
        if np.max(np.abs(artificials), initial=0.0) > self.FEASIBILITY_TOL * scale:
            return False

        primalResidual = np.max(np.abs(self.__A @ x - self.__b), initial=0.0)
        primalNegativity = max(0.0, -np.min(x, initial=0.0))
        reducedCosts = self.__Z - self.__A.T @ self.__y
        dualViolation = max(0.0, -np.min(reducedCosts, initial=0.0))
        primalObjective = float(self.__Z @ x)
        dualObjective = float(self.__b @ self.__y)
        optimalityGap = primalObjective - dualObjective
        objectiveScale = max(1.0, abs(primalObjective), abs(dualObjective))
        feasibilityTolerance = self.FEASIBILITY_TOL * max(scale, objectiveScale)
        optimalityTolerance = self.FEASIBILITY_TOL * max(
            1.0,
            float(np.sum(np.abs(x))),
            objectiveScale,
        )

        if primalResidual > feasibilityTolerance:
            self.__fail(f"Falha numérica: resíduo primal {primalResidual:.3g}", timeBegin)
            return True
        if primalNegativity > feasibilityTolerance:
            self.__fail(f"Falha numérica: variável primal negativa ({primalNegativity:.3g})", timeBegin)
            return True
        if dualViolation > feasibilityTolerance:
            self.__fail(f"Falha numérica: violação dual {dualViolation:.3g}", timeBegin)
            return True
        if optimalityGap < -optimalityTolerance or optimalityGap > optimalityTolerance:
            self.__fail(f"Falha numérica: diferença primal-dual {optimalityGap:.3g}", timeBegin)
            return True

        x[np.abs(x) <= self.FEASIBILITY_TOL] = 0.0
        self.__solution = x
        self.__dualSolution = self.__rowSigns * self.__y
        self.__objective = float(self.__Z @ x)
        self.__status = (
            "Solução ótima encontrada pelo primal-dual simplex em "
            f"{self.__outerIterations} iterações e {self.__simplexPivots} pivôs"
        )
        self.__executionTime = time.perf_counter() - timeBegin
        return True

    def Solver(self):
        timeBegin = time.perf_counter()

        if self.__rows == 0:
            if np.any(self.__Z < -self.REDUCED_COST_TOL):
                return self.__fail("Problema ilimitado: não há restrições e existe custo negativo", timeBegin)
            self.__solution = np.zeros(self.__columns)
            self.__dualSolution = np.zeros(0)
            self.__objective = 0.0
            self.__status = "Solução ótima encontrada pelo primal-dual simplex (problema sem restrições)"
            self.__executionTime = time.perf_counter() - timeBegin
            return self.__solution.copy(), self.__objective

        initialized, message = self.__initialDualFeasibleSolution()
        if not initialized:
            return self.__fail(
                f"Inicialização primal-dual falhou; o problema pode ser ilimitado ou inviável: {message}",
                timeBegin,
            )

        while True:
            self.__outerIterations += 1
            try:
                active = self.__activeSet()
            except ArithmeticError as error:
                return self.__fail(f"Falha no primal-dual simplex: {error}", timeBegin)

            restrictedA, restrictedB, restrictedC = self.__buildRestrictedProblem(active)
            reportProgress = self.__outerIterations <= 10 or self.__outerIterations % 25 == 0
            # if reportProgress:                   #muda aqui se quiser ver as interações!!!
            #     print(
            #         f"Primal-dual iteração {self.__outerIterations}: "
            #         f"|J|={active.size}/{self.__columns}, "
            #         f"subproblema={restrictedA.shape[0]}x{restrictedA.shape[1]}",
            #         flush=True,
            #     )
            restrictedStart = time.perf_counter()
            pricingRule = "dantzig" if np.any(self.__Z < 0.0) else "bland"
            restrictedSimplex = simplex.Simplex(
                restrictedA,
                restrictedB,
                restrictedC,
                pricingRule=pricingRule,
            )
            restrictedValues, _ = restrictedSimplex.Solver()
            restrictedElapsed = time.perf_counter() - restrictedStart
            self.__simplexPivots += restrictedSimplex.iterations

            if restrictedValues is None:
                return self.__fail(
                    f"O subproblema primal-dual falhou: {restrictedSimplex.status}",
                    timeBegin,
                )

            # if reportProgress:                #muda aqui se quiser ver a conclusao final!!!
            #     artificials = restrictedValues[active.size:]
            #     print(
            #         f"  Subproblema concluído: {restrictedSimplex.iterations} pivôs; "
            #         f"artificiais soma={np.sum(artificials):.6g}; "
            #         f"tempo={restrictedElapsed:.2f}s; calculando passo dual",
            #         flush=True,
            #     )

            if self.__finish(active, restrictedValues, timeBegin):
                if self.__solution is not None:
                    return self.__solution.copy(), self.__objective
                return None, None

            direction = restrictedSimplex.dualSol
            if direction is None or not np.all(np.isfinite(direction)):
                return self.__fail("O subproblema não forneceu uma direção dual válida", timeBegin)
            reducedCosts = self.__Z - self.__A.T @ self.__y
            oldDualObjective = float(self.__b @ self.__y)
            phaseOneValue = float(self.__b @ direction)
            if phaseOneValue <= self.FEASIBILITY_TOL:
                return self.__fail(
                    "O subproblema inviável não produziu melhoria dual positiva",
                    timeBegin,
                )

            if np.all(self.__Z >= 0.0):
                directionStep, theta, stepStatus = self.__maximizeDualStep(
                    active,
                    reducedCosts,
                    phaseOneValue,
                )
                if directionStep is None:
                    return self.__fail(f"Não foi possível avançar no dual: {stepStatus}", timeBegin)
            else:
                theta, stepStatus = self.__maximumFeasibleDualStep(
                    active,
                    reducedCosts,
                    direction,
                )
                if theta is None:
                    return self.__fail(f"Não foi possível avançar no dual: {stepStatus}", timeBegin)
                directionStep = theta * direction
            self.__y = self.__y + directionStep
            newReducedCosts = self.__Z - self.__A.T @ self.__y
            if np.min(newReducedCosts, initial=0.0) < -1e-6:
                return self.__fail("O passo dual ultrapassou a região dual factível", timeBegin)

            newDualObjective = float(self.__b @ self.__y)
            improvement = newDualObjective - oldDualObjective
            progressScale = max(1.0, abs(oldDualObjective), abs(newDualObjective))
            if improvement <= np.finfo(float).eps * progressScale:
                return self.__fail(
                    "O método primal-dual deixou de progredir numericamente; nenhuma solução alternativa foi usada",
                    timeBegin,
                )

            if self.__outerIterations <= 10 or self.__outerIterations % 25 == 0:
                print(
                    f"  Passo dual aplicado: theta={theta:.6g}, "
                    f"dual={newDualObjective:.8g}, "
                    f"pivôs acumulados={self.__simplexPivots}",
                    flush=True,
                )
