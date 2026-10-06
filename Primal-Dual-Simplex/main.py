import argparse
import pulp

from InstanceData import InstanceData
from primal import Primal
from pulpSolver import buildModelPulp
from stoerWagner import StoerWagner


DEFAULT_INSTANCE = "instances/instance2.min"


def _print_nonzero(title, names, values, tolerance=1e-7):
    print(title)
    nonzero = [
        (names[i] if i < len(names) else f"v_{i + 1}", float(value))
        for i, value in enumerate(values)
        if abs(float(value)) > tolerance
    ]
    if not nonzero:
        print("  (nenhuma)")
        return
    for name, value in nonzero:
        print(f"  {name} = {value:.8g}")


def _pulpSolver():
    # PuLP 4 uses COIN_CMD; older releases may also expose PULP_CBC_CMD.
    for solverName in ("COIN_CMD", "PULP_CBC_CMD"):
        solverClass = getattr(pulp, solverName, None)
        if solverClass is None:
            continue
        solver = solverClass(msg=False)
        if solver.available():
            return solver, solverName
    raise RuntimeError(
        "nenhum solver está disponível. Instale o CBC com "
        "python -m pip install 'pulp[cbc]' e execute novamente"
    )


def _solveWithPulp(instanceData, primalObjective):
    print("\n----- Solução PuLP (COIN-OR) -----")
    try:
        solver, solverName = _pulpSolver()
        problem, pulpVariables = buildModelPulp(instanceData)
        stats = problem.solve(solver)

        if hasattr(stats, "status_str"):
            status = stats.status_str
            isOptimal = status == "Optimal"
            hasSolution = bool(stats.has_solution)
            objective = stats.objective
        else:
            status = pulp.LpStatus.get(problem.status, str(problem.status))
            isOptimal = problem.status == pulp.LpStatusOptimal
            hasSolution = isOptimal or getattr(problem, "sol_status", None) == getattr(
                pulp, "LpSolutionIntegerFeasible", object()
            )
            objective = pulp.value(problem.objective) if hasSolution else None

        print(f"Solver: {solverName}")
        print(f"Status: {status}")
        if hasSolution and objective is not None:
            print(f"Valor ótimo: {float(objective):.8g}")
        else:
            print("O solver não retornou uma solução viável.")
            return

        if instanceData.problemType == "min":
            selectedEdges = [
                instanceData.edges[i]
                for i, variable in enumerate(pulpVariables)
                if variable.value() is not None and variable.value() > 0.5
            ]
            print(f"Arestas do corte: {selectedEdges}")

        if isOptimal and primalObjective is not None:
            difference = abs(float(objective) - float(primalObjective))
            print(f"Diferença para o primal-dual: {difference:.3g}")
            if difference <= 1e-6 * max(1.0, abs(float(objective)), abs(float(primalObjective))):
                print("Comparação: os valores ótimos coincidem.")
            else:
                print("Comparação: divergência entre as soluções.")
        elif not isOptimal:
            print("Comparação inconclusiva: PuLP não provou otimalidade.")
    except Exception as error:
        print(f"Comparação não realizada ({type(error).__name__}): {error}")


def main():
    parser = argparse.ArgumentParser(description="Resolve instâncias do trabalho de otimização em grafos.")
    parser.add_argument(
        "instance",
        nargs="?",
        default=DEFAULT_INSTANCE,
        help=f"caminho da instância (padrão: {DEFAULT_INSTANCE})",
    )
    args = parser.parse_args()

    instanceData = InstanceData()
    instanceData.readInstance(args.instance)
    print(
        "Problema de Corte Mínimo s-t"
        if instanceData.problemType == "min"
        else "Problema de Fluxo Máximo com Múltiplas Mercadorias"
    )
    print(f"Vértices: {instanceData.vertexNum}")
    print(f"Arestas: {instanceData.edgesNum}")

    if instanceData.problemType == "min":
        stoerWagner = StoerWagner(
            instanceData.edges,
            instanceData.vertexNum,
            instanceData.source,
            instanceData.sink,
        )
        globalCut = stoerWagner.Solver()
        print("\n----- Corte mínimo global (Stoer-Wagner) -----")
        print(f"Tempo de execução: {stoerWagner.executionTime:.3f}s")
        print(f"Valor retornado: {globalCut:.8g}")

    print("\n\n")

    A, b, Z = instanceData.InitPL()
    primal = Primal(A, b, Z, initialDual=instanceData.InitialDualFeasible())
    optimal, optimalZ = primal.Solver()

    print("\n----- Solução pelo Simplex primal-dual -----")
    print(primal.status)
    print(f"Tempo de execução: {primal.executionTime:.3f}s")

    if optimal is None:
        print("Solução ótima indisponível.")
    else:
        # The max-flow model is stored internally as min(-profit).
        objectiveSign = 1.0 if instanceData.problemType == "min" else -1.0
        reportedObjective = objectiveSign * float(optimalZ)
        reportedDual = objectiveSign * primal.dualSolution
        print(f"Valor ótimo: {reportedObjective:.8g}")
        dualObjective = sum(
            float(rhs) * float(multiplier)
            for rhs, multiplier in zip(b, primal.dualSolution)
        )
        reportedDualObjective = objectiveSign * dualObjective
        print(f"Valor dual: {reportedDualObjective:.8g}")
        print(f"Gap primal-dual: {abs(reportedObjective - reportedDualObjective):.3g}")
        _print_nonzero("Variáveis primais não nulas:", instanceData.variableNames, optimal)
        _print_nonzero("Variáveis duais não nulas:", instanceData.constraintNames, reportedDual)

        if instanceData.problemType == "min":
            edgeStart = instanceData.vertexNum
            optimalEdges = [
                instanceData.edges[i]
                for i in range(instanceData.edgesNum)
                if optimal[edgeStart + i] > 0.5
            ]
            print(f"Arestas do corte s-t: {optimalEdges}")

    _solveWithPulp(
        instanceData,
        reportedObjective if optimal is not None else None,
    )


if __name__ == "__main__":
    main()
