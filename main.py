from pathlib import Path
import time
from InstanceData import InstanceData
from primal import Primal
from pulpSolver import *
from stoerWagner import StoerWagner


def escolher_modo(all_files):
    print("\n" + "=" * 60)
    print("               MENU DE EXECUÇÃO DE INSTÂNCIAS")
    print("=" * 60)
    print("1 - Rodar uma instância específica (detalhado + variáveis)")
    print("2 - Rodar todas as instâncias (resumido)")
    print("3 - Rodar apenas instâncias de Corte Mínimo (.min)")
    print("4 - Rodar apenas instâncias de Múltiplas Mercadorias (.max)")
    print("=" * 60)


    while True:
        opcao = input("Escolha uma opção (1-4): ").strip()

        if opcao == "1":
            print("\nInstâncias encontradas:")
            for idx, f in enumerate(all_files, 1):
                print(f"  [{idx:2d}] {f.name}")

            while True:
                escolha = input(
                    "\nDigite o número ou o nome da instância: "
                ).strip()
                if escolha.isdigit() and 1 <= int(escolha) <= len(all_files):
                    return [all_files[int(escolha) - 1]], True
                encontrados = [f for f in all_files if f.name == escolha]
                if encontrados:
                    return encontrados, True
                print("Instância inválida. Tente novamente.")

        elif opcao == "2":
            return all_files, False

        elif opcao == "3":
            filtrados = [
                f
                for f in all_files
                if f.suffix == ".min"
                or (f.suffix == ".in" and "min" in f.name.lower())
            ]
            return filtrados, False

        elif opcao == "4":
            filtrados = [
                f
                for f in all_files
                if f.suffix == ".max"
                or (f.suffix == ".in" and "mcf" in f.name.lower())
            ]
            return filtrados, False

        else:
            print("Opção inválida! Escolha um valor entre 1 e 4.")


if __name__ == "__main__":
    instances_dir = Path("instances")

    valid_extensions = {".in", ".min", ".max"}
    all_files = sorted(
        [
            f
            for f in instances_dir.iterdir()
            if f.is_file() and f.suffix in valid_extensions
        ]
    )

    if not all_files:
        print("Nenhuma instância encontrada na pasta 'instances'.")
        exit()

    files, verbose = escolher_modo(all_files)
    results = []

    for file_path in files:
        filepath_str = str(file_path)

        if verbose:
            print(f"\n\n{'#'*65}")
            print(f"# INICIANDO INSTÂNCIA: {file_path.name}")
            print(f"{'#'*65}")
            print("\n--- ETAPA 1: Leitura e Preparação de Dados ---")
        else:
            print(f"Processando {file_path.name}...", end="", flush=True)

        # 1. Leitura
        instanceData = InstanceData()
        instanceData.readInstance(filepath_str, verbose=verbose)
        is_min = instanceData.problemType == "min"

        # 2. Stoer-Wagner (apenas para corte mínimo)
        sw_cut = None
        sw_time = None
        if is_min:
            if verbose:
                print("\n--- ETAPA 2: Stoer-Wagner (Corte Global) ---")
            stoerWagner = StoerWagner(
                instanceData.edges,
                instanceData.vertexNum,
                1,
                instanceData.vertexNum,
                verbose=verbose,
            )
            sw_cut = stoerWagner.Solver()
            sw_time = stoerWagner.executionTime

        # 3. Primal-Dual Simplex
        if verbose:
            print(
                f"\n--- ETAPA {'3' if is_min else '2'}: Primal-Dual Simplex ---"
            )
        
            
        A, b, Z = instanceData.InitPL(verbose=verbose)
        primal = Primal(A, b, Z, verbose=verbose)
        optimal, optimalZ = primal.Solver()

        # 4. PuLP (com logs do CBC silenciados via msg=False)
        if verbose:
            print(f"\n--- ETAPA {'4' if is_min else '3'}: Resolução com PuLP ---")
        prob, optimalEdges = buildModelPulp(instanceData)

        start_pulp = time.perf_counter()
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        pulp_time = time.perf_counter() - start_pulp

        pulp_status = pulp.LpStatus.get(prob.status, "Desconhecido")
        pulp_obj = (
            pulp.value(prob.objective) if prob.objective is not None else 0.0
        )

        if verbose:
            print(f"[PuLP] Status: {pulp_status}")
            print(f"[PuLP] Tempo de Execução: {pulp_time:.4f}s")
            print(f"[PuLP] Valor Ótimo: {pulp_obj:.4f}")

            # Variáveis
            print("\n" + "=" * 55)
            print(">> TODAS AS VARIÁVEIS - PRIMAL SIMPLEX <<")
            print("=" * 55)
            if optimal is not None:
                for i, val in enumerate(optimal):
                    if abs(val) > 1e-5:
                        print(f"  x[{i}] = {val:.4f}")

            print("\n" + "=" * 55)
            print(">> TODAS AS VARIÁVEIS - PuLP <<")
            print("=" * 55)
            for v in prob.variables():
                if v.varValue is not None and abs(v.varValue) > 1e-5:
                    print(f"  {v.name:<25} = {v.varValue:.4f}")
        else:
            print(" OK!")

        results.append(
            {
                "instance": file_path.name,
                "type": instanceData.problemType,
                "simplex_z": (
                    f"{optimalZ:.2f}" if optimalZ is not None else "N/A"
                ),
                "simplex_t": f"{primal.executionTime:.4f}s",
                "pulp_z": f"{pulp_obj:.2f}",
                "pulp_t": f"{pulp_time:.4f}s",
                "sw_z": f"{sw_cut:.2f}" if sw_cut is not None else "-",
                "sw_t": f"{sw_time:.4f}s" if sw_time is not None else "-",
            }
        )

    # Tabela comparativa final
    print("\n\n" + "=" * 105)
    print(
        f"{'Instância':<18} | {'Tipo':<5} | {'Z Simplex':>11} | {'T. Simplex':>11} | "
        f"{'Z PuLP':>11} | {'T. PuLP':>11} | {'Z Stoer-W':>11} | {'T. Stoer-W':>11}"
    )
    print("-" * 105)

    for r in results:
        print(
            f"{r['instance']:<18} | {r['type']:<5} | {r['simplex_z']:>11} | {r['simplex_t']:>11} | "
            f"{r['pulp_z']:>11} | {r['pulp_t']:>11} | {r['sw_z']:>11} | {r['sw_t']:>11}"
        )

    print("=" * 105)