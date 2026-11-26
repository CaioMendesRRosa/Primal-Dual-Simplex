# Primal-Dual Simplex & Otimização em Redes

Implementação em Python do método Simplex e do algoritmo Primal-Dual aplicados a dois problemas de otimização em redes:

- **Corte mínimo** (instâncias `.min`), com o algoritmo de Stoer-Wagner como referência;
- **Fluxo máximo com múltiplas mercadorias** (instâncias `.max`).

Os resultados são comparados com o solver [PuLP](https://coin-or.github.io/pulp/), incluindo o tempo de execução de cada método.

## Estrutura

| Arquivo | Conteúdo |
|---------|----------|
| `main.py` | Menu de execução das instâncias |
| `simplex.py` | Método Simplex (tableau) |
| `primal.py` | Algoritmo Primal-Dual |
| `stoerWagner.py` | Algoritmo de Stoer-Wagner para corte mínimo |
| `pulpSolver.py` | Modelos equivalentes no PuLP, usados na comparação |
| `InstanceData.py` | Leitura das instâncias |
| `instances/` | Instâncias de teste (`.in`, `.min`, `.max`) |

## Início Rápido

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows (PowerShell)
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
python main.py
```

O menu permite rodar uma instância específica (com as variáveis detalhadas), todas as instâncias, ou apenas as de corte mínimo ou de múltiplas mercadorias.
