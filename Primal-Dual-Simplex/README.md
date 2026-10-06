# Primal-Dual Simplex e Otimização em Redes

Implementação em Python do Simplex primal-dual para corte mínimo s-t e fluxo máximo com múltiplas mercadorias. O programa também mantém e executa a implementação original de Stoer-Wagner, exibindo seu resultado separadamente do corte s-t.

## Instalação

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

O extra `cbc` instala o solver COIN-OR CBC usado na comparação com PuLP.

## Execução

```powershell
python main.py
python main.py instances\instance2.min
python main.py instances\mc_instance2.max
```

Sem argumento, o programa executa `instances/instance2.min`. A saída inclui o valor ótimo, as variáveis primais e duais não nulas, e a comparação com PuLP. Para instâncias `.min`, também mostra o corte global de Stoer-Wagner, que pode ter valor diferente do corte s-t.
