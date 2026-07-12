# Vale Desenvolver 2026 — Análise Avançada de Dados

Solução de análise preditiva desenvolvida para o **Programa Vale Desenvolver 2026**, categoria Análise Avançada de Dados.

## Objetivo

Prever eventos **Don't Go** (alerta crítico que proíbe a saída do equipamento para operação) em equipamentos de mineração pesada com pelo menos **1 hora de antecedência**, utilizando dados de telemetria e apontamentos operacionais.

## Resultados do Modelo (Conjunto de Teste — Jun/2025)

| Métrica | Valor |
|---------|-------|
| **F1-Score** | **0.6741** |
| Precision | 0.7121 |
| Recall | 0.6400 |
| ROC-AUC | 0.9924 |
| PR-AUC | 0.7018 |
| Threshold (fixado na validação) | 0.9914 |
| Features | 53 |
| Árvores (early stopping por PR-AUC) | 233 |

Split temporal estrito: Treino Jan–Abr/2025 | Validação Mai/2025 | Teste Jun/2025.
O threshold de decisão é escolhido na **validação (Mai)** e aplicado intacto ao
teste (Jun) — nunca selecionado no próprio teste. A flag do evento atual
`Is_Dont_Go` foi removida das features por ser vazamento concorrente (54 → 53).

## Dados

| Dataset | Registros | Período |
|---------|-----------|---------|
| Telemetria (sensores/alarmes) | 37.164.054 | Jan–Jun 2025 |
| Apontamentos (estado operacional) | 377.907 | Jan–Jun 2025 |

**Equipamentos:** 47 unidades nos apontamentos, das quais 35 possuem telemetria no período (cobertas pelo modelo) — Caminhões 793-D (2S/3S/4S/5S) e Escavadeiras LeTourneau L 1850

## Stack Tecnológica

| Biblioteca | Papel |
|------------|-------|
| **Polars** | Processamento dos 37M registros (Lazy API) |
| **DuckDB** | Queries SQL analíticas direto nos parquets |
| **LightGBM** | Modelo preditivo principal |
| **SHAP** | Explicabilidade do modelo |
| **Plotly** | Visualizações interativas |

## Estrutura do Projeto

```
├── Base_Dados/              # Datasets originais (não modificar)
│   ├── datasets/
│   │   ├── telemetria/      # 6 arquivos parquet mensais (Jan-Jun 2025)
│   │   └── apontamentos/    # Ciclos operacionais (377K registros)
│   ├── Dicionario_Dados.xlsx
│   └── Alarmes - Regra de Negocio.xlsx
├── notebooks/                      # 18 notebooks executados
│   │  # Base — EDA, features, modelo, insights
│   ├── 01_EDA_apontamentos.ipynb   # Análise exploratória dos apontamentos
│   ├── 02_EDA_telemetria.ipynb     # Análise dos alarmes e Don't Go (hipóteses H1–H7)
│   ├── 03_regras_negocio.ipynb     # Regras OEM de disparo do Don't Go
│   ├── 04_feature_engineering.ipynb # Feature engineering e análise do dataset Gold
│   ├── 05_modelo_preditivo.ipynb   # Modelo final, métricas e SHAP
│   ├── 06_insights_negocio.ipynb   # Baseline, impacto operacional e recomendações
│   │  # Análises avançadas (Sprints 1–12)
│   ├── 05b_modelos_alternativos.ipynb   # Comparação de 5 modelos (Sprint 1)
│   ├── 06b_analise_erros_e_custo.ipynb  # Análise de erros FP/FN + custo (Sprint 2)
│   ├── 07_segmentacao_frota.ipynb       # Performance por frota (Sprint 2)
│   ├── 08_horizonte_calibracao.ipynb    # Multi-horizonte 60/120/240min (Sprint 3)
│   ├── 09_experimento_escavadeira.ipynb # Experimento negativo controlado (Sprint 5)
│   ├── 10_calibracao_isotonica.ipynb    # Calibração isotônica, Brier −89% (Sprint 6)
│   ├── 11_ca65926_temporal.ipynb        # Cronologia de degradação do CA65926 (Sprint 7)
│   ├── 12_drift_detection.ipynb         # Detecção de drift Page-Hinkley + KS (Sprint 8)
│   ├── 13_fleet_threshold_policy.ipynb  # Threshold custo-ótimo por frota (Sprint 9)
│   ├── 14_ensemble.ipynb                # Ensemble RF+LightGBM, resultado negativo (Sprint 10)
│   ├── 15_escavadeira_fingerprint.ipynb # Fingerprint da escavadeira por lift (Sprint 11)
│   └── 16_escavadeira_gru.ipynb         # PoC modelo sequencial GRU, negativo (Sprint 12)
├── src/                            # Pipeline + um script por sprint
│   ├── ingestion.py         # Raw → Bronze (carregamento e validação)
│   ├── transformation.py    # Bronze → Silver (joins, flags pré-DG)
│   ├── features.py          # Silver → Gold (feature engineering)
│   ├── models.py            # Treinamento, avaliação, SHAP, persistência
│   ├── visualization.py     # Dashboards HTML interativos (+ snapshots PNG)
│   ├── retrain_optimized.py # Script de retreino reproduzível
│   ├── generate_report.py   # Geração automática do relatório Word
│   ├── data_inspection.py   # Inspeção inicial + controle de alterações (CM 2.1/3.1)
│   ├── shap_example_waterfall.py # Waterfall SHAP de predição individual (CM 5.3)
│   └── sprint1..12_*.py     # Análises avançadas reproduzíveis (1 arquivo por sprint)
├── outputs/
│   ├── silver/              # Dados Bronze → Silver (222 MB, 6 meses)
│   ├── gold/                # Feature matrix + modelo (383 MB + lgbm_dontgo.pkl)
│   ├── dashboards/          # 9 dashboards HTML interativos
│   ├── figures/             # Gráficos PNG (ROC, SHAP, confusão, timeline)
│   └── reports/             # Relatório final Word + model_metrics.json
├── PRD.md                   # Definição do produto e objetivos
├── TECH_SPEC.md             # Arquitetura técnica detalhada
└── pyproject.toml           # Dependências (gerenciado por uv)
```

## Guia Rápido — O Que Rodar em Cada Situação

| Quero... | Comando |
|---|---|
| Só ver o dashboard (dados/modelo já prontos) | `uv run streamlit run app.py` |
| Reprocessar tudo do zero (ETL + modelo + dashboards) | Seguir "Como Reproduzir do Zero" abaixo, passos 1–6 |
| Só regenerar os dashboards HTML | Passo 6 |
| Só retreinar o modelo | Passo 4 |

## Como Reproduzir do Zero

**Pré-requisito:** [uv](https://docs.astral.sh/uv/) instalado (`pip install uv`).

### 1. Instalar dependências

```bash
uv sync
```

### 2. Executar notebooks de análise exploratória (leitura)

Abra o Jupyter Lab e execute em ordem:

```bash
uv run jupyter lab
```

- `notebooks/01_EDA_apontamentos.ipynb` — análise dos 377K apontamentos
- `notebooks/02_EDA_telemetria.ipynb` — análise dos 37M eventos de telemetria
- `notebooks/03_regras_negocio.ipynb` — interpretação das regras OEM de Don't Go

### 3. Gerar datasets Silver e Gold (ETL)

Os arquivos já estão em `outputs/silver/` e `outputs/gold/`. Para regenerar:

```bash
# Bronze → Silver (join telemetria + apontamentos + flags pré-DG)
uv run python3 -c "
import sys; sys.path.insert(0, 'src')
from transformation import build_silver_dataset
build_silver_dataset()
"

# Silver → Gold (feature engineering: frequência, fingerprint, temporais)
uv run python3 -c "
import sys; sys.path.insert(0, 'src')
from features import build_feature_matrix
build_feature_matrix()
"
```

### 4. Treinar o modelo (opcional — modelo já salvo)

O modelo final está em `outputs/gold/lgbm_dontgo.pkl`. Para retreinar:

```bash
uv run python3 src/retrain_optimized.py
```

Parâmetros do modelo final:
- `num_leaves=255`, `scale_pos_weight=40`, `learning_rate=0.05`
- `min_child_samples=50`, `neg_sample_ratio=5`
- Treino: Jan–Abr/2025 | Validação: Mai/2025 | Teste: Jun/2025

### 5. Executar notebooks de modelagem e insights

```bash
uv run jupyter lab
```

- `notebooks/04_feature_engineering.ipynb` — análise das 54 features Gold
- `notebooks/05_modelo_preditivo.ipynb` — avalia modelo salvo, gera SHAP e gráficos (F1=0.6741)
- `notebooks/06_insights_negocio.ipynb` — baseline vs LightGBM, impacto operacional, recomendações

### 6. Gerar dashboards HTML

```bash
uv run python3 -c "
import sys; sys.path.insert(0, 'src')
from visualization import generate_all_dashboards
generate_all_dashboards()
"
```

Os dashboards são gerados em `outputs/dashboards/` e podem ser abertos diretamente no navegador.

### 7. Visualizar os resultados

Abra os dashboards HTML diretamente no navegador:

| Dashboard | Conteúdo |
|-----------|----------|
| `01_fleet_risk.html` | Ranking de risco Don't Go por equipamento |
| `02_temporal_patterns.html` | Padrões temporais (mensal, hora×dia) |
| `03_alarm_fingerprint.html` | Alarmes que mais precedem Don't Go |
| `04_risk_timeline_CA65926_jun.html` | Timeline de probabilidade (equip. mais crítico) |
| `05_shap_global.html` | Importância global das features (SHAP) |
| `06_critical_CA65926.html` | Análise detalhada do equipamento CA65926 |
| `07_baseline_comparison.html` | Comparação LightGBM vs regra estática (gerado pelo notebook 06) |
| `08_alarm_fingerprint_narrative.html` | Fingerprint narrativo com nomes dos alarmes |
| **`09_story_dashboard.html`** | **Dashboard executivo: narrativa completa do projeto** |

### 8. Executar o dashboard interativo (Streamlit)

Não precisa reprocessar nada — o dashboard lê os artefatos já salvos em `outputs/gold/` e `outputs/reports/`.

```bash
uv run streamlit run app.py
```

- Abre em `http://localhost:8501` (no WSL2, o Windows encaminha a porta automaticamente — é só abrir essa URL no navegador).
- **Primeira carga demora ~25-30s** (import do `torch`, puxado pelo `shap`) — a tela fica no esqueleto cinza enquanto isso. Depois da primeira carga fica rápido (tudo cacheado). É esperado; não é travamento.
- Para rodar em outra porta: `uv run streamlit run app.py --server.port 8502`.
- Para parar: `Ctrl+C` no terminal onde está rodando (ou `pkill -f "streamlit run app.py"` se estiver em background).

> **Nota WSL2 (importante):** rodando a partir de `/mnt/c/...`, o file watcher do
> Streamlit trava o app em disk-sleep ao tentar vigiar os ~13 mil arquivos do
> `torch` pela ponte 9p. Isso já está resolvido em `.streamlit/config.toml`
> (`fileWatcherType = "none"`) — o comando acima funciona direto. O único efeito
> colateral é não haver hot-reload ao salvar `app.py` (basta reiniciar o app).
> Para eliminar de vez a lentidão da primeira carga, mova o projeto para o
> sistema de arquivos nativo do Linux (ex: `~/projetos/`) em vez de `/mnt/c/`.

## Arquitetura: Pipeline Medallion

```
Raw (parquet)
  → Bronze: tipagem, validação de schema, snake_case       [src/ingestion.py]
  → Silver: join tel+apo, flags pré-DG, estado operacional [src/transformation.py]
  → Gold:   features (frequência, fingerprint, temporais)   [src/features.py]
  → Modelo: LightGBM + SHAP + dashboards                   [src/models.py]
```

## Diferenciais da Solução

- **Alarm Fingerprint:** presença dos 30 alarmes mais preditivos como features binárias — o "DNA do Don't Go"
- **Predição com antecedência real:** target `is_dont_go_next_60m` — previsão 1h antes; multi-horizonte mantém ROC-AUC > 0.94 (≈0,95 em 4h) em todos os horizontes (Sprint 3)
- **Pipeline Medallion:** arquitetura de nível produção com camadas Bronze → Silver → Gold
- **Explicabilidade por SHAP:** justificativa de cada previsão para o negócio
- **Validação temporal estrita:** sem vazamento de informação futuro no treino
- **Comparação formal de 5 modelos (Sprint 1):** baseline trivial, regra de negócio (F1=0.153), Logistic L1, Random Forest e LightGBM — ganho de **+341% de F1** do ML sobre a heurística operacional (LightGBM e Random Forest empatam em F1=0.674; LightGBM mantido como principal pela explicabilidade SHAP)
- **Decisão por custo, não só por F1 (Sprint 2):** FN custa 62× mais que FP; medindo o custo **por episódio de Don't Go** (1 episódio = 1 parada física, não por linha de telemetria), o threshold custo-ótimo economiza **~R$26 milhões no mês de teste** capturando ~99,9% das paradas reais (5647/5652). A versão por-linha (~R$285 M) superestimava e foi descontinuada
- **Calibração isotônica (Sprint 6):** Brier −89% — probabilidades absolutas confiáveis para precificação de manutenção
- **Política de threshold por frota (Sprint 9):** F1 agregado +17% cortando ~46 mil falsos positivos; tabela de decisão deployável
- **Detecção de drift (Sprint 8):** Page-Hinkley + KS para disparar re-treino em produção
- **Rigor com resultados negativos (Sprints 10 e 12):** ensemble e modelo sequencial GRU testados e documentados como não-superiores — provam que o teto de F1≈0.69 é imposto pelo conjunto de features, não pelo algoritmo
- **Diagnóstico da limitação (Sprints 5, 11, 12):** o ponto cego da escavadeira LeTourneau é triangulado a uma causa estrutural (não-estacionariedade), com os alarmes de alto lift já identificados para correção
- **Dashboard executivo:** `09_story_dashboard.html` — narrativa completa do projeto para tomadores de decisão

## Requisitos de Hardware

O pipeline foi otimizado para rodar em máquinas com **16 GB de RAM**:

- Processamento mês a mês — nunca carrega os 37M registros completos em memória
- Pico estimado: ~4–6 GB durante feature engineering
- `build_silver_dataset()` e `build_feature_matrix()` liberam RAM após cada mês

## Status do Projeto

- [x] Fase 1 — EDA (notebooks 01, 02, 03)
- [x] Fase 2 — Pipeline ETL Silver (`src/ingestion.py`, `src/transformation.py`) → `outputs/silver/`
- [x] Fase 3 — Feature Engineering Gold (`src/features.py`) → `outputs/gold/`
- [x] Fase 4 — Modelagem ML (`src/models.py`, `src/retrain_optimized.py`) → F1=0.6741, ROC-AUC=0.9924, PR-AUC=0.7018 (233 árvores, threshold da validação, sem vazamento `Is_Dont_Go`)
- [x] Fase 5 — Visualizações e Relatório Final (`src/visualization.py`, `outputs/dashboards/`, `outputs/reports/`)
- [x] Fase 6 — Insights de Negócio (`notebooks/06_insights_negocio.ipynb`, `09_story_dashboard.html`)
- [x] **Sprints 1–12 — Análises avançadas** (`src/sprint1..12_*.py`, notebooks 05b–16): comparação de 5 modelos, análise de erros e custo, multi-horizonte, calibração isotônica, cronologia do CA65926, detecção de drift, política de threshold por frota, ensemble e modelo sequencial (resultados negativos documentados), diagnóstico do ponto cego da escavadeira

### Reproduzir as análises avançadas

```bash
# Cada sprint é um script independente e reproduzível
uv run python3 src/sprint1_compare_models.py        # comparação de 5 modelos
uv run python3 src/sprint9_fleet_threshold_policy.py # threshold por frota
# ... sprint2 a sprint12 seguem o mesmo padrão

# Sprint 12 (modelo sequencial) requer PyTorch CPU — dependência opcional:
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv run python3 src/sprint12_escavadeira_gru.py
```

## Testes

Uma suíte enxuta cobre as funções puras de apoio (formatação BR, detector de
drift Page-Hinkley, busca de threshold ótimo, baselines e seleção de features):

```bash
uv run pytest -q
```

## Licença e Confidencialidade

**Proprietário e Confidencial.** Os dados são reais e pertencem à Vale S.A.; o
material é de uso restrito ao Programa Vale Desenvolver 2026. Deploy público é
**proibido** — a demonstração deve ser local (`uv run streamlit run app.py`).
Ver [`LICENSE`](LICENSE).
