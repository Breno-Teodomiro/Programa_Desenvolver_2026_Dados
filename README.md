# Vale Desenvolver 2026 — Análise Avançada de Dados

Solução de análise preditiva desenvolvida para o **Programa Vale Desenvolver 2026**, categoria Análise Avançada de Dados.

## Objetivo

Prever eventos **Don't Go** (alerta crítico que proíbe a saída do equipamento para operação) em equipamentos de mineração pesada com pelo menos **1 hora de antecedência**, utilizando dados de telemetria e apontamentos operacionais.

## Resultados do Modelo (Conjunto de Teste — Jun/2025)

| Métrica | Valor |
|---------|-------|
| **F1-Score** | **0.6886** |
| Precision | 0.7581 |
| Recall | 0.6309 |
| ROC-AUC | 0.9923 |
| PR-AUC | 0.6739 |
| Threshold ótimo | 0.9324 |
| Features | 54 |

Split temporal estrito: Treino Jan–Abr/2025 | Validação Mai/2025 | Teste Jun/2025

## Dados

| Dataset | Registros | Período |
|---------|-----------|---------|
| Telemetria (sensores/alarmes) | 37.164.054 | Jan–Jun 2025 |
| Apontamentos (estado operacional) | 377.907 | Jan–Jun 2025 |

**Equipamentos:** 47 unidades — Caminhões 793-D (2S/3S/4S/5S) e Escavadeiras LeTourneau L 1850

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
├── notebooks/
│   ├── 01_EDA_apontamentos.ipynb   # Análise exploratória dos apontamentos
│   ├── 02_EDA_telemetria.ipynb     # Análise dos alarmes e Don't Go
│   ├── 03_regras_negocio.ipynb     # Regras OEM de disparo do Don't Go
│   ├── 04_feature_engineering.ipynb # Feature engineering e análise do dataset Gold
│   └── 05_modelo_preditivo.ipynb   # Modelo final, métricas e SHAP
├── src/
│   ├── ingestion.py         # Raw → Bronze (carregamento e validação)
│   ├── transformation.py    # Bronze → Silver (joins, flags pré-DG)
│   ├── features.py          # Silver → Gold (feature engineering)
│   ├── models.py            # Treinamento, avaliação, SHAP, persistência
│   ├── visualization.py     # Dashboards HTML interativos
│   ├── retrain_optimized.py # Script de retreino reproduzível
│   └── generate_report.py   # Geração automática do relatório Word
├── outputs/
│   ├── silver/              # Dados Bronze → Silver (222 MB, 6 meses)
│   ├── gold/                # Feature matrix + modelo (383 MB + lgbm_dontgo.pkl)
│   ├── dashboards/          # 6 dashboards HTML interativos
│   ├── figures/             # Gráficos PNG (ROC, SHAP, confusão, timeline)
│   └── reports/             # Relatório final Word + model_metrics.json
├── PRD.md                   # Definição do produto e objetivos
├── TECH_SPEC.md             # Arquitetura técnica detalhada
└── pyproject.toml           # Dependências (gerenciado por uv)
```

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

### 5. Executar notebook de modelagem e visualizações

```bash
# Notebook 04: análise das features Gold
# Notebook 05: avalia modelo salvo, gera SHAP e gráficos
uv run jupyter lab
```

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
- **Predição com antecedência real:** target `is_dont_go_next_60m` — previsão 1h antes
- **Pipeline Medallion:** arquitetura de nível produção com camadas Bronze → Silver → Gold
- **Explicabilidade por SHAP:** justificativa de cada previsão para o negócio
- **Validação temporal estrita:** sem vazamento de informação futuro no treino

## Requisitos de Hardware

O pipeline foi otimizado para rodar em máquinas com **16 GB de RAM**:

- Processamento mês a mês — nunca carrega os 37M registros completos em memória
- Pico estimado: ~4–6 GB durante feature engineering
- `build_silver_dataset()` e `build_feature_matrix()` liberam RAM após cada mês

## Status do Projeto

- [x] Fase 1 — EDA (notebooks 01, 02, 03)
- [x] Fase 2 — Pipeline ETL Silver (`src/ingestion.py`, `src/transformation.py`) → `outputs/silver/`
- [x] Fase 3 — Feature Engineering Gold (`src/features.py`) → `outputs/gold/`
- [x] Fase 4 — Modelagem ML (`src/models.py`, `src/retrain_optimized.py`) → F1=0.6886, ROC-AUC=0.9923
- [x] Fase 5 — Visualizações e Relatório Final (`src/visualization.py`, `outputs/dashboards/`, `outputs/reports/`)
