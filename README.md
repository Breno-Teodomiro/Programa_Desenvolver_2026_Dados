# Vale Desenvolver 2026 — Análise Avançada de Dados

Solução de análise preditiva desenvolvida para o **Programa Vale Desenvolver 2026**, categoria Análise Avançada de Dados.

## Objetivo

Prever eventos **Don't Go** (alerta crítico que proíbe a saída do equipamento para operação) em equipamentos de mineração pesada com pelo menos **1 hora de antecedência**, utilizando dados de telemetria e apontamentos operacionais.

## Dados

| Dataset | Registros | Período |
|---------|-----------|---------|
| Telemetria (sensores/alarmes) | 37.164.054 | Jan–Jun 2025 |
| Apontamentos (estado operacional) | 377.907 | Jan–Jun 2025 |

**Equipamentos:** 47 unidades — Caminhões 793-D (2S/3S/4S/5S) e Escavadeiras LeTourneau L 1850  
**Localidade:** Mina de Itabira

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
│   │   ├── telemetria/      # 6 arquivos parquet mensais
│   │   └── apontamentos/    # Ciclos operacionais
│   ├── Dicionario_Dados.xlsx
│   └── Alarmes - Regra de Negocio.xlsx
├── notebooks/
│   ├── 01_EDA_apontamentos.ipynb   # Análise exploratória dos apontamentos
│   ├── 02_EDA_telemetria.ipynb     # Análise dos alarmes e Don't Go
│   ├── 03_regras_negocio.ipynb     # Regras OEM de disparo do Don't Go
│   ├── 04_feature_engineering.ipynb
│   └── 05_modelo_preditivo.ipynb
├── src/
│   ├── ingestion.py         # Raw → Bronze
│   ├── transformation.py    # Bronze → Silver
│   ├── features.py          # Silver → Gold (feature engineering)
│   ├── models.py            # Treinamento e avaliação
│   └── visualization.py     # Visualizações reutilizáveis
├── outputs/
│   ├── figures/             # Gráficos gerados
│   └── reports/             # Relatório final
├── PRD.md                   # Definição do produto e objetivos
├── TECH_SPEC.md             # Arquitetura técnica
└── pyproject.toml           # Dependências (gerenciado por uv)
```

## Como Executar

**Pré-requisito:** [uv](https://docs.astral.sh/uv/) instalado.

```bash
# Instalar dependências
uv sync

# Iniciar Jupyter Lab
uv run jupyter lab
```

## Requisitos de Hardware

O pipeline foi otimizado para rodar em máquinas com **16 GB de RAM**:

- Processamento mês a mês — nunca carrega os 37M registros completos em memória
- Pico estimado: ~1–1.5 GB por mês durante feature engineering
- `build_silver_dataset()` libera cada mês da RAM após salvar em parquet
- `build_feature_matrix()` usa DuckDB para stats globais + processa um mês por vez com overlap de 4h para rolling windows corretas

## Diferenciais da Solução

- **Alarm Fingerprint:** identifica a sequência de alarmes que precede cada evento Don't Go
- **Predição com antecedência real:** prever Don't Go com 1–4h de antecedência
- **Pipeline Medallion:** arquitetura Raw → Bronze → Silver → Gold de nível produção
- **Explicabilidade:** SHAP values para justificar cada previsão ao negócio

## Fases de Desenvolvimento

- [x] Fase 1 — EDA (notebooks 01, 02, 03)
- [x] Fase 2 — Pipeline ETL (`src/ingestion.py`, `src/transformation.py`) → `outputs/silver/`
- [x] Fase 3 — Feature Engineering (`src/features.py`) → `outputs/gold/` *(notebook pendente)*
- [x] Fase 4 — Modelagem ML (`src/models.py`) *(notebook + execução pendentes)*
- [ ] Fase 5 — Visualizações e Relatório Final
