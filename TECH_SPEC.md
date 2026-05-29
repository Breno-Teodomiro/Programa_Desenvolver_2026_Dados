# TECH_SPEC — Especificação Técnica

**Versão:** 1.0  
**Data:** Abril/2026  
**Projeto:** Vale Desenvolver 2026 — Análise Avançada de Dados

---

## 1. Stack Tecnológica

### Justificativas de Escolha

**Polars** em vez de Pandas:
- 37M registros de telemetria × 51 colunas ≈ processamento de GB de dados
- Polars usa Apache Arrow internamente e executa em múltiplos cores por padrão
- Lazy API permite criar o plano de execução inteiro antes de rodar — otimizações automáticas
- Benchmark típico: 10-50x mais rápido para filtros e agregações em datasets grandes

**DuckDB** para queries analíticas:
- Executa SQL diretamente sobre arquivos `.parquet` sem carregá-los na memória
- Ideal para exploração: `SELECT COUNT(*) FROM 'telemetria/*.parquet' WHERE Id_Criticidade = 1`
- Integra com Polars e pandas nativamente
- Suporta janelas temporais e funções analíticas em SQL

**LightGBM** para o modelo:
- Gradient Boosting otimizado para dados tabulares
- Suporte nativo a dados desbalanceados (`scale_pos_weight`)
- Treinamento rápido mesmo com milhões de registros
- SHAP values calculados nativamente para explicabilidade

---

## 2. Arquitetura: Pipeline Medallion (4 Camadas)

```
┌─────────────────────────────────────────────────────────────────┐
│                          RAW LAYER                              │
│  Base_Dados/datasets/telemetria/*.parquet  (37M registros)      │
│  Base_Dados/datasets/apontamentos/*.parquet  (377K registros)   │
│  Base_Dados/Alarmes - Regra de Negocio.xlsx                     │
│  (Dados originais — NUNCA modificar)                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │ src/ingestion.py
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        BRONZE LAYER                             │
│  - Tipagem correta (datetime[us], categoricals, int8)           │
│  - Nomes de colunas padronizados (snake_case)                   │
│  - Validação de schema (sem colunas faltando)                   │
│  - Remoção de duplicatas óbvias                                 │
│  Saída: DataFrames Polars / parquet em outputs/bronze/          │
└──────────────────────────────┬──────────────────────────────────┘
                               │ src/transformation.py
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SILVER LAYER                             │
│  - Join telemetria ↔ apontamentos (TAG + janela temporal)       │
│  - Aplicação das regras de negócio de alarmes                   │
│  - Enriquecimento: turno, localidade, modelo de frota           │
│  - Criação da flag `is_pre_dont_go` (N horas antes do evento)   │
│  - Deduplicação e tratamento de valores nulos                   │
│  Saída: parquet em outputs/silver/                              │
└──────────────────────────────┬──────────────────────────────────┘
│ src/features.py
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         GOLD LAYER                              │
│  - Feature matrix para ML (uma linha por equipamento × janela)  │
│  - Features de frequência de alarmes (janelas 30min, 1h, 4h)    │
│  - Features de sequência (alarm fingerprint encoding)           │
│  - Features temporais (hora do dia, dia da semana, posição turno) │
│  - Features do equipamento (modelo, histórico recente)          │
│  - Target: `is_dont_go_next_1h` (binário)                       │
│  Saída: parquet em outputs/gold/                                │
└──────────────────────────────┬──────────────────────────────────┘
│ src/models.py + src/visualization.py
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUTS                                  │
│  - Modelo treinado (LightGBM .pkl)                              │
│  - Métricas de avaliação (F1, Precision, Recall, AUC-ROC)       │
│  - SHAP values (top features)                                   │
│  - Visualizações interativas (Plotly HTML)                      │
│  - Relatório final (Word)                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Módulos de Código (`src/`)

### `src/ingestion.py`

Responsabilidade: Raw → Bronze

```python
# Interface esperada:
def load_telemetry(months: list[str] = None) -> pl.LazyFrame:
    """Carrega telemetria como LazyFrame. months=['jan','feb'] ou None para todos."""

def load_apontamentos() -> pl.LazyFrame:
    """Carrega apontamentos como LazyFrame."""

def load_alarm_rules() -> pl.DataFrame:
    """Carrega e parseia as regras de negócio de alarmes do xlsx."""

def validate_schema(df: pl.LazyFrame, expected: dict) -> bool:
    """Valida que o schema do DataFrame corresponde ao esperado."""
```

### `src/transformation.py`

Responsabilidade: Bronze → Silver

```python
def join_telemetry_apontamentos(
    telemetry: pl.LazyFrame,
    apontamentos: pl.LazyFrame,
    tolerance: str = "1m"  # tolerância temporal para o join
) -> pl.LazyFrame:
    """Join assof: para cada evento de telemetria, encontra o apontamento ativo."""

def apply_alarm_rules(
    telemetry: pl.LazyFrame,
    rules: pl.DataFrame
) -> pl.LazyFrame:
    """Aplica regras de negócio para classificar e enriquecer alarmes."""

def flag_pre_dont_go_windows(
    telemetry: pl.LazyFrame,
    lead_times: list[int] = [60, 120, 240]  # minutos antes do Don't Go
) -> pl.LazyFrame:
    """Para cada evento Don't Go, marca os N minutos anteriores como janela de risco."""
```

### `src/features.py`

Responsabilidade: Silver → Gold (Feature Engineering)

```python
def compute_alarm_frequency_features(
    df: pl.LazyFrame,
    windows: list[str] = ["30m", "1h", "4h"]
) -> pl.LazyFrame:
    """Conta alarmes por tipo/criticidade nas N janelas temporais anteriores a cada evento."""

def compute_alarm_fingerprint(
    df: pl.LazyFrame,
    top_n_alarms: int = 20
) -> pl.LazyFrame:
    """Codifica a sequência dos últimos N alarmes como vetor de features."""

def compute_temporal_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """Extrai hora do dia, dia da semana, posição no turno, etc."""

def compute_equipment_history_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """Features baseadas no histórico recente de cada equipamento (dias desde última manutenção, etc.)."""

def build_feature_matrix(silver_df: pl.LazyFrame) -> pl.DataFrame:
    """Orquestra todas as funções acima e retorna a matriz final para ML."""
```

### `src/models.py`

Responsabilidade: Treinar, avaliar e salvar modelos

```python
def temporal_train_test_split(
    df: pl.DataFrame,
    train_months: list[str],
    val_months: list[str],
    test_months: list[str]
) -> tuple:
    """Split temporal: treino Jan-Abr, val Mai, teste Jun."""

def train_lightgbm(X_train, y_train, class_weight: str = "balanced") -> lgb.Booster:
    """Treina modelo com tratamento de desbalanceamento."""

def evaluate_model(model, X_test, y_test) -> dict:
    """Retorna F1, Precision, Recall, AUC-ROC, confusion matrix."""

def compute_shap_values(model, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Calcula SHAP values para explicabilidade."""
```

### `src/visualization.py`

Responsabilidade: Criar visualizações reutilizáveis

```python
def plot_alarm_timeline(df: pl.DataFrame, equipment_tag: str) -> go.Figure:
    """Timeline interativa de alarmes de um equipamento com marcação de Don't Go."""

def plot_alarm_heatmap(df: pl.DataFrame) -> go.Figure:
    """Heatmap: equipamento × tipo de alarme × frequência."""

def plot_feature_importance_shap(shap_values, feature_names: list) -> go.Figure:
    """Beeswarm plot de SHAP values."""

def plot_confusion_matrix(y_true, y_pred) -> go.Figure:
    """Matriz de confusão interativa."""

def plot_precision_recall_curve(model, X_test, y_test) -> go.Figure:
    """Curva Precision-Recall para modelo binário desbalanceado."""
```

---

## 4. Feature Engineering — Detalhamento

### 4.1 Janelas Temporais de Alarmes

Para cada evento, computar contagem de alarmes nas janelas anteriores:

```
Para cada (TAG, timestamp_t):
    freq_criticos_30m  = COUNT(alarmes com Id_Criticidade=1, nos 30min anteriores)
    freq_criticos_1h   = COUNT(alarmes com Id_Criticidade=1, na 1h anterior)
    freq_criticos_4h   = COUNT(alarmes com Id_Criticidade=1, nas 4h anteriores)
    freq_nao_criticos_1h = COUNT(alarmes com Id_Criticidade=2, na 1h anterior)
    alarmes_unicos_1h  = COUNT DISTINCT(Id_Alarme, na 1h anterior)
    ...
```

Implementar com `pl.Expr.rolling()` agrupado por TAG.

### 4.2 Alarm Fingerprint

Conceito: os eventos Don't Go são precedidos por uma "assinatura" de alarmes.

```
Para cada janela de 4h antes de um evento:
    1. Listar os IDs únicos de alarmes que ocorreram
    2. Codificar como vetor binário (top 30 alarmes mais frequentes)
    3. Usar como features de presença/ausência
```

Implementar com codificação one-hot esparsa dos top-N Id_Alarme.

### 4.3 Features de Transição de Estado

Capturar mudanças no padrão de alarmes (aceleração de ocorrências):

```
taxa_criticos_aceleracao = freq_criticos_1h / (freq_criticos_4h / 4 + epsilon)
```

Uma aceleração > 2x indica escalada rápida de problemas.

---

## 5. Modelo Preditivo

### Formulação do Problema

```
Target:  is_dont_go_next_1h ∈ {0, 1}
         1 = ocorrerá um Don't Go no equipamento TAG nas próximas 60 minutos
         0 = não ocorrerá

Granularidade: (TAG, timestamp) — uma observação por evento de telemetria ou janela de 15min

Classes: severamente desbalanceadas (Don't Go é evento raro)
```

### Validação Temporal

```
Dados de treino:  Janeiro a Abril 2025   (4 meses)
Dados de validação: Maio 2025            (1 mês — tuning de hiperparâmetros)
Dados de teste:   Junho 2025             (1 mês — avaliação final)
```

**Nunca usar dados do futuro para prever o passado** — o split deve ser estritamente temporal.

### Tratamento do Desbalanceamento

- Usar `scale_pos_weight = n_negativos / n_positivos` no LightGBM
- Otimizar pelo **F1-Score** (não accuracy) — accuracy seria enganosa com classes desbalanceadas
- Avaliar também Precision-Recall AUC (mais informativa que ROC-AUC com desbalanceamento)

### Hiperparâmetros Base (ajustar com validação)

```python
# Valores-base sugeridos. Os valores FINAIS ajustados (deployados) estão em
# outputs/reports/model_metrics.json e src/retrain_optimized.py:
#   num_leaves=255, scale_pos_weight=40, min_child_samples=50, lr=0.05, best_iter=50
lgb_params = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "num_leaves": 63,        # base; valor final ajustado: 255
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "scale_pos_weight": ...,  # final: 40
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}
```

---

## 6. Considerações de Performance

### Problema: 37M registros de telemetria não cabem confortavelmente em RAM com Pandas

| Abordagem             | Uso de RAM (est.) | Velocidade   | Quando usar                          |
|-----------------------|-------------------|--------------|--------------------------------------|
| `pl.scan_parquet()`   | ~2-4 GB           | Muito rápido | Processamento em pipeline            |
| `duckdb.query()`      | < 1 GB            | Rápido       | Queries SQL exploratórias            |
| `pl.read_parquet()`   | ~8-12 GB          | Rápido       | Quando precisar do DataFrame em memória |
| `pd.read_parquet()`   | ~15-25 GB         | Lento        | Evitar para telemetria               |

### Regras de ouro para este projeto:
1. Nunca usar `pl.read_parquet()` em todos os 6 meses de telemetria de uma vez
2. Processar em blocos mensais quando necessário
3. Fazer `.collect()` apenas no final do pipeline lazy, não em etapas intermediárias
4. Para features de janela temporal, usar `pl.Expr.rolling()` — não loops Python
5. Salvar outputs intermediários em parquet (Bronze/Silver/Gold) para não re-processar

---

## 7. Estrutura de Notebooks

| Notebook                         | Objetivo                                  | Output                           |
|----------------------------------|-------------------------------------------|----------------------------------|
| `01_EDA_apontamentos.ipynb`      | Explorar distribuições, padrões de uso    | Insights + visualizações         |
| `02_EDA_telemetria.ipynb`        | Explorar alarmes, frequências, localidades | Insights + validação de H1-H7   |
| `03_regras_negocio.ipynb`        | Parsear e validar regras do xlsx          | Módulo de regras testado         |
| `04_feature_engineering.ipynb`   | Desenvolver e validar features            | Dataset Gold salvo               |
| `05_modelo_preditivo.ipynb`      | Treinar, avaliar e explicar o modelo      | Modelo + métricas + SHAP plots   |

---

## 8. Ambiente e Dependências

> **Fonte de verdade das versões:** `pyproject.toml` + `uv.lock` (gerenciado por uv).
> As versões abaixo refletem o ambiente efetivamente usado no projeto.

```toml
# pyproject.toml (extrato — versões mínimas reais)
polars >= 1.40.1
duckdb >= 1.5.2
pyarrow >= 24.0.0
lightgbm >= 4.6.0
shap >= 0.51.0
plotly >= 6.7.0
scikit-learn >= 1.8.0
openpyxl >= 3.1.5       # para ler os .xlsx
jupyter >= 1.1.1
ipykernel >= 7.2.0
pandas >= 3.0.2          # necessário para SHAP e scikit-learn
numpy >= 2.4.4
streamlit >= 1.57.0      # app de demonstração local
```

> **Dependência opcional:** PyTorch (CPU) é usado apenas na PoC do Sprint 12
> (modelo sequencial GRU para a escavadeira) e **não integra o pipeline principal**.
> Por isso fica fora do `uv.lock` (evita arrastar a variante CUDA). Instalar sob demanda:
> `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Setup inicial

O projeto é gerenciado por [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`):

```bash
uv sync            # cria o .venv e instala todas as dependências do lockfile
uv run jupyter lab # abre os notebooks no ambiente do projeto
```

---

## 9. Padrão de Código nos Notebooks

```python
# Imports padrão no topo de cada notebook
import polars as pl
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Caminhos sempre relativos à raiz do projeto
ROOT = Path("..") if Path("..").joinpath("CLAUDE.md").exists() else Path(".")
DATA_DIR = ROOT / "Base_Dados" / "datasets"
TELEMETRY_GLOB = str(DATA_DIR / "telemetria" / "*.parquet")
APONTAMENTOS_PATH = str(DATA_DIR / "apontamentos" / "desenvolver_apontamentos.parquet")

# Padrão para carregar telemetria com Polars lazy
lf_telemetry = pl.scan_parquet(TELEMETRY_GLOB)

# Padrão para query exploratória com DuckDB
con = duckdb.connect()
result = con.execute(f"""
    SELECT Id_Criticidade, COUNT(*) as total
    FROM '{TELEMETRY_GLOB}'
    GROUP BY Id_Criticidade
    ORDER BY total DESC
""").pl()  # retorna Polars DataFrame direto
```
