# CLAUDE.md — Guia de Contexto do Projeto

## Sobre o Projeto

**Programa:** Vale Desenvolver 2026 — Análise Avançada de Dados  
**Objetivo:** Desenvolver uma solução de análise preditiva sobre dados reais de telemetria e apontamentos de equipamentos de mineração da Vale, visando uma colocação de destaque na competição interna.  
**Prazo:** Mais de 2 meses a partir de abril de 2026.  
**Entregável final:** Relatório no template `Desenvolver_Template.docx` + análises/notebooks + pipeline implementado.

---

## Stack Tecnológica

| Biblioteca     | Papel                                              | Por quê                                              |
|----------------|----------------------------------------------------|------------------------------------------------------|
| **Polars**     | Processamento principal de dados                   | 10-50x mais rápido que Pandas para os 37M registros  |
| **DuckDB**     | Queries SQL analíticas sobre parquet               | Executa sobre arquivos sem carregar tudo em memória  |
| **PyArrow**    | I/O parquet eficiente                              | Integração nativa com Polars e DuckDB               |
| **LightGBM**   | Modelo preditivo principal                         | Melhor custo-benefício para features tabulares       |
| **SHAP**       | Explicabilidade do modelo                          | Essencial para justificar decisões ao negócio        |
| **Plotly**     | Visualizações interativas                          | Gráficos de alta qualidade para o relatório          |
| **Jupyter**    | Notebooks de análise e apresentação                | Formato esperado para EDA e comunicação              |
| **scikit-learn** | Pré-processamento, métricas, pipelines ML        | Ecossistema padrão para ML                           |

**Python:** 3.11+

---

## Localização dos Dados

```
Base_Dados/
├── Dicionario_Dados.xlsx              # Schema completo de todas as colunas
├── Alarmes - Regra de Negocio.xlsx    # 148K+ regras de negócio de alarmes
└── datasets/
    ├── README.md                      # Sumário dos datasets
    ├── apontamentos/
    │   ├── desenvolver_apontamentos.parquet   # 377.907 registros (5.8 MB)
    │   └── desenvolver_apontamentos.xlsx      # Amostra de referência
    └── telemetria/
        ├── telemetry_jan.parquet      # Jan/2025 — 32.5 MB
        ├── telemetry_feb.parquet      # Fev/2025 — 31.6 MB
        ├── telemetry_mar.parquet      # Mar/2025 — 31.7 MB
        ├── telemetry_abr.parquet      # Abr/2025 — 36.0 MB
        ├── telemetry_may.parquet      # Mai/2025 — 33.7 MB
        ├── telemetry_jun.parquet      # Jun/2025 — 42.9 MB
        └── desenvolver_dontgo.xlsx    # Exemplo de sequência de eventos Don't Go
```

**Total de dados:** ~208 MB de parquet  
**Período:** 2025-01-01 a 2025-06-30

---

## Schemas dos Dados

### Apontamentos (377.907 registros)

| Coluna                  | Tipo             | Descrição                                         |
|-------------------------|------------------|---------------------------------------------------|
| `Id`                    | int64            | Identificador único do ciclo                      |
| `Inicio`                | datetime64[ns]   | Início do apontamento                             |
| `Fim`                   | datetime64[ns]   | Fim do apontamento                                |
| `Tag`                   | string           | Código de identificação do equipamento            |
| `Frota`                 | string           | Modelo/frota (ex: 793-D 5S, LeTourneau L 1850)    |
| `Tipo`                  | string           | Tipo do equipamento (Caminhão, Escavadeira)        |
| `Classe`                | string           | Classificação da atividade (Operando, Manutenção, Hibernando, Parado) |
| `Nome_Operador_Anon`    | string           | Código anonimizado do operador (formato OP_XXX)   |
| `Matricula_Operador_Hash` | string         | ID do operador hasheado                           |

### Telemetria (37.164.054 registros)

| Coluna                   | Tipo              | Descrição                                               |
|--------------------------|-------------------|---------------------------------------------------------|
| `Id_Eventos_Telemetria`  | int64             | Identificador único do evento                           |
| `Data_Evento`            | datetime64[us]    | Timestamp do evento (precisão de microssegundos)        |
| `Inicio_Turno`           | datetime64        | Início do turno                                         |
| `Fim_Turno`              | datetime64        | Fim do turno                                            |
| `Dia`                    | int               | Dia do mês                                              |
| `Localidade`             | string            | Mina/localização onde o equipamento opera               |
| `TAG`                    | string            | Identificação do equipamento                            |
| `Tag_Frota`              | string            | Identificação da frota                                  |
| `Tipo`                   | string            | Tipo do equipamento                                     |
| `Id_Alarme`              | int               | ID único do tipo de alarme                              |
| `Alarme`                 | string            | Nome/descrição do alarme                                |
| `Id_Criticidade`         | int               | Nível de criticidade (1=Crítico, 2=Não-Crítico, 3=Info, 4=Outro) |
| `Criticidade`            | string            | Descrição do nível de criticidade                       |
| `Valor`                  | numeric           | Valor do alarme no momento do evento                    |
| `Classe`                 | string            | Estado do alarme (Ativar/Inativar)                      |
| `Is_Dont_Go`             | int8              | Flag binário (1 = está na lista Don't Go, 0 = não está) |

---

## Conceitos-Chave do Domínio

- **Don't Go**: Alerta crítico que proíbe a saída do equipamento para operação. É o evento mais grave — gerado por combinação de alarmes críticos. É o principal alvo preditivo do projeto.
- **Apontamentos**: Registros manuais de estado dos equipamentos (operando, manutenção, hibernando, parado). Registram ciclos de trabalho e paradas planejadas/não planejadas.
- **Telemetria**: Dados em tempo real dos sensores dos equipamentos. Cada linha é um evento de alarme disparado.
- **Criticidade**: Hierarquia de severidade dos alarmes: 1=Crítico > 2=Não-Crítico > 3=Informativo > 4=Outro.
- **Frota**: Modelos de equipamentos — Caminhões (793-D 2S/3S/4S/5S) e Escavadeiras (LeTourneau L 1850).
- **Turno**: Janela de trabalho com início e fim definidos; os eventos de telemetria ocorrem dentro dessas janelas.
- **Alarm Fingerprint**: Padrão único de sequência de alarmes que precede um evento Don't Go — hipótese central do modelo preditivo.

---

## Estrutura de Pastas do Projeto

```
Analise_Avancada_de_Dados/
├── CLAUDE.md                          # Este arquivo
├── PRD.md                             # O que construir e por quê
├── TECH_SPEC.md                       # Como construir (arquitetura técnica)
├── Base_Dados/                        # Dados fornecidos (não modificar)
├── notebooks/
│   ├── 01_EDA_apontamentos.ipynb
│   ├── 02_EDA_telemetria.ipynb
│   ├── 03_regras_negocio.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_modelo_preditivo.ipynb
├── src/
│   ├── ingestion.py                   # Carregamento e validação dos dados
│   ├── transformation.py             # Transformações ETL (Bronze → Silver)
│   ├── features.py                   # Feature engineering (Silver → Gold)
│   ├── models.py                     # Treinamento e avaliação de modelos
│   └── visualization.py             # Funções de visualização reutilizáveis
└── outputs/
    ├── figures/                       # Gráficos e visualizações geradas
    └── reports/                       # Relatório final e documentação
```

---

## Padrões de Código

- **Use Polars Lazy API** para telemetria: sempre `pl.scan_parquet(...)` em vez de `pl.read_parquet(...)` antes do `.collect()`
- **Processe telemetria por mês** quando precisar de toda a série: evite carregar os 208 MB de uma vez
- **Use DuckDB** para queries exploratórias sobre múltiplos arquivos parquet: `duckdb.query("SELECT ... FROM 'Base_Dados/datasets/telemetria/*.parquet'")`
- **Notebooks reproduzíveis**: sempre definir seeds, não deixar variáveis de estado global entre células
- **Nomenclatura**: snake_case para variáveis e funções, PascalCase para classes
- **Não modificar** os arquivos em `Base_Dados/` — são os dados originais do desafio

---

## KPIs de Sucesso do Projeto

| Dimensão             | Métrica                                                    |
|----------------------|------------------------------------------------------------|
| Pipeline ETL         | Processamento completo dos 37M registros em < 5 min       |
| Modelo preditivo     | F1-Score ≥ 0.75 na previsão de Don't Go (validação temporal) |
| Antecedência         | Prever Don't Go com pelo menos 1h de antecedência          |
| Explicabilidade      | SHAP values para os top-10 features mais importantes       |
| Visualização         | Dashboard interativo com timeline de equipamentos e alarmes |
| Negócio              | ≥ 3 insights acionáveis sobre padrões operacionais         |

---

## Referências Internas

- `Edital 001_Projeto Desenvolver 2026.pdf` — critérios oficiais de avaliação
- `Estudo Guiado - Análise Avançada de Dados.pdf` — framework metodológico sugerido
- `Desenvolver_Template.docx` — template obrigatório do relatório final
- `CRONOGRAMA.jpg` — cronograma do programa
