# PRD — Produto: Sistema de Análise Preditiva de Equipamentos de Mineração

**Versão:** 1.0  
**Data:** Abril/2026  
**Programa:** Vale Desenvolver 2026 — Análise Avançada de Dados

---

> **Nota de status de entrega (documento-base):** este PRD é o documento de
> requisitos elaborado no início do projeto (Abril/2026) e é mantido como
> referência histórica. A solução foi **entregue e estendida** em 12 sprints de
> refinamento, documentados no `README.md`, no relatório final
> (`outputs/reports/`) e no `GUIA_APRESENTACAO.md`. Ajuste relevante de meta: o
> alvo inicial de **F1 ≥ 0,75** foi reavaliado empiricamente — o modelo final
> alcança **F1 = 0,689**, comprovadamente o teto do conjunto de features atual
> (validado pelo experimento de ensemble do Sprint 10, que não o superou; ganho
> adicional exigiria novas fontes de dados, não mais modelagem). As hipóteses
> H1–H7 da Seção 8 foram testadas, com **H4, H6 e H7 refutadas com evidência**.

---

## 1. Declaração do Problema

Equipamentos de mineração pesada (caminhões 793-D e escavadeiras LeTourneau) geram continuamente dados de telemetria — 37+ milhões de eventos de alarme em 6 meses. Esses alarmes são processados por sistemas OEM que emitem alertas **Don't Go** quando a combinação de falhas ultrapassa um limiar crítico de segurança/operação.

O problema atual: **a análise é retrospectiva**. Os alertas Don't Go são identificados apenas quando já ocorreram, resultando em paralisações não planejadas, custo de manutenção emergencial e risco operacional.

**Oportunidade:** Os dados de telemetria contêm padrões que precedem esses eventos. Uma solução que detecte esses padrões com antecedência transforma a operação de **reativa para preditiva**.

---

## 2. Objetivos do Projeto

### Objetivo Principal
Construir um sistema de análise avançada que integre telemetria e apontamentos para **prever eventos Don't Go com antecedência**, identificar padrões operacionais e gerar insights acionáveis para a gestão da frota.

### Objetivos Específicos
1. **ETL robusto**: Pipeline que ingere, limpa, integra e enriquece os dados de telemetria (37M registros) e apontamentos (377K registros) de forma eficiente e reproduzível.
2. **Análise exploratória profunda**: Revelar padrões de comportamento de equipamentos, operadores e turnos que não são óbvios pela análise manual.
3. **Modelo preditivo**: Classificar com antecedência (≥ 1 hora) a probabilidade de ocorrência de um evento Don't Go por equipamento.
4. **Visualizações comunicativas**: Dashboards e gráficos que transformam dados complexos em narrativas de negócio claras para tomadores de decisão.

---

## 3. Fontes de Dados

| Fonte                                | Registros     | Período           | Papel no sistema                        |
|--------------------------------------|---------------|-------------------|-----------------------------------------|
| Telemetria (parquet mensais)         | 37.164.054    | Jan–Jun 2025      | Sinais primários para detecção de padrões |
| Apontamentos (parquet)               | 377.907       | Jan–Jun 2025      | Estado real do equipamento (contexto)   |
| Regras de Negócio (xlsx)             | 148.421+ linhas | —               | Lógica de classificação dos alarmes     |
| Dicionário de Dados (xlsx)           | —             | —                 | Schema e semântica de todas as colunas  |

### Relação entre fontes
```
Telemetria (eventos de alarme)
    ↓ join por TAG + janela temporal
Apontamentos (estado do equipamento)
    ↓ enriquecimento por Id_Alarme
Regras de Negócio (classificação)
    ↓
Dataset unificado e enriquecido
    ↓
Features para ML + Análises de negócio
```

---

## 4. Personas e Stakeholders

| Persona                  | Interesse primário                                              |
|--------------------------|-----------------------------------------------------------------|
| Avaliadores do Desafio   | Solução técnica rigorosa, insights de negócio, apresentação clara |
| Gerentes de Operação     | Redução de paradas não planejadas, visibilidade da frota        |
| Mantenedores             | Alertas antecipados de falhas, priorização de manutenção        |
| Operadores               | Feedback sobre comportamento que precede falhas                 |

---

## 5. Entregáveis

| Entregável                        | Formato                  | Prazo        |
|-----------------------------------|--------------------------|--------------|
| Relatório final                   | `.docx` (template fornecido) | Fim do programa |
| Notebooks de análise              | `.ipynb` Jupyter         | Contínuo     |
| Pipeline ETL implementado         | Python (`.py`)           | Fase 2       |
| Modelo preditivo treinado         | LightGBM + SHAP          | Fase 3       |
| Visualizações e dashboard         | Plotly/HTML              | Fase 4       |

---

## 6. Critérios de Avaliação (inferidos do Edital)

Com base no template de relatório fornecido (`Desenvolver_Template.docx`), os critérios de avaliação incluem:

| Critério                        | Peso estimado | Como maximizar                                          |
|---------------------------------|---------------|---------------------------------------------------------|
| Business Understanding          | Alto          | Conectar dados às realidades operacionais da mineração  |
| Qualidade da Metodologia        | Alto          | Pipeline ETL bem documentado e decisões justificadas    |
| Resultados e Discussão          | Alto          | Insights não-óbvios, métricas sólidas, valor gerado     |
| Inovação / Fora da Caixa        | Alto          | Abordagem preditiva vs. apenas descritiva               |
| Qualidade da Apresentação       | Médio         | Visualizações claras, resumo executivo impactante       |

---

## 7. Estratégia de Diferenciação

A maioria das soluções em competições como esta entrega **análise descritiva/exploratória** (o que aconteceu). Nossa vantagem competitiva está em entregar **análise preditiva** (o que vai acontecer) com **explicabilidade de negócio** (por que vai acontecer).

### Diferenciais planejados

**D1 — Alarm Fingerprint (inovação conceitual)**  
Identificar a sequência e combinação específica de alarmes que invariavelmente precede um evento Don't Go. Criar um "DNA do Don't Go" por modelo de equipamento.

**D2 — Previsão com antecedência real**  
Em vez de detectar que o Don't Go ocorreu, prever com 1-4 horas de antecedência a probabilidade de ocorrência. Isso tem valor operacional imediato e mensurável.

**D3 — Análise de padrões operador-equipamento**  
Cruzar comportamento de operadores anônimos com frequência de alarmes para identificar se há correlação entre estilo de operação e degradação do equipamento.

**D4 — Pipeline de dados de nível produção**  
Arquitetura em camadas (Raw → Bronze → Silver → Gold) que demonstra maturidade de engenharia de dados além do esperado para um desafio acadêmico.

---

## 8. Hipóteses de Negócio a Validar

| # | Hipótese                                                                 | Dataset          |
|---|--------------------------------------------------------------------------|------------------|
| H1 | Existe uma sequência característica de alarmes que precede o Don't Go em até 4h | Telemetria    |
| H2 | A frequência de alarmes críticos aumenta progressivamente antes de um Don't Go | Telemetria    |
| H3 | Determinados modelos de frota (793-D 5S vs 2S) têm perfis de falha distintos | Telemetria     |
| H4 | Há diferença na frequência de alarmes entre turnos (dia vs noite)         | Telemetria       |
| H5 | O estado do apontamento (manutenção recente) influencia a ocorrência de Don't Go | Tel + Apont  |
| H6 | Determinadas localidades apresentam maior concentração de alarmes críticos | Telemetria      |
| H7 | Existe correlação entre operador (anonimizado) e perfil de alarmes         | Tel + Apont      |

---

## 9. Fases de Desenvolvimento

### Fase 1: Fundação e EDA (Semanas 1–2)
- Setup do ambiente (Polars, DuckDB, Jupyter)
- EDA dos apontamentos (`01_EDA_apontamentos.ipynb`)
- EDA da telemetria com amostragem (`02_EDA_telemetria.ipynb`)
- Validação das regras de negócio (`03_regras_negocio.ipynb`)
- **Entrega:** Compreensão profunda dos dados e primeiros insights

### Fase 2: Pipeline ETL (Semanas 3–4)
- Implementar `src/ingestion.py` (Raw → Bronze)
- Implementar `src/transformation.py` (Bronze → Silver): join telemetria+apontamentos, aplicação de regras
- Documentar decisões de arquitetura no `TECH_SPEC.md`
- **Entrega:** Pipeline reproduzível processando todos os 37M registros

### Fase 3: Feature Engineering e Modelo Preditivo (Semanas 5–7)
- Implementar `src/features.py`: janelas temporais, frequências de alarmes, alarm fingerprint
- Treinar e validar modelo LightGBM (`05_modelo_preditivo.ipynb`)
- Analisar SHAP values para explicabilidade
- Validação temporal: treino Jan–Abr, validação Mai, teste Jun
- **Entrega:** Modelo com F1 ≥ 0.75 e SHAP explicado

### Fase 4: Insights, Visualizações e Relatório (Semanas 8–10)
- Implementar `src/visualization.py` e dashboard interativo
- Validar todas as 7 hipóteses de negócio
- Redigir relatório final no template fornecido
- **Entrega:** Relatório completo + portfolio de visualizações

---

## 10. Riscos e Mitigações

| Risco                                              | Probabilidade | Mitigação                                          |
|----------------------------------------------------|---------------|----------------------------------------------------|
| Volume de dados ultrapassa memória disponível      | Médio         | Polars lazy API + DuckDB + processamento por mês   |
| Desequilíbrio de classes (Don't Go é evento raro)  | Alto          | Técnicas de oversampling (SMOTE) ou class_weight   |
| Regras de negócio complexas no xlsx               | Médio         | Processar em etapas, validar com amostra dontgo.xlsx |
| Prazo insuficiente para todos os diferenciais      | Baixo         | Priorizar D1 (fingerprint) e D2 (predição) sobre D3/D4 |
