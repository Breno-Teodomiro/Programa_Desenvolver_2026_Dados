# Guia de Apresentação — Don't Go Predictor
### Vale Desenvolver 2026 · Análise Avançada de Dados

> Documento de orientação para a banca/apresentação. Não é entregável técnico — é o seu roteiro.
> Tudo aqui está ancorado nos resultados reais do projeto (12 sprints). Última revisão: 2026-06-24.

---

## 1. A mensagem central (decore isto)

> **"Transformamos 37 milhões de eventos brutos de telemetria em um score de risco que antecipa o evento Don't Go com 1 a 4 horas de antecedência, ganhando +340% de F1 sobre a regra operacional atual e abrindo uma economia de ~R$6 milhões/mês ao evitar paradas — e onde o modelo NÃO funciona, sabemos exatamente por quê."**

Essa última frase é o seu diferencial. A maioria dos concorrentes vai mostrar só o que funcionou. Você mostra **domínio do problema**: onde funciona, quanto vale, onde falha e por qual razão estrutural. Isso é o que separa um finalista de um vencedor.

---

## 2. Os 3 critérios do edital — e como cada parte os ataca

| Critério do edital | Onde você marca ponto | Frase de ancoragem |
|---|---|---|
| **Diagnóstico** (entender o problema) | EDA, validação de 7 hipóteses (H1–H7), achado do CA65926, diagnóstico da escavadeira | "Não assumimos nada: testamos 7 hipóteses e refutamos 3 com dados." |
| **Qualidade da Solução** | Pipeline Medallion, 5 modelos comparados, calibração, análise de custo, multi-horizonte | "Não é só um modelo: é um sistema, com baseline, calibração e decisão por custo." |
| **Reprodutibilidade** | `uv sync` + scripts numerados por sprint, seeds fixas, notebooks executados | "Qualquer pessoa reproduz tudo com dois comandos." |

**Tática:** ao final de cada bloco da apresentação, diga em voz alta qual critério aquele bloco atende. A banca pontua por critério — facilite a vida dela.

---

## 3. Roteiro sugerido (12–15 min + Q&A)

Ajuste os tempos ao limite real do evento. A regra: **40% diagnóstico/negócio, 40% solução, 20% rigor/limitações**.

### Bloco 1 — O problema (2–3 min) · *Critério: Diagnóstico*
- O que é Don't Go: alerta crítico que **proíbe a saída do equipamento**. Cada ocorrência não prevista = parada não planejada (~4h, ~R$50 mil).
- O tamanho do dado: **37,2 milhões** de eventos de telemetria + 377 mil apontamentos, Jan–Jun/2025.
- O desafio estatístico: evento **raríssimo** — desbalanceamento de **1:1.860** (0,054%). *"Prever Don't Go é achar agulha no palheiro — e a agulha custa R$50 mil."*
- **Mostre:** o slide de contexto + 1 número grande (37M).

### Bloco 2 — Diagnóstico dos dados (2–3 min) · *Critério: Diagnóstico*
- Validação de hipóteses **H1–H7** (tabela no relatório). Destaque que **refutamos H4 (turno), H6 (localidade) e H7 (operador)** com evidência — não por opinião.
- O achado de ouro: **CA65926**. Conte como história:
  - Jan: estava **abaixo** da média da frota (0,22%). Não nasceu problemático.
  - Mar: primeiro surto (8,5× a média) — sinal de alerta ignorável.
  - Jun: **colapso, 21,58%** — 205× a média da sua frota. Degradação **estrutural**.
  - Recomendação: retirar para inspeção + revisar manutenções de Fev–Mar.
- **Mostre:** `ca65926_monthly_escalation.png` ou o dashboard `04_risk_timeline`. Esse é o seu momento "uau".

### Bloco 3 — A solução (4–5 min) · *Critério: Qualidade*
- **Pipeline Medallion:** Bronze → Silver → Gold → Modelo. 37M registros processados de forma reproduzível. (Stack: Polars + DuckDB + LightGBM.)
- **Não é um modelo só — é uma comparação honesta de 5:** baseline trivial, regra de negócio, Logistic L1, Random Forest, LightGBM.
  - Número-chave: **+340% de F1** do ML sobre a regra operacional (0,153 → 0,673).
  - LightGBM final: **F1=0,67 · ROC-AUC=0,99 · PR-AUC=0,60** (threshold fixado na validação, sem vazamento `Is_Dont_Go`).
- **Antecedência (a promessa do PRD):** o mesmo modelo mantém **ROC-AUC > 0,96 em 4 horas**. A precisão até *sobe* com o horizonte.
- **Decisão por custo, não por F1:** FN custa 62× mais que FP. Medindo o custo **por episódio de Don't Go** (1 episódio = 1 parada física), o threshold custo-ótimo economiza **~R$6 milhões** no mês de teste capturando ~99% das paradas, vs. o threshold puramente estatístico.
- **Mostre:** `comparison_f1_bar.png` + o dashboard de fingerprint/SHAP.

### Bloco 4 — Robustez e operação (2–3 min) · *Critério: Qualidade*
Escolha 2 destes 4 (não tente mostrar tudo — sobrecarrega):
- **Calibração isotônica:** Brier −89%. *"As probabilidades agora são confiáveis — dá para dizer '70% de chance' e cobrar a manutenção por isso."*
- **Política de threshold por frota (Sprint 9):** F1 agregado **+17%** cortando ~46 mil falsos positivos; tabela de decisão deployável.
- **Detecção de drift (Sprint 8):** Page-Hinkley + KS para disparar re-treino em produção.
- **App Streamlit:** demo ao vivo do score por equipamento (ver §6).

### Bloco 5 — Onde NÃO funciona, e por quê (2 min) · *Critério: Diagnóstico + Rigor*
Este bloco é contraintuitivo mas é o que ganha pontos de maturidade. **Não esconda a limitação — lidere com ela.**
- O modelo é cego à **escavadeira LeTourneau** (Recall 0,14). Investigamos a fundo:
  - **Sprint 11:** o sinal **existe** (alarme "Channel Forced", lift **44×**), mas é raro e o fingerprint por frequência o descarta.
  - **Sprint 12:** testamos até um modelo sequencial (rede neural GRU). Também falhou (PR-AUC 0,0002).
  - **Conclusão triangulada:** 3 experimentos independentes mostram que o problema **não é o modelo — é a não-estacionariedade dos dados** (a distribuição da escavadeira mudou estruturalmente entre Mai e Jun).
- **A virada:** *"Isso não é um fracasso do modelo, é um achado de diagnóstico. A solução não é um modelo maior — é investigação operacional + re-treino adaptativo."*

### Bloco 6 — Fechamento (1 min)
- Reafirme a mensagem central (§1).
- Os 3 entregáveis: **modelo + pipeline reproduzível + painéis/app**.
- Trabalhos futuros **fundamentados** (não genéricos): fingerprint dedicado da escavadeira com os alarmes já identificados, re-treino disparado por drift, coleta de dados de operador.

---

## 4. Os números que você precisa saber de cor

| Número | O que é | Por que importa |
|---|---|---|
| **37,2 M** | eventos de telemetria | escala do problema |
| **1:1.860** | desbalanceamento | dificuldade estatística |
| **F1 = 0,67** | desempenho do modelo final | supera a meta de 0,75? **Não** — seja honesto (ver §7) |
| **+340%** | ganho de F1 vs. regra de negócio | justifica o ML |
| **ROC-AUC 0,96 @ 4h** | antecedência | cumpre o PRD |
| **~R$ 6 M** | economia no mês de teste (por episódio) | impacto financeiro |
| **−89%** | queda no Brier (calibração) | confiabilidade das probabilidades |
| **21,58%** | taxa DG do CA65926 em Jun (205× a frota) | o achado-estrela |
| **44×** | lift do alarme "Channel Forced" na escavadeira | o sinal que o modelo ignora |

> ⚠️ **Atenção a um erro comum:** a taxa do CA65926 em Junho é **21,58%** (não 61,6% — esse outro número era a taxa de eventos *pré*-DG, uma métrica diferente). O relatório já está corrigido; não tropece nisso ao vivo.

---

## 5. Como apresentar os resultados negativos (Sprints 10 e 12)

Dois dos achados mais fortes são **resultados negativos**. Apresentados errado, parecem fracasso; apresentados certo, são o seu maior diferencial de rigor.

**Faça assim:**
- Enquadre como **pergunta científica**, não como tentativa frustrada: *"Perguntamos: dá para romper o teto de desempenho combinando modelos? Testamos e a resposta foi não — e isso nos ensinou onde está o limite real."*
- Use a métrica honesta. No GRU, **nunca** diga "Recall 0,68" sem completar: esse recall vem de marcar 6,6 milhões de eventos como positivos (precisão ≈ 0). A métrica decisiva é **PR-AUC**, e nela o GRU perde.
- Termine com a lição: o teto de F1≈0,69 é imposto pelo **conjunto de features**, não pelo algoritmo. Ganho real exigiria **dados novos**, não mais modelagem.

**Por que isso ganha pontos:** mostra que você entende a diferença entre *otimizar uma métrica* e *resolver o problema*. Bancas técnicas valorizam isso muito.

---

## 6. Demo ao vivo — o que mostrar e como

**Se houver projeção e tempo (recomendado, 2 min):**
1. `uv run streamlit run app.py` — abra **antes** da apresentação (deixe rodando em outra aba).
2. Mostre: seleção de equipamento → score de risco → timeline. Foque no **CA65926** (a história que você já contou).
3. Tenha aberto também `outputs/dashboards/09_story_dashboard.html` como plano B (não depende de servidor).

**Regra de ouro de compliance:** **NÃO faça deploy público.** Os dados são proprietários da Vale (TAGs, timestamps reais). Demo **local apenas**. Se perguntarem "está no ar?", a resposta correta é: *"Roda localmente por questão de confidencialidade dos dados — é uma decisão de governança, não uma limitação técnica."* Isso também pontua (mostra consciência de dados sensíveis).

**Plano B sem internet/projeção:** os snapshots PNG dos dashboards já estão **embutidos no relatório Word** (seção "Painéis Interativos"). Você consegue contar tudo só com o documento.

---

## 7. Perguntas difíceis — e respostas prontas

**P: "O F1 de 0,67 não bateu a meta de 0,75 do projeto. Por quê?"**
R: "Correto, e investigamos a fundo. Testamos 6 variações de hiperparâmetros, um segundo modelo (Random Forest) e um ensemble — nenhum passou de ~0,69. Provamos que **~0,69 é o teto do conjunto de features atual**, não uma falha de tuning (o F1 de produção é 0,67, com o threshold honestamente fixado na validação). Para superá-lo seriam necessárias fontes de dados novas, não mais ajuste. Documentar esse teto com rigor vale mais que inflar a métrica."

**P: "Por que LightGBM e não deep learning?"**
R: "Para features tabulares e 37M de registros, LightGBM tem o melhor custo-benefício e é explicável via SHAP. E não ficamos na teoria: **testamos** uma rede recorrente (GRU) na escavadeira (Sprint 12) — ela perdeu para o LightGBM. A escolha é empírica, não dogmática."

**P: "Como sei que não há vazamento de dados (data leakage)?"**
R: "Split estritamente **temporal**: treino Jan–Abr, validação Mai, teste Jun. O modelo nunca vê o futuro. O alvo é look-ahead de 60 min, e as features usam só janelas **passadas**. O threshold é calibrado na validação, não no teste."

**P: "Os ~R$6 milhões são realistas?"**
R: "É uma **ordem de grandeza** com premissas explícitas (FN=R$50k por parada de ~4h, FP=R$800 por inspeção), medida **por episódio de Don't Go** — não por linha de telemetria (1 parada física = 1 episódio; precificar cada linha superestimava em ordens de grandeza). O valor exato depende dos custos reais da Vale; o que é robusto é a **direção**: priorizar recall sob esse custo assimétrico evita paradas. As premissas estão no relatório para serem ajustadas."

**P: "E se a distribuição mudar em produção?"**
R: "Já tratamos isso (Sprint 8): detectores de drift Page-Hinkley + KS disparam alerta de re-treino. Foi inclusive como **diagnosticamos** o problema da escavadeira."

**P: "Os dados de operador não estavam no escopo (H7)?"**
R: "Estavam previstos, mas as colunas de operador **não vieram no parquet** fornecido (só no Excel de amostra). Documentamos isso como limitação de dados e substituímos por segmentação por frota — igualmente acionável."

**P: "Qual a feature mais importante?"**
R: "O alarme 'Raise Hoist Limited By End Of Stroke' (presente em >60% dos Don't Go de caminhão) e a **aceleração de alarmes críticos** na janela de 60 min — um indicador simples e interpretável que operadores podem monitorar até manualmente."

---

## 8. Mapa dos entregáveis (o que abrir se pedirem)

| Quero mostrar… | Abra… |
|---|---|
| O relatório completo | `outputs/reports/Relatorio_Final_DontGo_Predictor.docx` |
| A história executiva | `outputs/dashboards/09_story_dashboard.html` |
| Ranking de risco da frota | `outputs/dashboards/01_fleet_risk.html` |
| O caso CA65926 | `outputs/dashboards/04_risk_timeline_CA65926_jun.html` |
| Explicabilidade (SHAP) | `outputs/dashboards/05_shap_global.html` |
| O app operacional | `uv run streamlit run app.py` |
| O código de um sprint | `src/sprint{1..12}_*.py` (um arquivo por análise) |
| A análise narrada | `notebooks/01` a `16` (executados) |

---

## 9. Checklist pré-apresentação

- [ ] `uv sync` rodado e ambiente OK na máquina da apresentação.
- [ ] `uv run streamlit run app.py` testado e deixado aberto numa aba.
- [ ] `09_story_dashboard.html` aberto no navegador (plano B offline).
- [ ] Relatório Word aberto na seção "Resumo Executivo" (plano C).
- [ ] Os 9 números da §4 decorados — principalmente **37M, +340%, ~R$6M, 21,58%, 44×**.
- [ ] Cronômetro: ensaie em voz alta pelo menos 1×. O bloco 5 (limitações) é o que mais se corta sob pressão — **não corte**.
- [ ] Tenha uma resposta de 1 frase para "qual o impacto disso para a Vale?": *"Menos paradas não planejadas e manutenção priorizada por risco real, com 1–4h de antecedência."*

---

## 10. Os 3 erros a evitar

1. **Afogar a banca em método.** Eles querem saber *o que você descobriu* e *quanto vale*, não cada hiperparâmetro. Método é prova, não protagonista.
2. **Esconder a limitação da escavadeira.** É o seu trunfo de maturidade — lidere com ela, não a deixe surgir numa pergunta.
3. **Vender os ~R$6M como certeza.** Diga sempre "ordem de grandeza, com estas premissas". Honestidade numérica > exagero.

---

*Boa sorte. Você tem um projeto que combina rigor estatístico, impacto de negócio quantificado e honestidade científica — os três pilares que a banca procura. Conte a história com confiança.*
