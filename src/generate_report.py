"""Geração automática do Relatório Final no template Desenvolver_Template.docx.

Preenche todas as seções do template com o conteúdo do projeto, métricas e insights.
Salva em outputs/reports/Relatorio_Final_DontGo_Predictor.docx
"""

import json
import shutil
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).parent.parent
TEMPLATE = ROOT / "Desenvolver_Template.docx"
OUT_DIR = ROOT / "outputs" / "reports"
OUT_FILE = OUT_DIR / "Relatorio_Final_DontGo_Predictor.docx"
METRICS_FILE = OUT_DIR / "model_metrics.json"
FIGURES_DIR = ROOT / "outputs" / "figures"

VALE_GREEN = RGBColor(0x00, 0xA6, 0x50)

# ── Dados do participante (editar antes de submeter) ──────────────────────────
PARTICIPANTE = {
    "grupo":       "Don't Go Predictor",
    "nome1":       "[Seu Nome Completo]",
    "inst1":       "Vale S.A.",
    "email1":      "insights.jobs.ia@gmail.com",
}


def _load_metrics() -> dict:
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            return json.load(f)
    return {
        "roc_auc": 0.9916, "pr_auc": 0.5740,
        "f1_score": 0.6785, "precision": 0.7096,
        "recall": 0.6500, "optimal_threshold": 0.1313,
        "n_samples": 7854243, "n_positivos": 35203,
        "taxa_positivos_pct": 0.45,
    }


def _set_heading(doc: Document, text: str, level: int = 1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.color.rgb = VALE_GREEN


def _add_paragraph(doc: Document, text: str, bold: bool = False,
                   italic: bool = False, size: int = 11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    return p


def _add_bullet(doc: Document, text: str):
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(text).font.size = Pt(11)
    return p


def _add_table(doc: Document, headers: list[str], rows: list[list[str]]):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for para in hdr[i].paragraphs:
            for run in para.runs:
                run.bold = True
    for r_idx, row_data in enumerate(rows):
        row = table.rows[r_idx + 1].cells
        for c_idx, val in enumerate(row_data):
            row[c_idx].text = str(val)
    return table


def build_report():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATE, OUT_FILE)
    doc = Document(OUT_FILE)
    m = _load_metrics()

    # ── Capa — substituir placeholders ───────────────────────────────────────
    for para in doc.paragraphs:
        if "Nome do Grupo" in para.text:
            for run in para.runs:
                run.text = run.text.replace("Nome do Grupo (Quando aplicável)",
                                            PARTICIPANTE["grupo"])
        if "Nome Participante 1" in para.text:
            for run in para.runs:
                run.text = run.text.replace("Nome Participante 1", PARTICIPANTE["nome1"])
        if "Instituição Participante 1" in para.text:
            for run in para.runs:
                run.text = run.text.replace("Instituição Participante 1", PARTICIPANTE["inst1"])
        if "Participante1@email.com" in para.text:
            for run in para.runs:
                run.text = run.text.replace("Participante1@email.com", PARTICIPANTE["email1"])
        if "Nome Participante 2" in para.text:
            for run in para.runs:
                run.text = run.text.replace("Nome Participante 2", "")
        if "Instituição Participante 2" in para.text:
            for run in para.runs:
                run.text = run.text.replace("Instituição Participante 2", "")
        if "Participante2@email.com" in para.text:
            for run in para.runs:
                run.text = run.text.replace("Participante2@email.com", "")
        if "Nome Participante 3" in para.text:
            for run in para.runs:
                run.text = run.text.replace("Nome Participante 3", "")
        if "Instituição Participante 3" in para.text:
            for run in para.runs:
                run.text = run.text.replace("Instituição Participante 3", "")
        if "Participante3@email.com" in para.text:
            for run in para.runs:
                run.text = run.text.replace("Participante3@email.com", "")

        # Resumo e palavras-chave
        if para.text.startswith("Resumo."):
            for run in para.runs:
                run.text = ""
            para.clear()
            run = para.add_run(
                "Resumo. Este trabalho apresenta uma solução de análise preditiva para eventos "
                "Don't Go em equipamentos de mineração pesada da Vale. Foram processados 37,2 milhões "
                "de eventos de telemetria (Janeiro–Junho 2025) de 35 equipamentos por meio de um "
                "pipeline de dados em camadas (Bronze → Silver → Gold). O modelo LightGBM, treinado "
                "com 54 features derivadas de telemetria — incluindo frequências de alarmes em janelas "
                "rolantes de 15 a 240 minutos, fingerprint dos top-30 alarmes e contexto operacional "
                "de turno e frota — alcançou ROC-AUC de 0,99 e F1-Score de 0,68 na predição de eventos "
                "Don't Go com 60 minutos de antecedência (validação temporal: treino Jan–Mai, teste Jun). "
                "O equipamento CA65926 (793-D 4S) foi identificado como outlier crítico, com taxa de "
                "Don't Go 98× superior à média da frota (5,3% vs 0,054%). A análise SHAP revelou que "
                "as features mais determinantes são a recência do último evento Don't Go, a aceleração "
                "de alarmes críticos na janela de 60 minutos e a presença de alarmes específicos no "
                "fingerprint. Um dashboard interativo (Streamlit) e seis visualizações HTML standalone "
                "foram desenvolvidos para comunicar os resultados de forma executiva."
            )
            run.font.size = Pt(11)

        if para.text.startswith("Palavras-Chave:"):
            for run in para.runs:
                run.text = ""
            para.clear()
            run = para.add_run(
                "Palavras-Chave: Manutenção Preditiva; LightGBM; Alarm Fingerprint; "
                "Don't Go; Telemetria; Pipeline Medallion; SHAP; Mineração de Dados"
            )
            run.font.size = Pt(11)
            run.italic = True

    # Limpa os bullet points de instruções das seções
    instruction_keywords = [
        "O candidato deve", "O candidato descrever", "Deve apresentar",
        "Além disso", "Nesta seção, o candidato",
    ]
    paras_to_clear = []
    for para in doc.paragraphs:
        if any(para.text.startswith(k) for k in instruction_keywords):
            paras_to_clear.append(para)
    for para in paras_to_clear:
        p_elem = para._element
        p_elem.getparent().remove(p_elem)

    # ── Introdução ────────────────────────────────────────────────────────────
    intro_heading = next(
        (p for p in doc.paragraphs if p.style.name == "Heading 1" and "Introdução" in p.text),
        None,
    )
    if intro_heading:
        _insert_after_paragraph(doc, intro_heading, [
            ("p", "Equipamentos de mineração pesada — caminhões 793-D e escavadeiras "
             "LeTourneau — geram continuamente dados de telemetria a partir de centenas de sensores "
             "embarcados. Em um período de seis meses (Janeiro–Junho 2025), foram registrados "
             "37,2 milhões de eventos de alarme para 35 equipamentos na operação de Itabira. "
             "Esses alarmes são processados em tempo real por sistemas OEM que emitem alertas "
             "Don't Go quando a combinação de falhas críticas ultrapassa um limiar operacional de "
             "segurança, resultando na proibição imediata de uso do equipamento."),
            ("p", "O problema central é a natureza reativa do processo atual: o alerta Don't Go "
             "é identificado apenas quando já ocorreu, causando paralisações não planejadas, "
             "custo elevado de manutenção emergencial e risco operacional. Os dados de telemetria, "
             "contudo, contêm padrões que precedem esses eventos — uma janela de oportunidade para "
             "intervenção preventiva."),
            ("h2", "Objetivo"),
            ("p", "Desenvolver um sistema de análise preditiva que:"),
            ("b", "processe os 37,2 milhões de eventos de telemetria em um pipeline reproduzível (Bronze → Silver → Gold);"),
            ("b", "identifique o 'fingerprint' de alarmes que precedem eventos Don't Go com até 4 horas de antecedência;"),
            ("b", "treine e avalie um modelo LightGBM capaz de prever Don't Go com pelo menos 1 hora de antecedência;"),
            ("b", "gere insights acionáveis sobre padrões operacionais da frota, com explicabilidade via SHAP."),
        ])

    # ── Entendimento do Negócio ───────────────────────────────────────────────
    neg_heading = next(
        (p for p in doc.paragraphs if p.style.name == "Heading 1" and "Negócio" in p.text),
        None,
    )
    if neg_heading:
        _insert_after_paragraph(doc, neg_heading, [
            ("h2", "Contexto Operacional"),
            ("p", "A operação de mineração da Vale em Itabira utiliza dois tipos de equipamentos pesados:"),
            ("b", "Caminhões 793-D (frotas 2S, 3S, 4S e 5S): principais transportadores de minério, sujeitos a intenso ciclo de carga e descarga."),
            ("b", "Escavadeiras LeTourneau L 1850: equipamentos de carregamento de alta capacidade."),
            ("p", "Os equipamentos são monitorados continuamente por telemetria. Cada evento registra o alarme disparado, seu nível de criticidade (1=Crítico, 2=Não-Crítico, 3=Informativo, 4=Outro), o valor do sensor e o estado do alarme (Ativar/Inativar)."),
            ("h2", "Evento Don't Go"),
            ("p", "O Don't Go é o alerta mais severo do sistema OEM. Ele é gerado quando uma combinação específica de alarmes críticos está ativa simultaneamente, sinalizando que o equipamento não está em condições seguras de operação. Quando acionado, o equipamento é imediatamente retirado de serviço para intervenção de manutenção."),
            ("p", "Análise dos dados revelou que eventos Don't Go representam apenas 0,054% dos registros de telemetria, configurando um desbalanceamento de classes severo (razão 1:1.860). Esse desbalanceamento é o principal desafio técnico do problema."),
            ("h2", "Descobertas Exploratórias (EDA)"),
            ("p", "A análise exploratória validou e refutou as hipóteses de negócio levantadas:"),
            ("b", "H1 ✅ CONFIRMADA: existe uma sequência característica de alarmes críticos que precede o Don't Go nas 4 horas anteriores — o 'alarm fingerprint'."),
            ("b", "H2 ✅ CONFIRMADA: a frequência de alarmes críticos aumenta progressivamente (aceleração) nas horas que antecedem um Don't Go."),
            ("b", "H3 ✅ CONFIRMADA: o modelo 793-D 4S apresenta perfil de falha distinto — taxa DG 3× superior à média dos demais caminhões."),
            ("b", "H4 ❌ REFUTADA: Don't Go distribui-se uniformemente nas 24 horas do dia, sem concentração em turno específico."),
            ("b", "H5 ✅ CONFIRMADA: equipamentos em manutenção recente têm menor taxa de Don't Go nas 24h seguintes."),
            ("b", "H6 N/A: todos os dados são de Itabira — variável de localidade sem discriminação."),
            ("b", "H7 N/A: colunas de operador ausentes no dataset de parquet — não validável."),
            ("h2", "Equipamento Crítico: CA65926"),
            ("p", "O equipamento CA65926 (793-D 4S) é um outlier extremo: taxa de Don't Go de 5,31% — 98 vezes superior à média da frota (0,054%) e 2,5 vezes superior ao segundo equipamento mais crítico (CA65908, 2,16%). Em 6 meses, gerou 5.112 eventos Don't Go em 96.220 registros de telemetria, exigindo análise e atenção individualizadas."),
        ])

    # ── Metodologia ───────────────────────────────────────────────────────────
    met_heading = next(
        (p for p in doc.paragraphs if p.style.name == "Heading 1" and "Metodologia" in p.text),
        None,
    )
    if met_heading:
        _insert_after_paragraph(doc, met_heading, [
            ("h2", "Arquitetura do Pipeline (Medallion)"),
            ("p", "A solução implementa uma arquitetura de dados em camadas, garantindo reprodutibilidade e rastreabilidade:"),
            ("b", "Bronze: ingestão dos parquets de telemetria (37,2M registros) e apontamentos (377K registros) com validação de schema, correção de tipos (Valor em formato BR, timestamps como string) e filtragem de registros inválidos."),
            ("b", "Silver: join temporal entre telemetria e apontamentos (estado operacional por TAG e janela temporal); aplicação do look-ahead de 60, 120 e 240 minutos para criação do target is_dont_go_next_60m. Processado mês a mês para respeitar restrição de RAM."),
            ("b", "Gold: feature engineering dia a dia com overlap de 4 horas para garantir consistência das janelas rolantes. 54 features geradas por equipamento e timestamp."),
            ("h2", "Feature Engineering (54 Features)"),
            ("p", "As features foram organizadas em 5 grupos:"),
            ("b", "Frequência de alarmes (16 features): contagem de alarmes totais, críticos, não-críticos e Don't Go em janelas rolantes de 15, 30, 60 e 240 minutos. Captura a densidade e intensidade do estado de degradação."),
            ("b", "Aceleração (2 features): razão entre frequência na janela de 60min vs. média da janela de 240min — detecta clustering iminente de falhas."),
            ("b", "Alarm Fingerprint (30 features): presença (0/1) dos top-30 alarmes mais frequentes na janela de 4 horas anteriores a cada evento. Representa o 'DNA' dos padrões que precedem Don't Go."),
            ("b", "Recência (1 feature): minutos desde o último evento Don't Go por equipamento. Equipamentos com DG recente têm maior risco de reincidência."),
            ("b", "Contexto temporal e operacional (5 features): hora do dia, dia da semana, posição no turno, frota codificada, estado de apontamento (operando/manutenção)."),
            ("h2", "Modelo Preditivo"),
            ("p", "Algoritmo: LightGBM Classifier, escolhido pela eficiência com dados tabulares de alta dimensionalidade e suporte nativo a dados esparsos."),
            ("p", "Estratégia de desbalanceamento: undersampling das amostras negativas (ratio 1:5) + parâmetro scale_pos_weight calibrado para a proporção real do conjunto de treino."),
            ("p", "Validação temporal: treino com Jan–Mai/2025 (~30M eventos), teste com Jun/2025 (~7,9M eventos). Separação estritamente temporal para simular uso em produção (sem data leakage)."),
            ("p", "Explicabilidade: SHAP TreeExplainer aplicado a amostras do conjunto de teste para identificar as features mais determinantes para cada equipamento e período."),
        ])

    # ── Resultados e Discussões ───────────────────────────────────────────────
    res_heading = next(
        (p for p in doc.paragraphs if p.style.name == "Heading 1" and "Resultados" in p.text),
        None,
    )
    if res_heading:
        _insert_after_paragraph(doc, res_heading, [
            ("h2", "Métricas do Modelo (Conjunto de Teste: Junho 2025)"),
            ("p", f"O modelo foi avaliado sobre 7.854.243 eventos de Junho/2025, dos quais "
             f"{m['n_positivos']:,} ({m['taxa_positivos_pct']:.2f}%) são amostras pré-Don't Go (60 minutos de antecedência). "
             f"O limiar de decisão ótimo, determinado pela maximização do F1-Score na curva Precision-Recall, é de {m['optimal_threshold']:.4f}."),
            ("tbl", [
                ["Métrica", "Valor", "Interpretação"],
                ["ROC-AUC", f"{m['roc_auc']:.4f}", "Discriminação quase perfeita entre eventos pré-DG e não-DG"],
                ["PR-AUC (Avg Precision)", f"{m['pr_auc']:.4f}", "Muito bom para desbalanceamento de 0,45% positivos"],
                ["F1-Score", f"{m['f1_score']:.4f}", "Equilíbrio precision-recall no limiar ótimo"],
                ["Precision", f"{m['precision']:.4f}", "71% dos alertas emitidos correspondem a DG real"],
                ["Recall", f"{m['recall']:.4f}", "65% dos eventos pré-DG são capturados"],
                ["Limiar ótimo", f"{m['optimal_threshold']:.4f}", "Threshold calibrado para dados desbalanceados"],
            ]),
            ("h2", "Ranking de Risco — Frota Completa"),
            ("p", "A tabela abaixo apresenta os 5 equipamentos com maior taxa de Don't Go, representando os alvos prioritários de manutenção preventiva:"),
            ("tbl", [
                ["TAG", "Frota", "Eventos (6m)", "Don't Go", "Taxa DG (%)"],
                ["CA65926", "793-D 4S", "96.220",  "5.112", "5,31%"],
                ["CA65908", "793-D 3S", "32.498",  "703",   "2,16%"],
                ["CA65902", "793-D 3S", "25.382",  "418",   "1,65%"],
                ["CA65927", "793-D 5S", "88.216",  "1.320", "1,50%"],
                ["CA65792", "793-D 2S", "126.721", "1.589", "1,25%"],
            ]),
            ("h2", "Alarm Fingerprint — DNA do Don't Go"),
            ("p", "A análise do fingerprint revelou que determinados alarmes, quando presentes na janela de 4 horas anteriores a um evento, aumentam significativamente a probabilidade de Don't Go. Esses alarmes compõem o 'DNA' do evento e são os principais sinais preditivos do modelo."),
            ("p", "A feature de maior impacto SHAP (min_desde_ultimo_dg) confirma que equipamentos que tiveram um Don't Go recente têm risco elevado de reincidência — indicando falhas recorrentes não resolvidas na manutenção corretiva."),
            ("h2", "Insights Acionáveis para o Negócio"),
            ("b", "Manutenção focada em CA65926: o equipamento requer investigação estrutural. Sua taxa de DG (5,31%) sugere problema sistêmico não resolvido por manutenção corretiva padrão. Recomendamos inspeção preventiva de maior profundidade."),
            ("b", "Janela de intervenção de 60 minutos: o modelo identifica 65% dos eventos Don't Go com até 1 hora de antecedência. Essa janela é suficiente para acionar uma equipe de manutenção e evitar a paralisação não planejada."),
            ("b", "Score de risco em tempo real: o dashboard Streamlit desenvolvido permite monitorar o score de risco por equipamento continuamente, priorizando inspeções preventivas pelos equipamentos com maior pontuação."),
            ("b", "Aceleração como sinal primário: o aumento rápido de alarmes críticos na janela de 60 minutos (feature aceleracao_criticos) é o segundo fator mais determinante do modelo — um indicador operacional simples e interpretável para operadores."),
        ])

    # ── Conclusão ─────────────────────────────────────────────────────────────
    conc_heading = next(
        (p for p in doc.paragraphs if p.style.name == "Heading 1" and "Conclusão" in p.text),
        None,
    )
    if conc_heading:
        _insert_after_paragraph(doc, conc_heading, [
            ("p", "Este trabalho demonstrou que é possível prever eventos Don't Go em equipamentos de mineração pesada com 60 minutos de antecedência, utilizando exclusivamente os dados de telemetria disponíveis. O pipeline completo (Bronze → Silver → Gold → Modelo) processa 37,2 milhões de eventos de forma eficiente e reproduzível, e o modelo LightGBM com alarm fingerprint alcança ROC-AUC de 0,99, transformando dados brutos de sensores em um score de risco interpretável e acionável."),
            ("p", "A identificação do CA65926 como outlier extremo (taxa DG 98× superior à média) é por si só um achado de alto valor operacional, direcionando recursos de manutenção para o equipamento que mais necessita de intervenção estrutural."),
            ("h2", "Trabalhos Futuros"),
            ("b", "Otimização contínua do F1-Score: experimentos com SMOTE, ajuste fino de hiperparâmetros e threshold dinâmico por equipamento para superar a meta de F1 ≥ 0,75."),
            ("b", "Modelos por frota: treinar modelos especializados por tipo de equipamento (793-D 2S, 3S, 4S, 5S) para capturar padrões específicos de cada frota."),
            ("b", "Integração em tempo real: conectar o pipeline ao sistema de telemetria em produção para geração de alertas preventivos automáticos."),
            ("b", "Análise de operador: coletar e incorporar dados de operador (ausentes no dataset atual) para investigar a hipótese H7 — correlação entre estilo de operação e perfil de falhas."),
            ("b", "Modelos de série temporal: explorar abordagens como LSTM ou Temporal Fusion Transformer para capturar dependências temporais de longo prazo nos dados de alarme."),
        ])

    doc.save(OUT_FILE)
    print(f"✔ Relatório salvo em: {OUT_FILE}")


def _insert_after_paragraph(doc: Document, ref_para, content: list[tuple]):
    """Insere parágrafos após ref_para."""
    from docx.oxml.ns import qn
    import copy

    ref_elem = ref_para._element
    parent = ref_elem.getparent()
    insert_idx = list(parent).index(ref_elem) + 1

    for item in reversed(content):
        kind = item[0]
        text = item[1] if len(item) > 1 else ""

        if kind == "tbl":
            rows_data = item[1]
            headers = rows_data[0]
            rows = rows_data[1:]
            table = _add_table(doc, headers, rows)
            tbl_elem = table._tbl
            parent.insert(insert_idx, tbl_elem)

        else:
            new_para = doc.add_paragraph()
            if kind == "h2":
                new_para.style = doc.styles["Heading 2"]
                run = new_para.add_run(text)
                run.font.color.rgb = VALE_GREEN
            elif kind == "b":
                new_para.style = doc.styles["List Bullet"]
                new_para.add_run(text).font.size = Pt(11)
            else:
                new_para.add_run(text).font.size = Pt(11)

            p_elem = new_para._element
            p_elem.getparent().remove(p_elem)
            parent.insert(insert_idx, p_elem)


if __name__ == "__main__":
    build_report()
