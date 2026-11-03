"""Previsão de IPCA e Selic: comparação de modelos com backtest walk-forward.
Dados: API SGS do Banco Central (séries 433 = IPCA mensal, 432 = meta da Selic).
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import modelos

st.set_page_config(page_title="Previsão IPCA e Selic", page_icon="🔮", layout="wide")

CORES = {"painel": "#121C2E", "borda": "#26364D", "texto": "#F4F7FB", "sec": "#A9B8CC",
         "verde": "#10B981", "azul": "#3B82F6", "amarelo": "#F59E0B", "vermelho": "#EF4444"}
PALETA = ["#A9B8CC", "#F59E0B", "#3B82F6", "#8B5CF6", "#EF4444", "#EC4899"]
URL_BCB = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
SERIES = {
    "IPCA (variação mensal, %)": {"codigo": 433, "unidade": "% no mês", "agregar": None},
    "Selic (meta, % a.a.)": {"codigo": 432, "unidade": "% a.a.", "agregar": "last"},
}

st.markdown(
    f"""<style>
    :root {{ color-scheme: dark; }}
    .stApp {{ background: #0B1120; color: {CORES["texto"]}; }}
    .block-container {{ max-width: 1400px; padding-top: 1.1rem; }}
    [data-testid="stSidebar"] {{ background: {CORES["painel"]}; border-right: 1px solid {CORES["borda"]}; }}
    [data-testid="stMetric"] {{ background: linear-gradient(145deg,#162238,#111B2C);
        border: 1px solid {CORES["borda"]}; padding: 16px 18px; border-radius: 15px; }}
    [data-testid="stMetricLabel"] p {{ color: {CORES["sec"]} !important; }}
    [data-testid="stPlotlyChart"] {{ background: {CORES["painel"]}; border: 1px solid {CORES["borda"]};
        border-radius: 15px; padding: 7px; }}
    .hero {{ background: linear-gradient(120deg,#15243A,#1B1B3A); border: 1px solid #2C2F66;
        border-radius: 18px; padding: 22px 26px; margin-bottom: 20px; }}
    .hero-eyebrow {{ color: #8B5CF6; text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; font-weight: 700; }}
    .hero-title {{ color: #fff; font-size: clamp(1.6rem,3vw,2.2rem); font-weight: 800; }}
    .hero-sub {{ color: #fff; font-size: .94rem; margin-top: 6px; }}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner="Consultando a API do Banco Central...")
def baixar_serie(codigo: int, agregar: str | None, ano_inicial: int = 2012) -> pd.Series:
    """Baixa a série em janelas de 5 anos (a API limita consultas diárias a 10 anos)."""
    hoje, quadros, ano = date.today(), [], ano_inicial
    while ano <= hoje.year:
        fim = min(date(ano + 4, 12, 31), hoje)
        params = {"formato": "json", "dataInicial": f"01/01/{ano}", "dataFinal": fim.strftime("%d/%m/%Y")}
        r = requests.get(URL_BCB.format(codigo=codigo), params=params,
                         headers={"User-Agent": "previsao-ipca-selic"}, timeout=30)
        r.raise_for_status()
        quadros.append(pd.DataFrame(r.json()))
        ano += 5
    df = pd.concat(quadros, ignore_index=True)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    serie = df.dropna().drop_duplicates("data").set_index("data")["valor"].sort_index()
    if agregar == "last":
        serie = serie.resample("MS").last()
    return serie.asfreq("MS").dropna()


@st.cache_data(show_spinner="Rodando backtest walk-forward...", ttl=3600)
def avaliar(y: pd.Series, h: int, k: int, nomes: tuple):
    nomes = list(nomes)
    return modelos.backtest(y, h, k, nomes), modelos.prever_final(y, h, nomes)


def fmt(v: float, casas: int = 2) -> str:
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------- Barra lateral ----------------
st.sidebar.markdown("## 🔮 Previsão Macro")
st.sidebar.caption("IPCA E SELIC · BACKTEST DE MODELOS")
st.sidebar.divider()
nome_serie = st.sidebar.selectbox("Série", list(SERIES))
horizonte = st.sidebar.slider("Horizonte (meses)", 1, 12, 6)
cortes = st.sidebar.slider("Cortes do backtest", 6, 24, 12,
                           help="Quantas vezes o modelo é treinado no passado e testado nos meses seguintes.")
disponiveis = list(modelos.MODELOS)
escolhidos = st.sidebar.multiselect("Modelos", disponiveis, default=[m for m in disponiveis if m != "Chronos-Bolt"])
if "Ingênuo" not in escolhidos:
    escolhidos.insert(0, "Ingênuo")  # a referência é sempre incluída

config = SERIES[nome_serie]
try:
    y = baixar_serie(config["codigo"], config["agregar"])
except requests.RequestException as erro:
    st.error(f"Não foi possível consultar a API do Banco Central: {erro}")
    st.stop()

st.markdown(
    f"""<div class="hero"><div class="hero-eyebrow">Machine Learning · Séries Temporais</div>
    <div class="hero-title">Previsão de IPCA e Selic</div>
    <div class="hero-sub">{nome_serie} &nbsp;·&nbsp; {len(y)} meses de histórico
    &nbsp;·&nbsp; Fonte: Banco Central do Brasil</div></div>""",
    unsafe_allow_html=True,
)

try:
    erros, previsoes = avaliar(y, horizonte, cortes, tuple(escolhidos))
except ValueError as erro:
    st.warning(str(erro))
    st.stop()

tabela = modelos.resumo(erros)
melhor = tabela.iloc[0]["Modelo"]
futuro = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=horizonte, freq="MS")
inf, sup = modelos.intervalo(previsoes[melhor], erros[melhor])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Melhor modelo no backtest", melhor)
c2.metric(f"Previsão para {futuro[0]:%m/%Y}", f"{fmt(previsoes[melhor][0])} {config['unidade']}")
c3.metric("Erro médio (MAE)", f"{fmt(tabela.iloc[0]['MAE'], 3)} p.p.")
ganho = tabela.iloc[0]["Ganho vs. ingênuo (%)"]
c4.metric("Ganho vs. ingênuo", f"{fmt(ganho, 1)}%",
          help="Redução percentual do erro (MAE) em relação a repetir o último valor.")

aba_graf, aba_met, aba_prev, aba_cod = st.tabs(
    ["📈 Previsão", "🏁 Comparação de modelos", "🗂️ Tabela", "💻 Código dos modelos"]
)

with aba_graf:
    historico = y.iloc[-48:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historico.index, y=historico.values, name="Histórico",
                             line=dict(color=CORES["verde"], width=3)))
    fig.add_trace(go.Scatter(x=list(futuro) + list(futuro[::-1]), y=list(sup) + list(inf[::-1]),
                             fill="toself", fillcolor="rgba(139,92,246,0.18)", line=dict(width=0),
                             name=f"Intervalo 80% ({melhor})", hoverinfo="skip"))
    for i, (nome, prev) in enumerate(previsoes.items()):
        fig.add_trace(go.Scatter(x=futuro, y=prev, name=nome, mode="lines+markers",
                                 line=dict(color=PALETA[i % len(PALETA)], width=3 if nome == melhor else 1.5,
                                           dash="solid" if nome == melhor else "dot")))
    fig.update_layout(template="plotly_dark", height=470, paper_bgcolor=CORES["painel"],
                      plot_bgcolor=CORES["painel"], margin=dict(l=18, r=18, t=40, b=20),
                      title=f"{nome_serie}: histórico e previsões",
                      yaxis_title=config["unidade"], legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="rgba(169,184,204,0.10)")
    fig.update_yaxes(gridcolor="rgba(169,184,204,0.10)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("O intervalo é empírico: usa os erros que o melhor modelo cometeu no backtest, mês a mês.")

with aba_met:
    exibir = tabela.rename(columns={"MAE": "MAE (p.p.)", "RMSE": "RMSE (p.p.)"})
    st.dataframe(exibir.style.format({"MAE (p.p.)": "{:.3f}", "RMSE (p.p.)": "{:.3f}",
                                      "Ganho vs. ingênuo (%)": "{:+.1f}"}),
                 use_container_width=True, hide_index=True)
    barras = go.Figure(go.Bar(x=tabela["Modelo"], y=tabela["MAE"], marker_color=CORES["azul"]))
    barras.update_layout(template="plotly_dark", height=330, paper_bgcolor=CORES["painel"],
                         plot_bgcolor=CORES["painel"], title="Erro médio absoluto por modelo (menor é melhor)",
                         margin=dict(l=18, r=18, t=50, b=20))
    st.plotly_chart(barras, use_container_width=True)
    st.info("Ganho negativo significa que o modelo foi **pior** que simplesmente repetir o último valor. "
            "Isso é comum, principalmente em séries com poucos dados e muito ruído.")

with aba_prev:
    saida = pd.DataFrame({"Mês": futuro.strftime("%m/%Y"), f"Previsão ({melhor})": previsoes[melhor],
                          "Mínimo (80%)": inf, "Máximo (80%)": sup})
    for nome, prev in previsoes.items():
        if nome != melhor:
            saida[nome] = prev
    st.dataframe(saida.round(3), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Baixar CSV", saida.to_csv(index=False).encode("utf-8"),
                       file_name="previsoes.csv", mime="text/csv")

with aba_cod:
    st.caption("Modelos e backtest usados neste app (arquivo modelos.py).")
    st.code((Path(__file__).parent / "modelos.py").read_text(encoding="utf-8"), language="python")

st.divider()
st.caption("Projeto educacional: não é recomendação de investimento. Fonte: Banco Central do Brasil (API SGS).")
