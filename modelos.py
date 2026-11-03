"""Modelos de previsão e backtest walk-forward (sem dependência do Streamlit).

Todo modelo recebe uma série mensal `y` (pd.Series com índice mensal) e um
horizonte `h`, e devolve um array com as `h` próximas previsões.
"""

import importlib.util

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Modelos
# ---------------------------------------------------------


def prever_ingenuo(y: pd.Series, h: int) -> np.ndarray:
    """Repete o último valor observado. Referência mínima que todo modelo deve tentar bater."""
    return np.repeat(y.iloc[-1], h)


def prever_media_movel(y: pd.Series, h: int) -> np.ndarray:
    """Repete a média dos últimos 12 meses."""
    return np.repeat(y.iloc[-12:].mean(), h)


def _statsforecast(y: pd.Series, h: int, modelo: str) -> np.ndarray:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS

    df = pd.DataFrame({"unique_id": "serie", "ds": y.index, "y": y.values})
    algoritmo = AutoARIMA(season_length=12) if modelo == "arima" else AutoETS(season_length=12)
    sf = StatsForecast(models=[algoritmo], freq="MS")
    return sf.forecast(df=df, h=h).iloc[:, -1].to_numpy()


def prever_autoarima(y: pd.Series, h: int) -> np.ndarray:
    return _statsforecast(y, h, "arima")


def prever_autoets(y: pd.Series, h: int) -> np.ndarray:
    return _statsforecast(y, h, "ets")


def prever_lightgbm(y: pd.Series, h: int) -> np.ndarray:
    """Regressão com defasagens (lags) e médias móveis, previsão recursiva."""
    import lightgbm as lgb

    lags = [1, 2, 3, 6, 12]

    def atributos(v):
        return [v[-lag] for lag in lags] + [np.mean(v[-3:]), np.mean(v[-6:])]

    valores = list(y.values)
    X = np.array([atributos(valores[:i]) for i in range(12, len(valores))])
    alvo = np.array(valores[12:])

    modelo = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=5, verbose=-1
    )
    modelo.fit(X, alvo)

    previsoes = []
    for _ in range(h):
        p = float(modelo.predict(np.array([atributos(valores)]))[0])
        previsoes.append(p)
        valores.append(p)
    return np.array(previsoes)


_PIPELINE_CHRONOS = None


def prever_chronos(y: pd.Series, h: int) -> np.ndarray:
    """Modelo de fundação para séries temporais (zero-shot, sem treinar)."""
    global _PIPELINE_CHRONOS
    import torch
    from chronos import BaseChronosPipeline

    if _PIPELINE_CHRONOS is None:
        _PIPELINE_CHRONOS = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny", device_map="cpu", torch_dtype=torch.float32
        )
    contexto = torch.tensor(y.values, dtype=torch.float32)
    _, media = _PIPELINE_CHRONOS.predict_quantiles(
        context=contexto, prediction_length=h, quantile_levels=[0.5]
    )
    return media[0].numpy()


MODELOS = {
    "Ingênuo": prever_ingenuo,
    "Média móvel 12m": prever_media_movel,
    "AutoARIMA": prever_autoarima,
    "AutoETS": prever_autoets,
    "LightGBM": prever_lightgbm,
}
# O Chronos é opcional: só aparece se o pacote estiver instalado
if importlib.util.find_spec("chronos") is not None:
    MODELOS["Chronos-Bolt"] = prever_chronos


# ---------------------------------------------------------
# Backtest walk-forward
# ---------------------------------------------------------


def _prever_seguro(funcao, y: pd.Series, h: int) -> np.ndarray:
    try:
        return np.asarray(funcao(y, h), dtype=float)
    except Exception:
        return np.full(h, np.nan)


def backtest(y: pd.Series, h: int, k: int, nomes: list[str]) -> dict[str, np.ndarray]:
    """Treina no passado e prevê `h` meses à frente, repetindo em `k` pontos de corte.

    Devolve, para cada modelo, uma matriz (k, h) com os erros (real - previsto).
    """
    n = len(y)
    erros = {nome: np.full((k, h), np.nan) for nome in nomes}
    for i in range(k):
        fim_treino = n - h - (k - 1) + i
        if fim_treino < 36:
            raise ValueError("Série curta demais para esse horizonte e número de cortes.")
        treino = y.iloc[:fim_treino]
        real = y.iloc[fim_treino : fim_treino + h].to_numpy()
        for nome in nomes:
            erros[nome][i] = real - _prever_seguro(MODELOS[nome], treino, h)
    return erros


def prever_final(y: pd.Series, h: int, nomes: list[str]) -> dict[str, np.ndarray]:
    """Previsão para o futuro, treinando com toda a série."""
    return {nome: _prever_seguro(MODELOS[nome], y, h) for nome in nomes}


def resumo(erros: dict[str, np.ndarray]) -> pd.DataFrame:
    """Tabela de métricas (MAE e RMSE) e ganho percentual sobre o modelo ingênuo."""
    linhas = []
    for nome, e in erros.items():
        if np.isnan(e).all():
            continue
        linhas.append(
            {"Modelo": nome, "MAE": np.nanmean(np.abs(e)), "RMSE": np.sqrt(np.nanmean(e**2))}
        )
    tabela = pd.DataFrame(linhas).sort_values("MAE").reset_index(drop=True)
    base = tabela.loc[tabela["Modelo"] == "Ingênuo", "MAE"].iloc[0]
    tabela["Ganho vs. ingênuo (%)"] = (1 - tabela["MAE"] / base) * 100
    return tabela


def intervalo(previsao: np.ndarray, erros: np.ndarray, nivel: float = 0.8):
    """Intervalo empírico: soma à previsão os quantis dos erros do backtest, passo a passo."""
    inferior, superior = (1 - nivel) / 2, 1 - (1 - nivel) / 2
    q_inf = np.nanquantile(erros, inferior, axis=0)
    q_sup = np.nanquantile(erros, superior, axis=0)
    return previsao + q_inf, previsao + q_sup
