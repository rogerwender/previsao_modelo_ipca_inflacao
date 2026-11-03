# 🔮 Previsão de IPCA e Selic

App em **Streamlit** que prevê o **IPCA mensal** e a **meta da Selic** e compara vários modelos com **backtest walk-forward**, usando dados abertos da API do Banco Central (SGS).

🔗 **Demo:** _cole aqui o link do app publicado_

## O que o projeto faz

- Baixa as séries do Banco Central (IPCA: série 433; Selic meta: série 432) e transforma tudo em séries mensais.
- Compara modelos de naturezas diferentes:
  - **Ingênuo** (repete o último valor) e **média móvel de 12 meses**: referências simples.
  - **AutoARIMA** e **AutoETS** (`statsforecast`): estatística clássica.
  - **LightGBM** com lags e médias móveis: machine learning.
  - **Chronos-Bolt** (opcional): modelo de fundação para séries temporais, zero-shot.
- Avalia cada modelo com **backtest walk-forward**: treina no passado, prevê os meses seguintes e repete em vários pontos de corte.
- Mostra MAE, RMSE e o **ganho sobre o modelo ingênuo**, além de **intervalo de previsão empírico** (calculado dos erros do backtest).

## Estrutura

| Arquivo | Função |
|---|---|
| `app.py` | Interface em Streamlit e coleta de dados |
| `modelos.py` | Modelos, backtest e métricas (sem dependência do Streamlit) |
| `requirements.txt` | Dependências |

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Ativar o Chronos (opcional)

O Chronos depende do PyTorch, que é pesado e pode não caber na hospedagem gratuita. Para testar localmente:

```bash
pip install torch chronos-forecasting
```

Com o pacote instalado, "Chronos-Bolt" aparece automaticamente na lista de modelos.

## Como publicar (grátis)

1. Suba a pasta para um repositório público no GitHub.
2. No Streamlit Community Cloud (share.streamlit.io), crie um app apontando para `app.py`.
3. Cole o link no topo deste README e no LinkedIn.

## Limitações (e por que importam)

- Poucos dados mensais e muito ruído: o modelo ingênuo costuma ser difícil de bater. O app mostra isso de forma transparente.
- A previsão da Selic reflete o histórico e não as decisões futuras do Copom.
- Projeto educacional, não é recomendação de investimento.

## Próximos passos

- Comparar as previsões com as expectativas do mercado (boletim Focus, também no SGS).
- Adicionar variáveis explicativas (câmbio, expectativas) aos modelos de ML.
- Calibrar os hiperparâmetros do LightGBM dentro do backtest.
