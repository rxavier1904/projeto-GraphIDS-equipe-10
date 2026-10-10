# F4: Novo dataset (NF-ToN-IoT-v3), avaliação e figuras

**Responsável:** Felipe · **Peso:** ~20% · **Módulos:** `graphids/evaluation/`, `configs/NF-ToN-IoT-v3.yaml`, `notebooks/eda_ton_iot.ipynb`, `notebooks/figuras.ipynb`

Cuida do segundo requisito obrigatório (avaliar o GraphIDS em outro dataset) e de
todas as métricas e figuras de análise.

## Entregas

### 4.1 Análise exploratória e justificativa do NF-ToN-IoT-v3 (semana 1)
- Número de fluxos, hosts, proporção de ataques e categorias de ataque.
- Comparação com os datasets do artigo (escala, ambiente e ataques).
- Texto de justificativa para o relatório. Pontos principais:
  - mesmo conjunto de 53 features NetFlow v3, então o pipeline roda sem adaptação;
  - ambiente IoT/IIoT, diferente dos dois do artigo;
  - ataques que o artigo não cobre (ransomware, injection, XSS, MITM, backdoor, password);
  - **proporção de ataques alta**, que afeta a interpretação do PR-AUC (a linha de base
    de um classificador aleatório é a própria prevalência). Isso precisa ser discutido.
- Decidir se é preciso amostrar (`fraction`), conforme o tamanho e a memória disponível.

### 4.2 Métricas: `graphids/evaluation/metrics.py` (semana 2)
Contrato C4: PR-AUC (`average_precision_score`), F1 macro, matriz de confusão e
**taxa de detecção por categoria de ataque** (como nas Figuras 5 a 8 do artigo).

### 4.3 Figuras: `graphids/evaluation/plots.py` (semana 2, com dados sintéticos)
Tudo lido de `scores_*.npz` (contrato C2):
- curvas precisão-revocação por dataset (Figura 4);
- distribuição do score de anomalia por tipo de ataque com o threshold, em escala log
  (Figuras 5 a 8);
- t-SNE dos embeddings por tipo de ataque (Figuras 9 a 12), com amostragem para caber
  na memória;
- figuras salvas em PDF ou PNG em `results/figuras/`.

### 4.4 Configuração e execução no ToN-IoT (semanas 3 e 4)
- `configs/NF-ToN-IoT-v3.yaml` partindo da config do NF-UNSW-NB15-v3.
- Ajuste leve de hiperparâmetros **pela PR-AUC de validação** (nunca pelo teste):
  learning rate, dropout do GNN e `edim_out`. Pequena grade combinada com o Bernardo
  para as rodadas no servidor.
- Rodadas finais com 3 seeds e a mesma agregação da F3.

### 4.5 Relatório
Seção 6 (novo dataset: o que é, por que foi escolhido, splits e contagens por classe),
seção 7 (resultados no ToN-IoT, comparação com os datasets do artigo e ataques mais
difíceis) e apoio ao modelo de ameaça com os ataques do ToN-IoT.

## Testes
- Métricas comparadas com casos calculados à mão.
- As funções de figura rodam com um `.npz` sintético sem erro.

## Depende de
- F2 para as estatísticas por split (reaproveitar `stats.py`). F1 e F3 para as rodadas.
