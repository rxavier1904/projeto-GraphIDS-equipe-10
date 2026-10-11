# GraphIDS: reprodução (Equipe 10, CIN0114 2026.2)

Reprodução do artigo **"Self-Supervised Learning of Graph Representations for Network
Intrusion Detection"** (Guerra et al., NeurIPS 2025), feita do zero pela Equipe 10 da
disciplina CIN0114 (Técnicas de Ataque e Detecção de Intrusão, CIn-UFPE).

O GraphIDS detecta intrusões em tráfego de rede sem usar rótulos de ataque no treino.
Um E-GraphSAGE gera um embedding por fluxo com o contexto local do grafo de hosts, e um
autoencoder Transformer com máscara de atenção reconstrói esses embeddings. Fluxos com
erro de reconstrução alto são marcados como ataques.

## Equipe

| Integrante | Frente |
|---|---|
| Rafael | F1: núcleo do modelo, treino e integração (coordenação) |
| Danilo | F2: pipeline de dados e construção dos grafos |
| Bernardo | F3: infraestrutura no servidor do CIn e experimentos |
| Felipe | F4: novo dataset (NF-ToN-IoT-v3), avaliação e figuras |

## Documentação

- [Plano de desenvolvimento](docs/PLANO.md): escopo, cronograma e divisão do relatório
- [Frentes de trabalho](docs/frentes/)
- [Contratos entre módulos](docs/CONTRATOS.md)
- [Notas sobre o código oficial](docs/REFERENCIA-CODIGO-OFICIAL.md)
- [Como contribuir](docs/CONTRIBUINDO.md)

## Referência

L. Guerra, T. Chapuis, G. Duc, P. Mozharovskyi e V.-T. Nguyen, "Self-Supervised Learning
of Graph Representations for Network Intrusion Detection", em *Advances in Neural
Information Processing Systems (NeurIPS)*, vol. 38, 2025. Código oficial:
<https://github.com/lorenzo9uerra/GraphIDS>.

## Como rodar

```bash
pip install -e ".[dev]"     # instala o pacote graphids e as dependências
pytest                      # testes com dados sintéticos (rodam em segundos, sem GPU)

# treino + avaliação no dataset sintético
python -m graphids.cli --config configs/synthetic.yaml

# treino + avaliação num dataset do artigo (depois do pipeline de dados da F2)
python -m graphids.cli --config configs/NF-UNSW-NB15-v3.yaml --seed 1 --data-dir data/
```

Cada execução grava métricas, scores e checkpoint em `results/runs/<dataset>/seed<s>/`
(ver o [contrato C2](docs/CONTRATOS.md)).
