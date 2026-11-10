# Servidor do CIn: como rodar os experimentos

Guia da F3 (Bernardo). Os campos marcados com **PREENCHER** dependem do servidor e
são completados na primeira configuração.

## 1. Acesso

- Endereço: `ssh <login>@slurm-client1.cin.ufpe.br` (ou `slurm-client2`). Esses nomes
  só existem dentro da rede do CIn: fora do campus, conecte antes na VPN do CIn.
- Gerenciador de filas: **SLURM** (os scripts em `scripts/jobs/` usam `sbatch`).
  Partição usada: `short-simple` (até 2 dias por job). A `debug` recusa jobs com
  "Invalid qos specification". O nó de login **não tem GPU**: tudo que usa GPU vai
  por `srun`/`sbatch`.
- GPUs (conferido em 10/11/2026): NVIDIA A100 80GB, driver 590.48 (CUDA 13.1).
  - `cluster-node6` divide uma A100 em 7 fatias **MIG 1g.10gb** (9,5 GiB e ~1/7 do
    processamento cada). É onde o job cai quando as GPUs inteiras estão ocupadas.
  - GPUs inteiras: `cluster-node4`, `5`, `7` e `9` (2 cada), quase sempre ocupadas.
    Os nós 1, 2, 3, 8 e 10 estavam fora do ar.
  - Para exigir GPU inteira: `sbatch --exclude=cluster-node6 ...`. Para ver quando
    o job começaria sem submeter: `sbatch --test-only ...`.
  - Os tempos e a memória da Tabela 2 do artigo são numa A100 inteira. Se uma rodada
    usar MIG, registrar isso na tabela de custo.
- Pasta do projeto: `~/projeto-GraphIDS-equipe-10` (home em NFS, 15 TB livres).

## 2. Ambiente

O projeto não usa PyTorch Geometric, só PyTorch. Instale a versão do PyTorch que
combina com o driver CUDA do servidor (veja `nvidia-smi`, linha "CUDA Version"):

```bash
git clone https://github.com/rxavier1904/projeto-GraphIDS-equipe-10.git
cd projeto-GraphIDS-equipe-10
python3 -m venv .venv            # ou: uv venv --python 3.11 .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # ajustar cu1xx
pip install -e ".[dev]"
python scripts/check_env.py      # confere versões e faz um teste na GPU
pytest -q                        # testes sintéticos (segundos)
```

Versões instaladas no servidor (saída do `check_env.py` num nó com GPU, 10/11/2026;
vai para a seção 6 do relatório): Python 3.10.12, PyTorch 2.14.1+cu130 (CUDA 13.0,
cuDNN 9.24), NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, Ubuntu 22.04.

No servidor, o `pip install torch` padrão já traz o CUDA certo (não precisa do
`--index-url`). A instalação na home em NFS leva cerca de 10 minutos.

## 3. Datasets

Os CSVs ficam em `data/<dataset>/<dataset>.csv` (fora do git). Os links da UQ abrem
uma página, não o arquivo. Baixe o zip pelo navegador e copie para `data/raw/`:

```bash
# na sua máquina
scp NF-UNSW-NB15-v3.zip <login>@<servidor>:<pasta do projeto>/data/raw/
# no servidor: descompacta, renomeia o CIC-IDS2018 e registra tamanho e SHA-256
scripts/download_datasets.sh data
```

Se o navegador mostrar um link direto do zip, dá para baixar direto no servidor:
`URL_NF_UNSW_NB15_v3=<link> scripts/download_datasets.sh data NF-UNSW-NB15-v3`.

| Dataset | Fluxos (site da UQ) | CSV (bytes) | SHA-256 do zip |
|---|---|---|---|
| NF-UNSW-NB15-v2 | 2.390.275 | PREENCHER | PREENCHER |
| NF-UNSW-NB15-v3 | 2.365.424 | PREENCHER | PREENCHER |
| NF-CSE-CIC-IDS2018-v2 | 18.893.708 | PREENCHER | PREENCHER |
| NF-CSE-CIC-IDS2018-v3 | 20.115.529 | PREENCHER | PREENCHER |
| NF-ToN-IoT-v3 | 27.520.260 | PREENCHER | PREENCHER |

Os valores saem de `data/MANIFEST.tsv`, gravado pelo script.

Estatísticas da seção 6 (também gravam o cache dos grafos, que o treino reaproveita):

```bash
python -m graphids.data.stats --seed 1 \
    --config configs/NF-UNSW-NB15-v2.yaml --config configs/NF-UNSW-NB15-v3.yaml \
    --config configs/NF-CSE-CIC-IDS2018-v2.yaml --config configs/NF-CSE-CIC-IDS2018-v3.yaml
```

## 4. Rodadas

```bash
# teste rápido no servidor (poucas épocas), antes das rodadas oficiais
scripts/run_seeds.sh configs/NF-UNSW-NB15-v3.yaml 1 -- --set num_epochs=2 \
    --output-dir results/smoke

# rodadas oficiais com SLURM: 4 datasets × seeds 1, 2 e 3
scripts/jobs/submit_all.sh
squeue -u $USER                       # acompanhar
tail -f results/logs/<dataset>_seed<s>.log

# sem SLURM: uma sessão tmux por dataset
tmux new -s unsw3 'scripts/run_seeds.sh configs/NF-UNSW-NB15-v3.yaml 1 2 3'
```

Cada execução grava `results/runs/<dataset>/seed<s>/` (contrato C2). Uma seed com
`metrics.json` é pulada se o script rodar de novo. Para refazer, apague a pasta dela.

## 5. Tabelas

```bash
python scripts/aggregate_results.py   # results/tabela_{reproducao,custo}.{csv,tex}
```

Vão para o git: `results/runs/**/{metrics.json,history.json,config.yaml}` e as
tabelas. Os checkpoints, os `scores_*.npz` e os logs ficam fora (`.gitignore`).
Para a F4 gerar as figuras, copie os `scores_*.npz` do servidor com `rsync`.

## 6. Problemas conhecidos

- **PREENCHER** à medida que aparecerem (falta de memória, tempo do job etc.) e
  avisar a F1.
