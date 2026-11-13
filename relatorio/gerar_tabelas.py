"""Gera as tabelas LaTeX do relatório a partir dos CSVs e JSONs de ``results/``.

Nenhum número do relatório é digitado à mão: as tabelas com resultados nossos
saem daqui. Os únicos valores fixos neste arquivo são os publicados no artigo de
referência (Tabelas 2 e 3 de Guerra et al., 2025), usados para comparação.

Uso, a partir da raiz do repositório:

    python relatorio/gerar_tabelas.py

As tabelas são gravadas em ``relatorio/tabelas/*.tex`` (UTF-8).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RESULTS = RAIZ / "results"
SAIDA = Path(__file__).resolve().parent / "tabelas"

ARTIGO = ["NF-UNSW-NB15-v3", "NF-CSE-CIC-IDS2018-v3", "NF-UNSW-NB15-v2",
          "NF-CSE-CIC-IDS2018-v2"]  # fmt: skip
NOVO = "NF-ToN-IoT-v3"
CURTO = {
    "NF-UNSW-NB15-v2": "UNSW-NB15-v2",
    "NF-UNSW-NB15-v3": "UNSW-NB15-v3",
    "NF-CSE-CIC-IDS2018-v2": "CIC-IDS2018-v2",
    "NF-CSE-CIC-IDS2018-v3": "CIC-IDS2018-v3",
    "NF-ToN-IoT-v3": "ToN-IoT-v3",
}

# Tabela 3 do artigo: (PR-AUC média, desvio, F1 macro média, desvio).
TABELA3 = {
    "CBLOF": {
        "NF-UNSW-NB15-v3": (0.3658, 0.0634, 0.7319, 0.0225),
        "NF-CSE-CIC-IDS2018-v3": (0.2638, 0.0263, 0.6599, 0.0130),
        "NF-UNSW-NB15-v2": (0.2102, 0.0157, 0.7046, 0.0140),
        "NF-CSE-CIC-IDS2018-v2": (0.7822, 0.0198, 0.8889, 0.0068),
    },
    "SAFE": {
        "NF-UNSW-NB15-v3": (0.8946, 0.0279, 0.9236, 0.0086),
        "NF-CSE-CIC-IDS2018-v3": (0.6294, 0.0923, 0.4662, 0.0016),
        "NF-UNSW-NB15-v2": (0.2044, 0.0267, 0.5815, 0.0164),
        "NF-CSE-CIC-IDS2018-v2": (0.5222, 0.1896, 0.4684, 0.0001),
    },
    "Anomal-E": {
        "NF-UNSW-NB15-v3": (0.9032, 0.0041, 0.9459, 0.0009),
        "NF-CSE-CIC-IDS2018-v3": (0.2555, 0.0383, 0.6709, 0.0394),
        "NF-UNSW-NB15-v2": (0.7489, 0.0074, 0.9156, 0.0217),
        "NF-CSE-CIC-IDS2018-v2": (0.9287, 0.0265, 0.9410, 0.0161),
    },
    "T-MAE": {
        "NF-UNSW-NB15-v3": (0.9914, 0.0022, 0.9933, 0.0017),
        "NF-CSE-CIC-IDS2018-v3": (0.7398, 0.0777, 0.4707, 0.0049),
        "NF-UNSW-NB15-v2": (0.5995, 0.0155, 0.8039, 0.0447),
        "NF-CSE-CIC-IDS2018-v2": (0.7375, 0.0035, 0.8453, 0.0317),
    },
    "SimpleAE": {
        "NF-UNSW-NB15-v3": (0.9996, 0.0007, 0.9838, 0.0321),
        "NF-CSE-CIC-IDS2018-v3": (0.8458, 0.0498, 0.9223, 0.0201),
        "NF-UNSW-NB15-v2": (0.7864, 0.0608, 0.8680, 0.1579),
        "NF-CSE-CIC-IDS2018-v2": (0.9310, 0.0126, 0.9450, 0.0092),
    },
    "GraphIDS (artigo)": {
        "NF-UNSW-NB15-v3": (0.9998, 0.0007, 0.9961, 0.0084),
        "NF-CSE-CIC-IDS2018-v3": (0.8819, 0.0347, 0.9447, 0.0213),
        "NF-UNSW-NB15-v2": (0.8116, 0.0367, 0.9264, 0.0217),
        "NF-CSE-CIC-IDS2018-v2": (0.9201, 0.0238, 0.9431, 0.0131),
    },
}

# Tabela 2 do artigo: (parâmetros em milhões, treino em horas ± desvio,
# inferência em µs/amostra ± desvio).
TABELA2 = {
    "NF-UNSW-NB15-v2": (0.867, 0.39, 0.27, 2.27, 0.92),
    "NF-UNSW-NB15-v3": (0.874, 0.63, 0.32, 3.04, 1.25),
    "NF-CSE-CIC-IDS2018-v2": (0.570, 1.00, 0.55, 4.89, 0.50),
    "NF-CSE-CIC-IDS2018-v3": (0.572, 1.46, 0.59, 5.09, 1.09),
}


def br(x: float, casas: int = 4) -> str:
    """Número com vírgula decimal."""
    return f"{x:.{casas}f}".replace(".", ",")


def milhar(n: int | float) -> str:
    """Inteiro com ponto como separador de milhar."""
    return f"{int(round(n)):,}".replace(",", ".")


def pm(m: float, s: float, casas: int = 4) -> str:
    return f"${br(m, casas)} \\pm {br(s, casas)}$"


def ler_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def gravar(nome: str, linhas: list[str]) -> None:
    SAIDA.mkdir(exist_ok=True)
    (SAIDA / f"{nome}.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def tabela_datasets() -> None:
    linhas_csv = {r["Dataset"]: r for r in ler_csv(
        RESULTS / "estatisticas" / "resumo_datasets.csv")}  # fmt: skip
    amostra = {d: "100\\%" for d in ("NF-UNSW-NB15-v2", "NF-UNSW-NB15-v3")}
    amostra.update({d: "20\\%" for d in ("NF-CSE-CIC-IDS2018-v2",
                                          "NF-CSE-CIC-IDS2018-v3", NOVO)})  # fmt: skip
    out = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & Fluxos & Hosts & Ataques & Cat. & Amostra \\",
        r"\midrule",
    ]
    for d in ["NF-UNSW-NB15-v2", "NF-UNSW-NB15-v3", "NF-CSE-CIC-IDS2018-v2",
              "NF-CSE-CIC-IDS2018-v3", NOVO]:  # fmt: skip
        r = linhas_csv[d]
        if d == NOVO:
            out.append(r"\midrule")
        out.append(
            f"{CURTO[d]} & {milhar(int(r['Fluxos']))} & {milhar(int(r['Hosts']))} & "
            f"{br(float(r['Ataques (%)']), 2)}\\% & {r['Categorias de ataque']} & "
            f"{amostra[d]} \\\\"
        )
    out += [r"\bottomrule", r"\end{tabular}"]
    gravar("datasets", out)


def _contagens(d: str) -> list[dict[str, str]]:
    return ler_csv(RESULTS / "estatisticas" / f"{d}_contagem_por_split.csv")


def _nome_categoria(c: str) -> str:
    c = c.replace("_", " ")
    return tex_escape(c)


def tabela_contagens_par(nome: str, v2: str, v3: str) -> None:
    """Contagem por categoria e split de um par v2/v3 (mesmas categorias)."""
    a, b = _contagens(v2), _contagens(v3)

    def chave(c: str) -> str:
        return c.replace("_", " ").lower()

    mb = {chave(r["Categoria"]): r for r in b}
    out = [
        r"\begin{tabular}{l rrr rrr}",
        r"\toprule",
        f"& \\multicolumn{{3}}{{c}}{{{CURTO[v2]}}} & "
        f"\\multicolumn{{3}}{{c}}{{{CURTO[v3]}}} \\\\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Categoria & Treino & Val. & Teste & Treino & Val. & Teste \\",
        r"\midrule",
    ]
    for r in a:
        cat = r["Categoria"]
        s = mb[chave(cat)]
        if cat == "Ataques (total)":
            out.append(r"\midrule")
        if cat == "Hosts com arestas":
            out.append(r"\midrule")
        nome_cat = _nome_categoria(cat)
        if cat in ("Ataques (total)", "Total"):
            nome_cat = f"\\textbf{{{nome_cat}}}"
        vals = [r["Treino"], r["Validação"], r["Teste"],
                s["Treino"], s["Validação"], s["Teste"]]  # fmt: skip
        out.append(nome_cat + " & " + " & ".join(milhar(int(v)) for v in vals) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    gravar(nome, out)


def tabela_contagens_ton() -> None:
    out = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Categoria & Treino & Validação & Teste & Total \\",
        r"\midrule",
    ]
    for r in _contagens(NOVO):
        cat = r["Categoria"]
        if cat in ("Ataques (total)", "Hosts com arestas"):
            out.append(r"\midrule")
        nome_cat = _nome_categoria(cat)
        if cat in ("Ataques (total)", "Total"):
            nome_cat = f"\\textbf{{{nome_cat}}}"
        vals = [r["Treino"], r["Validação"], r["Teste"], r["Total"]]
        out.append(nome_cat + " & " + " & ".join(milhar(int(v)) for v in vals) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    gravar("contagens_ton", out)


def _lr(v: float) -> str:
    """Taxa de aprendizado em notação científica, por exemplo 1,1e-05."""
    mant, exp = f"{float(v):.1e}".split("e")
    return f"${mant} \\times 10^{{{int(exp)}}}$"


def _cabecalho_curto(d: str) -> str:
    """Nome do dataset quebrado em duas linhas, por exemplo UNSW-NB15 / v3."""
    nome, versao = CURTO[d].rsplit("-", 1)
    return f"\\makecell{{{nome}\\\\{versao}}}"


def tabela_hiperparametros() -> None:
    import yaml  # dependência do projeto (pyyaml)

    cfgs = {
        d: yaml.safe_load((RAIZ / "configs" / f"{d}.yaml").read_text(encoding="utf-8"))
        for d in [*ARTIGO, NOVO]
    }
    linhas = [
        ("edim\\_out", "edim_out", str),
        ("dropout do GNN", "gnn_dropout", str),
        ("embed\\_dim", "embed_dim", str),
        ("ff\\_dim", "ff_dim", str),
        ("dropout do AE", "ae_dropout", str),
        ("taxa de aprendizado", "learning_rate", _lr),
        ("weight decay do GNN", "gnn_weight_decay", str),
        ("weight decay do AE", "ae_weight_decay", str),
        ("lote do GNN", "gnn_batch_size", milhar),
        ("fraction", "fraction", lambda v: "--" if v is None else str(v)),
    ]
    out = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        "Parâmetro & " + " & ".join(
            _cabecalho_curto(d) for d in [*ARTIGO, NOVO])
        + r" \\",
        r"\midrule",
    ]  # fmt: skip
    for rotulo, chave, fmt in linhas:
        vals = [fmt(cfgs[d].get(chave)).replace(".", "{,}") for d in [*ARTIGO, NOVO]]
        if chave == "gnn_batch_size":
            vals = [milhar(cfgs[d][chave]) for d in [*ARTIGO, NOVO]]
        out.append(f"{rotulo} & " + " & ".join(vals) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    gravar("hiperparametros", out)


def tabela_reproducao() -> None:
    nosso = {r["dataset"]: r for r in ler_csv(RESULTS / "tabela_reproducao.csv")}
    out = [
        r"\begin{tabular}{l cc cc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Datasets v3} & \multicolumn{2}{c}{Datasets v2} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Modelo / métrica & UNSW-NB15 & CIC-IDS2018 & UNSW-NB15 & CIC-IDS2018 \\",
        r"\midrule",
    ]
    for modelo, por_ds in TABELA3.items():
        for i, metrica in enumerate(("PR-AUC", "F1 macro")):
            cel = [pm(por_ds[d][2 * i], por_ds[d][2 * i + 1]) for d in ARTIGO]
            rot = f"{modelo} -- {metrica}"
            out.append(f"{rot} & " + " & ".join(cel) + r" \\")
        if modelo == "SAFE" or modelo == "SimpleAE":
            out.append(r"\midrule")
    out.append(r"\midrule")
    for metrica, m, s in (("PR-AUC", "pr_auc_mean", "pr_auc_std"),
                          ("F1 macro", "f1_macro_mean", "f1_macro_std")):  # fmt: skip
        cel = [pm(float(nosso[d][m]), float(nosso[d][s])) for d in ARTIGO]
        out.append(
            f"\\textbf{{GraphIDS (nosso)}} -- {metrica} & " + " & ".join(cel) + r" \\"
        )
    out += [r"\bottomrule", r"\end{tabular}"]
    gravar("reproducao", out)


def _metricas(d: str) -> list[dict]:
    runs = sorted((RESULTS / "runs" / d).glob("seed*/metrics.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in runs]


def tabela_por_seed() -> None:
    out = [
        r"\begin{tabular}{lcrrrrr}",
        r"\toprule",
        r"Dataset & Seed & \makecell{PR-AUC\\val.} & \makecell{PR-AUC\\teste} & "
        r"\makecell{F1 macro\\teste} & \makecell{Melhor\\época} & "
        r"\makecell{Treino\\(min)} \\",
        r"\midrule",
    ]
    for d in [*ARTIGO, NOVO]:
        ms = _metricas(d)
        for k, m in enumerate(ms):
            nome = f"\\multirow{{{len(ms)}}}{{*}}{{{CURTO[d]}}}" if k == 0 else ""
            out.append(
                f"{nome} & {m['seed']} & {br(m['val_best_pr_auc'])} & "
                f"{br(m['test_pr_auc'])} & {br(m['test_f1_macro'])} & "
                f"{m['best_epoch']} & {br(m['train_time_s'] / 60, 1)} \\\\"
            )
        if d != NOVO:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    gravar("por_seed", out)


def tabela_ton() -> None:
    nosso = {r["dataset"]: r for r in ler_csv(RESULTS / "tabela_reproducao.csv")}
    resumo = {r["Dataset"]: r for r in ler_csv(
        RESULTS / "estatisticas" / "resumo_datasets.csv")}  # fmt: skip
    out = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dataset & PR-AUC & F1 macro & \makecell{PR-AUC de\\referência$^\ast$} \\",
        r"\midrule",
    ]
    for d in [*ARTIGO, NOVO]:
        r = nosso[d]
        if d == NOVO:
            out.append(r"\midrule")
        # Linha de base da PR-AUC = prevalência de ataques no teste.
        cont = {c["Categoria"]: c for c in _contagens(d)}
        prev = int(cont["Ataques (total)"]["Teste"]) / int(cont["Total"]["Teste"])
        out.append(
            f"{CURTO[d]} & {pm(float(r['pr_auc_mean']), float(r['pr_auc_std']))} & "
            f"{pm(float(r['f1_macro_mean']), float(r['f1_macro_std']))} & "
            f"{br(prev)} \\\\"
        )
    out += [r"\bottomrule", r"\end{tabular}"]
    _ = resumo
    gravar("ton", out)


def tabela_deteccao(nome: str, datasets: list[str]) -> None:
    linhas = ler_csv(RESULTS / "figuras" / "deteccao_por_ataque.csv")
    out = [
        r"\begin{tabular}{l" + "r" * (2 * len(datasets)) + "}",
        r"\toprule",
    ]
    if len(datasets) > 1:
        out.append(
            "& "
            + " & ".join(f"\\multicolumn{{2}}{{c}}{{{CURTO[d]}}}" for d in datasets)
            + r" \\"
        )
        regras = [
            f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(datasets))
        ]
        out.append("".join(regras))
    out.append("Categoria" + r" & Fluxos & Taxa (\%)" * len(datasets) + r" \\")
    out.append(r"\midrule")

    def chave(c: str) -> str:
        return c.replace("_", " ").lower()

    por_ds = {d: {chave(r["categoria"]): r for r in linhas if r["dataset"] == d}
              for d in datasets}  # fmt: skip
    base = [r for r in linhas if r["dataset"] == datasets[0]]
    for r in base:
        cat = r["categoria"]
        celulas = []
        for d in datasets:
            x = por_ds[d][chave(cat)]
            taxa = 100 * float(x["deteccao_media"])
            desvio = 100 * float(x["deteccao_desvio"])
            celulas.append(milhar(int(x["fluxos_teste"])))
            celulas.append(f"${br(taxa, 1)} \\pm {br(desvio, 1)}$")
        rot = _nome_categoria(cat)
        if cat == "Benign":
            rot = r"Benign (FPR)"
        out.append(rot + " & " + " & ".join(celulas) + r" \\")
        if cat == "Benign":
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    gravar(nome, out)


def tabela_custo() -> None:
    nosso = {r["dataset"]: r for r in ler_csv(RESULTS / "tabela_custo.csv")}
    out = [
        r"\begin{tabular}{l rr rr rr r}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Parâmetros (M)} & \multicolumn{2}{c}{Treino (min)} & "
        r"\multicolumn{2}{c}{Inferência ($\mu$s/fluxo)} & Mem. GPU \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Dataset & nosso & artigo & nosso & artigo & nosso & artigo & (MB) \\",
        r"\midrule",
    ]
    for d in [*ARTIGO, NOVO]:
        r = nosso[d]
        p = float(r["num_params_mean"]) / 1e6
        t = pm(float(r["train_time_min_mean"]), float(r["train_time_min_std"]), 1)
        i = pm(float(r["inference_us_per_sample_mean"]),
               float(r["inference_us_per_sample_std"]), 2)  # fmt: skip
        if d in TABELA2:
            pa, th, ts, ia, isd = TABELA2[d]
            art = (br(pa, 3), pm(60 * th, 60 * ts, 1), pm(ia, isd, 2))
        else:
            art = ("--", "--", "--")
        if d == NOVO:
            out.append(r"\midrule")
        out.append(
            f"{CURTO[d]} & {br(p, 3)} & {art[0]} & {t} & {art[1]} & {i} & {art[2]} & "
            f"{milhar(float(r['peak_gpu_mem_mb_mean']))} \\\\"
        )
    out += [r"\bottomrule", r"\end{tabular}"]
    gravar("custo", out)


def main() -> None:
    tabela_datasets()
    tabela_contagens_par("contagens_unsw", "NF-UNSW-NB15-v2", "NF-UNSW-NB15-v3")
    tabela_contagens_par(
        "contagens_cic", "NF-CSE-CIC-IDS2018-v2", "NF-CSE-CIC-IDS2018-v3"
    )
    tabela_contagens_ton()
    tabela_hiperparametros()
    tabela_reproducao()
    tabela_por_seed()
    tabela_ton()
    tabela_deteccao("deteccao_unsw", ["NF-UNSW-NB15-v2", "NF-UNSW-NB15-v3"])
    tabela_deteccao("deteccao_cic", ["NF-CSE-CIC-IDS2018-v2", "NF-CSE-CIC-IDS2018-v3"])
    tabela_deteccao("deteccao_ton", [NOVO])
    tabela_custo()
    print(f"Tabelas gravadas em {SAIDA}")


if __name__ == "__main__":
    main()
