# Relatório do projeto

Relatório final da Equipe 10 (CIN0114, 2026.2) no formato de artigo, em LaTeX:
reprodução do GraphIDS nos quatro datasets do artigo e avaliação no NF-ToN-IoT-v3.

| Arquivo | Conteúdo |
|---|---|
| `relatorio.pdf` | versão compilada do relatório |
| `main.tex` | texto, com as seções pedidas na especificação da disciplina |
| `referencias.bib` | referências (estilo IEEE, `IEEEtran.bst`) |
| `tabelas/*.tex` | tabelas geradas por `gerar_tabelas.py` a partir de `results/` |
| `figuras/*.pdf` | cópia das figuras de `results/figuras/` usadas no texto |

## Como compilar

```bash
python relatorio/gerar_tabelas.py      # na raiz do repositório: refaz as tabelas
cd relatorio
latexmk -pdf main.tex                  # gera main.pdf
cp main.pdf relatorio.pdf
```

Requer uma distribuição TeX com `IEEEtran`, `babel` (português), `algorithm2e`,
`tikz`, `booktabs`, `multirow` e `makecell` (no Ubuntu: `texlive-publishers`,
`texlive-lang-portuguese`, `texlive-latex-extra` e `texlive-science`).

## Números e figuras

Nenhum resultado nosso é digitado à mão: as tabelas saem dos CSVs e JSONs em
`results/` (rodadas com as seeds 1, 2 e 3). Os únicos valores fixos em
`gerar_tabelas.py` são os publicados no artigo (Tabelas 2 e 3), usados para
comparação. Se as rodadas mudarem, rode `gerar_tabelas.py` de novo e, se for o
caso, copie as figuras novas de `results/figuras/` para `figuras/`.

## Template da disciplina

O texto usa a classe `IEEEtran` (conferência, duas colunas). Para passar para a
cópia do template Overleaf da disciplina, basta enviar esta pasta para o projeto
e colar o conteúdo entre `\maketitle` e `\bibliography` de `main.tex` no arquivo
principal do template, mantendo os pacotes do preâmbulo.
