# Como contribuir

## Autoria

- Cada integrante faz **os commits da própria frente, com a própria conta do GitHub**.
  Isso garante que a participação de todos apareça no histórico.
- Antes do primeiro commit, configure nome e e-mail do GitHub no clone local:
  ```bash
  git config user.name  "<seu login do GitHub>"
  git config user.email "<id>+<login>@users.noreply.github.com"   # ou o e-mail da conta
  ```
- **Nunca** inclua em commits ou PRs linhas geradas por ferramentas, como
  "Generated with ...", `Co-Authored-By:` de ferramentas ou links de sessão.

## Fluxo de trabalho

1. Crie uma branch a partir de `main` no formato `<tipo>/resumo-da-tarefa` (kebab-case):
   - `feat/`: nova funcionalidade (ex.: `feat/egraphsage`)
   - `fix/`: ajuste ou correção (ex.: `fix/clip-normalizacao`)
   - `docs/`: só documentação
   - `refactor/`: reorganização sem mudar comportamento
   - `test/`: só testes
   - `chore/`: configuração, dependências, CI
2. Commits pequenos, com mensagens em português no imperativo
   (ex.: `Implementa agregação média do E-GraphSAGE`).
3. Abra um PR para `main`, descreva o que mudou e marque o Rafael como revisor.
   PRs que mexem nos [contratos](CONTRATOS.md) precisam de aprovação do Rafael.
4. Todo módulo novo vem com teste em `tests/` usando dados sintéticos (sem depender dos
   datasets reais).

## Código

- **Comentários e docstrings em português**: a especificação exige o código comentado.
  Cada função pública explica o que faz, as entradas e as saídas (com as formas dos
  tensores).
- Quando um trecho segue uma escolha do artigo, cite a seção ou equação
  (ex.: `# Eq. (1) do artigo: s_i = ||h_i - ĥ_i||²`).
- Formatação com `ruff format` e lint com `ruff check`.
- Não copie arquivos do repositório oficial. Use-o só como referência
  (ver [REFERENCIA-CODIGO-OFICIAL.md](REFERENCIA-CODIGO-OFICIAL.md)).

## Dados e artefatos

- Datasets, caches de grafo e checkpoints **não vão para o git** (ver `.gitignore`).
- Vão para o git: `results/runs/**/metrics.json`, as tabelas agregadas e as figuras do relatório.
