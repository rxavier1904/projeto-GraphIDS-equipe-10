"""Gerador de um dataset NetFlow sintético e pequeno.

Serve para testar o modelo e o treino sem depender dos datasets reais (que são
responsabilidade da F2). A ideia imita o cenário do artigo: hosts conversando
com padrões estáveis (tráfego benigno) e alguns hosts atacantes cujos fluxos
fogem do padrão, tanto nas features quanto na topologia (varredura de portas
e inundação de uma vítima).

Atenção: no E-GraphSAGE (como no código oficial), o embedding de um fluxo u -> v
vem só das médias de ENTRADA de u e de v. Por isso os ataques sintéticos miram
hosts que não aparecem no tráfego benigno (endereços sem serviço): é a
vizinhança anômala desses hosts que o modelo consegue perceber.
"""

from __future__ import annotations

import torch

from graphids.data.types import FlowGraph, GraphSplits

ATTACK_NAMES = ["Benign", "DoS", "Scan"]
N_VICTIMS = 10  # hosts extras, só alcançados por ataques


def _benign_flows(n: int, hosts: int, profiles: torch.Tensor, g):
    """Clientes falam com servidores; cada servidor tem um 'perfil' de serviço."""
    servers, num_features = profiles.shape
    src = torch.randint(servers, hosts, (n,), generator=g)
    dst = torch.randint(0, servers, (n,), generator=g)
    attr = profiles[dst] + 0.03 * torch.randn(n, num_features, generator=g)
    return src, dst, attr


def _attack_flows(n: int, hosts: int, num_features: int, g):
    """Metade DoS (inunda uma vítima) e metade Scan (varre as demais vítimas).

    Índices: ``hosts`` é o atacante; ``hosts+1 .. hosts+N_VICTIMS`` são vítimas.
    """
    n_dos = n // 2
    n_scan = n - n_dos
    attacker = hosts  # host extra que só aparece em validação/teste
    src = torch.full((n,), attacker, dtype=torch.long)
    dst = torch.cat(
        [
            torch.full((n_dos,), hosts + 1, dtype=torch.long),
            torch.randint(hosts + 2, hosts + 1 + N_VICTIMS, (n_scan,), generator=g),
        ]
    )
    attr = torch.cat(
        [
            0.95 + 0.03 * torch.randn(n_dos, num_features, generator=g),
            0.05 + 0.03 * torch.randn(n_scan, num_features, generator=g),
        ]
    )
    attack = torch.cat(
        [torch.full((n_dos,), 1), torch.full((n_scan,), 2)]
    ).long()
    return src, dst, attr, attack


def _make_graph(n_benign, n_attack, hosts, profiles, g) -> FlowGraph:
    num_features = profiles.shape[1]
    s_b, d_b, a_b = _benign_flows(n_benign, hosts, profiles, g)
    if n_attack > 0:
        s_a, d_a, a_a, k_a = _attack_flows(n_attack, hosts, num_features, g)
    else:
        s_a = d_a = k_a = torch.empty(0, dtype=torch.long)
        a_a = torch.empty(0, num_features)
    attack = torch.cat([torch.zeros(n_benign, dtype=torch.long), k_a])
    return FlowGraph(
        edge_index=torch.stack([torch.cat([s_b, s_a]), torch.cat([d_b, d_a])]),
        edge_attr=torch.cat([a_b, a_a]).clamp(-10, 10).float(),
        edge_label=(attack > 0).long(),
        edge_attack=attack,
        num_nodes=hosts + 1 + N_VICTIMS,
        attack_names=list(ATTACK_NAMES),
    )


def make_synthetic_splits(
    seed: int = 0,
    num_features: int = 8,
    hosts: int = 40,
    servers: int = 6,
    n_train: int = 4000,
    n_eval: int = 1500,
    attack_ratio: float = 0.1,
) -> GraphSplits:
    """Cria treino (só benigno), validação e teste (com ataques)."""
    g = torch.Generator().manual_seed(seed)
    n_att = int(n_eval * attack_ratio)
    # Os perfis de serviço são sorteados uma única vez, para que o tráfego
    # benigno dos três splits venha da mesma distribuição.
    profiles = torch.rand(servers, num_features, generator=g) * 0.6 + 0.2
    splits = []
    for n_b, n_a in [(n_train, 0), (n_eval - n_att, n_att), (n_eval - n_att, n_att)]:
        graph = _make_graph(n_b, n_a, hosts, profiles, g)
        # embaralha a ordem dos fluxos, como num capture real misturado
        perm = torch.randperm(graph.num_edges, generator=g)
        graph = FlowGraph(
            edge_index=graph.edge_index[:, perm],
            edge_attr=graph.edge_attr[perm],
            edge_label=graph.edge_label[perm],
            edge_attack=graph.edge_attack[perm],
            num_nodes=graph.num_nodes,
            attack_names=graph.attack_names,
        )
        splits.append(graph)
    return GraphSplits(
        train=splits[0],
        val=splits[1],
        test=splits[2],
        feature_names=[f"feat_{i}" for i in range(num_features)],
    )
