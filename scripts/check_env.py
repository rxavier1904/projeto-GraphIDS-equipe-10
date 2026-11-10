"""Confere o ambiente do servidor antes das rodadas (entrega 3.1 da F3).

Imprime as versões do Python, do PyTorch e das bibliotecas, a versão do CUDA e
as GPUs visíveis, e faz uma conta pequena na GPU para garantir que ela funciona.
A saída serve também para a seção 6 do relatório (hardware e software).

Uso:
    python scripts/check_env.py            # falha (código 1) se não houver GPU
    python scripts/check_env.py --cpu-ok   # só informa, sem exigir GPU
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys

LIBS = ("torch", "numpy", "pandas", "sklearn", "yaml", "graphids")


def lib_versions() -> dict[str, str]:
    """Versão de cada biblioteca de ``LIBS`` (ou ``"AUSENTE"``)."""
    versions = {}
    for name in LIBS:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "?")
        except ImportError:
            versions[name] = "AUSENTE"
    return versions


def gpu_report() -> list[str]:
    """Linhas descrevendo o CUDA e cada GPU. Lista vazia se não houver GPU."""
    import torch

    if not torch.cuda.is_available():
        return []
    lines = [
        f"CUDA (PyTorch): {torch.version.cuda}",
        f"cuDNN: {torch.backends.cudnn.version()}",
    ]
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        lines.append(
            f"GPU {i}: {props.name} | {props.total_memory / 2**30:.1f} GiB | "
            f"capacidade {props.major}.{props.minor}"
        )
    # Teste rápido: multiplicação de matrizes na GPU 0.
    x = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    lines.append(f"Teste na GPU 0: ok (soma = {(x @ x).sum().item():.1f})")
    return lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Confere o ambiente (F3).")
    p.add_argument("--cpu-ok", action="store_true", help="não exige GPU")
    args = p.parse_args(argv)

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Sistema: {platform.platform()}")
    versions = lib_versions()
    for name, version in versions.items():
        print(f"{name}: {version}")

    missing = [n for n, v in versions.items() if v == "AUSENTE"]
    if missing:
        print(f"\nERRO: bibliotecas ausentes: {missing}. Rode pip install -e '.[dev]'.")
        return 1

    gpu = gpu_report()
    if gpu:
        print("\n".join(gpu))
        return 0
    print("Nenhuma GPU visível para o PyTorch (torch.cuda.is_available() = False).")
    return 0 if args.cpu_ok else 1


if __name__ == "__main__":
    sys.exit(main())
