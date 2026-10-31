"""Organizador seguro do repositorio Federated XAI Lab.

Tres camadas de protecao:
1. Modo padrao somente-leitura (previa das exclusoes, desindexacoes e revisoes sugeridas).
2. Backup integral fora da raiz do repositorio com manifesto JSON e verificacao SHA-256
   de todos os arquivos cientificos, checkpoints, dados brutos locais, codigo-fonte,
   testes, docs e PDFs.
3. Varredura de segredos/tokens na arvore de trabalho e no historico Git completo antes
   de permitir commit/push, recusando URLs de remote com credenciais embutidas.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT_KEEP = {
    ".git",
    ".gitignore",
    ".venv",
    "Federated_XAI_Lab_Entregavel.zip",
    "LICENSE",
    "README.md",
    "app.py",
    "data",
    "docs",
    "human+activity+recognition+using+smartphones",
    "pyproject.toml",
    "requirements.txt",
    "results",
    "scripts",
    "src",
    "tests",
}

# Artefatos sinteticos da Fase 1 (referenciados/embutidos na versao atual do PDF do artigo).
# Preservados por padrao ate que o artigo seja atualizado com as 5 sementes reais do UCI HAR;
# podem ser arquivados futuramente com --arquivar-sinteticos-fase1.
PHASE1_SYNTHETIC_PDF_ARTIFACTS = [
    "results/demonstracao_seed_43",
    "results/demonstracao_seed_44",
    "results/fig_media_macro_f1_3_sementes.png",
    "results/metricas_sinteticas_3_sementes.csv",
    "results/resumo_concordancia_sintetica.csv",
    "results/resumo_sintetico_3_sementes.csv",
    "results/fig_heterogeneidade_6_clientes.png",
]

# Diretorios de dados brutos locais que devem existir apenas no disco local (fora do indice Git)
RAW_DATA_DIRS = [
    "data",
    "human+activity+recognition+using+smartphones",
]

REQUIRED_GITIGNORE_RULES = [
    ".venv/",
    "venv/",
    "ENV/",
    "__pycache__/",
    "*.py[cod]",
    ".pytest_cache/",
    "*.egg-info/",
    "build/",
    "dist/",
    "*.zip",
    "data/",
    "data/UCI HAR Dataset/",
    "**/data/UCI HAR Dataset/",
    "human+activity+recognition+using+smartphones/",
    "results/test_pipeline/",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*secret*",
    "*credentials*",
    "*token*",
]

SAFE_STAGE_TARGETS = [
    ".gitignore",
    "LICENSE",
    "README.md",
    "app.py",
    "pyproject.toml",
    "requirements.txt",
    "src/flxai",
    "tests",
    "scripts",
    "docs",
    "results",
]

SECRET_REGEXES: dict[str, re.Pattern[bytes]] = {
    "github_classic_token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    "github_fine_grained_token": re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    "gitlab_token": re.compile(rb"glpat-[A-Za-z0-9_-]{20,}"),
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "google_api_key": re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
    "private_key_block": re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    "url_embedded_credentials": re.compile(rb"https?://[^/\s:@\"'<>]+:[^/\s@\"'<>]{6,}@"),
}

GIT_GREP_SECRET_PATTERN = (
    r"(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"glpat-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|"
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----)"
)

PROTECTED_SEEDS = [42, 43, 44, 45, 46]
SKIP_WALK_DIRS = {".git", ".venv", "venv", "ENV", "data", "human+activity+recognition+using+smartphones"}
ARTICLE_SUFFIXES = {".pdf", ".doc", ".docx", ".tex", ".bib"}


def run_cmd(
    root: Path,
    cmd: list[str],
    check: bool = True,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    p = subprocess.run(
        cmd,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if check and p.returncode != 0:
        raise RuntimeError(f"Comando falhou ({' '.join(cmd)}):\n{p.stdout[-1500:]}")
    return p


def validate_root(root: Path) -> None:
    required = [
        root / "app.py",
        root / "README.md",
        root / "pyproject.toml",
        root / "requirements.txt",
        root / "src" / "flxai" / "experiment.py",
        root / "tests",
        root / "docs" / "RELATORIO_EXPERIMENTAL_UCI_HAR.md",
        root / "results" / "resumo_experimentos_reais.csv",
    ]
    missing = [str(p.relative_to(root)) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Raiz invalida do Federated XAI Lab; ausentes: " + ", ".join(missing))

    total_ckpts = 0
    for s in PROTECTED_SEEDS:
        ckpt_dir = root / "results" / f"uci_seed_{s}" / "checkpoints"
        pts = list(ckpt_dir.glob("*.pt")) if ckpt_dir.is_dir() else []
        if len(pts) != 12:
            raise RuntimeError(
                f"Semente {s} nao possui os 12 checkpoints esperados em {ckpt_dir} (encontrados: {len(pts)})."
            )
        total_ckpts += len(pts)
    if total_ckpts != 60:
        raise RuntimeError(f"Total de checkpoints .pt diferente de 60 ({total_ckpts}).")


def git_available(root: Path) -> bool:
    if shutil.which("git") is None:
        return False
    p = run_cmd(root, ["git", "rev-parse", "--is-inside-work-tree"], check=False)
    return p.returncode == 0 and "true" in p.stdout.lower()


def unsafe_remote(root: Path) -> str | None:
    p = run_cmd(root, ["git", "remote", "get-url", "origin"], check=False)
    if p.returncode != 0 or not p.stdout.strip():
        return "Remote 'origin' ainda nao esta configurado no Git"
    url = p.stdout.strip()
    if re.search(r"(gh[pousr]_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+|https?://[^/\s:@]+:[^/\s@]+@)", url):
        return "Remote 'origin' contem token ou senha embutida na URL (inseguro)"
    return None


def collect_tracked_raw_data(root: Path) -> list[str]:
    if not git_available(root):
        return []
    tracked: list[str] = []
    for raw_dir in RAW_DATA_DIRS:
        p = run_cmd(root, ["git", "ls-files", "--", raw_dir], check=False)
        if p.returncode == 0 and p.stdout.strip():
            for line in p.stdout.splitlines():
                s = line.strip()
                if s:
                    tracked.append(s)
    return sorted(set(tracked))


def scan_secrets(root: Path) -> dict[str, object]:
    wt_hits: list[dict[str, str]] = []
    wt_files_scanned = 0
    for p in sorted(root.rglob("*")):
        rel_parts = p.relative_to(root).parts
        if any(part in {".git", ".venv", "venv", "ENV"} for part in rel_parts):
            continue
        if p.is_file() and p.suffix.lower() not in {".pt", ".zip", ".png", ".pdf", ".docx"}:
            wt_files_scanned += 1
            raw = p.read_bytes()
            for rule_name, rx in SECRET_REGEXES.items():
                if rx.search(raw):
                    wt_hits.append({"regra": rule_name, "arquivo": p.relative_to(root).as_posix()})

    git_cfg = root / ".git" / "config"
    if git_cfg.is_file():
        raw_cfg = git_cfg.read_bytes()
        for rule_name, rx in SECRET_REGEXES.items():
            if rx.search(raw_cfg):
                wt_hits.append({"regra": rule_name, "arquivo": ".git/config"})

    hist_hits: list[str] = []
    commits_scanned = 0
    if git_available(root):
        revs = run_cmd(root, ["git", "rev-list", "--all"], check=False)
        commit_list = [c.strip() for c in revs.stdout.splitlines() if c.strip()]
        commits_scanned = len(commit_list)
        if commit_list:
            grep_res = run_cmd(
                root,
                ["git", "grep", "-I", "-l", "-E", GIT_GREP_SECRET_PATTERN, *commit_list],
                check=False,
            )
            if grep_res.returncode == 0 and grep_res.stdout.strip():
                hist_hits.extend([line.strip() for line in grep_res.stdout.splitlines() if line.strip()])

            log_msgs = run_cmd(root, ["git", "log", "--all", "--format=%H %B"], check=False)
            for rule_name, rx in SECRET_REGEXES.items():
                if rx.search(log_msgs.stdout.encode("utf-8", errors="ignore")):
                    hist_hits.append(f"commit_message:{rule_name}")

    return {
        "arquivos_texto_verificados": wt_files_scanned,
        "commits_historico_verificados": commits_scanned,
        "ocorrencias_arvore_trabalho": wt_hits,
        "ocorrencias_historico_git": hist_hits,
        "limpo": len(wt_hits) == 0 and len(hist_hits) == 0,
    }


def collect_removable(root: Path, include_phase1_synthetic: bool = False) -> list[tuple[Path, str]]:
    targets: list[tuple[Path, str]] = []
    claimed_dirs: list[Path] = []

    def _under_claimed(candidate: Path) -> bool:
        return any(candidate == d or d in candidate.parents for d in claimed_dirs)

    # 1. Caches e artefatos temporarios de raiz / build / testes antigos
    pytest_cache = root / ".pytest_cache"
    if pytest_cache.exists():
        targets.append((pytest_cache, "cache do pytest"))
        claimed_dirs.append(pytest_cache)

    old_test_out = root / "results" / "test_pipeline"
    if old_test_out.exists():
        targets.append((old_test_out, "saida temporaria antiga de teste de pipeline"))
        claimed_dirs.append(old_test_out)

    # 2. Versoes antigas de ZIP na raiz (preservando Federated_XAI_Lab_Entregavel.zip)
    for z in sorted(root.glob("*.zip")):
        if z.name != "Federated_XAI_Lab_Entregavel.zip":
            targets.append((z, "versao antiga de pacote ZIP na raiz"))

    # 3. Caches Python (__pycache__, *.egg-info, *.pyc/pyo/pyd, temporarios) fora de .venv/.git/data
    for p in sorted(root.rglob("*")):
        rel_parts = p.relative_to(root).parts
        if any(part in SKIP_WALK_DIRS for part in rel_parts):
            continue
        if _under_claimed(p):
            continue
        if p.is_dir() and p.name == "__pycache__":
            targets.append((p, "cache de bytecode Python (__pycache__)"))
            claimed_dirs.append(p)
        elif p.is_dir() and p.name.endswith(".egg-info"):
            targets.append((p, "metadados temporarios de build setuptools (*.egg-info)"))
            claimed_dirs.append(p)
        elif p.is_file() and (
            p.suffix.lower() in {".pyc", ".pyo", ".pyd", ".tmp", ".bak"}
            or p.name in {".coverage", "Thumbs.db", ".DS_Store"}
        ):
            targets.append((p, "arquivo temporario ou bytecode compilado"))

    # 4. Opcional: arquivar artefatos sinteticos da Fase 1 somente quando solicitado explicitamente
    if include_phase1_synthetic:
        for rel in PHASE1_SYNTHETIC_PDF_ARTIFACTS:
            cand = root / rel
            if cand.exists() and not _under_claimed(cand):
                targets.append((cand, "artefato sintetico legado da Fase 1 arquivado sob comando explicito"))
                if cand.is_dir():
                    claimed_dirs.append(cand)

    return targets


def expected_gitignore(root: Path) -> tuple[list[str], bool]:
    gi = root / ".gitignore"
    existing_lines = set()
    if gi.exists():
        for line in gi.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                existing_lines.add(s)
    missing = [rule for rule in REQUIRED_GITIGNORE_RULES if rule not in existing_lines]
    return missing, len(missing) > 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def protected_hashes(root: Path, removable_paths: set[Path]) -> dict[str, str]:
    def _is_removable(path: Path) -> bool:
        return any(path == r or r in path.parents for r in removable_paths)

    hashes: dict[str, str] = {}

    # Arquivos de raiz protegidos (inclui o PDF do artigo; exclui .gitignore que recebe regras novas)
    for item in sorted(root.iterdir()):
        if item.is_file() and item.name != ".gitignore" and not _is_removable(item):
            if item.name in ROOT_KEEP or item.suffix.lower() in ARTICLE_SUFFIXES:
                hashes[item.relative_to(root).as_posix()] = sha256_file(item)

    # Diretorios cientificos, codigo-fonte, scripts, testes e documentacao
    for sub in ("src/flxai", "tests", "scripts", "docs", "results"):
        base = root / sub
        if not base.exists():
            continue
        for f in sorted(base.rglob("*")):
            if not f.is_file() or _is_removable(f):
                continue
            if "__pycache__" in f.parts or any(part.endswith(".egg-info") for part in f.parts):
                continue
            hashes[f.relative_to(root).as_posix()] = sha256_file(f)

    # Tambem conferir os arquivos de dados locais desindexados para provar preservacao no disco
    for raw_rel in collect_tracked_raw_data(root):
        raw_path = root / raw_rel
        if raw_path.is_file():
            hashes[raw_path.relative_to(root).as_posix()] = sha256_file(raw_path)

    return hashes


def _python_runner(root: Path) -> str:
    win_py = root / ".venv" / "Scripts" / "python.exe"
    nix_py = root / ".venv" / "bin" / "python"
    if win_py.is_file():
        return str(win_py)
    if nix_py.is_file():
        return str(nix_py)
    return sys.executable


def do_tests(root: Path, exigir_24: bool) -> None:
    py = _python_runner(root)
    print(f"\n[TESTES] Executando suite pytest sem geracao de cache/bytecode com: {py}")
    p = run_cmd(
        root,
        [py, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        check=False,
        extra_env={"PYTHONDONTWRITEBYTECODE": "1"},
    )
    print(p.stdout.strip())
    if p.returncode != 0:
        raise RuntimeError("A suite pytest falhou; commit/push abortado.")
    if exigir_24:
        m_pass = re.search(r"(\d+)\s+passed", p.stdout)
        m_skip = re.search(r"(\d+)\s+skipped", p.stdout)
        n_pass = int(m_pass.group(1)) if m_pass else 0
        n_skip = int(m_skip.group(1)) if m_skip else 0
        if n_pass != 24 or n_skip != 0:
            raise RuntimeError(
                f"Exigencia de 24 testes aprovados e 0 ignorados nao atendida (obtido: {n_pass} passed, {n_skip} skipped)."
            )
    print("[TESTES] Suite validada com sucesso (24 passed, 0 skipped).")


def untrack_raw_data(root: Path, raw_tracked: list[str], backup: Path) -> list[str]:
    untracked: list[str] = []
    for rel in raw_tracked:
        src = root / rel
        if src.is_file():
            snap = backup / "dados_locais_desindexados_copia" / rel
            snap.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, snap)
        res = run_cmd(root, ["git", "rm", "--cached", "--", rel], check=False)
        if res.returncode != 0:
            raise RuntimeError(f"Falha ao retirar dado bruto do indice Git ({rel}):\n{res.stdout}")
        if not src.exists():
            raise RuntimeError(f"ERRO CRITICO: arquivo local {rel} desapareceu apos git rm --cached!")
        untracked.append(rel)
    return untracked


def do_commit(root: Path, moved: list[dict[str, str]]) -> bool:
    existing_targets = [t for t in SAFE_STAGE_TARGETS if (root / t).exists()]
    add = run_cmd(root, ["git", "add", "--", *existing_targets], check=False)
    if add.returncode:
        raise RuntimeError("git add falhou: " + add.stdout[-1200:])
    for item in moved:
        rel = Path(item["origem"]).as_posix()
        tracked = run_cmd(root, ["git", "ls-files", "--", rel], check=False)
        if tracked.returncode == 0 and tracked.stdout.strip():
            deleted = run_cmd(root, ["git", "rm", "-r", "--cached", "--ignore-unmatch", "--", rel], check=False)
            if deleted.returncode:
                raise RuntimeError("Falha ao registrar exclusao de " + rel)
    staged = run_cmd(root, ["git", "diff", "--cached", "--name-only"], check=True).stdout.strip()
    if not staged:
        print("\n[GIT] Sem alteracoes para commitar.")
        return False
    print("\n[GIT] Arquivos preparados para commit:\n" + staged[:4000])
    p = run_cmd(
        root,
        [
            "git",
            "commit",
            "-m",
            "chore: organizar repositorio, remover temporarios e desindexar dados brutos locais",
        ],
        check=False,
    )
    print(p.stdout[-2000:])
    if p.returncode:
        raise RuntimeError("Commit falhou; confira autor do Git e estado dos arquivos.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Organizacao segura do Federated XAI Lab")
    parser.add_argument("--raiz", type=Path, default=Path.cwd(), help="Diretorio raiz do projeto")
    parser.add_argument("--aplicar", action="store_true", help="Efetivamente mover descartaveis para backup fora da raiz")
    parser.add_argument(
        "--arquivar-sinteticos-fase1",
        action="store_true",
        help="Mover tambem os 7 artefatos sinteticos da Fase 1 para o backup externo (usar apenas apos atualizar o PDF do artigo)",
    )
    parser.add_argument(
        "--manter-dados-no-git",
        action="store_true",
        help="Nao executar git rm --cached em arquivos de dados brutos rastreados",
    )
    parser.add_argument("--testes", action="store_true", help="Executar a suite pytest")
    parser.add_argument("--exigir-24-testes", action="store_true", help="Exigir 24 passed e 0 skipped")
    parser.add_argument("--commit", action="store_true", help="Commitar apenas codigo, docs e resultados aprovados")
    parser.add_argument("--push", action="store_true", help="Enviar o commit ao remote origin via Git autenticado")
    args = parser.parse_args()
    root = args.raiz.expanduser().resolve()
    try:
        validate_root(root)
        if args.push and not args.commit:
            raise RuntimeError("--push requer --commit")
        if args.commit and not args.aplicar:
            raise RuntimeError("--commit requer --aplicar para evitar commits durante uma previa")
        if (args.commit or args.push) and not git_available(root):
            raise RuntimeError("Git nao configurado para esta raiz.")
        if args.push:
            msg = unsafe_remote(root)
            if msg:
                raise RuntimeError(msg + ". Configure um remote HTTPS sem token, ou SSH.")

        secret_report = scan_secrets(root)
        if not secret_report["limpo"]:
            raise RuntimeError(
                f"Segredos/credenciais detectados! Arvore: {secret_report['ocorrencias_arvore_trabalho']} | "
                f"Historico: {secret_report['ocorrencias_historico_git']}"
            )

        targets = collect_removable(root, include_phase1_synthetic=args.arquivar_sinteticos_fase1)
        raw_tracked = [] if args.manter_dados_no_git else collect_tracked_raw_data(root)
        missing, ignore_changed = expected_gitignore(root)

        print(f"Federated XAI Lab: {root}")
        print("\nPRESERVADOS: artigo/PDFs, dados locais no disco, src, tests, docs, 5 sementes UCI HAR,")
        print("             60 checkpoints, resultados consolidados, demo, .venv, .git e ZIP entregavel.")
        print(
            f"[SEGURANCA] Varredura limpa: 0 segredos em {secret_report['arquivos_texto_verificados']} arquivos "
            f"e {secret_report['commits_historico_verificados']} commits do historico Git."
        )
        print(f"\n{len(targets)} itens descartaveis identificados:")
        for p, why in targets:
            print(f"  [REMOVER DA RAIZ] {p.relative_to(root)}  ({why})")
        if not targets:
            print("  Nenhum cache ou arquivo obsoleto seguro identificado.")

        if raw_tracked:
            print(f"\n{len(raw_tracked)} arquivo(s) de dados brutos rastreado(s) para retirar apenas do indice Git (preservados no disco):")
            for rel in raw_tracked:
                print(f"  [GIT RM --CACHED] {rel}")

        if ignore_changed:
            print(f"\n.gitignore: adicionar {len(missing)} regras ausentes, sem remover regras existentes.")

        if not args.arquivar_sinteticos_fase1:
            kept_syn = [rel for rel in PHASE1_SYNTHETIC_PDF_ARTIFACTS if (root / rel).exists()]
            if kept_syn:
                print(
                    f"\nPRESERVADOS POR REFERENCIA NO ARTIGO PDF ATUAL ({len(kept_syn)} artefatos sinteticos da Fase 1):"
                )
                for rel in kept_syn:
                    print(f"  [MANTIDO + COPIA NO BACKUP] {rel}")

        if not args.aplicar:
            print("\n[PREVIA] Nenhuma alteracao realizada. Use --aplicar para efetivar com backup.")
            return 0

        removable_set = {p for p, _ in targets}
        pre_hash = protected_hashes(root, removable_set)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = root.parent / f"{root.name}_backup_limpeza_{stamp}"
        backup.mkdir(exist_ok=False)
        moved: list[dict[str, str]] = []

        if ignore_changed and (root / ".gitignore").exists():
            safe = backup / "antes_da_limpeza" / ".gitignore"
            safe.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / ".gitignore", safe)

        # Copia preventiva dos 7 artefatos sinteticos referenciados pelo PDF atual
        copied_phase1: list[str] = []
        if not args.arquivar_sinteticos_fase1:
            for rel in PHASE1_SYNTHETIC_PDF_ARTIFACTS:
                src = root / rel
                if not src.exists():
                    continue
                dst = backup / "copia_seguranca_sinteticos_artigo_pdf" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
                copied_phase1.append(rel)

        for p, why in targets:
            if not p.exists():
                continue
            dest = backup / "removidos" / p.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest))
            moved.append({"origem": str(p.relative_to(root)), "backup": str(dest.relative_to(backup)), "motivo": why})

        untracked_raw: list[str] = []
        if raw_tracked and git_available(root):
            untracked_raw = untrack_raw_data(root, raw_tracked, backup)

        if ignore_changed:
            ignore = root / ".gitignore"
            with ignore.open("a", encoding="utf-8") as f:
                f.write("\n# Federated XAI Lab - protecao de dados e temporarios\n" + "\n".join(missing) + "\n")

        # Como untrack_raw_data desindexa do Git, collect_tracked_raw_data pos-limpeza retorna vazio;
        # conferimos explicitamente os mesmos caminhos de pre_hash no disco local!
        post_hash: dict[str, str] = {}
        for rel_posix in pre_hash:
            target_file = root / Path(rel_posix)
            if target_file.is_file():
                post_hash[rel_posix] = sha256_file(target_file)

        # Tambem verificar se algum arquivo novo inesperado surgiu ou sumiu nos diretorios protegidos
        current_protected = protected_hashes(root, removable_set)
        for k, v in current_protected.items():
            post_hash[k] = v

        if pre_hash != post_hash:
            missing_or_changed = sorted(
                set(pre_hash) ^ set(post_hash)
                | {k for k in pre_hash.keys() & post_hash.keys() if pre_hash[k] != post_hash[k]}
            )
            raise RuntimeError(
                "Integridade dos arquivos protegidos mudou! NAO PUBLIQUE. Arquivos: "
                + ", ".join(missing_or_changed[:15])
            )

        report = {
            "root": str(root),
            "timestamp": stamp,
            "backup": str(backup),
            "itens_movidos": moved,
            "dados_brutos_desindexados_git_preservados_disco": untracked_raw,
            "copia_preventiva_sinteticos_artigo_pdf": copied_phase1,
            "varredura_segredos": secret_report,
            "total_arquivos_preservados_conferidos": len(post_hash),
            "sha256_preservados": post_hash,
            "artigo_modificado": False,
            "gitignore_alterado": bool(ignore_changed),
        }
        (backup / "manifesto_limpeza.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[LIMPEZA] {len(moved)} itens temporarios movidos para: {backup}")
        if untracked_raw:
            print(f"[DADOS LOCAIS] {len(untracked_raw)} arquivo(s) retirado(s) do indice Git e preservado(s) no disco.")
        print(f"[INTEGRIDADE] {len(post_hash)} arquivos cientificos/codigo/artigo/dados conferidos por SHA-256: sem alteracoes.")

        if args.testes or args.commit:
            do_tests(root, args.exigir_24_testes)
        if args.commit:
            do_commit(root, moved)
        if args.push:
            p = run_cmd(root, ["git", "push", "-u", "origin", "HEAD"], check=False)
            print(p.stdout[-3000:])
            if p.returncode:
                raise RuntimeError("Push falhou: revise gh auth login, remote e permissoes.")
            print("[GITHUB] Push concluido pelo Git configurado localmente.")
        print("\nConcluido. O artigo nao foi editado; backup/manifesto disponiveis fora da raiz.")
        return 0
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        print(f"\n[ERRO] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
