"""Publicacao segura do Federated XAI Lab no GitHub (Lucas-715/Federated-XAI-Lab).

Garantias de seguranca:
- Modo padrao somente-leitura (previa completa sem alterar arquivos, branches ou remotes).
- Nao contem nem aceita tokens/credenciais embutidas em URLs ou arquivos.
- Nao apaga arquivos locais, nao modifica o artigo em PDF e jamais executa git push --force.
- Exige pelo menos 24 testes aprovados e 0 ignorados/falhos antes de publicar.
- Verifica arquivos indevidos, possiveis credenciais (indice + historico) e limite de 100 MB do GitHub.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

REPOSITORY = "https://github.com/Lucas-715/Federated-XAI-Lab.git"
BRANCH = "main"
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024  # 100 MB

FORBIDDEN_INDEX_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(^|/)\.venv/"), "ambiente virtual (.venv)"),
    (re.compile(r"(^|/)venv/"), "ambiente virtual (venv)"),
    (re.compile(r"(^|/)__pycache__/"), "cache de bytecode (__pycache__)"),
    (re.compile(r"\.py[cod]$"), "bytecode Python compilado"),
    (re.compile(r"(^|/)\.pytest_cache/"), "cache do pytest"),
    (re.compile(r"\.egg-info/"), "metadados temporarios setuptools (*.egg-info)"),
    (re.compile(r"^results/test_pipeline/"), "saida temporaria antiga (results/test_pipeline)"),
    (re.compile(r"^data/"), "diretorio de dados brutos (data/)"),
    (re.compile(r"^human\+activity\+recognition\+using\+smartphones/"), "pacote bruto local do UCI HAR"),
    (re.compile(r"\.zip$", re.IGNORECASE), "arquivo compactado (.zip)"),
    (re.compile(r"(^|/)\.env(\.|$)"), "arquivo de variaveis de ambiente (.env)"),
    (re.compile(r"\.(pem|key|p12|pfx)$", re.IGNORECASE), "chave privada ou certificado"),
)

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("github_classic_token", re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("github_fine_grained_token", re.compile(rb"github_pat_[A-Za-z0-9_]{20,}")),
    ("gitlab_token", re.compile(rb"glpat-[A-Za-z0-9_-]{20,}")),
    ("aws_access_key", re.compile(rb"AKIA[0-9A-Z]{16}")),
    ("google_api_key", re.compile(rb"AIza[0-9A-Za-z_-]{35}")),
    ("private_key_block", re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("url_embedded_credentials", re.compile(rb"https?://[^/\s:@\"'<>]+:[^/\s@\"'<>]{6,}@")),
)

GIT_GREP_SECRET_REGEX = (
    r"(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"glpat-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|"
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----)"
)


def run(*args: str, cwd: Path, check: bool = True, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        list(args),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if check and proc.returncode != 0:
        details = (proc.stdout + "\n" + proc.stderr).strip()
        raise RuntimeError(f"Comando falhou ({' '.join(args)}):\n{details[-1500:]}")
    return proc


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run("git", *args, cwd=root, check=check)


def root_check(root: Path) -> None:
    required = [
        root / "app.py",
        root / "README.md",
        root / "pyproject.toml",
        root / "requirements.txt",
        root / "LICENSE",
        root / "src" / "flxai" / "experiment.py",
        root / "tests",
        root / "docs" / "RELATORIO_EXPERIMENTAL_UCI_HAR.md",
        root / "results" / "resumo_experimentos_reais.csv",
    ]
    missing = [str(p.relative_to(root)) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Pasta invalida do Federated XAI Lab; itens ausentes: " + ", ".join(missing))

    ckpts = 0
    for seed in (42, 43, 44, 45, 46):
        cdir = root / "results" / f"uci_seed_{seed}" / "checkpoints"
        pts = list(cdir.glob("*.pt")) if cdir.is_dir() else []
        if len(pts) != 12:
            raise RuntimeError(f"Semente {seed} nao contem os 12 checkpoints esperados ({len(pts)} encontrados).")
        ckpts += len(pts)
    if ckpts != 60:
        raise RuntimeError(f"Total de checkpoints diferente de 60 ({ckpts}).")

    inside = git(root, "rev-parse", "--is-inside-work-tree", check=False)
    if inside.returncode != 0 or "true" not in inside.stdout.lower():
        raise RuntimeError("Esta pasta nao possui um repositorio Git inicializado.")
    head = git(root, "rev-parse", "--verify", "HEAD", check=False)
    if head.returncode != 0:
        raise RuntimeError("Repositorio Git sem commits existentes.")


def remote_state(root: Path) -> bool:
    res = git(root, "remote", "get-url", "origin", check=False)
    if res.returncode != 0 or not res.stdout.strip():
        return False
    current = res.stdout.strip()
    if re.search(r"(gh[pousr]_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+|https?://[^/\s:@]+:[^/\s@]+@)", current):
        raise RuntimeError("O remote 'origin' contem credenciais/token embutidos na URL! Remova-o antes de prosseguir.")
    norm_cur = current.rstrip("/").removesuffix(".git")
    norm_exp = REPOSITORY.rstrip("/").removesuffix(".git")
    if norm_cur.lower() != norm_exp.lower():
        raise RuntimeError(
            f"O remote 'origin' ja aponta para outro endereco ({current}). "
            f"Esperado: {REPOSITORY}. Revise manualmente para evitar envio ao destino errado."
        )
    return True


def inspect_files(root: Path) -> tuple[int, int, bool]:
    # Usa -z para lidar com acentos e espacos sem escape octal do Git
    listed = git(root, "ls-files", "-z").stdout
    files = [entry for entry in listed.split("\0") if entry]
    if not files:
        raise RuntimeError("Nenhum arquivo encontrado no indice Git.")

    total_bytes = 0
    article_in_git = False
    oversized: list[str] = []
    forbidden_hits: list[str] = []
    secret_hits: list[str] = []

    for rel in files:
        rel_posix = rel.replace("\\", "/")
        for rx, reason in FORBIDDEN_INDEX_PATTERNS:
            if rx.search(rel_posix):
                forbidden_hits.append(f"{rel_posix} ({reason})")

        if rel_posix.lower().endswith(".pdf"):
            article_in_git = True

        fpath = root / Path(rel_posix)
        if not fpath.is_file():
            continue
        size = fpath.stat().st_size
        total_bytes += size
        if size > MAX_GITHUB_FILE_BYTES:
            oversized.append(f"{rel_posix} ({size / (1024 * 1024):.1f} MB)")

        if fpath.suffix.lower() not in {".pt", ".png", ".pdf", ".zip"}:
            raw = fpath.read_bytes()
            for rule_name, srx in SECRET_PATTERNS:
                if srx.search(raw):
                    secret_hits.append(f"{rel_posix} [{rule_name}]")

    # Varredura adicional no historico de commits e .git/config
    git_cfg = root / ".git" / "config"
    if git_cfg.is_file():
        cfg_bytes = git_cfg.read_bytes()
        for rule_name, srx in SECRET_PATTERNS:
            if srx.search(cfg_bytes):
                secret_hits.append(f".git/config [{rule_name}]")

    revs = [c.strip() for c in git(root, "rev-list", "--all").stdout.splitlines() if c.strip()]
    if revs:
        hist_grep = git(root, "grep", "-I", "-l", "-E", GIT_GREP_SECRET_REGEX, *revs, check=False)
        if hist_grep.returncode == 0 and hist_grep.stdout.strip():
            for line in hist_grep.stdout.splitlines():
                if line.strip():
                    secret_hits.append(f"historico:{line.strip()}")

        log_msgs = git(root, "log", "--all", "--format=%H %B").stdout.encode("utf-8", errors="ignore")
        for rule_name, srx in SECRET_PATTERNS:
            if srx.search(log_msgs):
                secret_hits.append(f"commit_message [{rule_name}]")

    if forbidden_hits:
        raise RuntimeError("Arquivos indevidos detectados no indice Git:\n  - " + "\n  - ".join(forbidden_hits[:15]))
    if oversized:
        raise RuntimeError("Arquivos acima do limite de 100 MB do GitHub:\n  - " + "\n  - ".join(oversized))
    if secret_hits:
        raise RuntimeError("Possiveis credenciais/segredos detectados:\n  - " + "\n  - ".join(secret_hits[:15]))

    return len(files), total_bytes, article_in_git


def update_readme(root: Path) -> bool:
    file = root / "README.md"
    original = file.read_text(encoding="utf-8-sig")
    result = re.sub(
        r"<URL-DO-SEU-REPOSITORIO>[ \t]*#[ \t]*substituir pela URL real ap[oó]s a publica[cç][aã]o",
        REPOSITORY,
        original,
    )
    result = result.replace("<URL-DO-SEU-REPOSITORIO>", REPOSITORY)
    result = re.sub(
        r"https://github\.com/(?:SEU-USUARIO|seu-usuario|usuario)/Federated-XAI-Lab(?:\.git)?",
        REPOSITORY,
        result,
    )
    if result == original:
        return False
    file.write_text(result, encoding="utf-8", newline="")
    return True


def tests(root: Path) -> None:
    python = root / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        python = root / ".venv" / "bin" / "python"
    if not python.exists():
        raise RuntimeError("Ambiente .venv não encontrado. Configure-o com o UCI HAR antes de publicar.")
    print("\n[TESTES] Executando a suíte em ambiente local...")
    result = run(
        str(python),
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests",
        cwd=root,
        check=False,
        extra_env={"PYTHONDONTWRITEBYTECODE": "1"},
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    print("\n".join(output.splitlines()[-12:]))
    # Rejeita a execução parcial, como '9 passed, 15 skipped'.
    match = re.search(r"(?<!\d)(\d+)\s+passed\b", output)
    if (
        result.returncode
        or not match
        or int(match.group(1)) < 24
        or re.search(r"\b\d+\s+(?:skipped|failed|error|errors)\b", output)
    ):
        raise RuntimeError("Testes insuficientes ou com falhas: exigidos pelo menos 24 aprovados, 0 ignorados.")
    print("[OK] Pelo menos 24 testes aprovados e nenhum ignorado/falho.")


def check_remote_history(root: Path) -> None:
    # Falha de rede/autenticação é erro, jamais interpretada como 'repositório vazio'.
    refs = git(root, "ls-remote", "--heads", "origin").stdout.strip().splitlines()
    branches = {line.split("\t")[-1].removeprefix("refs/heads/") for line in refs if "\t" in line}
    if not branches:
        print("[OK] Repositório remoto sem branches: primeiro push permitido.")
        return
    if BRANCH not in branches:
        raise RuntimeError(
            "O GitHub já contém uma branch diferente de 'main'. Não farei push automático "
            "para evitar históricos paralelos. Revise o remoto primeiro."
        )
    git(root, "fetch", "--no-tags", "origin", BRANCH)
    ancestor = git(root, "merge-base", "--is-ancestor", f"origin/{BRANCH}", "HEAD", check=False)
    if ancestor.returncode != 0:
        raise RuntimeError(
            "A branch 'main' do GitHub possui commits não incorporados ao projeto local. "
            "Não farei push/force. Faça merge ou rebase conscientemente."
        )
    print("[OK] O histórico remoto pode receber um push sem sobrescrita.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Publicar o Federated XAI Lab no GitHub de Lucas-715")
    parser.add_argument("--publicar", action="store_true", help="Executa commit e push; sem a flag, somente prévia.")
    parser.add_argument("--raiz", type=Path, help="Caminho da raiz do projeto (padrão: pasta acima de scripts/).")
    args = parser.parse_args()
    root = (args.raiz or Path(__file__).resolve().parent.parent).resolve()
    print("=== FEDERATED XAI LAB — PUBLICAÇÃO NO GITHUB ===")
    print(f"Projeto: {root}")
    print(f"Destino: {REPOSITORY}")
    print("Segurança: não utiliza token, não apaga arquivos, não faz push --force.")
    root_check(root)
    has_origin = remote_state(root)
    branch = git(root, "branch", "--show-current").stdout.strip()
    status = git(root, "status", "--short").stdout.strip()
    preview_count, preview_bytes, preview_pdf = inspect_files(root)
    print(f"Branch local: {branch or '(sem branch)'}")
    print(f"Remote origin: {'já configurado e correto' if has_origin else 'a configurar'}")
    print(f"Estado: {status if status else 'sem modificações'}")
    print(f"Índice Git validado: {preview_count} arquivos ({preview_bytes/1_000_000:.1f} MB), 0 segredos, 0 acima de 100 MB.")
    if preview_pdf:
        print("[AVISO] Há PDF do artigo versionado no Git; ele ficará público caso publicado.")
    print(f"README: será atualizado com {REPOSITORY} se possuir endereço provisório.")
    print("O artigo e os resultados científicos não serão alterados por este script.")
    if not args.publicar:
        print("\n[PRÉVIA] Nenhum arquivo alterado; nenhum commit ou push efetuado.")
        print("Para publicar: .\\.venv\\Scripts\\python scripts\\publicar_federated_xai_github.py --publicar")
        return 0

    # Confirmação explícita antes de qualquer alteração.
    print("\nATENÇÃO: o repositório remoto foi configurado como PÚBLICO no GitHub.")
    print("Se o PDF do artigo já estiver versionado, ele também poderá ficar público.")
    print("Revogue o token compartilhado anteriormente em https://github.com/settings/tokens")
    if input("Digite PUBLICAR para autorizar a publicação: ").strip() != "PUBLICAR":
        print("Cancelado sem alterações.")
        return 0

    tests(root)
    if update_readme(root):
        print("[OK] URL do GitHub aplicada ao README.md.")
    git(root, "add", "-A")
    count, total_bytes, article_in_git = inspect_files(root)
    print(f"[OK] {count} arquivos no índice Git ({total_bytes/1_000_000:.1f} MB no disco).")
    if article_in_git:
        print("[AVISO] Há PDF(s) já incluído(s) no índice; serão publicados conforme seu histórico Git.")
    if not has_origin:
        git(root, "remote", "add", "origin", REPOSITORY)
        print("[OK] origin configurado.")
    # Muda somente o nome da branch local; preserva todos os commits.
    if branch != BRANCH:
        git(root, "branch", "-M", BRANCH)
        print("[OK] Branch local renomeada para main.")

    # Se GitHub foi inicializado com README próprio, recusar sobrescrever o histórico.
    check_remote_history(root)
    staged = git(root, "diff", "--cached", "--name-only").stdout.strip()
    if staged:
        git(root, "commit", "-m", "docs: preparar publicacao do Federated XAI Lab no GitHub")
        print("[OK] Commit criado.")
    else:
        print("[OK] Nenhuma nova alteração para commit; histórico existente preservado.")

    print("[PUSH] Enviando main para origin pelo Git Credential Manager/SSH...")
    # Não usa --force nem tokens; Git pode abrir o navegador para autenticar.
    git(root, "push", "-u", "origin", BRANCH)
    local = git(root, "rev-parse", "HEAD").stdout.strip()
    refs = git(root, "ls-remote", "origin", "refs/heads/main").stdout.strip().splitlines()
    remote = refs[0].split()[0] if refs else ""
    if local != remote:
        raise RuntimeError("Push retornou, mas o commit remoto não corresponde ao local. Confira manualmente!")
    print("\nPUBLICAÇÃO CONFIRMADA")
    print(f"GitHub: {REPOSITORY.removesuffix('.git')}")
    print(f"Branch: main | commit remoto: {remote[:12]} | arquivos versionados: {count}")
    print("O artigo não foi modificado pelo script. Você já pode trabalhar em sua revisão separadamente.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, EOFError) as exc:
        print(f"\n[INTERROMPIDO] {exc}", file=sys.stderr)
        raise SystemExit(1)

