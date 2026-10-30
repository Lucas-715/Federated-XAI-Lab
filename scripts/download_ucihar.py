"""Download e extracao do UCI HAR Dataset, com dois niveis de verificacao distintos.

1. Verificacao CRIPTOGRAFICA (SHA-256): aplicada somente ao arquivo 'UCI HAR Dataset.zip'
   obtido durante o download. Se o pacote baixado nao contiver esse ZIP interno, o script
   NAO consegue verificar o hash e emite um AVISO explicito (ver `verify_hash`).
2. Verificacao ESTRUTURAL: presenca e tamanho minimo de 8 arquivos obrigatorios. Esta e a
   unica verificacao aplicada a dados ja extraidos em disco; ela NAO prova integridade
   dos bytes, apenas que a arvore de arquivos esta completa.

Sobre o hash de referencia: o valor abaixo foi obtido do arquivo 'UCI HAR Dataset.zip'
baixado do repositorio da UCI durante este projeto (60.999.314 bytes). Nao foi confirmado
em uma segunda fonte independente; confira-o contra uma referencia confiavel antes de
usa-lo como garantia de autenticidade.
"""
import os
import sys
import zipfile
import urllib.request
import hashlib
from pathlib import Path

UCI_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
# SHA-256 observado para o arquivo interno 'UCI HAR Dataset.zip' (60.999.314 bytes).
# Hash de referencia do projeto; NAO confirmado em fonte independente (ver docstring).
EXPECTED_INNER_ZIP_SHA256 = "2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0"

MANDATORY_FILES = {
    "train/X_train.txt": 50_000_000,       # ~66.0 MB
    "train/y_train.txt": 10_000,           # ~14.7 KB
    "train/subject_train.txt": 15_000,     # ~20.1 KB
    "test/X_test.txt": 20_000_000,         # ~26.4 MB
    "test/y_test.txt": 4_000,              # ~5.8 KB
    "test/subject_test.txt": 5_000,        # ~7.9 KB
    "features.txt": 10_000,                # ~15.7 KB
    "activity_labels.txt": 50              # ~80 B
}

def compute_sha256(filepath: Path) -> str:
    """Calcula o hash SHA-256 de um arquivo em blocos de 64 KB."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()

def verify_hash(zip_path: Path) -> bool:
    """Verificacao criptografica do ZIP interno. Retorna True somente se o hash coincidir."""
    actual = compute_sha256(zip_path)
    print(f"  Hash obtido:    {actual}")
    print(f"  Hash esperado:  {EXPECTED_INNER_ZIP_SHA256}")
    return actual.lower() == EXPECTED_INNER_ZIP_SHA256.lower()

def verify_extracted_dataset(dataset_dir: Path) -> bool:
    """Verifica a presença e o tamanho mínimo de todos os arquivos obrigatórios."""
    if not dataset_dir.is_dir():
        return False
    
    missing = []
    too_small = []
    for rel_path, min_size in MANDATORY_FILES.items():
        p = dataset_dir / rel_path
        if not p.is_file():
            missing.append(rel_path)
        elif p.stat().st_size < min_size:
            too_small.append((rel_path, p.stat().st_size, min_size))
            
    if missing:
        print(f"Arquivos obrigatórios ausentes em {dataset_dir}: {missing}")
        return False
    if too_small:
        print(f"Arquivos corrompidos ou incompletos em {dataset_dir}: {too_small}")
        return False
        
    return True

def download_and_extract_dataset(target_dir="data", strict=False):
    """Baixa/extrai o dataset.

    strict=True: so retorna sucesso se o SHA-256 do ZIP interno tiver sido efetivamente
    verificado durante este download.
    """
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    
    extracted_dataset = target_path / "UCI HAR Dataset"
    if verify_extracted_dataset(extracted_dataset):
        print("Dataset ja presente em disco: verificacao ESTRUTURAL aprovada (8 arquivos obrigatorios,")
        print("tamanhos minimos). Nenhuma verificacao criptografica foi feita nesta execucao;")
        print("para verificar o SHA-256 apague 'data/UCI HAR Dataset' e execute novamente.")
        print(f"  -> {extracted_dataset}")
        return not strict

    zip_dest = target_path / "uci_har_download.zip"
    print(f"Baixando pacote oficial da UCI:")
    print(f"  URL: {UCI_URL}")
    print(f"  Destino: {zip_dest}")
    
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = min(100.0, readsofar * 100.0 / totalsize)
            s = f"\rProgresso: {percent:5.1f}% [{readsofar/(1024*1024):.1f}/{totalsize/(1024*1024):.1f} MB]"
            sys.stdout.write(s)
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(UCI_URL, zip_dest, reporthook=reporthook)
        print("\nDownload concluído com sucesso!")
    except Exception as e:
        print(f"\nErro no download automático: {e}")
        print("Você pode baixar manualmente em:")
        print(f"  {UCI_URL}")
        print(f"e extrair o conteúdo na pasta: {target_path}")
        return False

    print("Extraindo arquivo baixado...")
    with zipfile.ZipFile(zip_dest, "r") as z:
        z.extractall(target_path)
    zip_dest.unlink(missing_ok=True)
    
    # O pacote oficial da UCI pode conter o arquivo 'UCI HAR Dataset.zip' internamente
    nested_zip = target_path / "UCI HAR Dataset.zip"
    if not nested_zip.exists():
        # Procura em subpastas caso extraído com pasta intermediária
        for candidate in target_path.glob("**/UCI HAR Dataset.zip"):
            nested_zip = candidate
            break

    hash_verified = False
    if nested_zip.exists():
        print(f"Verificando SHA-256 de '{nested_zip.name}'...")
        if not verify_hash(nested_zip):
            print("ERRO DE INTEGRIDADE: o SHA-256 do ZIP interno nao coincide com o hash de referencia!")
            return False
        hash_verified = True
        print("  -> SHA-256 coincide com o hash de referencia do projeto.")
        
        print(f"Extraindo '{nested_zip.name}'...")
        with zipfile.ZipFile(nested_zip, "r") as z:
            z.extractall(target_path)
        nested_zip.unlink(missing_ok=True)
    else:
        print("AVISO: o pacote baixado nao contem 'UCI HAR Dataset.zip'; o SHA-256 NAO foi verificado.")
        print("       Apenas a verificacao estrutural sera aplicada.")
        if strict:
            print("Modo --strict: abortando por nao ser possivel verificar o hash.")
            return False

    # Verificação pós-extração de todos os arquivos essenciais
    if verify_extracted_dataset(extracted_dataset):
        print("\nDataset UCI HAR instalado. Verificacoes realizadas nesta execucao:")
        print(f"  - SHA-256 do ZIP interno: {'APROVADO' if hash_verified else 'NAO VERIFICADO'}")
        print("  - Estrutural (8 arquivos obrigatorios, tamanhos minimos): APROVADO")
        for rel_path, _ in MANDATORY_FILES.items():
            sz_mb = (extracted_dataset / rel_path).stat().st_size / (1024 * 1024)
            print(f"  [OK] {rel_path} ({sz_mb:.2f} MB)")
        return True
    else:
        print(f"\nErro: Extração incompleta ou estrutura inesperada em {target_path}")
        return False

if __name__ == "__main__":
    ok = download_and_extract_dataset(strict="--strict" in sys.argv[1:])
    sys.exit(0 if ok else 1)
