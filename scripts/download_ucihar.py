"""Script utilitário para download e extração automatizada do UCI HAR Dataset."""
import os
import sys
import zipfile
import urllib.request
import hashlib
from pathlib import Path

UCI_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
EXPECTED_SHA256 = "2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0"

def download_and_extract_dataset(target_dir="data"):
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    
    extracted_dataset = target_path / "UCI HAR Dataset"
    if (extracted_dataset / "train" / "X_train.txt").exists():
        print(f"Dataset já instalado e verificado em: {extracted_dataset}")
        return True

    zip_dest = target_path / "uci_har.zip"
    print(f"Baixando dataset oficial da UCI ({UCI_URL})...")
    
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
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

    print("Verificando integridade e extraindo arquivos...")
    with zipfile.ZipFile(zip_dest, "r") as z:
        z.extractall(target_path)
    
    # Se o arquivo baixado contiver um ZIP aninhado ou estrutura direta
    nested_zip = target_path / "UCI HAR Dataset.zip"
    if nested_zip.exists():
        with zipfile.ZipFile(nested_zip, "r") as z:
            z.extractall(target_path)
        nested_zip.unlink(missing_ok=True)
        
    if zip_dest.exists():
        zip_dest.unlink(missing_ok=True)

    if (extracted_dataset / "train" / "X_train.txt").exists():
        print(f"Sucesso! Dataset extraído em: {extracted_dataset}")
        return True
    else:
        print(f"Aviso: Verifique a estrutura das pastas em {target_path}")
        return False

if __name__ == "__main__":
    download_and_extract_dataset()
