import os
import zipfile
from pathlib import Path

def create_deliverable_zip():
    root_dir = Path(".").resolve()
    zip_path = root_dir / "Federated_XAI_Lab_Entregavel.zip"
    
    # Files and folders to include relative to root
    include_paths = [
        "src",
        "tests",
        "scripts",
        "results",
        "docs",
        "app.py",
        "pyproject.toml",
        "requirements.txt",
        "LICENSE",
        "README.md"
    ]
    
    # Exclude patterns
    exclude_dirs = {
        "__pycache__",
        ".pytest_cache",
        ".git",
        ".venv",
        "data",
        "federated_xai_lab.egg-info"
    }
    exclude_extensions = {".pyc", ".pyo", ".zip"}
    
    print(f"Gerando arquivo {zip_path.name}...")
    entry_count = 0
    total_uncompressed_bytes = 0
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for item in include_paths:
            item_path = root_dir / item
            if not item_path.exists():
                print(f"Aviso: {item} nao existe!")
                continue
                
            if item_path.is_file():
                arcname = item
                zipf.write(item_path, arcname)
                entry_count += 1
                total_uncompressed_bytes += item_path.stat().st_size
            else:
                for root, dirs, files in os.walk(item_path):
                    # Filter out directories
                    dirs[:] = [d for d in dirs if d not in exclude_dirs]
                    
                    for file in files:
                        file_path = Path(root) / file
                        if file_path.suffix.lower() in exclude_extensions:
                            continue
                        if file_path.name == zip_path.name:
                            continue
                        
                        arcname = str(file_path.relative_to(root_dir)).replace("\\", "/")
                        zipf.write(file_path, arcname)
                        entry_count += 1
                        total_uncompressed_bytes += file_path.stat().st_size

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Sucesso! {zip_path.name} criado com {entry_count} entradas.")
    print(f"Tamanho comprimido: {zip_size_mb:.2f} MB")
    print(f"Tamanho descompactado: {total_uncompressed_bytes / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    create_deliverable_zip()
