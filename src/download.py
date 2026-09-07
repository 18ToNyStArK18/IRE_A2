"""Download raw MIND and EB-NeRD files.

MIND lives in a *gated* HuggingFace dataset repo (yjw1029/MIND): you must
accept the dataset's terms on huggingface.co while logged in, then supply an
access token here via --hf-token or the HF_TOKEN env var.

EB-NeRD is a public S3 bucket and needs no auth.

Idempotent: re-running skips any dataset whose extraction marker already
exists, unless --force is passed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from src import config

MARKER_NAME = ".extracted"


def _already_done(dest_dir: Path) -> bool:
    return (dest_dir / MARKER_NAME).exists()


def _mark_done(dest_dir: Path) -> None:
    (dest_dir / MARKER_NAME).write_text("ok\n")


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    # Some archives (e.g. MIND) wrap their contents in a single top-level
    # subdirectory; flatten it so downstream code has stable paths.
    entries = [p for p in dest_dir.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        nested = entries[0]
        for item in nested.iterdir():
            shutil.move(str(item), str(dest_dir / item.name))
        nested.rmdir()


def _download_http(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest_path.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def download_mind(hf_token: str | None = None, force: bool = False) -> None:
    from huggingface_hub import hf_hub_download

    token = hf_token or os.environ.get("HF_TOKEN")
    for split, filename in config.MIND_FILES.items():
        split_dir = config.MIND_RAW_DIR / split
        if _already_done(split_dir) and not force:
            print(f"[mind:{split}] already extracted, skipping")
            continue
        print(f"[mind:{split}] downloading {filename} from {config.MIND_HF_REPO} ...")
        try:
            local_path = hf_hub_download(
                repo_id=config.MIND_HF_REPO,
                repo_type="dataset",
                filename=filename,
                token=token,
            )
        except Exception as e:  # noqa: BLE001 - surface a clear actionable message
            raise RuntimeError(
                f"Failed to download MIND file '{filename}'. MIND is a gated HF "
                "dataset: log in at huggingface.co, accept the dataset terms at "
                f"https://huggingface.co/datasets/{config.MIND_HF_REPO}, generate a "
                "token at huggingface.co/settings/tokens, and pass it via --hf-token "
                f"or the HF_TOKEN env var. Original error: {e}"
            ) from e
        print(f"[mind:{split}] extracting to {split_dir}")
        _extract_zip(Path(local_path), split_dir)
        _mark_done(split_dir)


def download_ebnerd(bundle: str = config.EBNERD_BUNDLE, force: bool = False) -> None:
    filename = config.EBNERD_BUNDLE_FILES[bundle]
    dest_dir = config.EBNERD_RAW_DIR / bundle
    if _already_done(dest_dir) and not force:
        print(f"[ebnerd:{bundle}] already extracted, skipping")
        return
    url = f"{config.EBNERD_BASE_URL}/{filename}"
    zip_path = config.EBNERD_RAW_DIR / filename
    print(f"[ebnerd:{bundle}] downloading {url} ...")
    _download_http(url, zip_path)
    print(f"[ebnerd:{bundle}] extracting to {dest_dir}")
    _extract_zip(zip_path, dest_dir)
    zip_path.unlink()
    _mark_done(dest_dir)


def download_ebnerd_artifact(name: str = config.EBNERD_PROVIDED_ARTIFACT, force: bool = False) -> None:
    rel_path = config.EBNERD_ARTIFACT_FILES[name]
    filename = rel_path.split("/")[-1]
    dest_dir = config.EBNERD_RAW_DIR / "artifacts" / name
    if _already_done(dest_dir) and not force:
        print(f"[ebnerd:artifact:{name}] already extracted, skipping")
        return
    url = f"{config.EBNERD_BASE_URL}/{rel_path}"
    zip_path = config.EBNERD_RAW_DIR / "artifacts" / filename
    print(f"[ebnerd:artifact:{name}] downloading {url} ...")
    _download_http(url, zip_path)
    print(f"[ebnerd:artifact:{name}] extracting to {dest_dir}")
    _extract_zip(zip_path, dest_dir)
    zip_path.unlink()
    _mark_done(dest_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for gated MIND repo")
    parser.add_argument("--ebnerd-bundle", choices=["demo", "small"], default=config.EBNERD_BUNDLE)
    parser.add_argument("--force", action="store_true", help="Re-download even if already extracted")
    args = parser.parse_args()

    if args.dataset in ("mind", "all"):
        download_mind(hf_token=args.hf_token, force=args.force)
    if args.dataset in ("ebnerd", "all"):
        download_ebnerd(bundle=args.ebnerd_bundle, force=args.force)


if __name__ == "__main__":
    main()
