"""Download only the Replogle assets, with pins, resume, and SHA256 verification."""
import hashlib
import json
from pathlib import Path
import requests
import zstandard as zstd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
PIN = "10654f24cfb539d619e8221ec61eb7e878364cec"
ASSETS = {
    "replogle.h5ad.zst": (f"https://huggingface.co/datasets/katarinayuan/PerturbDiff_data/resolve/{PIN}/perturb_data/finetune_data/nadig_processed_data/replogle.h5ad.zst", None),
    "replogle_real_selected_genes.pkl": (f"https://huggingface.co/datasets/katarinayuan/PerturbDiff_data/resolve/{PIN}/perturb_data/selected_genes/replogle_real_selected_genes.pkl", None),
    "esm2_human.pt": ("https://huggingface.co/minwoosun/uce-misc/resolve/bffb91084e4476698984e7e01f6170ce291f4074/protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt", "a210e1cc7901513999b2bca3836ba9e2f203cd008be4e9a9d6412a2267de9748"),
    "hgnc_complete_set.tsv": ("https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt", None),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''): h.update(b)
    return h.hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    tree = requests.get(f'https://huggingface.co/api/datasets/katarinayuan/PerturbDiff_data/tree/{PIN}?recursive=true&limit=1000', timeout=90)
    tree.raise_for_status()
    tree = {x['path'].split('/')[-1]: x for x in tree.json()}
    manifest = []
    for name, (url, checksum) in ASSETS.items():
        p = RAW / name
        meta = tree.get(name, {})
        checksum = checksum or meta.get('lfs', {}).get('oid')
        expected = meta.get('size')
        if not p.exists() or (expected is not None and p.stat().st_size != expected):
            start = p.stat().st_size if p.exists() else 0
            with requests.get(url, headers={'Range': f'bytes={start}-'} if start else {}, stream=True, timeout=120) as r:
                r.raise_for_status()
                mode = 'ab' if start and r.status_code == 206 else 'wb'
                with open(p, mode) as f:
                    for b in r.iter_content(8 << 20): f.write(b)
        digest = sha256(p)
        if checksum and digest != checksum: raise ValueError(f'Checksum mismatch: {name}')
        manifest.append({'name': name, 'url': url, 'bytes': p.stat().st_size, 'sha256': digest})
        print(name, digest, flush=True)
    out = RAW / 'replogle.h5ad'
    raw_hash = '2c53aadd1bbbaee7604b579b2592034a0b9d9bfa9eecfc6cdc622a9d2e205c77'
    if not out.exists() or out.stat().st_size != 22297813040:
        tmp = out.with_suffix('.h5ad.tmp')
        with open(RAW / 'replogle.h5ad.zst', 'rb') as f, open(tmp, 'wb') as g:
            zstd.ZstdDecompressor().copy_stream(f, g)
        tmp.replace(out)
    digest = sha256(out)
    if digest != raw_hash: raise ValueError('Decompressed data differs from Arc release')
    manifest.append({'name': out.name, 'bytes': out.stat().st_size, 'sha256': digest,
                     'same_as': 'https://huggingface.co/datasets/arcinstitute/Replogle-Nadig-Preprint/blob/833d2be/replogle.h5ad'})
    (RAW / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Verified: decompressed dataset is byte-identical to Arc Institute release.', flush=True)


if __name__ == '__main__': main()

