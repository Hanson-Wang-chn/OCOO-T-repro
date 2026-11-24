"""Remove only the verified, redundant decompressed H5AD; retain its archive."""
import json
from pathlib import Path
import numpy as np
from ocoot.runtime import file_sha256,readme_event,timestamp


def main():
    raw=Path('data/raw');archive=raw/'replogle.h5ad.zst';expanded=raw/'replogle.h5ad'
    expected='5f42388fff4a50414e67bc1b5b3685208a3b1b0b92920a6eb87bf1004e8976d8'
    if file_sha256(archive)!=expected:raise ValueError('Canonical archive verification failed; nothing removed')
    assert np.load('data/processed/expression.npy',mmap_mode='r').shape==(643413,2000)
    report=json.loads(Path('data/processed/data_report.json').read_text())
    assert report['train']['cells']==572545 and report['test']['cells']==26878
    if expanded.exists():
        assert expanded.stat().st_size==22297813040
        expanded.unlink()
        note={'time':timestamp(),'removed':'replogle.h5ad (redundant decompressed copy)',
              'archive_retained':str(archive),'archive_sha256':expected,
              'restore_command':'python scripts/download_data.py',
              'training_cache_retained':'data/processed/expression.npy'}
        (raw/'storage_note.json').write_text(json.dumps(note,indent=2))
        readme_event('核验压缩原始数据 SHA256 后，移除约 22.30 GB 的重复解压副本以预留检查点空间；原始压缩包、训练缓存及所有实验权重均保留。需要完整 H5AD 时运行 `python scripts/download_data.py` 即可本地恢复。')
        print(json.dumps(note,indent=2),flush=True)


if __name__=='__main__':main()

