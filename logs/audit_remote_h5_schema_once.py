"""Inspect only HDF5 structure via bounded, validated public HTTP ranges."""
import io
import json
from pathlib import Path
import time

import h5py
import requests


class RangeReader(io.RawIOBase):
    def __init__(self, url, size, block_size=65536, budget=8 << 20):
        self.url, self.size, self.block_size, self.budget = url, size, block_size, budget
        self.position, self.transferred = 0, 0
        self.cache, self.ranges = {}, []
        self.session = requests.Session()

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0: raise ValueError('negative offset')
        return self.position

    def read(self, size=-1):
        if size < 0 or size > self.budget:
            raise ValueError('Large or unbounded reads are forbidden in this metadata audit')
        end = min(self.size, self.position + size)
        chunks = []
        while self.position < end:
            block = self.position // self.block_size
            start, stop = block * self.block_size, min(self.size, (block + 1) * self.block_size)
            if block not in self.cache:
                if self.transferred + stop - start > self.budget:
                    raise ValueError('Metadata transfer budget exceeded')
                with self.session.get(self.url, headers={'Range': f'bytes={start}-{stop-1}'},
                                      timeout=(15, 30), stream=True) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise ValueError('Server did not honor the range; refusing a full download')
                    if response.headers.get('Content-Range') != f'bytes {start}-{stop-1}/{self.size}':
                        raise ValueError('Unexpected Content-Range')
                    content = response.raw.read(stop - start + 1)
                if len(content) != stop - start:
                    raise ValueError('Unexpected range length')
                self.cache[block] = content
                self.transferred += len(content)
                self.ranges.append([start, stop - 1])
            count = min(end - self.position, stop - self.position)
            offset = self.position - start
            chunks.append(self.cache[block][offset:offset + count])
            self.position += count
        return b''.join(chunks)

    def readinto(self, buffer):
        content = self.read(len(buffer))
        buffer[:len(content)] = content
        return len(content)


def describe(value):
    result = {'kind': 'dataset' if isinstance(value, h5py.Dataset) else 'group'}
    if isinstance(value, h5py.Dataset):
        result.update(shape=value.shape, dtype=str(value.dtype), compression=value.compression)
    else:
        result['keys'] = list(value.keys())
        if 'shape' in value.attrs:
            result['shape_attribute'] = value.attrs['shape'].tolist()
    for key in ('encoding-type', 'encoding-version'):
        if key in value.attrs:
            result[key] = str(value.attrs[key])
    return result


def main():
    repository = 'arcinstitute/Replogle-Nadig-Preprint'
    metadata = requests.get(f'https://huggingface.co/api/datasets/{repository}/revision/833d2be', timeout=30)
    metadata.raise_for_status()
    revision = metadata.json()['sha']
    url = f'https://huggingface.co/datasets/{repository}/resolve/{revision}/replogle.h5ad'
    reader = RangeReader(url, 22297813040)
    with h5py.File(reader, 'r') as handle:
        schema = {'root_keys': list(handle.keys()), 'root': {key: describe(handle[key]) for key in handle}}
        for group in ('raw', 'obsm', 'layers'):
            if group in handle and isinstance(handle[group], h5py.Group):
                schema[group] = {key: describe(handle[group][key]) for key in handle[group]}
    result = {'time': time.strftime('%Y-%m-%dT%H:%M:%S+08:00'), 'url': url, 'revision': revision,
              'file_size': reader.size, 'file_sha256_from_pinned_lfs_metadata': '2c53aadd1bbbaee7604b579b2592034a0b9d9bfa9eecfc6cdc622a9d2e205c77',
              'metadata_bytes_transferred': reader.transferred, 'http_ranges': reader.ranges,
              'expression_datasets_read': False, 'schema': schema}
    Path('paper/replogle_h5_schema_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
