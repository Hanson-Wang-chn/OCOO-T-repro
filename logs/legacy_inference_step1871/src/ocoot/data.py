import json
from pathlib import Path
import numpy as np
import torch
from .model import ModelConfig, OCOOT


class BenchmarkData:
    def __init__(self, path='data/processed'):
        self.path=Path(path)
        self.x=np.load(self.path/'expression.npy',mmap_mode='r')
        self.meta=dict(np.load(self.path/'metadata.npz'))
        self.vocab=json.loads((self.path/'vocab.json').read_text())
        self.train=np.flatnonzero(self.meta['split']==0)
        self.val=np.flatnonzero(self.meta['split']==1)
        self.test=np.flatnonzero(self.meta['split']==2)
        self.control=np.flatnonzero(self.meta['split']==3)

    def batch(self, indices, device='cuda'):
        return (torch.from_numpy(np.array(self.x[indices],dtype=np.float32)).to(device),
                *[torch.from_numpy(self.meta[k][indices]).to(device) for k in ['pert','cell','batch']])

    def build_model(self, cfg):
        emb=torch.load(self.path/'embeddings.pt',map_location='cpu',weights_only=True)
        return OCOOT(ModelConfig(**cfg),emb['embeddings'],torch.tensor(self.meta['gene_indices']),len(self.vocab['cell_lines']),len(self.vocab['batches']),emb['missing'])

    def evaluation_indices(self, split='validation', cap=None, seed=123):
        idx=self.val if split=='validation' else self.test
        if cap is None: return idx
        rng=np.random.default_rng(seed);result=[]
        for p in np.unique(self.meta['pert'][idx]):
            v=idx[self.meta['pert'][idx]==p]
            result.extend(sorted(rng.choice(v,min(cap,len(v)),replace=False)))
        return np.array(result,dtype=np.int64)

