import json
from types import SimpleNamespace
import numpy as np
from ocoot.evaluate import official_metrics


def test_capped_generation_uses_full_heldout_reference_without_leakage(tmp_path):
    data=SimpleNamespace(
        x=np.array([[100,100,100],[2,1,1],[4,1,1],[1,2,1],[1,4,1],[1,1,1],[1,1,1],[200,200,200]],dtype=np.float32),
        meta={'cell':np.zeros(8,dtype=int),'pert':np.array([0,0,0,1,1,2,2,0]),'split':np.array([0,1,1,1,1,3,3,2])},
        control=np.array([5,6]),val=np.array([1,2,3,4]),test=np.array([7]),
        vocab={'all_genes':['X','Y','non-targeting'],'genes':['G1','G2','G3']})
    # Negative continuous outputs are valid for array-metric diagnostics.
    pred=np.array([[-.1,1,1],[1,3,1]],dtype=np.float32)
    scores=official_metrics(data,np.array([1,3]),pred,tmp_path)
    assert json.loads((tmp_path/'population_counts.json').read_text())=={
        'real_perturbed_cells':4,'generated_cells':2,'shared_real_controls':2}
    np.testing.assert_array_equal(np.load(tmp_path/'real_indices.npy'),[1,2,3,4])
    assert abs(scores['MSE']-(3.1**2/3)/2)<1e-6

