import numpy as np


def compare_graphs(learned, fixed, top_k=10):
    scores = learned.copy()
    np.fill_diagonal(scores, -np.inf)
    neighbours = np.argsort(scores, axis=1)[:, -top_k:]
    sources = np.repeat(np.arange(len(scores)), top_k)
    targets = neighbours.ravel()
    physical = fixed[sources, targets] > 0

    return {
        "top_k": top_k,
        "learned_links": len(targets),
        "in_physical_graph": int(physical.sum()),
        "absent_from_physical_graph": int((~physical).sum()),
        "physical_overlap_percent": float(100 * physical.mean()),
    }
