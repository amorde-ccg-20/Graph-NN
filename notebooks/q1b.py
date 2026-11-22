from itertools import product
import numpy as np


K = 8
BASE = K + 1

phi = {-1: BASE**2, 0: BASE, 1: 1}

counts = np.array(
    [t for t in product(range(K + 1), repeat=3)
     if 1 <= sum(t) <= K],
    dtype=np.int64,
)

codes = counts @ np.array([phi[-1], phi[0], phi[1]], dtype=np.int64)
decoded = np.column_stack(
    (codes // BASE**2, (codes // BASE) % BASE, codes % BASE)
)

assert np.array_equal(decoded, counts), "A count triple did not decode correctly"
assert np.unique(codes).size == len(counts), "Two triples share a code"
print(f"K={K}, base={BASE}: {len(counts)} triples decoded with all codes unique")

nonsteady_phi = {-1: 1, 0: 0, 1: 1}
assert all(nonsteady_phi[x] == abs(x) for x in (-1, 0, 1))
print("Nonsteady count for any neighbourhood size: sum(abs(x) for x in neighbours)")
