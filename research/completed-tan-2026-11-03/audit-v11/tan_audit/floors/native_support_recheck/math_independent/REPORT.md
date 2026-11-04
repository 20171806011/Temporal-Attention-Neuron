# Independent rectangle-integral recheck

The proposed native-output population floors are arithmetically correct for the intended rational reconstruction levels:

| Native output superset | Exact relaxed E_comp floor |
| --- | --- |
| Count {-2,0} | 13/9 |
| Latency {-2.85+0.30j : j=7,...,19} union {0} | 1243/1600 |

This check does not validate the dynamical argument that these supersets contain every native-task output; that is the primary audit's separate responsibility. It independently validates the integration conditional on those supports.

Let the targets X,Y be independent uniform variables on [-3,3]. For a fixed alphabet, nearest-symbol reconstruction independently minimizes |error X| and |error Y|, and therefore minimizes their maximum even if a decoder is granted both exact targets and arbitrary coupled decisions. Partition [-3,3] into nearest-symbol Voronoi cells and split every cell at its reconstruction value. A half-cell of length a has distance coordinate x in [0,a], with unit Jacobian. For any pair of half-cells with lengths a,b, set m=min(a,b), M=max(a,b). Directly integrating max(x,y) over their distance rectangle gives

`integral_[0,a] integral_[0,b] max(x,y) dy dx = m*M^2/2 + m^3/6`.

This follows by integrating over the square of side m and the remaining rectangle. Summing over all ordered half-cell pairs and dividing by 36, the joint uniform support area, produces the population oracle loss. No distance CDF or previous floor-audit function is used.

For count, half-lengths are [1,1,1,3]; the 16 rectangle contributions sum to 52, giving 52/36=13/9. For latency, the 28 half-lengths are recorded in results.json; the 784 rectangle contributions sum to 11187/400, giving (11187/400)/36=1243/1600.

Silence contributes the value zero to the unconditional composition error without an extra abstention penalty, so zero must be included in these E_comp alphabets. Allowing the oracle to select zero freely is a valid relaxation. It does not establish successful delivery: a silence event remains an unsuccessful delivery even when the target is zero or close to zero. A different metric that charged abstention would require another oracle calculation.

These are population bounds for independent uniform targets and the supplied output supersets, not lower bounds on every realized finite-sample mean. They do not apply to extra analog outputs, per-trial output rescaling, altered alphabets, changed target priors, or different composition scoring.

The exact fractions use the intended decimal codebook as rational numbers. For completeness, treating actual Python binary floating reconstruction constants as exact real numbers gives a latency-support floor smaller by approximately 3.3306690738754163e-18. results.json records the exact rational difference. Thus the decimal/rational wording should not be used to assert literal arithmetic exactness of the full floating pipeline; this difference is immaterial to the registered 0.05 threshold. No conceptual or arithmetic defect was found in the relaxed population calculations under the stated assumptions.

Executed successfully with `PYTHONDONTWRITEBYTECODE=1 python3 rectangle_integral.py`. Deliverables: rectangle_integral.py, results.json, run.log. No network, repository edits, archive access, or unresolved blockers.
