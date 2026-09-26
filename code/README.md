# G2P-WAM

Compact tensor-level reference implementation for review; not a full reproduction package.

| Module | Scope |
| --- | --- |
| `teachers.py` | Tensor contracts for external VGGT features and SpatialTrackerV2 tracks |
| `geosft.py` | Training-only static alignment and dynamic displacement readouts |
| `diagnosis.py` | 4RC confidence aggregation, outcome AUC, score–motion association |
| `preferences.py` | Outcome-stratified confidence rank filtering and pair construction |
| `geodpo.py` | Dual-stream preference loss with a frozen-reference prediction anchor |

Core inputs are caller-supplied tensors. Static targets must already share the
student token grid; tracks use one common coordinate frame. Explicit group IDs
define pairing scope, not a guarantee of identical resets. The reference and
policy predictions are supplied separately. No experiment results are encoded.

## CPU checks

With Python 3.10+ and PyTorch installed, from this directory:

```sh
python -B -m unittest discover -s tests -v
```
