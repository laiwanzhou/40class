# Thermal Generation-2 A2 Environment Probe

**Status:** `passed`

No formal model training was run. Each student received one throwaway bfloat16 optimizer step in memory; no checkpoint, optimizer state, or learned weight file was written.

## Runtime results

| Model | Physical / effective batch | Parameters | FP32 median latency | BF16 peak allocated | State dict | Complete package |
|---|---:|---:|---:|---:|---:|---:|
| B-X3D-XS | 2 / 8 | 3,056,634 | 17.52 ms/trial | 1,423.06 MiB | 12,570,073 bytes | 12,577,218 bytes |
| A multi-stream | 2 / 8 | 3,669,874 | 30.54 ms/trial | 2,770.52 MiB | 15,049,075 bytes | 21,327,781 bytes |

Both models produced finite 40-class logits. No batch-size fallback was required. Peak allocated memory remained below the strict 7,300 MiB gate, and both deployment proxies remained below 95,000,000 bytes.

Latency covers model execution over already prepared tensors. Online YOLO context/pose preprocessing latency was not measured.

## Time projection

- B-X3D-XS, 50 epochs: approximately 3.04 hours from one measured optimizer step.
- A-direct, 50 epochs: approximately 4.34 hours from one measured optimizer step.
- A-KD, 50 epochs: approximately 4.43 hours using a 2% heuristic KL overhead.
- C1 R(2+1)D-18, 30 epochs: approximately 13.03 hours using an unmeasured 5x multiplier.
- Conditional C2 VideoMAE-S, 30 epochs: approximately 26.05 hours using an unmeasured 10x multiplier.

Only B and A timing are measured. Route C estimates are planning heuristics and require an independent hardware probe.

## A3 blockers

1. The formal Route B report is absent.
2. The complete Thermal pose cache is absent.
3. `authorization.a_direct_training` remains false.

A2 is complete, but A3 must not start until all three conditions are resolved through their explicit gates.
