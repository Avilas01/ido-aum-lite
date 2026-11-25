# Measured experiment comparison

CIFAR-10 | 20,000 train / 10,000 test | 6 warm-up + 6 continuation epochs | seed 42 (single seed)

Single-seed exploratory observations from verified saved evaluation outputs. No significance test or reliable-superiority claim.

| Noise | Method | Test accuracy (%) | Difference vs Standard (pp) |
| --- | --- | ---: | ---: |
| 20% | Standard | 70.42 | +0.00 |
| 20% | IDO-lite (WE) | 75.26 | +4.84 |
| 20% | Hybrid (WE + AUM) | 74.76 | +4.34 |
| 40% | Standard | 63.80 | +0.00 |
| 40% | IDO-lite (WE) | 68.61 | +4.81 |
| 40% | Hybrid (WE + AUM) | 68.48 | +4.68 |

Detector metrics use the shared, prespecified final warm-up epoch; they are identical across continuation methods by design.

| Noise | WE ROC-AUC | -AUM ROC-AUC | Hybrid ROC-AUC |
| --- | ---: | ---: | ---: |
| 20% | 0.899201 | 0.943642 | 0.943767 |
| 40% | 0.867467 | 0.921886 | 0.919509 |

Source experiment commit: `695f50fef1915f0ab2d370ba6c8cda1282e9dae2`.
Configuration SHA256: `a949134949d25219405ea573349acee3f46191429c5542b7b46a4a33d517acc0`.

Standard-versus-adaptive differences compare whole training strategies. WE-only versus Hybrid uses the same adaptive objective and views; only suspicion scoring differs.
