# Registered paper results

| Method | Macro horizontal RMSE (m) | Risk AUC |
|---|---:|---:|
| Physical WLS | 27.224475 | — |
| GNC-TLS | 24.059802 | — |
| Residual FDE | 24.621221 | — |
| 9-input TDL-GNSS | 19.313849 | — |
| Null prior | 18.640288 | 0.500000 |
| Hard Jev | 16.653916 | 0.752305 |
| Soft Jev | 16.684938 | 0.765017 |

The positioning difference of Soft Jev minus Null prior is `[-3.346704, -0.547562]` m. Soft Jev minus the random prior is `[-1.108160, +0.609777]` m. The second interval crosses zero.

This file is the frozen reference. A run writes its own copy to `results/paper-v1/RESULTS.md`.
