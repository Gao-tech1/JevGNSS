# Method

The student network is `12→64→ReLU→5`, with 1,157 parameters. The 9-input TDL baseline is `9→64→128→64→2`, with 17,346 parameters. The 14-input positioning networks are `14→64→128→64→2`, with 17,666 trainable parameters. The student is frozen during positioning training.

If the student probabilities are `q`, the soft risk is `sum(q[k] * k/4)`. Hard Jev uses `onehot(argmax(q))`. The positioning network predicts a nonnegative weight factor and a correction subtracted from the pseudorange. The four standardized continuous inputs of the 14-input networks use `5*tanh(z/5)`.

A risk positive is `trusted_target < 0.5`, which means the reference relative code error exceeds three physical standard deviations. It is not an NLOS label and not a calibrated probability of failure.

| Method | Role |
|---|---|
| Physical WLS | Baseline with physical variances only |
| GNC-TLS | Continuous robust reweighting |
| Residual FDE | Hard exclusion after a residual test |
| 9-input TDL-GNSS | Learned baseline without a Jev channel |
| Null prior | Same 14-input network, five extra channels fixed at 0 |
| Hard Jev | One-hot grade from the frozen student |
| Soft Jev | Full five-level distribution from the frozen student |

The same training run also fits a random prior. The reproduction check still loads that ablation. The public data archive does not include it.
