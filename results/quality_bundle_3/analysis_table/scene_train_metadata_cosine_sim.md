# Scene train similarity

Encoder: `sentence-transformers/all-MiniLM-L6-v2`

## Scenario titles

- **D_train_1**: Compact sedan gasoline ICE calibration
- **D_train_2**: Compact sedan hybrid power-split tuning
- **D_train_3**: Chemical process yield tuning
- **D_train_4**: Polymer formulation screening
- **D_train_5**: Aircraft wing aerodynamic design

### Train MiniLM cosine similarity (automotive pairs)

| | D_train_1 | D_train_2 | D_train_3 | D_train_4 | D_train_5 |
|---|---:|---:|---:|---:|---:|
| **D_train_1** | 1.000 | 0.405 | 0.247 | 0.088 | 0.172 |
| **D_train_2** | 0.405 | 1.000 | 0.211 | 0.023 | 0.252 |
| **D_train_3** | 0.247 | 0.211 | 1.000 | 0.428 | 0.230 |
| **D_train_4** | 0.088 | 0.023 | 0.428 | 1.000 | 0.138 |
| **D_train_5** | 0.172 | 0.252 | 0.230 | 0.138 | 1.000 |
