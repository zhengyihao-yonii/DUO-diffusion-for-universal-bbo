# Task metadata (English text for conditioning)

Place **one UTF-8 text file per task** used in training. Filename = task key + `.txt`.

The diffusion trainer (when `use_text_condition=True`) loads these strings, encodes them with a **frozen** sentence embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`, 384-d), and feeds the vectors into `TemporalUnet` **in addition to** the existing task one-hot / classifier-free setup. Existing behaviour is unchanged when `use_text_condition=False`.

## Required files

For `train_tasks=ant,dkitty,...` you need at least:

- `ant.txt`
- `dkitty.txt`
- …

GTOPX tasks: `gtopx2.txt`, `gtopx3.txt`, …

## Content guidelines

- Short paragraph: domain, objective, design dimensionality, reward meaning, constraints.
- English is recommended (embedding models are pretrained primarily on English).

## Example

See `ant.txt` and `dkitty.txt` in this folder.
