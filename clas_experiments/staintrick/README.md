Support training the classification network on histology datasets. 
Explore multiple tricks based on stain variations.

## Code Organizations

Run [`main.py`](main.py) to train the models.

Tune all the hyper-parameters in [`config.yaml`](config.yaml).
- `train_root`: Path to the training set.
- `test_root`: Path to the test set.
- `output_path`: Path to the output. Output files will be exported to a folder created in `output_path` started with the date, hence no worry for overriding.

## Dataset

Datasets can be downloadee use [`preprocess/download.py`](preprocess/download.py).

- Normalized-v1: Stain normalized with template `NORM-AAAWMSFI.tif` (from training set).
- Normalized-v2: Stain normalized with template `STR-AAEILWWE.tif` (from training set).
- Normalized-v3: Stain normalized with template `NORM-TCGA-AASSYQPA.tif` (from test set).

## Methods
1. `LabPreNorm`: Learnable normalization parameters (i.e., channel mean and channel std) of the template in LAB color space, and use the Reinhard's normalization method.
3. `LabEMAPreNorm`: Use EMA to update the normalization template. Hyper-parameter: lambda. When lambda=0, degenerates to vanilla Reinhard's normalization method; when lambda=1, degenerates to a speical case of `LabRandNorm`.
2. `LabRandNorm`: Randomly select template in each mini-batch, and use the Reinhard's normalization method.

## Results
| *ResNet-18* | w/o Pretrain | w/ Pretrain |
| -- | -- | -- |
| w/o Norm    | 64.958 | 58.788 |
| w/ Norm v1  | 78.914 | 78.106 |
| w/ Norm v3  | 89.624 | 89.262 |
| w/ RandNorm | 88.454 | |
| w/ PreNorm  | 92.549 | |
| w/ EMAPreNorm (lambd=0) | 91.504 | |

