Codes of our paper "Learnable Color Space Conversion and Fusion for Stain Normalization in Pathology Image".
Support training classification networks with the proposed `LStainNorm` inserted at the top on histology datasets. 
In the same way, LStainNorm can be applied to other downstream diagnostic tasks.

## Code Organizations

Run [`main.py`](main.py) to train the models.

Tune all the hyper-parameters in [`config-SA1.yaml`](config-SA1.yaml).
- `train_root`: Path to the training set.
- `test_root`: Path to the test set.
- `output_path`: Path to the output. Output files will be exported to a folder created in `output_path` started with the date, hence no worry for overriding.

## Dataset

Datasets can be downloaded by links shared in our paper "Learnable Color Space Conversion and Fusion for Stain Normalization in Pathology Image".

## Methods
1. `LabPreNorm`: Learnable normalization parameters (i.e., channel mean and channel std) of the template in LAB color space.
2. `TemplateNorm`: Fixed normalization parameters in LAB space.
3. `SA3`: Our proposed LStainNorm, including color space conversion and fusion.

