# meth-SemiCancer: a cancer subtype classification framework via Semi-supervised learning utilizing DNA methylation profiles
meth-SemiCancer is a semi-supervised cancer subtype classification framework using both labeled and unlabeled DNA methylation profiles. meth-SemiCancer framework was first pre-trained based on the methylation datasets with the cancer subtype labels. After pre-training, meth-SemiCancer generated the pseudo-subtypes for the cancer datasets without subtype information based on the model’s prediction. Finally, fine-tuning was performed utilizing both the labeled and unlabeled datasets.
![cancer_subtype_classificaiton](https://user-images.githubusercontent.com/48755108/184848640-ed88ce6e-76dd-4212-8b37-8912c8b62352.png)

## Requirements
* Tensorflow (>= 1.8.0)
* Python (>= 2.7)
* Python packages : numpy, pandas

## Usage
Clone the repository or download source code files.

## Preprocessing
* Reference: ./preprocessing/
1. Edit **"run_preproc.sh"** file having DNA methylation beta value dataset files and cancer subtype file. Modify each variable values in the bash file with filename for your own dataset. Cancer subtype file for labeled dataset should contain cancer subtype annotation for each sample, where each row and column represent **sample ID** and **cancer subtype**, respectively. Example for data format is described below.
```
sample_ID,cancer_subtype
sample_1,LumA
sample_2,LumB
sample_3,Her2
...
sample_n,LumA
```
Each DNA methylation file for labeled and unlabeled datasets should contain matrix of DNA methylation beta value, where each row and column represent **CpG site** and **sample ID**, respectively :
```
cpg,sample_1,sample_2,...,sample_n
cg00000029,0.249737,0.464333,...,0.061501
cg00000165,0.347463,0.115849,...,0.216793
cg00000236,0.917430,0.881644,...,0.908840
...
```

2. Use **"run_preproc.sh"** to perform preprocessing.
3. You will get outputs **"train_X.csv", "train_Y.csv", "test_X.csv", "test_Y.csv", "unlabel_X.csv", "unlabel_Y.csv"**.

## Semi-supervised cancer subtype classification
* Reference: ./meth_SemiCancer/
1. Use "run_meth_SemiCancer.sh" to predict the cancer subtypes. If you want to use the confidence threshold option for pseudo-labeling, you can edit the confidence_threshold variable.
2. You can get the final output **"result_for_test_dataset.csv"** with classified cancer subtypes for test dataset.

## Contact
If you have any question or problem, please send an email to joungmin@vt.edu
