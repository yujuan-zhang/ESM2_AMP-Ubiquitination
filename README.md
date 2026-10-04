# ESM2-Ubiquitination Prediction
## Introduction
### Author Contact Information:
Author 1: Junhao Liu, Email: 895232226@qq.com

Author 2: Zeyu Luo, Email: 1024226968@qq.com, https://orcid.org/0000-0001-6650-9975

Author 3: Rui Wang, Email: 2219312248@qq.com

Author 4：Yujuan Zhang, Email: yujuan.zhang418@gmail.com
## Data available
### Inference Test Data

To evaluate the performance of the EUP models, we have prepared a independent test dataset:

- **[ESM2_3B_2560](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Inference_test_data/ESM2_3B_2560)**: This directory contains pre-extracted ESM2 features (`test_features.npy`) and aligned sample information (`test_info.csv`) for local model evaluation. For raw FASTA input, use the online analysis guide below.

## EUP Environment Setup
You can follow the instructions provided at [ESM2-Ubiquitination-Prediction/Instruction/EUP Online Analysis Guide.md](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Instruction) to use the [web server](https://eup.aibtit.com) for online predictions.
## Models

In this project, we have constructed four deep learning models for the ubiquitination site prediction task, specifically including:

- **[DNNLinerModel](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Model/DNNLinerModel)**: A linear model based on fully connected layers (Dense Layer). This model directly learns the prediction rules of ubiquitination sites from raw input data.
- **[ResDNNModel](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Model/ResDNNModel)**: A deep neural network model that introduces residual blocks (Residual Block). Through the residual learning mechanism, this model can effectively alleviate the problem of vanishing gradients, enhancing the model's ability to learn from deep structures.
- **[cVAE_DNNLinerModel](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Model/cVAE_DNNLinearModel)**: A conditional VAE model combining a Residual Variational Autoencoder (ResVAE) with DNN_LinerModel as the classification head. This model framework is trained with both reconstruction and classification objectives. During prediction, the features are directly input into the model framework, and ubiquitination site prediction is performed in the classification head DNN_LinerModel.
- **[cVAE_ResDNNModel](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Model/cVAE_ResDNNModel)**: A conditional VAE model combining a Residual Variational Autoencoder (ResVAE) with ResDNNModel as the classification head. This model framework is trained with both reconstruction and classification objectives. During prediction, the features are directly input into the model framework, and ubiquitination site prediction is performed in the classification head ResDNNModel.
- **ESMc support**: Please see this [document](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/blob/main/update_new.md) for detail. 

## Reference

## Citation

If our work has contributed to your research, we would greatly appreciate it if you could cite our work as follows.

Liu J, Luo Z, Wang R, Li X, Sun Y, et al. (2025) EUP: Enhanced cross-species prediction of ubiquitination sites via a conditional variational autoencoder network based on ESM2. PLOS Computational Biology 21(7): e1013268. https://doi.org/10.1371/journal.pcbi.1013268

## Acknowledgments

We are acknowledge the contributions of the open-source community and the developers of the Python libraries used in this study.

## Related Works
If you are interested in feature extraction and model interpretation for large language models, you may find our previous work helpful:
- **Interpretable feature extraction and dimensionality reduction in ESM2 for protein localization prediction**: [GitHub Repository](https://github.com/yujuan-zhang/feature-representation-for-LLMs)



## Choose an entry point

For raw protein FASTA, follow [the online analysis guide](Instruction/EUP%20Online%20Analysis%20Guide.md).
It describes example downloads, upload, the extraction code, and Excel/image
results. Local scripts below instead evaluate **pre-extracted ESM2 features**.

## Local evaluation prerequisites

The bundled inputs are `Inference_test_data/ESM2_3B_2560/test_features.npy`
(2560 features per row) and `test_info.csv` (including `ID` and `Label`).
Rows must correspond between the two files. Local scripts require NumPy,
pandas, PyTorch and scikit-learn in a compatible environment. A complete tested
local environment specification is not supplied here.

Trained checkpoints are not bundled with these four scripts. Obtain a checkpoint
matching the chosen architecture from the project authors/upstream release;
do not substitute or rename another model's checkpoint. For the DNN example:

```bash
git clone https://github.com/yujuan-zhang/ESM2_AMP-Ubiquitination.git
cd ESM2_AMP-Ubiquitination
python Model/DNNLinerModel/DNNLiner_ptidiction.py --checkpoint /path/to/DNNLinermodel_checkpoint_epoch_34.pth
```

Replace `/path/to/...` with your downloaded checkpoint. Input paths are resolved
from the script location; the repository folder does not need to be renamed.
The other three model scripts also accept `--checkpoint`. If omitted, each
looks for its original named checkpoint beside the script.
Evaluation prints MCC, F1, Recall, Accuracy, AUC and PR metrics to the console;
it does not export per-site predictions. Runtime has not been measured here.
