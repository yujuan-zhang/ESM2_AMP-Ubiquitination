## Best Models in Species Data

This directory contains the weight files for the best-performing models on each species' test data. To use them, simply place the respective model weight file into the corresponding code as follows:

1. **[DNNLinerModel]** corresponds to the code script: [DNNLiner_ptidiction.py](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/blob/main/Model/DNNLinerModel/DNNLiner_ptidiction.py)
2. **[ResDNNModel]** corresponds to the code script: [ResDNN_ptidiction.py](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/blob/main/Model/ResDNNModel/ResDNN_ptidiction.py)
3. **[CVAE_DNNLinerModel]** corresponds to the code script: [CVAEDNNLiner_ptidiction.py](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/blob/main/Model/cVAE_DNNLinearModel/CVAEDNNLiner_ptidiction.py)
4. **[CVAE_ResDNNModel]** corresponds to the code script: [CVAEResDNN_ptidiction.py](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/blob/main/Model/cVAE_ResDNNModel/CVAEResDNN_ptidiction.py)

Based on the above model frameworks and their corresponding scripts, simply insert the path to the weight file into the respective weight loading section of the code to use the models.