# The Framework of Our Model
![image](https://github.com/user-attachments/assets/8c9dd355-300c-4464-bbb5-e7ed09880dd7)
The SpectrumFM framework comprises three key stages. First, in the data collection and processing stage, diverse
spectrum data from multiple sources are gathered and 
preprocessed to ensure consistency and compatibility across datasets.
Second, during the pre-training stage, the model learns fundamental spectrum representations through self-supervised learning
tasks, namely, masked reconstruction and next-slot signal prediction. Finally, in the fine-tuning stage, the pre-trained model is
adapted to specific downstream tasks, including AMC, WTC, SS, and AD.
# Experimental Settings
The hyperparameters for SpectrumFM
are as follows. The mask ratio r is set to 15%, the number of
signal symbols is set to 128, the number of attention heads H
is set to 4, the latent dimension d is set to 256, the feedforward
dimension dfeed is set to 512, and the number of SpectrumFM
encoder layers L is set to 16. The pre-training phase consists
of 10 epochs with a batch size of 256 and a learning rate of
0.001. The AdamW optimizer is employed for optimization,
and early stopping is utilized to prevent overfitting. During
the fine-tuning stage, the same learning rate of 0.001 and
the AdamW optimizer are used to further adapt the model
to specific downstream tasks.
# Results
## Automatic Modulation Classification Task
![image](https://github.com/user-attachments/assets/7fcbe167-b320-4215-82b6-426d28aea512)
## Wireless Technology Classification Task
![image](https://github.com/user-attachments/assets/6d1fc9ed-2d65-4a2a-bda3-5d5685d080fd)
## Spectrum Sensing Task
![image](https://github.com/user-attachments/assets/6e2a5e35-884d-440a-b6e9-5d6db6be00be)
## Anomaly Detection Task
![image](https://github.com/user-attachments/assets/571d68c9-db05-4d1e-8cf1-fa1d5ba3040f)
# The Code
The pretraining code is located in the `pretrain.py` file, while the fine-tuning code can be found in the `amc.py` file.
# Cite Our Paper
```
@misc{zhou2025spectrumfmfoundationmodelintelligent,
      title={SpectrumFM: A Foundation Model for Intelligent Spectrum Management}, 
      author={Fuhui Zhou and Chunyu Liu and Hao Zhang and Wei Wu and Qihui Wu and Derrick Wing Kwan Ng and Tony Q. S. Quek and Chan-Byoung Chae},
      year={2025},
      eprint={2505.06256},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2505.06256}, 
}
