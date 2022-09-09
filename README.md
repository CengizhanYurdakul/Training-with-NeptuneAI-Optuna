# Training with NeptuneAI and Optuna

<p align="center">
  <img src="assets/logo.png" width="350" title="logo">
</p>

**Abstract:** This project includes training pipeline, experiment tracking, model versioning and hyperparameter tuning based on gender classification problem. The focus is on practicing the tools used, not the problem that model solves. It is easy to implement different dataloaders and modify last layer of model etc.

# Installation
## Local
```
conda create --name genderTrain python==3.7.13
conda activate genderTrain
pip install -r requirements.txt

# For Cuda 11.1
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# For Cuda 10.2
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```
## Docker
```
docker build -f Docker/Cuda111.dockerfile -t train .
docker run --runtime=nvidia -it train train.py
```

#TODO Add installation
#TODO Define .yml file
#TODO Show outputs of NeptuneAI and Optuna
#TODO Define .ini file
#TODO Add download link for dataset
#TODO Add dockerfile

docker build -f Docker/Cuda111.dockerfile -t train .