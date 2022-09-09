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
```
## Docker
```
# Build and train manually
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