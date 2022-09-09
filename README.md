# Training with NeptuneAI and Optuna

<p align="center">
  <img src="assets/logo.png" width="350" title="logo">
</p>

**Abstract:** This project includes training pipeline, experiment tracking, model versioning and hyperparameter tuning based on gender classification problem. The focus is on practicing the tools used, not the problem that model solves. It is easy to implement different dataloaders and modify last layer of model etc.

# Installation
## Environment
### Local
```
conda create --name genderTrain python==3.7.13
conda activate genderTrain
pip install -r requirements.txt

# For Cuda 11.1
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# For Cuda 10.2
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```
### Docker
```
docker build -t train .
docker run --runtime=nvidia -it train train.py
```
## Dataset
CelebAHQ dataset was used for training and WIKI dataset was used for validation. Processed images can be downloaded from [here](https://drive.google.com/file/d/1karOtT_tB34SW8I-dcQZ-vGR5NKMJqxY/view?usp=sharing) then unzip file to [src/Data](src/Data)
```
CelebAHQ: 30000 images
WIKI: 38455 images
```
## Configs
All training and hyperparameter tuning parameters can be changed from the [trainConfig.yml](src/Options/trainConfig.yml) file. Descriptions of all parameters are available in the file.

### Secret Variables for NeptuneAI
After creating account on Neptune.ai, API token is required to connect to the project. There are several ways to find your keys. Here is the [link](https://docs.neptune.ai/getting-started/installation#authentication-neptune-api-token). You need to add your informations to `secrets.ini` like [example](secretsExample.ini).

#TODO Show outputs of NeptuneAI and Optuna