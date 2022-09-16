# v1
- [x] Add [Neptune.ai](https://neptune.ai/) for experiment tracking and model versioning
    - [x] Add instruction for creating account and project on Neptune.ai
- [x] Add [Optuna](https://optuna.org/) for hyperparameter tuning while training
- [x] Add `resnet18`, `resnet34`, `resnet50`, `resnet101` and `resnet152` backbones to [model factory](src/Models/ModelFactory.py)
    - [x] Add last layer modifier with respect to number of class to [model factory](src/Models/ModelFactory.py#L62)
- [x] Add `BCEWithLogitsLoss`, `L1Loss` and `CrossEntropyLoss` to [loss factory](src/Losses/LossFactory.py)
- [x] Add `Adam` and `SGD` to [optimizer factory](src/Optimizers/OptimizerFactory.py)
- [x] Add description to [config file](src/Options/trainConfig.yml) for easy use
- [x] Describe [secrets.ini](secretsExample.ini) for use keys easily
- [x] Add docker and local installation guide
- [x] Add dataset link as public
- [x] Add [torch2onnx converter and onnx simplifier](src/Trainer.py#L175)
- [x] Add tests to Github actions

# v2
- [ ] Add false predictions to Neptune.ai for inspect model
- [x] Add `mobilenet_v3_small`, `mobilenet_v3_large` and `mobilenet_v2` backbones to [model factory](src/Models/ModelFactory.py)
    - [x] Add last layer modifier with respect to number of class to [model factory](src/Models/ModelFactory.py#L62)
- [ ] Add dataset versioning to Neptune.ai
- [ ] Add `processor` to initialize best model that trained before to any platform like [Streamlit](https://streamlit.io/) or [Gradio](https://gradio.app/)