import optuna

from src.utils import *
from src.Trainer import Trainer

def train(trial):
    trainer = Trainer(trial, args)
    validationAccuracy = trainer.main()
    return validationAccuracy

args = parseYaml("src/Options/trainConfig.yml")
    
study = optuna.create_study(direction="maximize")
study.optimize(train, n_trials=args["trialNumber"])