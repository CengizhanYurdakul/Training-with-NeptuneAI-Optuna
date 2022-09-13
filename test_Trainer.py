from src.utils import *
from src.Trainer import Trainer

args = parseYaml("src/Mocking/trainConfig.yml")

def testTrainer():
    trainer = Trainer(trial=None, args=args)
    return trainer