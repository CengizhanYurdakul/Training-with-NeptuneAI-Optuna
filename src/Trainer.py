import os
import onnx
import torch
import optuna
import onnxsim
import configparser
import neptune.new as neptune
from neptune.new.types import File
from torch.utils.data import DataLoader

from src.Models.ModelFactory import ModelFactory
from src.Datasets.GenderDataset import GenderDataset
from src.Datasets.ValidationDataset import ValidationDataset
from src.Losses.LossFactory import LossFunctionFactory
from src.Optimizers.OptimizerFactory import OptimizerFactory

class Trainer:
    def __init__(self, trial=None, args=None):
        
        self.args = args
        self.trial = trial
        
        self.initTrials()
        
        if trial is not None:
            self.initConfigs()
            
        self.initGlobalVariables()
        self.initModel()
        self.initDataset()
        self.initOptimizer()
        self.initLossFunction()
        self.initDevice()
        
        if trial is not None:
            self.initNeptune()
            
    def initTrials(self):
        if self.trial is not None:
            self.args["lr"] = self.trial.suggest_float("lr", self.args["trialLearningRate"][0], self.args["trialLearningRate"][1])
            self.args["optimizer"] = self.trial.suggest_categorical("optimizer", self.args["trialOptimizer"])
            self.args["backbone"] = self.trial.suggest_categorical("backbone", self.args["trialBackbone"])
            self.args["pretrained"] = self.trial.suggest_categorical("pretrained", self.args["trialPretrained"])
            
            if self.args["optimizer"] == "SGD":
                self.args["momentum"] = self.trial.suggest_float("momentum", self.args["trialMomentum"][0], self.args["trialMomentum"][1])
            else:
                self.args["momentum"] = "-"
        
    def initConfigs(self):
        config = configparser.ConfigParser()
        config.read(self.args["configFile"])
        
        self.neptuneName = config["NeptuneAI"]["ProjectName"]
        self.neptuneToken = config["NeptuneAI"]["APIToken"]
        self.neptuneModelKey = config["NeptuneAI"]["ModelKey"]
        
    def initGlobalVariables(self):
        self.trainIter = 0
        self.testIter = 0
        self.validationIter = 0
        
        self.testAccuracy = 0
        self.trainAccuracy = 0
        self.validationAccuracy = 0
        
        self.truePredicted = 0
        
        self.bestValidationAccuracy = 0
        
    def initNeptune(self):
        self.runNeptune = neptune.init(project=self.neptuneName,
                                       api_token=self.neptuneToken
                                       )
        
        try:
            neptune.init_model(name="Gender classifier",
                               key=self.neptuneModelKey,
                               project=self.neptuneName,
                               api_token=self.neptuneToken)
        except:
            print("Model already initialized before to NeptuneAI!")
                
        self.runNeptune["parameters"] = self.args
        
    def initModel(self):
        classifierFactory = ModelFactory()
        self.classifier = classifierFactory.getModel(self.args["backbone"], self.args["pretrained"], self.args["numberClass"])
    
    def initDataset(self):
        datasetTrain = GenderDataset(isTrain=True, seed=self.args["seed"], imagePath=self.args["imagePath"], informationCSV=self.args["informationCSV"], imageSize=self.args["imageSize"], testSize=self.args["testSize"])
        datasetTest = GenderDataset(isTrain=False, seed=self.args["seed"], imagePath=self.args["imagePath"], informationCSV=self.args["informationCSV"], imageSize=self.args["imageSize"], testSize=self.args["testSize"])
        datasetValidation = ValidationDataset(seed=self.args["seed"], validationImagePath=self.args["validationImagePath"], validationInformationCSV=self.args["validationInformationCSV"], imageSize=self.args["imageSize"])
        
        self.dataLoaderTrain = DataLoader(
            datasetTrain,
            batch_size=self.args["batchSize"],
            num_workers=self.args["numWorkers"]
        )
        
        self.dataLoaderTest = DataLoader(
            datasetTest,
            batch_size=self.args["batchSize"],
            num_workers=self.args["numWorkers"]
        )
        
        self.dataLoaderValidation = DataLoader(
            datasetValidation,
            batch_size=self.args["batchSize"],
            num_workers=self.args["numWorkers"]
        )
        
    def initOptimizer(self):
        optimizerFactory = OptimizerFactory()
        self.optimizer = optimizerFactory.getOptimizer(optimizerName=self.args["optimizer"], lr=self.args["lr"], momentum=self.args["momentum"], modelParameters=self.classifier.parameters())
        
    def initLossFunction(self):
        lossFunctionFactory = LossFunctionFactory()
        self.lossFunction = lossFunctionFactory.getLossFunction(self.args["lossFunction"])
    
    def initDevice(self):
        if (torch.cuda.is_available() is False) and ("cuda" in self.args["device"]):
            raise Exception("CUDA is not available for PyTorch!")
        else:
            self.classifier.to(self.args["device"])
            self.lossFunction.to(self.args["device"])

    def logModelNeptune(self):
        self.neptuneModelVersion = neptune.init_model_version(model=self.runNeptune._short_id.split("-")[0] + "-" + self.neptuneModelKey,
                                                            project=self.neptuneName,
                                                            api_token=self.neptuneToken
                                                            )
        
        self.neptuneModelVersion["model"].upload(os.path.join(self.args["savePath"], "PyTorchModel_%s_%s.pth" % (self.args["backbone"], self.epoch)))
        
        self.neptuneModelVersion["testAccuracy"] = self.testAccuracy
        self.neptuneModelVersion["validationAccuracy"] = self.validationAccuracy
        self.neptuneModelVersion["trainAccuracy"] = self.trainAccuracy
        
        self.neptuneModelVersion["lr"] = self.args["lr"]
        self.neptuneModelVersion["backbone"] = self.args["backbone"]
        self.neptuneModelVersion["momentum"] = self.args["momentum"]
        self.neptuneModelVersion["optimizer"] = self.args["optimizer"]
        self.neptuneModelVersion["pretrained"] = self.args["pretrained"]
        
        self.neptuneModelVersion["epoch"] = self.epoch
        
    def logImageNeptune(self, tag, tensor):
        self.runNeptune[tag].log(File.as_image(tensor))
    
    def logMetricNeptune(self, tag, value):
        self.runNeptune[tag].log(value)
    
    def compareLabels(self, predictedLabels, gtLabels):
        predictedClass = torch.argmax(predictedLabels, 1)
        gtClass = torch.argmax(gtLabels, 1)
        
        self.truePredicted += (gtClass == predictedClass).sum()
                        
    def calculateAccuracy(self, type):
        if type == "Train":
            accuracy = (self.truePredicted/(self.dataLoaderTrain.__len__() * self.args["batchSize"])).item()
        elif type == "Test":
            accuracy = (self.truePredicted/(self.dataLoaderTest.__len__() * self.args["batchSize"])).item()
        elif type == "Validation":
            accuracy = (self.truePredicted/(self.dataLoaderValidation.__len__() * self.args["batchSize"])).item()
        return accuracy
    
    def saveModel(self):
        if not os.path.exists(self.args["savePath"]):
            os.makedirs(self.args["savePath"])
            
        torch.save(self.classifier, os.path.join(self.args["savePath"], "PyTorchModel_%s_%s.pth" % (self.args["backbone"], self.epoch)))
        
    def checkSave(self):
        if (self.validationAccuracy > self.bestValidationAccuracy) and (self.validationAccuracy >= self.args["accuracyThreshold"]):
            self.saveModel()
            self.logModelNeptune()
            self.torch2onnx()
            print("[Current Validation Accuracy: %s > Best Validation Accuracy: %s] Model saved!" % (round(self.validationAccuracy, 3), round(self.bestValidationAccuracy, 3)))
            self.bestValidationAccuracy = self.validationAccuracy
            
    def torch2onnx(self):
        self.classifier.cpu()
        self.classifier.eval()
        
        dummyInput = torch.autograd.Variable(torch.randn((1, 3, self.args["imageSize"], self.args["imageSize"])))
        
        torch.onnx.export(self.classifier,
                          dummyInput,
                          os.path.join(self.args["savePath"], "PyTorchModel_%s_%s.onnx" % (self.args["backbone"], self.epoch)),
                          opset_version=11,
                          input_names=["input"],
                          output_names=["output"],
                          dynamic_axes={"input": {0 : "batch_size"},
                                        "output": {0 : "batch_size"}})
        
        model = onnx.load(os.path.join(self.args["savePath"], "PyTorchModel_%s_%s.onnx" % (self.args["backbone"], self.epoch)))
        
        inputShape = {model.graph.input[0].name : list((1, 3, self.args["imageSize"], self.args["imageSize"]))}
        
        model, _ = onnxsim.simplify(model, overwrite_input_shapes=inputShape)
        
        onnx.save(model, os.path.join(self.args["savePath"], "PyTorchModel_%s_%s_simp.onnx" % (self.args["backbone"], self.epoch)))
        
        os.remove(os.path.join(self.args["savePath"], "PyTorchModel_%s_%s.onnx" % (self.args["backbone"], self.epoch)))
    
    def validate(self):
        self.classifier.eval()
        self.truePredicted = 0
        for c, batch in enumerate(self.dataLoaderValidation):
            inputImage, gtLabel = batch["inputImage"].to(self.args["device"]), batch["label"].to(self.args["device"])
            
            predictedLabel = self.classifier(inputImage)
            
            loss = self.lossFunction(predictedLabel, gtLabel)
            
            self.compareLabels(predictedLabel, gtLabel)
            
            if (c % (int(self.dataLoaderValidation.__len__() * self.args["logInterval"])) == 0) and (c != 0):
                self.logMetricNeptune("Validation/Loss", loss.item())
                
                print(
                    "[Validation] - [Epoch: %s / %s] - [Iter: %s / %s] - [Loss: %s]"
                    % (self.epoch, self.args["numEpochs"], c, self.dataLoaderValidation.__len__(), loss.item())
                )
            
            self.validationIter += 1
        
        self.validationAccuracy = self.calculateAccuracy("Validation")
        
        self.logMetricNeptune("Validation/Accuracy", self.validationAccuracy)
        
    def test(self):
        self.classifier.eval()
        self.truePredicted = 0
        for c, batch in enumerate(self.dataLoaderTest):
            inputImage, gtLabel = batch["inputImage"].to(self.args["device"]), batch["label"].to(self.args["device"])
            
            predictedLabel = self.classifier(inputImage)
            
            loss = self.lossFunction(predictedLabel, gtLabel)
            
            self.compareLabels(predictedLabel, gtLabel)
            
            if (c % (int(self.dataLoaderTest.__len__() * self.args["logInterval"])) == 0) and (c != 0):
                self.logMetricNeptune("Test/Loss", loss.item())
                
                print(
                    "[Test] - [Epoch: %s / %s] - [Iter: %s / %s] - [Loss: %s]"
                    % (self.epoch, self.args["numEpochs"], c, self.dataLoaderTest.__len__(), loss.item())
                )
            
            self.testIter += 1
        
        self.testAccuracy = self.calculateAccuracy("Test")
        
        self.logMetricNeptune("Test/Accuracy", self.testAccuracy)
    
    def train(self):
        self.classifier.train()
        self.truePredicted = 0
        for c, batch in enumerate(self.dataLoaderTrain):
            inputImage, gtLabel = batch["inputImage"].to(self.args["device"]), batch["label"].to(self.args["device"])
            
            self.optimizer.zero_grad()
            
            predictedLabel = self.classifier(inputImage)
            
            loss = self.lossFunction(predictedLabel, gtLabel)
            
            self.compareLabels(predictedLabel, gtLabel)
            
            loss.backward()
            self.optimizer.step()
            
            if (c % (int(self.dataLoaderTrain.__len__() * self.args["logInterval"])) == 0) and (c != 0):
                self.logMetricNeptune("Train/Loss", loss.item())         

                print(
                    "[Train] - [Epoch: %s / %s] - [Iter: %s / %s] - [Loss: %s]"
                    % (self.epoch, self.args["numEpochs"], c, self.dataLoaderTrain.__len__(), loss.item())
                )

            self.trainIter += 1
        
        self.trainAccuracy = self.calculateAccuracy("Train")
        
        self.logMetricNeptune("Train/Accuracy", self.trainAccuracy)
        
    def main(self):
        for self.epoch in range(self.args["numEpochs"]):
            self.classifier.to(self.args["device"])
            
            self.train()
            self.test()
            self.validate()
            self.checkSave()
            
            self.trial.report(self.validationAccuracy, self.epoch)
                
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                    
        self.runNeptune.stop()
        return self.validationAccuracy