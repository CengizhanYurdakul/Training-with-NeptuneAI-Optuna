import torch.nn as nn
import torchvision.models as models

class ModelFactory:
    def __init__(self):        
        self.modelList = [
            "resnet152",
            "resnet101",
            "resnet50",
            "resnet34",
            "resnet18",
        ]
    
    def getModel(self, modelName:str, pretrained:bool, numberClass:int)->models:
        """
        Prepares the desired network to be used in training.

        Args:
            modelName (str): Model architecture name to use for training.

        Raises:
            Exception: The model to be used has not been implemented yet. 

        Returns:
            models: Network to be used for training.
        """
        
        if modelName == "resnet152":
            self.model = models.resnet152(pretrained=pretrained)
        elif modelName == "resnet101":
            self.model = models.resnet101(pretrained=pretrained)
        elif modelName == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
        elif modelName == "resnet34":
            self.model = models.resnet34(pretrained=pretrained)
        elif modelName == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
        elif modelName == "squeezenet1_0":
            self.model = models.squeezenet1_0(pretrained=pretrained)
        elif modelName == "squeezenet1_1":
            self.model = models.squeezenet1_1(pretrained=pretrained)
        elif modelName == "densenet201":
            self.model = models.densenet201(pretrained=pretrained)
        elif modelName == "densenet169":
            self.model = models.densenet169(pretrained=pretrained)
        elif modelName == "densenet161":
            self.model = models.densenet161(pretrained=pretrained)
        elif modelName == "densenet121":
            self.model = models.densenet121(pretrained=pretrained)
        elif modelName == "inception_v3":
            self.model = models.inception_v3(pretrained=pretrained)
        else:
            raise Exception("Model %s not in list!\nAvailable models: %s" % (modelName, self.modelList))
        
        if numberClass != 1000:
            self.editFinalLayer(modelName, numberClass)
        
        print("%s initialized successfully!" % modelName)
        
        return self.model
    
    def editFinalLayer(self, modelName, numberClass)->models:
        if (modelName == "resnet152") or (modelName == "resnet101") or (modelName == "resnet50") or (modelName == "resnet34") or (modelName == "resnet18"):
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, numberClass)
            )
        else:
            raise Exception("%s editing is not available!" % modelName)
