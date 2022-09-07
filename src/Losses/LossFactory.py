import torch.nn as nn

class LossFunctionFactory:
    def __init__(self):        
        self.lossFunctionList = [
            "BCEWithLogitsLoss",
            "L1Loss",
            "CrossEntropyLoss"
        ]
    
    def getLossFunction(self, lossFunctionName:str):
        """
        Prepares the desired loss function to be used in training.

        Args:
            lossFunctionName (str): Loss function name to use for training.

        Raises:
            Exception: The loss function to be used has not been implemented yet.

        Returns:
            torch.nn.Loss: Loss function to be used for training.
        """
        
        if lossFunctionName == "BCEWithLogitsLoss":
            lossFunction = nn.BCEWithLogitsLoss()
        elif lossFunctionName == "L1Loss":
            lossFunction = nn.L1Loss()
        elif lossFunctionName == "CrossEntropyLoss":
            lossFunction = nn.CrossEntropyLoss()
        else:
            raise Exception("Loss function not in list!\nAvailable loss functions: %s" % self.lossFunctionList)
        
        print("%s function initialized successfully!" % lossFunctionName)
        
        return lossFunction