import torch

class OptimizerFactory:
    def __init__(self):
        self.optimizerList = [
            "Adam",
            "SGD"
        ]
    
    def getOptimizer(self, optimizerName:str, lr:float, momentum:float, modelParameters:torch.Generator)->torch.optim:
        """
        Prepares the desired optimizer to be used in training.

        Args:
            optimizerName (str): Optimizer algorithm name to use for training.
            modelParameters (torch.Generator): Parameters of network that use in training.

        Raises:
            Exception: The optimizer to be used has not been implemented yet.

        Returns:
            torch.optim: Optimizer to be used for training.
        """

        if optimizerName == "Adam":
            optimizer = torch.optim.Adam(modelParameters, lr=lr)
        elif optimizerName == "SGD":
            optimizer = torch.optim.SGD(modelParameters, lr=lr, momentum=momentum)
        elif optimizerName == "RMSprop":
            optimizer = torch.optim.RMSprop(modelParameters, lr=lr, momentum=momentum)
        else:
            raise Exception("Optimizer not in list!\nAvailable optimizers: %s" % self.optimizerList)
        
        print("%s optimizer initialized successfully!" % optimizerName)
        
        return optimizer