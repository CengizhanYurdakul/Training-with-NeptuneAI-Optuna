import os
import torch
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

class ValidationDataset(data.Dataset):
    def __init__(self, seed:int, validationImagePath:str, validationInformationCSV:str, imageSize:int):
        
        np.random.seed(seed)
        
        self.imageSize = imageSize
        self.validationImagePath = validationImagePath
        self.validationInformationCSV = validationInformationCSV
        
        self.imageNames = None
        self.transform = None
        self.genderInformations = None

        self.initInformations()
        self.initTransform()
        self.initComplete()
            
    def __getitem__(self, index):        
        inputImage = None
        label = None
        
        while inputImage is None or label is None:
            # Read input image and label
            imageName = self.imageNames[index]
            inputImage = Image.open(os.path.join(self.validationImagePath, imageName))
            
            label = self.genderInformations.loc[imageName]["Genders"]
            
            if inputImage is None:
                raise Exception("Error reading image %s" % imageName)
            
            if label is None:
                raise Exception("Error reading label %s" % imageName)
            
        transformedImage = self.transform(inputImage)
        
        # Male - Female
        if label == -1:
            transformedLabel = torch.Tensor([1, 0])
        elif label == 1:
            transformedLabel = torch.Tensor([0, 1])
        else:
            raise Exception("Error unknown label %s!" % label)
        
        batch = {
            "inputImage": transformedImage,
            "label": transformedLabel
        }
        
        return batch
            
    def __len__(self):
        return len(self.imageNames)

    def initTransform(self):
        """
        Transfrom PIL.Image to torch.tensor for tranining/inference.
        """
        transformList = [
            transforms.Resize(self.imageSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        self.transform = transforms.Compose(transformList)
    
    def initInformations(self):
        """
        Get image names and gender labels from CSV that include dataset information.
        """
        self.imageNames = glob(os.path.join(self.validationImagePath, "**/*.jpg"))
        self.imageNames = [imageName.replace(self.validationImagePath + "/", "") for imageName in self.imageNames]
        self.genderInformations = pd.read_csv(self.validationInformationCSV, index_col=0)
        
        np.random.shuffle(self.imageNames)
    
    def initComplete(self):
        """
        Show if dataloader is initialized successfully.
        """
        print("Validation dataloader initialized with %s images!" % self.__len__())