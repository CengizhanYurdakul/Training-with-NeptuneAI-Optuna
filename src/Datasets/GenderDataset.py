import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
import torch.utils.data as data
import torchvision.transforms as transforms

class GenderDataset(data.Dataset):
    def __init__(self, isTrain:bool, seed:int, imagePath:str, informationCSV:str, imageSize:int, testSize:float):

        self.isTrain = isTrain
        self.testSize = testSize
        self.imageSize = imageSize
        self.imagePath = imagePath
        self.informationCSV = informationCSV
        
        np.random.seed(seed)
        
        self.imageNames = None
        self.transform = None

        self.initInformations()
        self.initTransform()
        self.initComplete()
            
    def __getitem__(self, index):        
        inputImage = None
        label = None
        
        while inputImage is None or label is None:
            # Read input image and label
            imageName = self.imageNames[index]
            inputImage = Image.open(os.path.join(self.imagePath, imageName))

            label = self.genderInformations.loc[imageName]["Genders"]
            
            if inputImage is None:
                raise Exception("Error reading image %s" % imageName)
            
            if label is None:
                raise Exception("Error reading label %s" % imageName)
            
        transformedImage = self.transform(inputImage)
        
        # Male - Female
        if label == -1:
            transformedLabel = torch.Tensor([0, 1])
        elif label == 1:
            transformedLabel = torch.Tensor([1, 0])
        else:
            raise Exception("Error unknown label %s!" % label)
        
        batch = {
            "inputImage": transformedImage,
            "label": transformedLabel
        }
        
        return batch
            
    def __len__(self):
        return len(self.imageNames)
    
    def splitTrain(self):
        """
        Split training images from overall dataset.
        """
        self.imageNames = self.imageNames[int(len(self.imageNames) * self.testSize):]
    
    def splitTest(self):
        """
        Split test images from overall dataset.
        """
        self.imageNames = self.imageNames[:int(len(self.imageNames) * self.testSize)]
    
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
        self.imageNames = glob(os.path.join(self.imagePath, "**/*.jpg"))
        self.imageNames = [imageName.replace(self.imagePath + "/", "") for imageName in self.imageNames]
        self.genderInformations = pd.read_csv(self.informationCSV, index_col=0)
        
        np.random.shuffle(self.imageNames)
        
        if self.isTrain:
            self.splitTrain()
        else:
            self.splitTest()
    
    def initComplete(self):
        """
        Show if dataloader is initialized successfully.
        """
        if self.isTrain:
            print("Train dataloader initialized with %s images!" % self.__len__())
        else:
            print("Test dataloader initialized with %s images!" % self.__len__())