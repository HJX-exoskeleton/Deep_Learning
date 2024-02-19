from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms

class ReadData():
    def __init__(self,data_path,image_size=64):
        self.root=data_path
        self.image_size=image_size
        self.dataset=self.getdataset()
    def getdataset(self):
        #3.dataset
        dataset = datasets.ImageFolder(root=self.root,
                                   transform=transforms.Compose([
                                       transforms.Resize(self.image_size),
                                       transforms.CenterCrop(self.image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        # Create the dataloader

        print(f'Total Size of Dataset: {len(dataset)}')
        return dataset

    def getdataloader(self,batch_size=128):
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0)

        return dataloader

if __name__ =='__main__':
    dset=ReadData('D:/Python_file/HJX_file/DCGAN_hjx/data/faces')
    print('-------------------------------')
    dloader=dset.getdataloader()
