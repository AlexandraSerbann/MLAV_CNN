import numpy as np
# Nu trebuie tf pentru citirea datelor, dar trebuie pentru tot restul
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
class DatasetMNIST(Dataset):
    def __init__(self, cale_catre_date, cale_catre_etichete):
        # Completati aici
        # Hint: va ajuta functia folosita pana acum
        # pentru citirea MNIST
        f = open(cale_catre_date,'r',encoding = 'latin-1')
        g = open(cale_catre_etichete,'r',encoding = 'latin-1')
    
        byte = f.read(16) #4 bytes magic number, 4 bytes nr imag, 4 bytes nr linii, 4 bytes nr coloane
        byte_label = g.read(8) #4bytes magic number, 4 bytes nr labels
    
        mnist_data = np.fromfile(f,dtype=np.uint8).reshape(-1,1,28,28)
        #print(mnist_data.shape)
        mnist_labels = np.fromfile(g,dtype=np.uint8)
    
    
        mnist_data = mnist_data/255
    
        # Conversii pentru a se potrivi cu procesul de antrenare    
        self.mnist_data = mnist_data.astype(np.float32)
        self.mnist_labels = mnist_labels.astype(np.int64)
        
        
        
    def __len__(self):
        # Completati aici
        return len(self.mnist_data)
        
    def __getitem__(self, idx):
        # Ca ajutor, daca aceasta clasa va fi folosita
        # de alte unelte PyTorch, idx s-ar putea sa fie
        # un tensor, nu un intreg sau o lista. Trebuie 
        # facuta o conversie in acest caz
        
        #if torch.is_tensor(idx):
            #idx = idx.tolist()
            
        # Completati aici.
        data = self.mnist_data[idx] 
        labels = self.mnist_labels[idx]
        
        # Conventia este sa returnati un dictionar care sa
        # contina separat datele si etichetele.         
        # Ex: mnist_batch = {'date': <datele>, 'etichete': <etichetele>}
        mnist_batch = {'date': data, 'etichete': labels}
        
        return mnist_batch
  


class Retea_CNN(nn.Module):
    
    def __init__(self, nr_clase):
        
        # Pentru a putea folosi mai departe reteaua, este recomandata mostenirea
        # clasei de baza nn.Module
        super(Retea_CNN,self).__init__()
       
       # out = (in- kernel+ 2*padding)// stride +1  
       
        #1x28x28
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size =[3,3],stride=[1,1],padding=[1,1])
        self.relu1 = nn.ReLU()
        
        #3x28x28
        
        self.maxpool1 = nn.MaxPool2d(kernel_size = [2,2], stride = [2, 2])
        # 3x14x14
        
        self.conv2 = nn.Conv2d(in_channels = 3, out_channels = 9, kernel_size =[3,3],stride=[1,1],padding=[1,1])
        self.relu2 = nn.ReLU()
        
        self.drop1 = nn.Dropout(p=0.5)
        #9x14x14
        self.maxpool2 = nn.MaxPool2d(kernel_size = [2,2], stride = [2, 2])
    
        #9x7x7
        
        self.fc1 =nn.Linear(in_features = 9*7*7, out_features = 128)
        self.relu3 = nn.ReLU()
        
        self.out =nn.Linear(in_features =128, out_features = nr_clase)
   
    def forward(self,input_batch):
        
        x = self.conv1(input_batch)
        x = self.relu1(x)
        
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        x= self.drop1(x)
        
        x = self.maxpool2(x)
        #print(x.shape)
        x = torch.flatten(x, 1,3)
        x = self.fc1(x)
        x = self.relu3(x)
        out = self.out(x)
        
        return out  






mnistTrain = DatasetMNIST(r'train-images.idx3-ubyte', r'train-labels.idx1-ubyte')
mnistTest = DatasetMNIST(r't10k-images.idx3-ubyte', r't10k-labels.idx1-ubyte')


# select an image to display
#image_index = 0
#mnist_dict = mnistTrain.__getitem__(image_index)
#print(mnist_dict['date'])

trainLoader = DataLoader(mnistTrain, batch_size=128, shuffle=True, num_workers=0)
testLoader = DataLoader(mnistTest, batch_size=128, shuffle=False, num_workers=0)


cnn = Retea_CNN(10)

# Specificarea optimizatorului
optim = torch.optim.Adam(cnn.parameters(), lr=1e-3)

#Specificare functie loss
loss_function = nn.CrossEntropyLoss(reduction='sum')

#Specificare scheduler

scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optim, step_size = 4, gamma=0.1)

nr_epoci = 20

cnn.train()

for ep in range(nr_epoci):
    predictii = []
    etichete = []

    for batch in trainLoader:
        
        batch_data = batch['date']
        batch_labels = batch['etichete']
        # Se calculeaza predictia retelei pentru datele curente (forward pass/ propagare inainte)
        current_predict = cnn.forward(batch_data)

        # Se calculeaza valoarea momentana a functiei loss
        loss = loss_function(current_predict, batch_labels) 
        
        # Se memoreaza predictiile si etichetele aferente batch-ului actual (pentru calculul acuratetii)
        current_predict = np.argmax(current_predict.detach().numpy(), axis=1)
        predictii = np.concatenate((predictii,current_predict))
        etichete = np.concatenate((etichete,batch_labels))
        
        # Antrenarea propriu-zisa
        
            # 1. Se sterg toti gradientii calculati anteriori, pentru toate variabilele antrenabile
            # deoarece, metoda <backward> acumuleaza noile valori, in loc sa le inlocuiasca.
        optim.zero_grad()
            # 2. Calculul tuturor gradientilor. Backpropagation
        loss.backward()
            # 3. Actualizarea tuturor ponderilor, pe baza gradientilor.
        optim.step()
        
    scheduler.step()  

    # Calculam acuratetea
    acc = np.sum(predictii==etichete)/len(predictii)
    print( 'Acuratetea la epoca {} este {}%'.format(ep+1,acc*100) )

cnn.eval()

predictii = []
test_labels = []

for batch in testLoader:
    batch_data = batch['date']
    batch_labels = batch['etichete']

    current_predict = cnn.forward(batch_data)
    current_predict = np.argmax(current_predict.detach().numpy(),axis=1)
    predictii = np.concatenate((predictii,current_predict))
    test_labels = np.concatenate([test_labels, batch_labels])

acc = np.sum(predictii==test_labels)/len(predictii)
print( 'Acuratetea la test este {}%'.format(acc*100) )