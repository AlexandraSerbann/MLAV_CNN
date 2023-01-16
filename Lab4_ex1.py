import struct
import numpy as np
# Nu trebuie torch pentru citirea datelor, dar trebuie pentru tot restul
import torch

from torch.utils.data import Dataset, DataLoader


class DatasetMNIST(Dataset):
    def __init__(self, cale_catre_date, cale_catre_etichete, nr_imag=-1):
        
        f = open(cale_catre_date,'r',encoding = 'latin-1')
        g = open(cale_catre_etichete,'r',encoding = 'latin-1')
        
        byte = f.read(16) #4 bytes magic number, 4 bytes nr imag, 4 bytes nr linii, 4 bytes nr coloane
        byte_label = g.read(8) #4 bytes magic number, 4 bytes nr labels
        
        
        mnist_data = np.fromfile(f,dtype=np.uint8).reshape(-1, 28 *28)
        mnist_labels = np.fromfile(g,dtype=np.uint8)

        if nr_imag != -1:
            mnist_data = mnist_data[:nr_imag]
            mnist_labels = mnist_labels[:nr_imag]
            
        # Conversii pentru a se potrivi cu procesul de antrenare    
        self.mnist_data = mnist_data.astype(np.float32)
        self.mnist_labels = mnist_labels.astype(np.int64)
    
    def __len__(self):
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
        label = self.mnist_labels[idx]

        mnist_batch = {'date': data, 'etichete': label}

        return mnist_batch



# Modulul nn contine o multitudine de elemente
# esentiale construirii unei retele neuronale
import torch.nn as nn

class Retea_MLP(nn.Module):
    
    def __init__(self, nr_clase):
        
        # Pentru a putea folosi mai departe reteaua, este recomandata mostenirea
        # clasei de baza nn.Module
        super(Retea_MLP,self).__init__()
        
        self.fc1 = nn.Linear(in_features=28*28, out_features=512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(in_features=128, out_features=nr_clase)
    
    def forward(self,input_batch):

        x = self.fc1(input_batch)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        out = self.out(x)
        
        return out
    
# Instantiem reteaua
mlp = Retea_MLP(10)

# Specificarea functiei loss
loss_function = nn.CrossEntropyLoss(reduction='sum')

# Specificarea optimizatorului
optim = torch.optim.Adam(mlp.parameters(), lr=1e-4)

mnistTrain = DatasetMNIST(r'Lab II\train-images.idx3-ubyte', r'Lab II\train-labels.idx1-ubyte', 200)
mnistTest = DatasetMNIST(r'Lab II\t10k-images.idx3-ubyte', r'Lab II\t10k-labels.idx1-ubyte')

trainLoader = DataLoader(mnistTrain, batch_size=128, shuffle=True, num_workers=0)
testLoader = DataLoader(mnistTest, batch_size=128, shuffle=False, num_workers=0)

nr_epoci = 40

for ep in range(nr_epoci):
    predictii = []
    etichete = []

    for batch in trainLoader:
        batch_data = batch['date']
        batch_labels = batch['etichete']
        # Se calculeaza predictia retelei pentru datele curente (forward pass/ propagare inainte)
        current_predict = mlp.forward(batch_data)

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
        
        

    # Calculam acuratetea
    acc = np.sum(predictii==etichete)/len(predictii)
    print( 'Acuratetea la epoca {} este {}%'.format(ep+1,acc*100) )

predictii = []
test_labels = []
for batch in testLoader:
    batch_data = batch['date']
    batch_labels = batch['etichete']

    current_predict = mlp.forward(batch_data)
    current_predict = np.argmax(current_predict.detach().numpy(),axis=1)
    predictii = np.concatenate((predictii,current_predict))
    test_labels = np.concatenate([test_labels, batch_labels])

acc = np.sum(predictii==test_labels)/len(predictii)
print( 'Acuratetea la test este {}%'.format(acc*100) )