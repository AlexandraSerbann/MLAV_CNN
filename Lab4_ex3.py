import struct
import numpy as np
# Nu trebuie torch pentru citirea datelor, dar trebuie pentru tot restul
import torch
from torchvision import models, transforms



from torch.utils.data import Dataset, DataLoader


class DatasetMNIST(Dataset):
    def __init__(self, cale_catre_date, cale_catre_etichete, nr_imag=-1):
        
        f = open(cale_catre_date,'r',encoding = 'latin-1')
        g = open(cale_catre_etichete,'r',encoding = 'latin-1')
        
        byte = f.read(16) #4 bytes magic number, 4 bytes nr imag, 4 bytes nr linii, 4 bytes nr coloane
        byte_label = g.read(8) #4 bytes magic number, 4 bytes nr labels
        
        
        mnist_data = np.fromfile(f,dtype=np.uint8).reshape(-1, 28, 28, 1) # nr_imag, linii, coloane, canale
        mnist_labels = np.fromfile(g,dtype=np.uint8)

        mnist_data = np.tile(mnist_data, [1, 1, 1, 3]) # Se repeta ultimul canal pentru a fi in format RGB

        if nr_imag != -1:
            mnist_data = mnist_data[:nr_imag]
            mnist_labels = mnist_labels[:nr_imag]
            
        # Conversii pentru a se potrivi cu procesul de antrenare    
        self.mnist_data = mnist_data
        self.mnist_labels = mnist_labels.astype(np.int64)

        self.transf = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize([224,224]),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    def __len__(self):
        return len(self.mnist_data)
        
    def __getitem__(self, idx):
        # Ca ajutor, daca aceasta clasa va fi folosita
        # de alte unelte PyTorch, idx s-ar putea sa fie
        # un tensor, nu un intreg sau o lista. Trebuie 
        # facuta o conversie in acest caz
        
        #if torch.is_tensor(idx):
            #idx = idx.tolist()

        data = self.mnist_data[idx]
        data = self.transf(data)
        label = self.mnist_labels[idx]

        mnist_batch = {'date': data, 'etichete': label}

        return mnist_batch



# Modulul nn contine o multitudine de elemente
# esentiale construirii unei retele neuronale
import torch.nn as nn

    
# Instantiem reteaua
cnn = models.alexnet(pretrained=True)
for param in cnn.parameters():
    param.requires_grad = False

cnn.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)
# Specificarea functiei loss
loss_function = nn.CrossEntropyLoss(reduction='sum')

# Specificarea optimizatorului
optim = torch.optim.Adam(cnn.parameters(), lr=1e-4)

mnistTrain = DatasetMNIST(r'Lab II\train-images.idx3-ubyte', r'Lab II\train-labels.idx1-ubyte', 200)
mnistTest = DatasetMNIST(r'Lab II\t10k-images.idx3-ubyte', r'Lab II\t10k-labels.idx1-ubyte')

trainLoader = DataLoader(mnistTrain, batch_size=128, shuffle=True, num_workers=0)
testLoader = DataLoader(mnistTest, batch_size=128, shuffle=False, num_workers=0)

nr_epoci = 10
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