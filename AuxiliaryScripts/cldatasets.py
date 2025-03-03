import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms
from sklearn.utils import shuffle
import copy

def get_splitCIFAR(seed=0, pc_valid=0.10, task_num = 0, split=""):

    if os.path.isfile(("../data/split_cifar/" + str(task_num) + "/x_train.bin")) == False:
        print("No dataset detected for similarity subsets. Creating new set prior to loading task.")
        make_splitcifar(seed=seed, pc_valid=pc_valid)


    data={}
    data['x']=torch.load(os.path.join(os.path.expanduser(('../data/split_cifar/' + str(task_num))), ('x_'+split+'.bin')))
    data['y']=torch.load(os.path.join(os.path.expanduser(('../data/split_cifar/' + str(task_num))), ('y_'+split+'.bin')))

    return data
    


def get_TinyImagenetCIFAR(seed=0, pc_valid=0.10, task_num = 0, split=""):

    """
    Sequence: 
        0:Tiny Imagenet, 
        1:CIFAR10, 
        2:CIFAR100 split, 
        3:CIFAR100 split, 
        4:CIFAR100 split, 
        5:CIFAR100 split, 
    """


    data={}
    if task_num == 0:
        data['x']=torch.load('../data/Tiny Imagenet/' + split + '/X.pt')
        data['y']=torch.load('../data/Tiny Imagenet/' + split + '/y.pt')
    else:
        data['x']=torch.load(os.path.join(os.path.expanduser(('../data/split_cifar/' + str(task_num-1))), ('x_'+split+'.bin')))
        data['y']=torch.load(os.path.join(os.path.expanduser(('../data/split_cifar/' + str(task_num-1))), ('y_'+split+'.bin')))

    return data
    


def get_mixedCIFAR_PMNIST(seed = 0, pc_valid=0.1, task_num=0, split=""):
    if os.path.isfile(("../data/split_cifar/" + str(task_num) + "/x_train.bin")) == False:
        print("No dataset detected for cifar subsets. Creating new set prior to loading task.")
        make_splitcifar(seed=seed, pc_valid=pc_valid)
    if os.path.isfile(("../data/PMNIST/" + str(task_num) + "/x_train.bin")) == False:
        print("No dataset detected for PMNIST subsets. Creating new set prior to loading task.")
        make_PMNIST(seed=seed, pc_valid=pc_valid)


    data={}

    mnisttasks = [0,2,4]
    if task_num in mnisttasks:
        data['x']=torch.load(os.path.join(os.path.expanduser(('../data/PMNIST/' + str(task_num))), ('x_'+split+'.bin')))
        data['y']=torch.load(os.path.join(os.path.expanduser(('../data/PMNIST/' + str(task_num))), ('y_'+split+'.bin')))
    else:
        data['x']=torch.load(os.path.join(os.path.expanduser(('../data/split_cifar/' + str(task_num))), ('x_'+split+'.bin')))
        data['y']=torch.load(os.path.join(os.path.expanduser(('../data/split_cifar/' + str(task_num))), ('y_'+split+'.bin')))

    return data
    
    

def get_mixedCIFAR_KEFMNIST(seed = 0, pc_valid=0.1, task_num=0, split = ""):


    """
    Sequence: 
        0:KMNIST-49, 
        1:CIFAR100 split, 
        1:EMNIST-Balanced, 
        3:CIFAR100 split, 
        4:Fashion MNIST, 
        5:CIFAR100 split, 
    """

    data={}
    if task_num in [1,3,5]:
        data['x']=torch.load(os.path.join(os.path.expanduser(('../data/split_cifar/' + str(task_num))), ('x_'+split+'.bin')))
        data['y']=torch.load(os.path.join(os.path.expanduser(('../data/split_cifar/' + str(task_num))), ('y_'+split+'.bin')))
        
    elif task_num == 0:
        data['x'] = torch.load('../data/K49/' + split + "/X.pt")
        data['y'] = torch.load('../data/K49/' + split + "/y.pt")
    elif task_num == 2:
        data['x'] = torch.load('../data/EMNIST/' + split + "/X.pt")
        data['y'] = torch.load('../data/EMNIST/' + split + "/y.pt")
    elif task_num == 4:
        data['x'] = torch.load('../data/FashionMNIST/' + split + "/X.pt")
        data['y'] = torch.load('../data/FashionMNIST/' + split + "/y.pt")

    return data
    
 







def make_splitcifar(seed=0, pc_valid=0.2):
    data={}
    taskcla=[]
    size=[3,32,32]
    
    
    # CIFAR10
    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]
    
    # CIFAR10
    dat={}
    dat['train']=datasets.CIFAR10('../data/',train=True,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.CIFAR10('../data/',train=False,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    data[0]={}
    data[0]['name']='cifar10'
    data[0]['ncla']=10
    data[0]['train']={'x': [],'y': []}
    data[0]['test']={'x': [],'y': []}
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        for image,target in loader:
            data[0][s]['x'].append(image)
            data[0][s]['y'].append(target.numpy()[0])
    
    # "Unify" and save
    for s in ['train','test']:
        data[0][s]['x']=torch.stack(data[0][s]['x']).view(-1,size[0],size[1],size[2])
        data[0][s]['y']=torch.LongTensor(np.array(data[0][s]['y'],dtype=int)).view(-1)
    
    # CIFAR100
    dat={}
    
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    
    dat['train']=datasets.CIFAR100('../data/',train=True,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.CIFAR100('../data/',train=False,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    
    for n in range(1,11):
        data[n]={}
        data[n]['name']='cifar100'
        data[n]['ncla']=10
        data[n]['train']={'x': [],'y': []}
        data[n]['test']={'x': [],'y': []}
    
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        for image,target in loader:
            task_idx = target.numpy()[0] // 10 + 1
            data[task_idx][s]['x'].append(image)
            data[task_idx][s]['y'].append(target.numpy()[0]%10)
    
    
    
    for t in range(1,11):
        for s in ['train','test']:
            data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
            data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
            
    os.makedirs('../data/split_cifar/' ,exist_ok=True)
    
    for t in range(0,11):
      # Validation
      r=np.arange(data[t]['train']['x'].size(0))
      r=np.array(shuffle(r,random_state=seed),dtype=int)
      nvalid=int(pc_valid*len(r))
      ivalid=torch.LongTensor(r[:nvalid])
      itrain=torch.LongTensor(r[nvalid:])
      data[t]['valid']={}
      data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
      data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
      data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
      data[t]['train']['y']=data[t]['train']['y'][itrain].clone()
    
    
      for s in ['train','valid','test']:
        os.makedirs(('../data/split_cifar/' + str(t)) ,exist_ok=True)
        torch.save(data[t][s]['x'], ('../data/split_cifar/'+ str(t) + '/x_' + s + '.bin'))
        torch.save(data[t][s]['y'], ('../data/split_cifar/'+ str(t) + '/y_' + s + '.bin'))
    
    
def make_PMNIST(seed=0, pc_valid=0.1):
    
    mnist_train = datasets.MNIST('../data/', train = True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),transforms.Resize((32,32))]), download = True)        
    mnist_test = datasets.MNIST('../data/', train = False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),transforms.Resize((32,32))]), download = True)        


    dat={}
    data={}
    taskcla=[]
    size=[1,32,32]    
    os.makedirs('../data/PMNIST', exist_ok =True)
    
    dat['train']=mnist_train
    dat['test']=mnist_test
    
    ### Prepare the data variable and lists of label indices for further processing
    for t in range(0,6):
      data[t]={}
      data[t]['name']='PMNIST'
      data[t]['ncla']=10
      data[t]['train']={'x': [],'y': []}
      data[t]['test']={'x': [],'y': []}



    for t in range(0,6):
        torch.manual_seed(t)
        taskperm = torch.randperm((32*32))
        # ### Extract only the appropriately labeled samples for each of the subsets
        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            ### For each image we will flatten it, permute it according to taskperm, and then reshape it and convert it to produce (3,32,32) image shape
            for image,target in loader:   
                ### Flatten the (1,32,32) image into (1,1024)
                image = torch.flatten(image)
                image = image[taskperm]
                image = image.view(1,32,32)
                ### Gives shape (3,32,32)
                image = torch.cat((image,image,image), dim=0)

                data[t][s]['x'].append(image)
                data[t][s]['y'].append(target.numpy()[0])
      
            data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,3,32,32)
            data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)

    ### Splitting validation off from training rather than test is fine here, so long as both sets are preprocessed identically
    for t in range(0,6):
      # Validation
      torch.manual_seed(t)
      taskperm = torch.randperm(data[t]['train']['x'].size(0))

      nvalid=int(pc_valid*len(taskperm))
      ivalid=torch.LongTensor(taskperm[:nvalid])
      itrain=torch.LongTensor(taskperm[nvalid:])
      data[t]['valid']={}
      data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
      data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
      ### Only overwrites the train key after its been used to create the valid subset
      data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
      data[t]['train']['y']=data[t]['train']['y'][itrain].clone()
    
      for s in ['train','valid','test']:
        os.makedirs(('../data/PMNIST/' + str(t)) ,exist_ok=True)
        torch.save(data[t][s]['x'], ('../data/PMNIST/'+ str(t) + '/x_' + s + '.bin'))
        torch.save(data[t][s]['y'], ('../data/PMNIST/'+ str(t) + '/y_' + s + '.bin'))
    
    
    
#!# Add a "make Tiny Imagenet"
#!# Add a "make KMNIST, EMNIST, FashionMNIST"











































def get_joint_TinyImagenetCIFAR(seed=0, pc_valid=0.10, task_num = 0, split=""):

    if os.path.isfile(("../data/split_cifar_64/" + str(task_num) + "/x_train.bin")) == False:
        print("No dataset detected for similarity subsets. Creating new set prior to loading task.")
        make_splitcifar_64(seed=seed, pc_valid=pc_valid)


    """
    Sequence: 
        0:Tiny Imagenet, 
        1:CIFAR10, 
        2:CIFAR100 split, 
        3:CIFAR100 split, 
        4:CIFAR100 split, 
        5:CIFAR100 split, 
    """


    data={}
    if task_num == 0:
        data['x']=torch.load('../data/Tiny Imagenet/' + split + '/X.pt')
        data['y']=torch.load('../data/Tiny Imagenet/' + split + '/y.pt')
    else:
        data['x']=torch.load(os.path.join(os.path.expanduser(('../data/split_cifar_64/' + str(task_num-1))), ('x_'+split+'.bin')))
        data['y']=torch.load(os.path.join(os.path.expanduser(('../data/split_cifar_64/' + str(task_num-1))), ('y_'+split+'.bin')))

    return data
    



def make_splitcifar_64(seed=0, pc_valid=0.2):
    data={}
    taskcla=[]
    size=[3,64,64]
    
    
    # CIFAR10
    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]
    
    # CIFAR10
    dat={}
    dat['train']=datasets.CIFAR10('../data/',train=True,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std),transforms.Resize((64,64))]))
    dat['test']=datasets.CIFAR10('../data/',train=False,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std),transforms.Resize((64,64))]))
    data[0]={}
    data[0]['name']='cifar10'
    data[0]['ncla']=10
    data[0]['train']={'x': [],'y': []}
    data[0]['test']={'x': [],'y': []}
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        for image,target in loader:
            data[0][s]['x'].append(image)
            data[0][s]['y'].append(target.numpy()[0])
    
    # "Unify" and save
    for s in ['train','test']:
        data[0][s]['x']=torch.stack(data[0][s]['x']).view(-1,size[0],size[1],size[2])
        data[0][s]['y']=torch.LongTensor(np.array(data[0][s]['y'],dtype=int)).view(-1)
    
    # CIFAR100
    dat={}
    
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    
    dat['train']=datasets.CIFAR100('../data/',train=True,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std),transforms.Resize((64,64))]))
    dat['test']=datasets.CIFAR100('../data/',train=False,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std),transforms.Resize((64,64))]))
    
    for n in range(1,11):
        data[n]={}
        data[n]['name']='cifar100'
        data[n]['ncla']=10
        data[n]['train']={'x': [],'y': []}
        data[n]['test']={'x': [],'y': []}
    
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        for image,target in loader:
            task_idx = target.numpy()[0] // 10 + 1
            data[task_idx][s]['x'].append(image)
            data[task_idx][s]['y'].append(target.numpy()[0]%10)
    
    
    
    for t in range(1,11):
        for s in ['train','test']:
            data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
            data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
            
    os.makedirs('../data/split_cifar/' ,exist_ok=True)
    
    for t in range(0,11):
      # Validation
      r=np.arange(data[t]['train']['x'].size(0))
      r=np.array(shuffle(r,random_state=seed),dtype=int)
      nvalid=int(pc_valid*len(r))
      ivalid=torch.LongTensor(r[:nvalid])
      itrain=torch.LongTensor(r[nvalid:])
      data[t]['valid']={}
      data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
      data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
      data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
      data[t]['train']['y']=data[t]['train']['y'][itrain].clone()
    
    
      for s in ['train','valid','test']:
        os.makedirs(('../data/split_cifar_64/' + str(t)) ,exist_ok=True)
        torch.save(data[t][s]['x'], ('../data/split_cifar_64/'+ str(t) + '/x_' + s + '.bin'))
        torch.save(data[t][s]['y'], ('../data/split_cifar_64/'+ str(t) + '/y_' + s + '.bin'))
    
    
    
    
    
    


def get_joint_mixedCIFAR_KEFMNIST(seed = 0, pc_valid=0.1, task_num=0, split = ""):


    """
    Sequence: 
        0:KMNIST-49, 
        1:CIFAR100 split, 
        1:EMNIST-Balanced, 
        3:CIFAR100 split, 
        4:Fashion MNIST, 
        5:CIFAR100 split, 
    """

    data={}
    if task_num in [1,3,5]:
        data['x']=torch.load(os.path.join(os.path.expanduser(('../data/split_cifar/' + str(task_num))), ('x_'+split+'.bin')))
        data['y']=torch.load(os.path.join(os.path.expanduser(('../data/split_cifar/' + str(task_num))), ('y_'+split+'.bin')))
        
    elif task_num == 0:
        data['x'] = torch.load('../data/K49/' + split + "/X.pt")
        data['y'] = torch.load('../data/K49/' + split + "/y.pt")
    elif task_num == 2:
        data['x'] = torch.load('../data/EMNIST/' + split + "/X.pt")
        data['y'] = torch.load('../data/EMNIST/' + split + "/y.pt")
    elif task_num == 4:
        data['x'] = torch.load('../data/FashionMNIST/' + split + "/X.pt")
        data['y'] = torch.load('../data/FashionMNIST/' + split + "/y.pt")

    return data
    
 

