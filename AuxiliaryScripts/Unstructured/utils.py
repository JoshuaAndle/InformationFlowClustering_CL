import os
import copy

# from scipy.special        import *


import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from AuxiliaryScripts import DataGenerator as DG
from AuxiliaryScripts import cldatasets
from AuxiliaryScripts.Unstructured import network as net








#####################################################
###    Activation Functions
#####################################################






acts = {}

### Returns a hook function directed to store activations in a given dictionary key "name"
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        acts[name] = output.detach().cpu()
    return hook

### Create forward hooks to all layers which will collect activation state
### Collected from ReLu layers when possible, but not all resnet18 trainable layers have coupled relu layers
def get_all_layers(net, hook_handles:list, relu_idxs:list[int]):
    for module_idx, module in enumerate(net.shared.modules()):
        if module_idx in relu_idxs:
            hook_handles.append(module.register_forward_hook(getActivation(module_idx)))


### Process and record all of the activations for the given pair of layers
def activations(data_loader, model: nn.Module, ardict:dict, use_raw_acts:bool = False):
    temp_op       = None
    temp_label_op = None

    parents_op  = None
    labels_op   = None

    handles     = []

    ### We will collect activations from the relu layers and store them for the corresponding preceding weight layer for use in measuring connectivity
    ### If "use_raw_acts" is set to True then we use the same weight layer idx for both collecting and storing the raw outputs
    act_idxs = list(ardict.keys())
    relu_idxs = list(ardict.values())

    if use_raw_acts == False:
        ### Set hooks in all tunable layers
        get_all_layers(model, handles, relu_idxs)
    else:
        get_all_layers(model, handles, act_idxs)
    
    ### A dictionary for storing the activations
    actsdict = {}
    labels = None

    for i in act_idxs: 
        actsdict[i] = None
    
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            x_input, y_label = data
            model(x_input.cuda())

            if step == 0:
                labels = y_label.detach().cpu()
                for key in acts.keys():
                    
                    ### We need to convert from relu idxs to trainable layer idxs for future masking purposes
                    acts_idx = act_idxs[relu_idxs.index(key)]
                    
                    ### For all conv layers we average over the feature maps, this makes them compatible when comparing with linear layers and reduces memory requirements
                    if len(acts[key].shape) > 2:
                        actsdict[acts_idx] = acts[key].mean(dim=3).mean(dim=2)
                    else:
                        actsdict[acts_idx] = acts[key]
            else: 
                labels = torch.cat((labels, y_label.detach().cpu()),dim=0)
                for key in acts.keys():
                    acts_idx = act_idxs[relu_idxs.index(key)]
                    if len(acts[key].shape) > 2:
                        actsdict[acts_idx] = torch.cat((actsdict[acts_idx], acts[key].mean(dim=3).mean(dim=2)), dim=0)
                    else:
                        actsdict[acts_idx] = torch.cat((actsdict[acts_idx], acts[key]), dim=0)

            
    # Remove all hook handles
    for handle in handles:
        handle.remove()    

    return actsdict, labels






















#####################################################
###    Saving Function
#####################################################



### Saves a checkpoint of the model
def save_ckpt(manager, savename:str):
    # Prepare the ckpt.
    ckpt = {
        'args': manager.args,
        'network': manager.network,
    }

    # Save to file.
    torch.save(ckpt, savename)





#####################################################
###    Masking Functions
#####################################################

### Get a binary mask where all previously frozen weights are indicated by a value of 1
### After pruning on the current task, this will still return the same masks, as the new weights aren't frozen until the task ends
def get_frozen_mask(weights: torch.Tensor, module_idx:int, all_task_masks:dict, task_num:int):
    mask = torch.zeros(weights.shape)
    ### Include all weights used in past tasks (which would have been subsequently frozen)
    for i in range(0, task_num):
        if i == 0:
            mask = all_task_masks[i][module_idx].clone().detach()
        else:
            mask = torch.maximum(all_task_masks[i][module_idx], mask)
    return mask
        
    
### Get a binary mask where all weights available for training are indicated by a value of 1
### Unlike get_frozen_mask(), this mask will change after pruning since the pruned weights are no longer trainable for the current task
def get_trainable_mask(module_idx:int, all_task_masks:dict, task_num:int):
    mask = all_task_masks[task_num][module_idx].clone().detach()
    frozen_mask = get_frozen_mask(mask, module_idx, all_task_masks, task_num)
    mask[frozen_mask.eq(1)] = 0
    return mask
    

  
### Get a binary mask with a value of 1 for all weights that were shared with the current task by frozen subnetworks
def get_shared_mask(module_idx:int, all_task_masks:dict, task_num:int):
    ### Get all weight indices included in the current task
    mask = all_task_masks[task_num][module_idx].clone().detach()
    frozen_mask = get_frozen_mask(mask, module_idx, all_task_masks, task_num)
    
    ### We want all frozen weights which are included in the current task subnetwork
    frozen_mask[mask.eq(0)] = 0
    return frozen_mask







#####################################################
###    Dataset Functions
#####################################################



def get_taskinfo(dataset:str):
    print("Dataset: ", dataset)
    if dataset == 'MPC':
        numclasses = [10,10,10,10,10,10]
        tasknames = ["pmnist0", "cifar100a", "pmnist2", "cifar100c", "pmnist4", "cifar100e"]
    elif dataset == 'KEF':
        numclasses = [49,10,47,10,10,10]
        tasknames = ["kmnist", "cifar100a", "emnist", "cifar100c", "fashion-mnist", "cifar100e"]
    elif dataset == "TIC":
        numclasses = [200,10,10,10,10,10]
        tasknames = ["tiny-imagenet", "cifar10", "cifar100a", "cifar100b", "cifar100c", "cifar100d"]
    return numclasses, tasknames
    
    
### Returns a dictionary of "train", "valid", and "test" data+labels for the appropriate cifar subset
def get_dataloader(dataset:str, batch_size:int, num_workers:int=4, pin_memory:bool=False, task_num:int=0, set:str="train"):
    
    # standard split CIFAR-10/100 sequence of tasks

    if dataset == "MPC":
        dataset = cldatasets.get_mixedCIFAR_PMNIST(task_num=task_num, split = set)
    elif dataset == "TIC":
        dataset = cldatasets.get_TinyImagenetCIFAR(task_num=task_num, split = set)
    elif dataset == "KEF":
        dataset = cldatasets.get_mixedCIFAR_KEFMNIST(task_num=task_num, split = set)
    else: 
        print("Incorrect dataset for get_dataloader()")
        return -1
        
    ### Makes a custom dataset for a given dataset through torch
    generator = DG.DataGenerator(dataset['x'],dataset['y'])

    ### Loads the custom data into the dataloader
    if set == "train":
        return data.DataLoader(generator, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory=pin_memory)
    else:
        return data.DataLoader(generator, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory=pin_memory)










### These dictionaries have been hardcoded for resnet particularly due to the difficulties of automating for the skip layers, 
###      but we provide here the general function used to put together the dictionaries mapping for vgg16
###   parent->child layers and activation->relu layers (ardict) in vgg16
def compute_idx_dictionaries(model:nn.Module, modeltype:str="vgg16"):
    if modeltype in ['vgg16']:
        parent_idxs = []
        child_idxs = []
        ardict={}
        acttemp=-1
        for module_idx, module in enumerate(model.shared.modules()):
            if isinstance(module, nn.Conv2d):
              acttemp = module_idx
            elif isinstance(module, nn.ReLU) and acttemp != -1:
              ardict[acttemp] = module_idx
              acttemp = -1
            ### Using ReLu for connectivity calculation often leads to NaNs in linear layers, so we stick to using the raw outputs
            elif isinstance(module, nn.Linear):
                ardict[module_idx] = module_idx
        if acttemp != -1:
          print("Appending to ardict for missing relu on layer: ", acttemp)
          ardict[acttemp] = acttemp
                 
        
        keyslist = list(ardict.keys())
        for idx in range(len(keyslist)):
          if idx != len(keyslist)-1:
            parent_idxs.append(keyslist[idx])
            child_idxs.append(keyslist[idx+1])
        
        print("act-relu dictionary: ", ardict)
        print("parents: ", parent_idxs)
        print("children: ", child_idxs)
        return parent_idxs,child_idxs, ardict
    # elif modeltype == "modresnet18":
    #     pcdict={}
    #     ardict={}
    #     acttemp=-1
    #     for module_idx, module in enumerate(model.shared.modules()):
    #         if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
    #           acttemp = module_idx
    #         if isinstance(module, nn.ReLU):
    #           ardict[acttemp] = module_idx
    #           acttemp = -1
    #     if acttemp != -1:
    #       ardict[acttemp] = acttemp
        
    #     keyslist = list(ardict.keys())
    #     for idx in range(len(keyslist)):
    #       if idx != len(keyslist)-1:
    #         pcdict[keyslist[idx]] = keyslist[idx+1]
        
    #     print("act-relu dictionary: ", ardict)
    #     print("parents: ", parent_idxs)
    #     print("children: ", child_idxs)
    #     return parent_idxs,child_idxs, ardict
