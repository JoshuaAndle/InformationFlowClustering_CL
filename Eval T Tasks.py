"""
Does standard subnetwork training on all tasks

"""

from __future__ import division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import warnings
import copy
import time

import numpy as np
import torch
import torch.nn as nn


import AuxiliaryScripts.Structured.manager as manager_structured
import AuxiliaryScripts.Unstructured.manager as manager_unstructured

import AuxiliaryScripts.Structured.utils as utils_structured
import AuxiliaryScripts.Unstructured.utils as utils_unstructured

# To prevent PIL warnings.
warnings.filterwarnings("ignore")

###General flags
FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--run_id', type=str, default="000", help='Id of current run.')
FLAGS.add_argument('--arch', choices=['modresnet18', 'vgg16'], default='modresnet18', help='Architectures')
FLAGS.add_argument('--dataset', type=str, choices=['MPC', 'KEF','TIC'], default='MPC', help='Name of dataset')
FLAGS.add_argument('--num_filters', type=int, default=32, help='Base number of filters for Resnet18')
FLAGS.add_argument('--task_order', nargs='+', type=int, default=[], help='Order in which tasks are trained')
FLAGS.add_argument('--num_tasks', type=int, default=6, help='Number of tasks being run, used if we only want to run up to a given task in the sequence for certain experiments')
FLAGS.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
FLAGS.add_argument('--task_num', type=int, default=0, help='Current task number.')
FLAGS.add_argument('--save_prefix', type=str, default='../checkpoints/', help='Location to save model')
FLAGS.add_argument('--use_raw_acts', action='store_true', default=False, help='Whether or not to use relu idxs or raw activations for capturing activations')


# Training options.
FLAGS.add_argument('--lr', type=float, default=0.1, help='Learning rate')
FLAGS.add_argument('--lr_min', type=float, default=0.001, help='Minimum learning rate below which training is stopped early')
FLAGS.add_argument('--lr_patience', type=int, default=5, help='Patience term to dictate when Learning rate is decreased during training')
FLAGS.add_argument('--lr_factor', type=int, default=5, help='Factor by which to reduce learning rate during training')
FLAGS.add_argument('--share_sparsity_offset', type=float, default=0.0, help='% by which sparsity is decreased if not sharing any past subnetworks')

FLAGS.add_argument('--batch_size', type=int, default=64, help='Batch size')

# Pruning options.
### Note: We only use structured pruning. I mostly kept this in (and the accompanying if/else statements) to clarify this decision
FLAGS.add_argument('--prune_method', type=str, default='structured', choices=['structured', 'unstructured'], help='Pruning method to use. Unstructured pruning is not implemented.')
FLAGS.add_argument('--prune_perc_per_layer', type=float, default=0.65, help='% of neurons to prune per layer')


### Subnetwork sharing flags
FLAGS.add_argument('--shareorder', choices=['lowest', 'highest'], default='highest', help='Order in which to select shared tasks by mean conn value')
FLAGS.add_argument('--num_shared', type=int, default=1, help='Number of past tasks to share')
FLAGS.add_argument('--similarity_method', choices=['linear', 'squared'], default='squared', help='Order in which to select shared tasks by mean conn value')
FLAGS.add_argument('--similarity_type', choices=['cka', 'mi', 'hsic', 'corr', 'acts'], default='acts', help='Metric to use in deciding task similarity')
FLAGS.add_argument('--share_type', choices=['transfer', 'standard', 'omit', 'optimalmanual', 'clustering'], default='standard', help='If transfer, only shares frozen weights during the final task')
FLAGS.add_argument('--score_threshold', type=float, default=2.0, help='The ratio of KMeans score needed for a past subnetwork to be shared through clustering')
FLAGS.add_argument('--num_cluster_layers', type=int, default=4, help='Number of trainable layers included during clustering of connectivities')
FLAGS.add_argument('--shared_layers', type=int, default=-1, help='Number of trainable layers used when sharing frozen weights, starting from the input layer')
FLAGS.add_argument('--manual_share_tasks', nargs='+', type=int, default=[], help='Which tasks to share with the last task for transfer learning experiments')



### Experiment setup flags

# Misc options
FLAGS.add_argument('--no_reinit', action='store_true', default=False, help='Dont reinitialize pruned weights to non-zero values')

###################################################################################################################################################
###
###     Main function
###
###################################################################################################################################################

def main():
    args = FLAGS.parse_args()

    ### Early termination conditions
    assert args.prune_perc_per_layer > 0., print("non-positive prune perc",flush = True) 
    assert args.num_filters > 0, print("non-positive base model width",flush = True)
        
    assert args.task_num >= 0, print("Task number must be 0 or greater",flush = True)
    assert args.num_tasks >= args.task_num, print(f"Starting task number {args.task_num} > number of tasks {args.num_tasks}", flush=True)

    assert args.lr >= 0.0, print("lr must be non-zero", flush=True)
    assert args.lr_min >= 0.0, print("lr_min must be non-zero", flush=True)
    assert args.lr_patience > 0, print("lr patience must be greater than zero", flush=True)
    assert args.lr_factor > 0, print("lr factor must be greater than zero", flush=True)

    assert args.batch_size > 0, print("batch_size must be greater than zero", flush=True)

    assert args.num_shared >= 0 , print("num_shared must be non-negative", flush=True)
    assert args.num_cluster_layers > 0 , print("num_cluster_layers must be greater than zero", flush=True)
    assert args.shared_layers >= -1 , print("shared_layers must be -1 or greater", flush=True)







    torch.cuda.set_device(0)
    
    
    if args.prune_method == "structured":
        utils = utils_structured
        manager_module = manager_structured
    else:
        utils = utils_unstructured
        manager_module = manager_unstructured


    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('-'*100)    
    
    acchistory = {}
    epochhistory = {}

    ### Determines which tasks are included in the overall sequence
    if args.dataset in ["MPC", 'KEF', 'TIC']: 
        taskset = [*range(0,6,1)]
        for task in taskset:
            acchistory[task] = {'train':{}, 'finetune':{}}
            epochhistory[task] = {'train':{}, 'finetune':{}}
    else: 
        print("Incorrect dataset name for args.dataset")
        return 0

    num_classes_by_task, task_names = utils.get_taskinfo(args.dataset)

    ###################
    ##### Prepare Checkpoint and Manager
    ###################

    args.save_prefix = os.path.join("../checkpoints/", (args.dataset + "_" + args.arch + "_" + args.prune_method), str(args.prune_perc_per_layer), 
                                                        (args.similarity_type + "_" + args.similarity_method),  str(args.num_shared), str(args.run_id))
    os.makedirs(args.save_prefix, exist_ok = True)



    ### If we're resuming midway through the sequence then we need to update the save/load path accordingly to account for nested directories 
    for t in range(0, args.task_num):
        priortaskid = args.task_order[t]
        ### Append the nested directories from the earliest task to the most recent
        args.save_prefix = os.path.join(args.save_prefix, str(priortaskid))
        

    ### If no checkpoint is found, the default value will be None and a new one will be initialized in the Manager
    ckpt = None
    if args.task_num != 0:
        ### Path to load previous task's checkpoint, if not starting at task 0
        previous_task_path = os.path.join(args.save_prefix, "trained.pt")
        
        ### Reloads checkpoint depending on where you are at for the current task's progress (t->c->p)    
        if os.path.isfile(previous_task_path) == True:
            ckpt = torch.load(previous_task_path, weights_only=False)
        else:
            print("No checkpoint file found at ", previous_task_path)
            return 0
    
    ### Initialize the manager using the checkpoint.
    manager = manager_module.Manager(args, ckpt)








    ### To evaluate six-task experiments primarily. Loop over all tasks and print out the final networks accuracy on them by applying masks
    accs = []
    for task in range(0,args.num_tasks):
        #!# We shouldnt HAVE to reload each time, but for some reason I was seeing issues otherwise IIRC. Should look into that given the time
        ckpt = torch.load(previous_task_path, weights_only=False)
        print("\nloaded checkpoint again")
        ### Initialize the manager using the final checkpoint
        manager = manager_module.Manager(args, ckpt)
        
        ### Unmasks all weights prior to eval(). Eval will reapply the appropriate mask
        manager.network.unmask_network()

        manager.task_num = task
        
        if args.cuda:
            manager.network.model = manager.network.model.cuda()

        taskid = args.task_order[task]

        ### Prepare dataloaders for new task
        train_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="train")
        val_data_loader =  utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="valid")
        test_data_loader =  utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="test")
        manager.train_loader = train_data_loader
        manager.val_loader = val_data_loader
        manager.test_loader = test_data_loader
    
       
        print("Task: ", task, "Task ID: ", taskid)
        
        manager.network.set_dataset(task)
        ### Don't mask any of the weights, but swap out the classifier for the current task
        manager.network.model.eval()  
        errors = manager.eval()
        taskacc = 100-errors[0]
        accs.append(taskacc)
        
    print("Done")
    print(accs)














if __name__ == '__main__':
    main()
