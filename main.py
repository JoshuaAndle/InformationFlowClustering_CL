"""
Does standard subnetwork training on all tasks

"""

from __future__ import division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
import warnings
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler  import MultiStepLR

# from AuxiliaryScripts import utils
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

FLAGS.add_argument('--train_epochs', type=int, default=40, help='Number of epochs to train for')
FLAGS.add_argument('--finetune_epochs', type=int, default=30, help='Number of epochs to finetune for after pruning')
FLAGS.add_argument('--batch_size', type=int, default=64, help='Batch size')

# Pruning options.
### Note: We only use structured pruning. I mostly kept this in (and the accompanying if/else statements) to clarify this decision
FLAGS.add_argument('--prune_method', type=str, default='structured', choices=['structured', 'unstructured'], help='Pruning method to use. Unstructured performes IFC-US.')
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
    assert args.prune_perc_per_layer > 0., "non-positive prune perc",flush = True
    assert args.num_filters > 0, "non-positive base model width",flush = True
        
    assert args.task_num >= 0, "Task number must be 0 or greater",flush = True
    assert args.num_tasks > args.task_num, "Starting task number args.task_num > number of tasks specified", flush=True

    assert args.lr >= 0.0, "lr must be non-zero", flush=True
    assert args.lr_min >= 0.0, "lr_min must be non-zero", flush=True
    assert args.lr_patience > 0, "lr patience must be greater than zero", flush=True
    assert args.lr_factor > 0, "lr factor must be greater than zero", flush=True

    assert args.train_epochs >= 0, "train_epochs must be greater than zero", flush=True
    assert args.finetune_epochs >= 0, "finetune_epochs must be greater than zero", flush=True
    assert args.batch_size > 0, "batch_size must be greater than zero", flush=True

    assert args.num_shared >= 0 , "num_shared must be non-negative", flush=True
    assert args.num_cluster_layers > 0 , "num_cluster_layers must be greater than zero", flush=True
    assert args.shared_layers >= -1 , "shared_layers must be -1 or greater", flush=True


    ### Get the appropriate packages for structured vs unstructured pruning
    if args.prune_method == "structured":
        utils = utils_structured
        manager = manager_structured
    else:
        utils = utils_unstructured
        manager = manager_unstructured

    torch.cuda.set_device(0)
    
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('-'*100)    
    
    acchistory, epochhistory = {}, {}

    ### Determines which tasks are included in the overall sequence
    if args.dataset in ["MPC", 'KEF', 'TIC']: 
        taskset = [*range(0,6,1)]
        for task in taskset:
            acchistory[task] = {'train':{}, 'finetune':{}}
            epochhistory[task] = {'train':{}, 'finetune':{}}
    else: 
        print("Dataset {} not implemented".format(args.dataset))
        return 0

    num_classes_by_task, task_names = utils.get_taskinfo(args.dataset)







    ###################
    ##### Prepare Checkpoint and Manager
    ###################
    args.save_prefix = os.path.join("../checkpoints/", (args.dataset + "_" + args.arch + "_" + args.prune_method), 
                                                        str(args.prune_perc_per_layer), (args.similarity_type + "_" + args.similarity_method),  
                                                        str(args.num_shared), str(args.run_id))
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
        previous_task_path = os.path.join(args.save_prefix, "final.pt")
        
        ### Reloads checkpoint depending on where you are at for the current task's progress (t->c->p)    
        if os.path.isfile(previous_task_path) == True:
            ckpt = torch.load(previous_task_path)
        else:
            print("No checkpoint file found at ", previous_task_path)
            return 0
    
    ### Initialize the manager using the checkpoint.
    manager = manager.Manager(args, ckpt)








    ###################
    ##### Loop Through Tasks
    ###################
    
    ### Logic for looping over remaining tasks
    for task in taskset[args.task_num:args.num_tasks]:
        
        ### By default task order is sequential 0:num_tasks
        taskid, taskclasses, taskname = args.task_order[task], num_classes_by_task[task], task_names[task]
        ### Append the current task to the nested subdirectories for task order
        args.save_prefix = os.path.join(args.save_prefix, str(taskid))

        print("Task ID: ", taskid, " #", task, " in sequence for dataset: ", args.dataset)


        ### Update paths as needed for each new task
        os.makedirs(args.save_prefix, exist_ok = True)
        trained_path = os.path.join(args.save_prefix, "trained.pt")
        finetuned_path = os.path.join(args.save_prefix, "final.pt")

        ### For manual sharing experiments this just adjusts the file name to keep clearer records
        if (task+1) == args.num_tasks and args.share_type == "transfer":
            ### Save the checkpoint and move on to the next task if required
            print("Training on final task")
            manualshare = ""
            for i in args.manual_share_tasks:
                manualshare += str(i)
            trained_path = (args.save_prefix + "/trained"+manualshare+".pt")
            finetuned_path = (args.save_prefix + "/final"+manualshare+".pt")
            print("New training path: ", trained_path)

        manager.task_num = task
        manager.share_fkt[task] = []
        manager.share_bkt[task] = []

        ### Prepare dataloaders for new task
        train_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="train")
        val_data_loader =  utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="valid")
        test_data_loader =  utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=taskid, set="test")
        manager.train_loader, manager.val_loader, manager.test_loader = train_data_loader, val_data_loader, test_data_loader

    


        ### This is for producing and setting the classifier layer for a given task's # classes
        ### We still want to pass the sequential/ordinal task number for the key of the classifier, to make for cleaner indexing when we go to retrieve it later on
        manager.network.add_dataset(task, taskclasses)
        manager.network.set_dataset(task)

        ### Need to just make sure everything is on same device prior to sharing decision
        if args.cuda:
            manager.network.model = manager.network.model.cuda()


        ### Add the new classifier's task mask after the classifier has been added to the network
        manager.network.make_taskmask(task)
        manager.network.unmask_network()

        if task != 0:
            ### Decide which frozen weights to mask to zero for the task
            manager.pick_shared_task()

            if args.no_reinit == False:
                manager.network.reinit_statedict(task)
        

        ### Train for new task
        print("Training", flush = True)
        acchistory[task]['train'],epochhistory[task]['train'], val_acc_history  = manager.train(args.train_epochs, save=True, savename=trained_path)


        print(val_acc_history)
        torch.save(acchistory, args.save_prefix + "test_acc_history.pt")
        torch.save(epochhistory, args.save_prefix + "epoch_history.pt")
        torch.save(val_acc_history, args.save_prefix + "val_acc_history.pt")

        utils.save_ckpt(manager, trained_path)

        ### Prune unecessary weights or nodes
        manager.prune()

        print('\nPost-prune eval:')
        errors = manager.eval()
        accuracy = 100 - errors[0]  # Top-1 accuracy.

        utils.save_ckpt(manager, finetuned_path)

        ### Do final finetuning to improve results on pruned network.
        if args.finetune_epochs:
            print('Doing some extra finetuning...')
            acchistory[task]['finetune'],epochhistory[task]['finetune'], val_acc_history_finetune = manager.train(args.finetune_epochs, save=True, savename=finetuned_path, best_accuracy=0, finetune=True)

        print('-' * 16, 'Pruning summary:')
        manager.network.check(True)
        print('-' * 16, "\n\n\n\n")

        ### Save the checkpoint and move on to the next task if required
        utils.save_ckpt(manager, savename=finetuned_path)
        print(acchistory)        

        ### Train and store KMeans models and scores for current task
        manager.create_clusters()














if __name__ == '__main__':
    main()
