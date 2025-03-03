"""
Evaluate a composite network. Intended for use with two-task experiments
Calculates either accuracies only or both accuracy and subnetwork A usefulness measure
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


from AuxiliaryScripts import utils
from AuxiliaryScripts.manager import Manager

# To prevent PIL warnings.
warnings.filterwarnings("ignore")

###General flags
FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--run_id', type=str, default="000", help='Id of current run.')
FLAGS.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
FLAGS.add_argument('--arch', choices=['modresnet18', 'vgg16'], default='modresnet18', help='Architectures')
FLAGS.add_argument('--dataset', type=str, choices=['splitCIFAR', 'MPC', 'KEF','TIC'], default='MPC', help='Name of dataset')
FLAGS.add_argument('--task_num', type=int, default=0, help='Current task number.')
FLAGS.add_argument('--num_tasks', type=int, default=6, help='Number of tasks being run, used if we only want to run up to a given task in the sequence for certain experiments')
FLAGS.add_argument('--save_prefix', type=str, default='../checkpoints/', help='Location to save model')
FLAGS.add_argument('--task_order', nargs='+', type=int, default=[], help='Order in which tasks are trained')
FLAGS.add_argument('--num_filters', type=int, default=64, help='Base number of filters for Resnet18')

# Training options.
FLAGS.add_argument('--lr', type=float, default=0.1, help='Learning rate')
FLAGS.add_argument('--lr_patience', type=int, default=20, help='Patience term to dictate when Learning rate is decreased during training')
FLAGS.add_argument('--lr_factor', type=int, default=2, help='Factor by which to reduce learning rate during training')
FLAGS.add_argument('--lr_min', type=float, default=0.001, help='Minimum learning rate below which training is stopped early')
FLAGS.add_argument('--batch_size', type=int, default=512, help='Batch size')
FLAGS.add_argument('--prune_perc_per_layer', type=float, default=0.5, help='% of neurons to prune per layer')


### Subnetwork sharing flags
FLAGS.add_argument('--shareorder', choices=['lowest', 'highest'], default='highest', help='Order in which to select shared tasks by mean conn value')
FLAGS.add_argument('--num_shared', type=int, default=1, help='Number of past tasks to share')
FLAGS.add_argument('--similarity_method', choices=['linear', 'squared'], default='linear', help='Order in which to select shared tasks by mean conn value')
FLAGS.add_argument('--similarity_type', choices=['cka', 'mi', 'hsic', 'corr', 'acts', 'hsic_cka', 'hsic_cka_layer'], default='acts', help='Metric to use in deciding task similarity')

FLAGS.add_argument('--manual_share_tasks', nargs='+', type=int, default=[], help='Which tasks to share with the last task for transfer learning experiments')

FLAGS.add_argument('--share_type', choices=['transfer', 'standard', 'omit', 'optimalmanual'], default='standard', help='If transfer, only shares frozen weights during the final task')

FLAGS.add_argument('--taska', type=int, default=0, help='Task 0 in sequence')
FLAGS.add_argument('--taskb', type=int, default=1, help='Task 1 in sequence')

FLAGS.add_argument('--getsimilarity', action='store_true', default=False, help='Whether or not to collect data or only run eval accuracy')
FLAGS.add_argument('--use_raw_acts', action='store_true', default=False, help='Whether or not to use relu idxs or raw activations for capturing activations')


FLAGS.add_argument('--shared_layers', type=int, default=-1, help='Number of trainable layers used when sharing frozen weights, starting from the input layer')



###################################################################################################################################################
###
###     Main function
###
###################################################################################################################################################


def main():
    args = FLAGS.parse_args()
    torch.cuda.set_device(0)
    
    
    ### Determines which tasks are included in the overall sequence
    if args.dataset in ["MPC", "KEF", "TIC"]: 
        taskset = [*range(0,6,1)]
    else: 
        print("Incorrect dataset name for args.dataset")
        return 0
        
    final_task_num = taskset[-1]

    
    num_classes_by_task, task_names = utils.get_taskinfo(args.dataset)
    numclasses = num_classes_by_task[args.taskb]

 
       
    filename = "trained.pt"

    #!# Note I've hard coded acts/linear for the trained pathway, this was just what I'd set it to for the paper 2-task experiments. Sharing was manual though so
    ###    the fact that this was acts and squared was arbitrary and they're only included for consistent pathway structure between experiments
    args.save_prefix = os.path.join("../checkpoints/", (str(args.dataset) + "_" + str(args.arch)), str(args.prune_perc_per_layer), ("acts_linear"), str(args.num_shared), args.run_id , str(args.taska), str(args.taskb))
    
    
    finalpath = os.path.join(args.save_prefix, filename)
    print(finalpath)

    if os.path.isfile(finalpath) == False:
        print("No trained checkpoint found for task number: ", args.taska,"  ", args.taskb)
        return 0
    else:
        ckpt = torch.load(finalpath)

    ### Initialize the manager using the final checkpoint
    manager = Manager(args, ckpt)

    
    if args.cuda:
        manager.network.model = manager.network.model.cuda()


    manager.task_num = 1

    ### Prepare dataloaders for new task
    train_data_loader = utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=args.taskb, set="train")
    val_data_loader =  utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=args.taskb, set="valid")
    test_data_loader =  utils.get_dataloader(args.dataset, args.batch_size, pin_memory=args.cuda, task_num=args.taskb, set="test")
    manager.train_loader = train_data_loader
    manager.val_loader = val_data_loader
    manager.test_loader = test_data_loader

    if args.getsimilarity == True:
        ### This is for producing and setting the classifier layer for a given task's # classes
        manager.network.set_dataset(0)
        ### Don't mask any of the weights, but swap out the classifier for the current task
        manager.network.model.eval()  
        
        if args.cuda:
            manager.network.model = manager.network.model.cuda()

        if args.similarity_type == "corr":
            past_task_dict = manager.get_dict_task_conns()
            if args.similarity_method == "linear":
                savepath = os.path.join(args.save_prefix, "connslinear.pt")
            elif args.similarity_method == "squared":
                savepath = os.path.join(args.save_prefix, "connssquared.pt")
        elif args.similarity_type == "acts":
            past_task_dict = manager.get_dict_task_acts()
            if args.similarity_method == "linear":
                savepath = os.path.join(args.save_prefix, "actslinear.pt")
            elif args.similarity_method == "squared":
                savepath = os.path.join(args.save_prefix, "actssquared.pt")
        elif args.similarity_type in ["mi", "hsic", "cka"]:
            savepath = os.path.join(args.save_prefix, (args.similarity_type + "_dict.pt"))
            past_task_dict = manager.get_dict_task_metrics(similarity_type=args.similarity_type)
        elif args.similarity_type in ["hsic_cka", "hsic_cka_layer"]:
            savepath = os.path.join(args.save_prefix, (args.similarity_type + "_dict.pt"))
            past_task_dict = manager.get_dict_task_metrics(similarity_type=args.similarity_type)
        else:
            print("Invalid similarity type argument")

        print("Similarity Dict: ", past_task_dict)
        torch.save(past_task_dict, savepath)
    else:
        # ### This is for producing and setting the classifier layer for a given task's # classes
        print("network classifiers: ", manager.network.classifiers.keys())
        # manager.network.set_dataset(args.taskb)
        manager.network.set_dataset(1)
        # ### Don't mask any of the weights, but swap out the classifier for the current task
        manager.network.model.eval()  
        test_errors = manager.eval(verbose=False)
        test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.
        savepath = os.path.join(args.save_prefix, ("testacc.pt"))
        torch.save(test_accuracy, savepath)
    
if __name__ == '__main__':
    main()
