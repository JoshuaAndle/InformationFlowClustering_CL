"""
Handles all the pruning, training, and connectivity functions. Pruning steps are adapted from: https://github.com/arunmallya/packnet/blob/master/src/prune.py
Connectivity steps and implementation of connectivity into the pruning steps are part of our contribution
"""
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm

import collections
import time
import copy
import random
import multiprocessing
import json
import copy
from math import floor

import sklearn
from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler  import MultiStepLR
import torchnet as tnt

# Custom imports
from AuxiliaryScripts.Structured import network as net
from AuxiliaryScripts import clmodels, mi_estimator, hsic_estimator
from AuxiliaryScripts.Structured import utils
from AuxiliaryScripts.Structured.utils import activations


class Manager(object):
    """Performs pruning on the given model."""
    ### Relavent arguments are moved to the manager to explicitly show which arguments are used by it
    def __init__(self, args, checkpoint):
        self.args = args
        self.task_num = args.task_num
        self.train_loader = None 
        self.val_loader = None 
        self.test_loader = None 
        
        self.share_fkt = {}
        self.share_bkt = {}
        ### Models hold the kmeans models and scores hold the in and cross-task score values for each model
        self.conn_scoring = {"models": {}, "scores": {}, "scoresnorm": {}}

        
        self.acts_dict = {}
        self.labels = None
        self.use_raw_acts = args.use_raw_acts
        
        ### These are the hardcoded connections within a ResNet18 model, for simplicity of code. 
        
        """
        Explanation of indices lists:
        Parent and Child_idxs:
            Record all connected trainable layers' idxs. For a given index, parent_idxs[idx] and child_idxs[idx] record the connected parent and child layer
            Duplicates in ResNet are due to residual connections which have multiple incoming/outgoing layer connections where they branch or add back in
        ReLu idxs:
            The layer idxs from which activations are collected. Where possible these are relu layers, though resnet18 has some conv2d layers which dont match relu layers
                and in these cases the conv2d layer output is used
        Act_idxs:
            All trainable layer (conv2d and linear) idxs. After activations are gathered theyre stored in the corresponding acts_idx indexes.
            Rather than storing them by relu index, this is done so that the idxs are compatible with the weight mask idxs
        
        NOTE: As implemented we take the outputs of the conv layers instead of the relu layers for vgg16 as well as resnet. This has been left in to clarify how exactly we collect activations however 
        """


        ### Either load from a checkpoint or initialize the necessary network
        if checkpoint != None:
            self.network = checkpoint['network']
        else:
            ### This is for producing and setting the classifier layer for a given task's # classes
            self.network = net.Network(args)
        
                
        if args.arch == "vgg16":
            self.parent_idxs, self.child_idxs, self.ardict = utils.compute_idx_dictionaries(self.network.model, args.arch)
            self.skip_layers = {}

        elif args.arch == "modresnet18":
            ### Hardcoding the layer relationships due to the complexity introduced by skip layers
            ### The parent and child idxs for the updated relu layers with in_place=False
            self.parent_idxs = [1,1, 7, 10,10,13,17,20,20,23,28,31,31,34,38,41,41,44,49,52,52,55,59,62,62,65,70,73,73,76,80]
            self.child_idxs =  [7,13,10,17,23,17,20,28,34,28,31,38,44,38,41,49,55,49,52,59,65,59,62,70,76,70,73,80,86,80,83]

            ### Since skip layers dont have their own relu layer they are set to using the raw values from their conv2d output
            self.ardict =  { 1:3,7:9,10:15,13:13,17:19,20:25,23:23,28:30,31:36,34:34,38:40,41:46,44:44,49:51,52:57,55:55,59:61,62:67,65:65,70:72,73:78,76:76,80:82,83:88,86:86}
            ### Store each skip layer and the corresponding parent as a key:value pair
            self.skip_layers = {13:1,23:10,34:20,44:31,55:41,65:52,76:62,86:73}
            

        print("\n#######################################################################")
        print("Finished Initializing Manager")
        print("All task Masks keys: ", self.network.all_task_masks.keys())
        print("Dataset: " + str(self.args.dataset))
        print("#######################################################################")

    
     
    """
    ###########################################################################################
    #####
    #####  Connectivity Functions
    #####
    ###########################################################################################
    """

    ### Run evaluation of val or test loader and get all activations stored as acts_dict
    def calc_activations(self):
        self.network.model.eval()
        self.acts_dict, self.labels = activations(self.val_loader, self.network.model, self.args.cuda, self.ardict, self.args.use_raw_acts)
        self.network.model.train()
        print("Done collecting activations")

    ### Calculate activation correlations for all layers in the network up to the max_idx layer
    def calc_corr(self, max_idx = -1):
        all_conns = {}
        ### Gets the activations. Network is unmasked prior to calling calc_corr
        self.calc_activations()
        if max_idx == -1:
            max_idx = len(self.parent_idxs)
        for key_id in range(0,max_idx): 
            parent_key, child_key = self.parent_idxs[key_id], self.child_idxs[key_id]
            p1_op = copy.deepcopy(self.acts_dict[parent_key]).numpy() 
            c1_op = copy.deepcopy(self.acts_dict[child_key]).numpy()
    
            parent_aves = []
    

            if np.count_nonzero(np.isnan(p1_op)) > 0 or np.count_nonzero(np.isnan(c1_op)) > 0:
                print("Raw activations are nan")
                
            ### Connectivity is standardized by class mean and stdev
            for label in list(np.unique(self.labels.numpy())):
                parent_mask = np.ones(p1_op.shape,dtype=bool)
                child_mask = np.ones(c1_op.shape,dtype=bool)
    
                ### masks out the activations of all but one class, as well as filters which are omitted (all activations are 0)
                parent_mask[self.labels != label] = False
                parent_mask[:,np.all(np.abs(p1_op) < 0.0001, axis=0)] = False
                child_mask[self.labels != label] = False
                child_mask[:,np.all(np.abs(c1_op) < 0.0001, axis=0)] = False
                
                p1_op[parent_mask] -= np.mean(p1_op[parent_mask])
                p1_op[parent_mask] /= np.std(p1_op[parent_mask])
    
                c1_op[child_mask] -= np.mean(c1_op[child_mask])
                c1_op[child_mask] /= np.std(c1_op[child_mask])
    
    
    
            """
            Code for averaging conns by parent prior by layer
            """
            parents_by_class = []
            conn_aves = []
            parents = []
            for cl in list(np.unique(self.labels.numpy())):
                p1_class = p1_op[self.labels == cl]
                c1_class = c1_op[self.labels == cl]
                
                ### Parents is a 2D list of all of the connectivities of parents and children for a single class
                coefs = np.corrcoef(p1_class, c1_class, rowvar=False).astype(np.float32)


                parents = []
                ### Loop over the cross correlation matrix for the rows corresponding to the parent layer's filters
                for i in range(0, len(p1_class[0])):
                    ### Append the correlations to all children layer filters for the parent filter i. We're indexing the upper-right quadrant of the correlation matrix between x and y
                    ### Nans: If a parent is omitted, this entire set will be NaN, if a child is omitted, then only the corresponding correlation is nan
                    ###    Note: These NaNs are expected and not an issue since they dont appear in the indexed values for the current subnetwork/task
                    parents.append(coefs[i, len(p1_class[0]):])
                ### We take the absolute value because we only care about the STRENGTH of the correlation, not the 
                parents = np.abs(np.asarray(parents))
                parents = np.nan_to_num(parents)
                ### This is a growing list of each p-c connectivity for all activations of a given class
                ###     The dimensions are (class, parent, child)
                parents_by_class.append(parents)
            
            conn_aves = np.mean(np.asarray(parents_by_class), axis=0)
            
            all_conns[key_id] = conn_aves

        return all_conns
        
        
    ### Create KMeans clusters and scores for current task    
    ### With structured constraints we collect all subnets' activations at once since they dont interfere with eachother, then we mask the connectivity values
    def create_clusters(self):
        self.conn_scoring['models'][self.task_num] = {}
        self.conn_scoring['scores'][self.task_num] = {}
        self.conn_scoring['scoresnorm'][self.task_num] = {}
        taskmodels = {}
        ### How many layers are we considering for clustering, avoids running needless computations
        num_idxs = min(self.args.num_cluster_layers,len(self.child_idxs))
        
        self.network.unmask_network()

        if self.args.similarity_type == "corr":
            connsdict = self.calc_corr(max_idx = num_idxs)
        taskconnsdict = {}
        taskmask = self.network.all_task_masks[self.task_num]
        
        taskscore = []
        ### We want to get the conns for just the current subnetwork for clustering
        ### We loop over only the layers which we're interested in producing kmeans models for
        for i in range(0, num_idxs):
            ### Use the task's layer masks to mask for weights in the chosen task
            parent_mask, child_mask = taskmask[self.parent_idxs[i]].eq(1), taskmask[self.child_idxs[i]].eq(1)
            ### Need to convert to a shape of 2 to match the the connectivity dictionary shape            
            if parent_mask.dim() == 4:
                parent_mask = parent_mask.all(dim=3).all(dim=2)
            if child_mask.dim() == 4:
                child_mask = child_mask.all(dim=3).all(dim=2)
            parent_mask = parent_mask.any(dim=1).cpu()
            child_mask = child_mask.any(dim=1).cpu()

            ### Get the connectivities to set the clusters
            lconns = connsdict[i][parent_mask][:,child_mask]
            ### For each included layer we create and store a KMeans model for the 2D connectivity data of the given layer
            taskmodels[i] = KMeans(n_clusters=2, random_state=0, n_init=1).fit(lconns)
            ### We then store the scores for distances to the centroids
            taskscore.append(taskmodels[i].score(lconns)/lconns.shape[0])
        
        ### Once all included layers are calculated, store the trained KMeans models and resulting in-distribution score for the task
        self.conn_scoring['models'][self.task_num] = taskmodels
        self.conn_scoring['scores'][self.task_num][self.task_num] = torch.tensor(taskscore).mean()
        self.conn_scoring['scoresnorm'][self.task_num][self.task_num] = torch.tensor(taskscore).mean()
        
        return

        

        

    ### Get the correlations of past subnetworks on the current data to compute connectivity
    def get_task_conns_corr(self):  
        taskconnsdict = {}

        connsdict = self.calc_corr()

        ### loop over all past tasks to collect subnetwork connectivities
        for task in range(0,self.task_num):
            taskmask = self.network.all_task_masks[task]
            task_conn_ave = 0

            for i in range(0, len(self.child_idxs)):
                parent_idx = self.parent_idxs[i]
                child_idx = self.child_idxs[i]

                ### Use the task mask to mask for weights in the chosen task
                bool_mask = taskmask[child_idx].eq(1)
                
                ### Need to convert to a shape of 2 to match the the connectivity dictionary shape            
                if bool_mask.dim() == 4:
                    ### True for all filters in the parent layer which have at least one outgoing, non-zero weight in the current task
                    bool_mask = bool_mask.any(dim=3).any(dim=2)

                
                bool_mask = bool_mask.transpose(0,1)
    
                
                ### Index into connectivity dictionary based on current child layer index. Works for layers that only appear once in child_idxs
                task_conns = torch.from_numpy(connsdict[i])[bool_mask.cpu()]
                ### For now I just want to use the average conns over the layer
                if task_conns.numel() > 0:
                    if self.args.similarity_method == 'linear':
                        task_conn_ave += torch.mean(task_conns)
                    elif self.args.similarity_method == 'squared':
                        task_conn_ave += torch.mean(torch.square(task_conns))
                    else:
                        print("Incorrect argument for similarity method")
                                                                            
                taskconnsdict[task] = task_conn_ave/len(self.child_idxs)
            
        ### Returns task-specific network connectivity averages. This is the average connectivity over all weights used during a given task
        return taskconnsdict


    ### Determine the scores for past task's subnetworks on current data with stored kmeans models
    def score_past_subnets(self):
        num_idxs = min(self.args.num_cluster_layers,len(self.child_idxs))
        if self.args.similarity_type == "corr":
            connsdict = self.calc_corr(max_idx = num_idxs)
        else: 
            print("clustering not implemented for similarity type: ", self.args,similarity_type)
        
        most_similar_tasks = []
        ### We loop over the subnetworks of all past tasks to compare scores on stored kmeans models
        for pasttask in range(0,self.task_num):
            taskmodels = self.conn_scoring['models'][pasttask]
            IDscore = self.conn_scoring['scores'][pasttask][pasttask]
            taskmask = self.network.all_task_masks[pasttask]

            taskscore = 0
            taskscores = []
            for i in range(0, num_idxs):
                ### Use the task's layer masks to mask for weights in the chosen task
                parent_mask, child_mask = taskmask[self.parent_idxs[i]].eq(1), taskmask[self.child_idxs[i]].eq(1)
                ### Need to convert to a shape of 2 to match the the connectivity dictionary shape            
                if parent_mask.dim() == 4:
                    parent_mask = parent_mask.all(dim=3).all(dim=2)
                if child_mask.dim() == 4:
                    child_mask = child_mask.all(dim=3).all(dim=2)
                parent_mask = parent_mask.any(dim=1).cpu()
                child_mask = child_mask.any(dim=1).cpu()
    
                ### Get the connectivities to set the clusters
                lconns = connsdict[i][parent_mask][:,child_mask]
                ### We first normalize the scores based on the number of samples, since they're calculated as a squared loss term
                taskscores.append(taskmodels[i].score(lconns)/lconns.shape[0])
            
            ### We normalize the scores based on the "In-Distribution" score of the original task data of the given subnetwork for consistency
            taskscore = torch.tensor(taskscores).mean()/IDscore
            
            print("For subnet ", pasttask, " taskscores raw are ", torch.tensor(taskscores).mean(), " IDScore is ", IDscore, " and taskscore is: ", taskscore)

            ### For record keeping we just store the calculated score in the appropriate dict without normalizing
            self.conn_scoring['scores'][pasttask][self.task_num] = torch.tensor(taskscores).mean()
            self.conn_scoring['scoresnorm'][pasttask][self.task_num] = taskscore
            
            ### If the loss score of the past subnetwork is low enough compared to its ID value then we'll share that subnetwork
            if taskscore < self.args.score_threshold:
                most_similar_tasks.append(pasttask)
            
        return most_similar_tasks



    def pick_shared_task(self):
        if self.args.share_type == "clustering":
            print("Sharing using automated clustering")
            most_similar_task_keys = self.score_past_subnets()

        elif self.args.share_type == 'optimalmanual':
            print("Sharing using manually dictated optimal setup")
            if (self.args.dataset == "MPC" or self.args.dataset == "KEF"):
                print("Using optimal share for MPC/KEF dataset")
                most_similar_task_dict = {1:None, 2:[0], 3:[1], 4:[0,2], 5:[1,3]}
            elif self.args.share_type == 'optimalmanual' and self.args.dataset == "TIC":
                print("Using optimal share for TIC dataset")
                most_similar_task_dict = {1:[0], 2:[0], 3:[0], 4:[0], 5:[0]}
            
            most_similar_task_keys = most_similar_task_dict[self.task_num]
    
        else:
            print("Sharing using similarity measures")
            if self.args.similarity_type == "corr":
                past_task_dict = self.get_task_conns_corr()
            elif self.args.similarity_type == "acts":
                past_task_dict = self.get_task_acts()
            elif self.args.similarity_type in ["mi", 'cka', 'hsic']:
                past_task_dict = self.get_task_conns_metrics()
            else:
                print("Invalid similarity type argument")

            print("Similarity Dict: ", past_task_dict)
            ### If we're doing three-task experiments then we only want to share during the last task in the training sequence
            if self.args.share_type == "transfer" and ((self.task_num + 1) < self.args.num_tasks):
                N = 0
                print("Not in final task, setting N to 0")
            ### For six-task experiments where we share all but N omitted tasks
            elif self.args.share_type == "omit":
                ### Repurposing num_shared as num_omitted, should rename though
                N = max(0, len(past_task_dict) - self.args.num_shared)
            ### Standard sharing of N tasks
            else:
                ### Number of tasks to be shared
                N = min(self.args.num_shared, len(past_task_dict))
            
            print("Sharing ", N, " tasks")

            ### If we're manually selecting the tasks to share for 3-task experiments and manual 6-task share orders
            if self.args.manual_share_tasks != [] and N != 0:
                most_similar_task_keys = self.args.manual_share_tasks        
                print("Using manual keys")
                print("N is: ", N)        
            ### Order of connectivity or activations to share by (using subnetwork averages)
            elif self.args.shareorder == 'highest' and N != 0 :
                most_similar_tasks = sorted(past_task_dict.items(), key=lambda x: x[1], reverse=True)[:N]
                most_similar_task_keys = [item[0] for item in most_similar_tasks]
            elif self.args.shareorder == 'lowest' and N != 0:
                most_similar_tasks = sorted(past_task_dict.items(), key=lambda x: x[1])[:N]
                most_similar_task_keys = [item[0] for item in most_similar_tasks]
            else:
                most_similar_task_keys = None

        print("Tasks deemed most similar: ", most_similar_task_keys)
        
        if most_similar_task_keys != None:
            self.share_fkt[self.task_num] = most_similar_task_keys

        ### Update the task mask to share and omit the appropriate frozen subnetworks
        self.network.update_shared_mask(most_similar_task_keys, self.task_num)
        
        
        ### Set all omitted weights to 0 with the updated task mask
        self.network.apply_mask(self.task_num)


        
    """
    ##########################################################################################################################################
    Pruning Functions
    ##########################################################################################################################################
    """
    ### Goes through and calls prune_mask for each layer and stores the results
    ### Then applies the masks to the weights
    def prune(self):
        print('Pruning for dataset idx: %d' % (self.task_num))
        print('Pruning each layer by removing %.2f%% of values' % (100 * self.args.prune_perc_per_layer))
        one_dim_prune_mask = 0
        four_dim_prune_mask = 0
        
        for module_idx, module in enumerate(self.network.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                ### If this isn't a skip layer, prune as normal
                if module_idx not in self.skip_layers.keys() or self.args.arch != "modresnet18":
                    trainable_mask = utils.get_trainable_mask(module_idx, self.network.all_task_masks, self.task_num)
                  
                    ### Get the pruned mask for the current layer
                    pruned_mask = self.pruning_mask(module.weight.data.clone().detach(), trainable_mask, module_idx)

                    for module_idx2, module2 in enumerate(self.network.backupmodel.shared.modules()):
                        if module_idx == module_idx2:
                            module2.weight.data[pruned_mask.eq(1)] = 0.0
                            
                    # Set pruned weights to 0.
                    module.weight.data[pruned_mask.eq(1)] = 0.0
                    self.network.all_task_masks[self.task_num][module_idx][pruned_mask.eq(1)] = 0

                    ### Store the prune mask to make sure its reused for the appropriate skip junction which will be re-added to the network, such that pruned filters match
                    ### Note: We store the mask of the layer with which the skip layers activations are added, which is always the preceding conv2d layer in resnet18
                    if self.args.arch == "modresnet18":
                        one_dim_prune_mask =  torch.amax(pruned_mask, dim=(1,2,3))
                    
        
                ### For skip layers, reuse the prune mask from the layer that they'll be added together with, expanded to the appropriate weight shape
                ### This is done to avoid re-adding residuals into frozen feature maps, which would cause feature drift
                else:
                    four_dim_prune_mask = one_dim_prune_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(module.weight.data.shape)
                    module.weight.data[four_dim_prune_mask.eq(1)] = 0.0
                    self.network.all_task_masks[self.task_num][module_idx][four_dim_prune_mask.eq(1)] = 0
                    
                    for module_idx2, module2 in enumerate(self.network.backupmodel.shared.modules()):
                        if module_idx == module_idx2:
                            module2.weight.data[four_dim_prune_mask.eq(1)] = 0.0
                                    
            
        ### Once all pruning is done, we need to make sure that the weights going out of newly pruned filters are zeroed to avoid future issues when sharing subnetworks
        self.network.zero_outgoing_omitted(self.task_num)  

 
    def pruning_mask(self, weights, trainable_mask, layer_idx):
        """
            Ranks prunable filters by magnitude. Sets all below kth to 0.
            Returns pruned mask.
        """
        ### Setting to less-than will allow for fewer new weights to be frozen if frozen weights are deemed sufficient
        ###    With simple weight-based pruning though this should probably not be implemented
        
        """
            weight_magnitudes: 2D weight magnitudes
            trainable_mask: 2D boolean mask, True for trainable weights
            task_mask: 2D boolean mask, 1 for all weights included in current task (not omitted incoming weights)
        """

        if len(weights.size()) > 2 and self.args.prune_method == "structured":
            weight_magnitudes = torch.mean(weights.abs(), axis=(2,3))
            ### This checks on a connection basis, not entire filters
            trainable_mask = trainable_mask.eq(1).all(dim=3).all(dim=2)
            task_mask = self.network.all_task_masks[self.task_num][layer_idx].eq(1).any(dim=3).any(dim=2)
        else:
            print("Weight size <= 2")
            weight_magnitudes = weights.abs()
            trainable_mask = trainable_mask.eq(1)
            task_mask = self.network.all_task_masks[self.task_num][layer_idx].eq(1)

        weights_sum = weight_magnitudes.sum(dim=1)


        ### Calculate the number of incoming weights that haven't been omitted for each filter prior to averaging
        weights_num = trainable_mask.long().sum(dim=1)
        ### This is the average weight values for ALL filters in current layer (not counting omitted incoming weights)
        current_task_weight_averages = torch.where(weights_num.gt(0), weights_sum/weights_num, weights_sum)
        # current_task_weight_averages = weights_simple_mean
        ### This is done to further mask out any frozen filters, since we want to set a pruning threshold based on new features
        included_weights = current_task_weight_averages[trainable_mask.any(dim=1)]

        prune_sparsity = self.args.prune_perc_per_layer
        ### If no past tasks could be shared then we want to afford more filters for the current task
        if len(self.share_fkt[self.task_num]) < 1: 
            prune_sparsity -= self.args.share_sparsity_offset
            
        
        ### Now we use our masked set of averaged 1D feature weights to get a pruning threshold
        prune_rank = round(prune_sparsity * included_weights.size(dim=0))

        prune_value = included_weights.view(-1).cpu().kthvalue(prune_rank)[0]

        ### Now that we have the pruning threshold, we need to get a mask of all filters who's average incoming weights fall below it        
        weights_to_prune = current_task_weight_averages.le(prune_value)

            
        prune_mask = torch.zeros(weights.shape)
        ### The frozen mask has 1's indicating frozen weight indices
        if len(weights.size()) > 2 and self.args.prune_method == "structured":
            # print("Using structured prune mask expand")
            expanded_prune_mask = weights_to_prune.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(prune_mask.shape)
            prune_mask[expanded_prune_mask]=1        
        else:
            expanded_prune_mask = weights_to_prune.unsqueeze(-1).expand(prune_mask.shape)
            prune_mask[expanded_prune_mask]=1        
                        
        ### Prevent pruning of any non-trainable weights (frozen or omitted)
        prune_mask[trainable_mask.eq(0)]=0
    
    
    
            
        ### Check how many weights are being chosen for pruning
        print('Layer #',layer_idx, ' pruned ',prune_mask.eq(1).sum(), '/', prune_mask.numel() ,
                '(',100 * prune_mask.eq(1).sum() / prune_mask.numel(),'%%)', ' (Total in layer: ', weights.numel() ,')')
            #   (layer_idx, prune_mask.eq(1).sum(), prune_mask.numel(), 100 * prune_mask.eq(1).sum() / prune_mask.numel(), weights.numel()))

        return prune_mask
        
        
        
    """
    ##########################################################################################################################################
    Train and Evaluate Functions
    ##########################################################################################################################################
    """

    def eval(self, verbose=True, mode="test", applymask=True):
        """Performs evaluation."""
        if mode == 'test':
            dataloader=self.test_loader
        elif mode == 'val':
            dataloader=self.val_loader
        if verbose==True:
            print("Task number in Eval: ", self.task_num)
            print("Applying dataset mask for current dataset: ", self.task_num)

        if applymask==True:
            self.network.apply_mask(self.task_num)

        self.network.model.eval()


        error_meter = None
        for batch, label in dataloader:
            if self.args.cuda:
                batch = batch.cuda()
                label = label.cuda()
    
            output = self.network.model(batch)
    
            if error_meter is None:
                topk = [1]
                if output.size(1) > 5:
                    topk.append(5)
                error_meter = tnt.meter.ClassErrorMeter(topk=topk)
            error_meter.add(output.data, label)

        errors = error_meter.value()
        print('Top1 eval accuracy: %0.2f'%
            (100 - errors[0]))       
        self.network.model.train()
        return errors


    ### Train the model for the current task, using all past frozen weights as well
    def train(self, epochs, save=True, savename='', best_accuracy=0, finetune=False):

        """Performs training."""
        best_model_acc = best_accuracy
        best_model = copy.deepcopy(self.network.model.state_dict())
        val_acc_history = []
        patience = self.args.lr_patience
        if self.args.cuda:
            self.network.model = self.network.model.cuda()

        # Get optimizer with correct params.
        params_to_optimize = self.network.model.parameters()

        lr = self.args.lr
        lrmin = self.args.lr_min

        optimizer = optim.SGD(params_to_optimize, lr=lr, momentum=0.9, weight_decay=0.0, nesterov=True)

        loss = nn.CrossEntropyLoss()

        self.network.model.train()
        for idx in range(epochs):
            epoch_idx = idx + 1
            print('Epoch: ', epoch_idx, '       Learning rate:', optimizer.param_groups[0]['lr'], flush=True)
            
            for x, y in self.train_loader:
                if self.args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                x = Variable(x)
                y = Variable(y)
    
                # Set grads to 0.
                self.network.model.zero_grad()
        
                # Do forward-backward.
                output = self.network.model(x)
                loss(output, y).backward()

                # Set frozen param grads to 0.
                self.network.make_grads_zero(self.task_num)

                # Update params.
                optimizer.step()

        
            val_errors = self.eval(verbose=False, mode='val', applymask=False)
            val_accuracy = 100 - val_errors[0]  # Top-1 accuracy.
            val_acc_history.append(val_accuracy)



            # Save best model, if required.
            if save and best_model_acc < val_accuracy:
                print('Best model so far, Accuracy: %0.2f%% -> %0.2f%%' %
                      (best_model_acc, val_accuracy))

                best_model_acc = val_accuracy
                best_model = copy.deepcopy(self.network.model.state_dict())
                # self.network.update_backup(self.task_num)
                patience = self.args.lr_patience
                # utils.save_ckpt(self, savename)
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.args.lr_factor
                    if lr < lrmin:
                        break
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= self.args.lr_factor  
                    patience = self.args.lr_patience
                        
        self.network.model.load_state_dict(copy.deepcopy(best_model))
        ### Commit the changes to the weights to the backup composite model
        self.network.update_backup(self.task_num)

        test_errors = self.eval(verbose=False)
        test_acc = 100 - test_errors[0]
        print('Finished finetuning...')
        print('Best val/test accuracy: %0.2f%%, %0.2f%%' %
              (best_model_acc, test_acc))
        print('-' * 16)

        return test_acc, epoch_idx, val_acc_history




































        
    """
    ##########################################################################################################################################
    Testing Functions for Usefulness Metric Quantification. Used in supplementary experiments, not the actual IFC method with Pearson Correlation
    ##########################################################################################################################################
    """       
       

        
    ### Calculate the mutual information for a given pair of layers
    def calc_mi(self, p1_op, c1_op):
        mi_est = 0
        for cl in list(np.unique(self.labels.numpy())):
            p1_class = p1_op[self.labels == cl]
            c1_class = c1_op[self.labels == cl]
            label_weight = np.shape(p1_class)[0]/np.shape(p1_op)[0]

            act_parent = np.reshape(p1_class, (np.shape(p1_class)[0], -1))
            act_child = np.reshape(c1_class, (np.shape(c1_class)[0], -1))
  
            mi_est += label_weight * mi_estimator.EDGE(act_parent, act_child, normalize_epsilon=False, L_ensemble=1, stochastic=True)

        return mi_est
        

    
    def get_task_conns_metrics(self):  
        ### loop over all past tasks, current task hasn't been trained yet so calculating it wouldn't be informative
        ### Note: Because taskmask gives the incoming connections for a layer and connsdict gives the outgoing connections,
        ###       we need to offset the layer index by 1 and reshape the taskmask to properly index into connsdict
        self.calc_activations()
        taskconnsdict = {}
        for task in range(0,self.task_num):
            taskmask = self.network.all_task_masks[task]

            taskActs = copy.deepcopy(self.acts_dict)

            """ 
            Loops twice, once to apply masks to activations to form taskActs, 
            then again to calculate the metrics for each pair of connected layers
            """
            ### First loop, apply masks to all activations to only compute the metric for the filters used in the given task
            for key in self.ardict.keys():
                ### Use the task mask to mask for weights in the chosen task
                bool_mask = taskmask[key].eq(1)
                
                # print("boolmask for key ", key , " has shape: ", bool_mask.size())
                # print("Corresponding acts shape: ", taskActs[key].size())
                ### Need to convert to a shape of 2 to match the the connectivity dictionary shape            
                if bool_mask.dim() == 4:
                    ### True for all filters in the parent layer which have at least one incoming, non-zero weight in the current task
                    bool_mask = bool_mask.any(dim=3).any(dim=2).any(dim=1)

                taskActs[key] = taskActs[key][:,bool_mask.cpu()]


            ### Second loop, for each parent-child connection, compute and add the selected metric
            task_metric=0
            for i in range(0, len(self.parent_idxs)):
                parent_idx = self.parent_idxs[i]
                child_idx = self.child_idxs[i]

                # print("taskAct shape layer for calc: ", taskActs[parent_idx].size())
                layer_metric = 0
                if self.args.similarity_type == "hsic":
                    layer_metric = hsic_estimator.hsic_computation(taskActs[parent_idx], taskActs[child_idx])
                elif self.args.similarity_type == "cka":
                    layer_metric = hsic_estimator.cka_computation(taskActs[parent_idx], taskActs[child_idx])
                elif self.args.similarity_type == "mi":
                    layer_metric = self.calc_mi(copy.deepcopy(taskActs[parent_idx]).numpy(), copy.deepcopy(taskActs[child_idx]).numpy())

                task_metric += layer_metric

            taskconnsdict[task] = task_metric
            
        ### Returns task-specific network connectivity averages. This is the average connectivity over all weights used during a given task
        return taskconnsdict


    def get_task_acts(self):  
        self.acts_dict = {}
        self.calc_activations()
        taskactsdict = {}
        ### loop over all past tasks, current task hasn't been trained yet so calculating it wouldn't be informative
        for task in range(0,self.task_num):
            taskmask = self.network.all_task_masks[task]
            task_acts_ave = 0
            
            act_idxs = list(self.ardict.keys())
            for idx in act_idxs:

                ### Use the task mask to mask for weights in the chosen task
                bool_mask = taskmask[idx].eq(1)
                
                ### Need to convert to a shape of 2 to match the the connectivity dictionary shape            
                if bool_mask.dim() == 4:
                    ### True for all filters in the parent layer which have at least one outgoing, non-zero weight in the current task
                    bool_mask = bool_mask.any(dim=3).any(dim=2)
    
                ### We just want the acts from the filters which werent omitted. Acts aren't tied to weights, so no need to mask individual weights
                bool_mask = bool_mask.any(dim=1)
    

                ### Index into acts to get activations of given task's filters. Need to transpose acts to get them into [#filters, #samples] shape
                task_acts = self.acts_dict[idx].transpose(0,1)[bool_mask.cpu()]

                ### Not standardizing the activations since I want them to be unaffected by the various unmasked subnetworks, and standardizing them within a given task's subnetwork
                ###     would always just lead to a mean of 0 which wouldn't allow for selection. 
                
                ### For now I just want to use the average conns over the layer
                if task_acts.numel() > 0:
                    if self.args.similarity_method == 'linear':
                        task_acts_ave += torch.mean(task_acts.abs())
                    elif self.args.similarity_method == 'squared':
                        task_acts_ave += torch.mean(torch.square(task_acts))
                    else:
                        print("Incorrect argument for similarity method")
                          
            taskactsdict[task] = task_acts_ave
            
        ### Returns task-specific network connectivity averages. This is the average connectivity over all weights used during a given task
        return taskactsdict

       
       
       
       
       
       
       
    ### Gets a dictionary of all connectivity values throughout the network for all input samples, for visualization and quantification
    def get_dict_task_conns(self):  
        connsdict = self.calc_corr()
        taskconnsdict = {}

        ### loop over all past tasks, current task hasn't been trained yet so calculating it wouldn't be informative
        ### Note: Because taskmask gives the incoming connections for a layer and connsdict gives the outgoing connections,
        ###       we need to offset the layer index by 1 and reshape the taskmask to properly index into connsdict. This is the reason for parent_idx and child_idx

        for task in range(0,self.task_num):
            taskmask = self.network.all_task_masks[task]
            task_conn_ave = 0
            taskconnsdict[task] = {}
            for i in range(0, len(self.child_idxs)):

                taskconnsdict[task][i] = {}

                parent_idx = self.parent_idxs[i]
                child_idx = self.child_idxs[i]

                ### Use the task mask to mask for weights in the chosen task
                bool_mask = taskmask[child_idx].eq(1)
                
                ### Need to convert to a shape of 2 to match the the connectivity dictionary shape            
                if bool_mask.dim() == 4:
                    ### True for all filters in the parent layer which have at least one outgoing, non-zero weight in the current task
                    bool_mask = bool_mask.any(dim=3).any(dim=2)
    
                bool_mask = bool_mask.transpose(0,1)
    
                        
                ### Index into connectivity dictionary based on current child layer index. Works for layers that only appear once in child_idxs
                task_conns = torch.from_numpy(connsdict[i])[bool_mask.cpu()]

                if self.args.similarity_method == 'linear':
                    taskconnsdict[task][i] = task_conns
                elif self.args.similarity_method == 'squared':
                    taskconnsdict[task][i] = torch.square(task_conns)
                else:
                    print("Incorrect argument for similarity method")
                    
        ### Returns task-specific network connectivity averages. This is the average connectivity over all weights used during a given task
        return taskconnsdict






    ### Depending on our chosen measure, we calculate connectivity between filters in subsequent layers
    def get_dict_task_metrics(self, similarity_type='mi'):  
        self.calc_activations()
        taskconnsdict = {}
        for task in range(0,self.task_num):
            taskmask = self.network.all_task_masks[task]
            taskconnsdict[task] = {}

            taskActs = copy.deepcopy(self.acts_dict)

            """ 
            Loops twice, once to apply masks to activations to form taskActs, 
            then again to calculate the metrics for each pair of connected layers
            """
            ### First loop, apply masks to all activations to only compute the metric for the filters used in the given task
            for key in self.ardict.keys():
                ### Use the task mask to mask for weights in the chosen task
                bool_mask = taskmask[key].eq(1)
                
                ### Need to convert to a shape of 2 to match the the connectivity dictionary shape            
                if bool_mask.dim() == 4:
                    ### True for all filters in the parent layer which have at least one incoming, non-zero weight in the current task
                    bool_mask = bool_mask.any(dim=3).any(dim=2).any(dim=1)
                # bool_mask = bool_mask.transpose(0,1)
                
                taskActs[key] = taskActs[key][:,bool_mask.cpu()]
                

            ### Second loop, for each parent-child connection, compute and add the selected metric
            task_metric=0
            for i in range(0, len(self.parent_idxs)):
                parent_idx = self.parent_idxs[i]
                child_idx = self.child_idxs[i]

                # print("taskAct shape layer for calc: ", taskActs[parent_idx].size())
                layer_metric = 0
                if similarity_type == "hsic":
                    layer_metric = hsic_estimator.hsic_computation(taskActs[parent_idx], taskActs[child_idx])
                elif similarity_type == "cka":
                    layer_metric = hsic_estimator.cka_computation(taskActs[parent_idx], taskActs[child_idx])
                elif similarity_type == "hsic_cka":
                    layer_metric = {}
                    layer_metric['hsic'], layer_metric['cka'] = hsic_estimator.hsic_and_cka_computation(taskActs[parent_idx], taskActs[child_idx])
                elif similarity_type == "hsic_cka_layer":
                    layer_metric = {}
                    layer_metric['hsic'], layer_metric['cka'] = hsic_estimator.hsic_and_cka_layers_computation(taskActs[parent_idx], taskActs[child_idx])
                elif similarity_type == "mi":
                    layer_metric = self.calc_mi(copy.deepcopy(taskActs[parent_idx]).numpy(), copy.deepcopy(taskActs[child_idx]).numpy())
                # print("Layer ", parent_idx, " metric value: ", layer_metric, flush=True)
                taskconnsdict[task][i] = layer_metric

        return taskconnsdict




















    ### Gets a dictionary of all activation values throughout the network for all input samples, for visualization and quantification
    def get_dict_task_acts(self):  
        self.acts_dict = {}
        self.calc_activations()
        taskactsdict = {}

        for task in range(0,self.task_num):
            taskmask = self.network.all_task_masks[task]
            task_acts_ave = 0
            taskactsdict[task] = {}
    
            act_idxs = list(self.ardict.keys())
            for i in range(0, len(act_idxs)):
                taskactsdict[task][i] = {}
                parent_idx = act_idxs[i]

                ### Use the task mask to mask for weights in the chosen task
                bool_mask = taskmask[parent_idx].eq(1)
                
                ### Need to convert to a shape of 2 to match the the connectivity dictionary shape            
                if bool_mask.dim() == 4:
                    ### True for all filters in the parent layer which have at least one outgoing, non-zero weight in the current task
                    bool_mask = bool_mask.any(dim=3).any(dim=2)
    
                ### We just want the acts from the filters which werent omitted. Acts aren't tied to weights, so no need to mask individual weights
                bool_mask = bool_mask.any(dim=1)
    
                ### Index into acts to get activations of given task's filters. Need to transpose acts to get them into [#filters, #samples] shape
                task_acts = self.acts_dict[parent_idx].transpose(0,1)[bool_mask.cpu()]

                ### Not standardizing the activations since I want them to be unaffected by the various unmasked subnetworks, and standardizing them within a given task's subnetwork
                ###     would always just lead to a mean of 0 which wouldn't allow for selection. 
                
                ### For now I just want to use the average conns over the layer
                if self.args.similarity_method == 'linear':
                    taskactsdict[task][i] = task_acts
                elif self.args.similarity_method == 'squared':
                    taskactsdict[task][i] = torch.square(task_acts)
                else:
                    print("Incorrect argument for similarity method")
                        
        ### Returns task-specific network connectivity averages. This is the average connectivity over all weights used during a given task
        return taskactsdict

       
       
       
       
       
       
       


