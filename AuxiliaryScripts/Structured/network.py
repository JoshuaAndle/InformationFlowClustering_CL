import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
from typing import Optional

from AuxiliaryScripts import clmodels
from AuxiliaryScripts.Structured import utils


"""
The Network class is responsible for low-level functions which manipulate the model, such as training, evaluating, or selecting the classifier layer
"""
class Network():
    def __init__(self, args, pretrained="True"):
        self.args = args
        self.arch = args.arch
        self.cuda = args.cuda

        self.preprocess = None
        self.model = None
        self.pretrained = pretrained


        self.all_task_masks = {}
        self.classifiers, self.classifier_masks = {}, {}

        if self.arch == "modresnet18":
            self.model = clmodels.modifiedresnet18(nf=args.num_filters)
        elif self.arch == "vgg16":
            self.model = clmodels.vgg16()   
        else:
            print("Unsupported architecture {} in Network init".format(self.arch))
            raise ValueError


        if self.cuda:
            self.model = self.model.cuda()
    
        """
            When to use the backup statedict:
                Update it at the end of a task, once all the weights for the task are finalized. 
                    Use get_trainable_mask() to overwrite only indices of the newly trained/frozen weights. No other weights need be updated since they were frozen
                Read it at the start of a task, resetting the network masking to include all previously frozen weights.
                Note:
                    There should be no reason to need to load the full network backup midtask, which would be equivalent to essentially undoing the sharing/omitting decision
        """
        
        self.backupmodel = copy.deepcopy(self.model).cuda()
        ### This is a very inefficient way to reinitialize pruned weights. It can almost certainly be done without storing a full copy of the model, 
        ###   but this is simpler for the purpose of this work. At the start of a new task all newly trainable weights will be reloaded from this statedict
        self.initialmodel = copy.deepcopy(self.model).cuda()
    
        
        
    
        
    """
    ##########################################################################################################################################
    Functions for Weight, Gradient, and Classifier Changes
    ##########################################################################################################################################
    """


    ### Add a new classifier layer for a given task
    def add_dataset(self, dataset:int, num_classes:int):
        if dataset not in self.classifiers.keys():
            if self.arch in ["modresnet18"]:
                self.classifiers[dataset] = (nn.Linear(self.args.num_filters*8, num_classes))
            elif self.arch in ['vgg16']:
                self.classifiers[dataset] = (nn.Linear(4096, num_classes))
                
            self.classifier_masks[dataset] = torch.ByteTensor(self.classifiers[dataset].weight.data.size()).fill_(1)


    ### Set the networks classifier layer to one of the available tasks'
    def set_dataset(self, dataset:int):
        assert dataset in self.classifiers.keys(), "Dataset not in classifiers for set_dataset"
        self.model.classifier = self.classifiers[dataset]
        self.backupmodel.classifier = self.classifiers[dataset]




    """
    Need to adjust make_grads_zero to also ensure that all incoming weights to a frozen filter are zeroed, and all weights out of an omitted filter are zeroed as well
    """
    ### Set all frozen and pruned weights' gradients to zero for training
    def make_grads_zero(self, tasknum:int):

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                trainable_mask = utils.get_trainable_mask(module_idx, self.all_task_masks, tasknum)
                
                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    module.weight.grad.data[trainable_mask.eq(0)] = 0
                if tasknum>0 and module.bias is not None:
                    module.bias.grad.data.fill_(0)



    ### Reset the previously pruned weight values for training the current task
    def reinit_statedict(self, tasknum:int):
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                ### Get indices of all trainable weights to reinitialize, with the frozen_filters used to mask out any frozen weights
                new_weights = utils.get_trainable_mask(module_idx, self.all_task_masks, tasknum)
                frozen_filters = utils.get_frozen_mask(module.weight.data, module_idx, self.all_task_masks, tasknum)
                if len(frozen_filters.shape) > 2:
                    frozen_filters = frozen_filters.eq(1).any(dim=3).any(dim=2).any(dim=1)
                else:
                    frozen_filters = frozen_filters.eq(1).any(dim=1)
                new_weights[frozen_filters] = 0

                for module_idx2, module2 in enumerate(self.initialmodel.shared.modules()):
                    if module_idx2 == module_idx:
                        module.weight.data[new_weights.eq(1)] = module2.weight.data.clone()[new_weights.eq(1)]




    ### Just checks how many parameters per layer are now 0 post-pruning
    def check(self, verbose:bool=False):
        print('Checking...')
        for layer_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_zero = weight.view(-1).eq(0).sum()
                if len(weight.shape) > 2:
                    filter_mask = torch.abs(weight).le(0.000001).all(dim=3).all(dim=2).all(dim=1)
                else:
                    filter_mask = torch.abs(weight).le(0.000001).all(dim=1)
                
                num_filters = filter_mask.numel()
                num_pruned_filters = filter_mask.view(-1).sum()


                if verbose:
                    print('Layer #%d: Pruned Weights %d/%d (%.2f%%), Pruned Filters %d/%d (%.2f%%)' %
                          (layer_idx, num_zero, num_params, 100 * num_zero / num_params, num_pruned_filters, num_filters, 100 * num_pruned_filters / num_filters))










        
    """
    ##########################################################################################################################################
    Functions for Handling Masks
    ##########################################################################################################################################
    """

    ### For all omitted filters we want to zero any weights which depend on them to prevent interference if they become unmasked for later sharing
    def zero_outgoing_omitted(self, tasknum:int):
        ### A mask of all weights connected to omitted filters in the parent layer
        omitmask = utils.get_omitted_outgoing_mask(self.all_task_masks, tasknum, self.model, self.args.arch)

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                
                ### Set the omitted weights to 0. This can include frozen or trainable weights
                module.weight.data[omitmask[module_idx].eq(0)] = 0.0
                
                ### Then apply the changes on ONLY THE TRAINABLE OMITTED WEIGHTS to the backup
                ### Note: Otherwise it will reset all of the weights in omitted frozen subnetworks as well
                self.update_backup(tasknum)
                
                ### delist the omitted weights from the current task mask once they've been reset to prevent training
                self.all_task_masks[tasknum][module_idx][omitmask[module_idx].eq(0)] = 0

        
        ### Get the flattened representation of the final shared layer for masking the classifier
        if self.args.arch == "modresnet18":
            final_omitted = self.all_task_masks[tasknum][83]
            ### Note: This works because the adaptive pooling at the end of the feature extractor results in a 1x1 feature map per filter, so here we flatten to 1 value per filter
            final_omitted_flattened = final_omitted.eq(0).all(dim=3).all(dim=2).all(dim=1).flatten()
        elif self.args.arch == "vgg16":
            final_omitted = self.all_task_masks[tasknum][49]
            final_omitted_flattened = final_omitted.eq(0).all(dim=1).flatten()
       

            
        ### Get all filters that are omitted for the current task and flatten the mask for use with the subsequent classifier layer
        ### This is likely redundant but just to be safe applying to both the list and set model classifier
        self.model.classifier.weight.data[:,final_omitted_flattened] = 0.0
        self.classifiers[tasknum].weight.data[:,final_omitted_flattened] = 0.0
        self.classifier_masks[tasknum][:,final_omitted_flattened] = 0



    ### For all omitted filters we want to zero any weights which depend on them to prevent interference if they become unmasked for later sharing
    def zero_shared_omitted(self, tasknum:int):
        ### A mask of all weights connected to omitted filters in the parent layer
        sharedmask = utils.get_shared_outgoing_mask(self.all_task_masks, tasknum, self.model, self.args.arch)

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                
                ### Set the omitted weights to 0. This can include frozen or trainable weights
                module.weight.data[omitmask[module_idx].eq(0)] = 0.0
                
                ### Then apply the changes on ONLY THE TRAINABLE OMITTED WEIGHTS to the backup
                ### Note: Otherwise it will reset all of the weights in omitted frozen subnetworks as well
                self.update_backup(tasknum)
                
                ### delist the omitted weights from the current task mask once they've been reset to prevent training
                self.all_task_masks[tasknum][module_idx][omitmask[module_idx].eq(0)] = 0

        
        ### Get the flattened representation of the final shared layer for masking the classifier
        if self.args.arch == "modresnet18":
            final_omitted = self.all_task_masks[tasknum][83]
            ### Note: This works because the adaptive pooling at the end of the feature extractor results in a 1x1 feature map per filter, so here we flatten to 1 value per filter
            final_omitted_flattened = final_omitted.eq(0).all(dim=3).all(dim=2).all(dim=1).flatten()
        elif self.args.arch == "vgg16":
            final_omitted = self.all_task_masks[tasknum][49]
            final_omitted_flattened = final_omitted.eq(0).all(dim=1).flatten()

            
        ### Get all filters that are omitted for the current task and flatten the mask for use with the subsequent classifier layer
        ### This is likely redundant but just to be safe applying to both the list and set model classifier
        self.model.classifier.weight.data[:,final_omitted_flattened] = 0.0
        self.classifiers[tasknum].weight.data[:,final_omitted_flattened] = 0.0
        self.classifier_masks[tasknum][:,final_omitted_flattened] = 0





    ### Makes the taskmask for a newly encountered task, note that the masks are stored sequentially rather than by name, under the assumption past tasks aren't re-encountered during training. This can be changed if needed however
    def make_taskmask(self, tasknum:int):
        ### Creates the task-specific mask during the initial weight allocation
        task_mask = {}
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = torch.ByteTensor(module.weight.data.size()).fill_(1)
                layer_mask = layer_mask.cuda()
                task_mask[module_idx] = layer_mask

        ### Initialize the new tasks' inclusion map with all 1's
        self.all_task_masks[tasknum] = task_mask
        
        print("Exiting finetuning mask")

       
       


    """
        Mask manipulation can be done by the combination of three functions:
        1. update_backup: After editing weights in a subnetwork, this commits the changes to the composite backup model
        2. unmask_network: When changing subnetworks, this unmasks all weights, reverting them to their frozen values from self.backupmodel
        3. apply_mask: Apply masking for a given subnetwork by zeroing all omitted weights
    """

    ### This just reloads the backup composite network
    def unmask_network(self):
        self.model.shared = copy.deepcopy(self.backupmodel.shared).cuda()             
       
       
    ### Done after training or finetuning. Updates the backup model to reflect changes in the model from training, pruning, or merging
    ### Note that this only works because we also update the backup model when removing weights from the taskmask, as during pruning. Otherwise this wouldnt update those weights
    def update_backup(self, tasknum:int = -1):
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layermask = self.all_task_masks[tasknum][module_idx]

                for module_idx2, module2 in enumerate(self.backupmodel.shared.modules()):
                    if module_idx2 == module_idx:
                        module2.weight.data[layermask.eq(1)] = module.weight.data.clone()[layermask.eq(1)]

       
    ### Applies appropriate mask to recreate given task's model for inference
    def apply_mask(self, tasknum:int):
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                module.weight.data[self.all_task_masks[tasknum][module_idx].eq(0)] = 0.0


    def update_shared_mask(self, most_similar_task_keys:Optional[list[int]], tasknum:int):
        print("Most similar task keys:", most_similar_task_keys, " for current task number: ", tasknum)
        shared_count = 0
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                new_weights = utils.get_trainable_mask(module_idx, self.all_task_masks, tasknum)
             
                frozen_filters = utils.get_frozen_mask(module.weight.data, module_idx, self.all_task_masks, tasknum)
                if len(frozen_filters.shape) > 2:
                    frozen_filters = frozen_filters.eq(1).any(dim=3).any(dim=2).any(dim=1)
                else:
                    frozen_filters = frozen_filters.eq(1).any(dim=1)
                    
                new_weights[frozen_filters] = 0
                
                ### This will omit all frozen weights
                self.all_task_masks[tasknum][module_idx] = new_weights
                
                
                #*# Currently omits entire layers, but can also use zero_shared_omitted instead to only omit the new connections from forming
                if shared_count < self.args.shared_layers or self.args.shared_layers == -1:
                    shared_count += 1
                    ### If any tasks were similar, re-add their masks to the current task mask
                    if most_similar_task_keys != None:
                        for t in most_similar_task_keys:
                            print("Sharing layer: ", module_idx, " from task: ", t)  
                            ### This is so that for manually shared keys we only attempt to share the ones that have already been trained on
                            if tasknum > t:
                                shared_weights = self.all_task_masks[int(t)][module_idx].clone().detach()
                                ### This will omit any weights which weren't used in the task being shared while keeping all trainable weights
                                self.all_task_masks[tasknum][module_idx] = torch.max(new_weights, shared_weights)
                
        ### Account for any outgoing weights from omitted filters
        self.zero_outgoing_omitted(tasknum)      
        
        