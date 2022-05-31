import torch
import torch.nn as nn

from tqdm import tqdm
from utils import *

def gradCAM_clf(model, layer_id,  cat_id, inp, scores=None, features=None):
    # get layers
    conv_layers = []
    relu_layers = []

    # compute gradients of score for class {part_class} wrt the layer
    if scores is None or features is None:
        out_dict = model(inp)
        scores, features = out_dict['scores'], out_dict['features']
        print("predicted class: ", torch.argmax(scores))
    num_points = inp.shape[2]

    # compute for each point individually
    pooled_gradients = []

    # method 2
    for _ in tqdm(range(num_points), desc="computing gradients..."):
        pt_score = scores[:, cat_id]
        pt_score.backward(retain_graph=True) # bottleneck
        gradients = features['feature_4'].grad.clone() # the score of a point -> grad map of size [1, num_channels, num_points]
        features['feature_4'].grad.zero_()
        #gradients[gradients<0.] = 0

        grads = gradients.max(dim=1).values
        pooled_gradients.append(grads)

    return sum(pooled_gradients)/len(pooled_gradients)

#TODO: integrate this part into run_nst.py
#TODO: implement layers cohice as hyperparameters
#TODO: implement a model wrapper that summarises that available layers of the input model
def gradCAM_seg(model, inp, part_class, part_specific=False, layer_id=None):
    """
    :param model: the segmentation model to be tested
    :param inp: the input to the model
    :param part_class: the target part class' id (not a list, could be, need to implement)
    :param part_specific: if True, this function computes the gradients wrt the points belonging to the
                          predicted target part
    :param layers: the layers that will be used to generate gradient attributions (not implemented yet)
    :return:
    """
    # get layers
    conv_layers = []
    relu_layers = []

    for _, module in model._modules.items():
        for l in module.modules():
            if isinstance(l, nn.ReLU):
                relu_layers.append(l)
            elif isinstance(l, nn.Conv1d) or isinstance(l, nn.Conv2d):
                conv_layers.append(l)

    # register layer hooks
    bwd_gradients = []
    fwd_maps = []
    def backward_hook_function(module, ten_in, ten_out):
        bwd_gradients.append(ten_in) # only for one layers

    def forward_hook_function(module, ten_in, ten_out):
        fwd_maps.append(ten_out)

    layer = conv_layers[layer_id]
    #layer = relu_layers[7] # layer choice
    layer.register_full_backward_hook(backward_hook_function)
    conv_layers[layer_id-1].register_forward_hook(forward_hook_function) ###

    # compute gradients of score for class {part_class} wrt the layer
    # or optionally, take all points that belong to the target part
    feats, scores = model(inp)

    # compute for each point individually
    pooled_gradients = []

    # method 2, compute avg attribution
    if part_specific:
        part_ids = []
        for i in range(scores.shape[2]):
            if torch.max(scores[:, :, i], dim=1).indices[0] == part_class:
                part_ids.append(i)

        # for now only for one part
        for n in tqdm(range(len(part_ids)), desc="computing part specific gradients"):
            pt_score = scores[:, part_class, part_ids[n]]
            pt_score.backward(retain_graph=True)
            target_feature = bwd_gradients[-1][0]
            gradients = target_feature.clone()
            grads = gradients.max(dim=1).values
            grads[grads < 0] = 0
            pooled_gradients.append(grads)
            bwd_gradients = []

    else:
        num_points = scores.shape[2]
        for n in tqdm(range(num_points), desc="computing all gradients..."):
            pt_score = scores[:, part_class, n]
            pt_score.backward(retain_graph=True)
            target_feature = bwd_gradients[-1][0]
            gradients = target_feature.clone()
            gradients = gradients * torch.squeeze(fwd_maps[0][0], 0)  # HiResCAM modification
            grads = gradients.mean(dim=1)
            grads[grads < 0] = 0
            pooled_gradients.append(grads)
            bwd_gradients = []

    # only for now
    return sum(pooled_gradients)/len(pooled_gradients), feats, scores

