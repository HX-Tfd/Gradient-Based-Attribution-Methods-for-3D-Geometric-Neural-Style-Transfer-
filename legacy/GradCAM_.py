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

        grads = gradients.max(dim=1).values# try max, .values, sum(dim=1)
        pooled_gradients.append(grads)
    #
    # #  method 2, use global pooling layer
    return sum(pooled_gradients)/len(pooled_gradients)


#TODO: implement layers cohice as hyperparameters
#TODO: implement a model wrapper that summarises that available layers of the input model
def gradCAM_seg(model, inp, part_class, part_specific=False
                , layer_id=None):
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
    # assert layer_id in range(num_modules)

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
    scores_ = nn.Softmax(dim=1)(scores)
    #print(scores[:, :, 0])

    # compute for each point individually
    pooled_gradients = []

    other_class=47
    other_part = []
    target_part_scores = []
    other_part_scores = []
    large_diff_scores_target = []
    large_diff_scores_other= []
    all_scores = {}
    #scores_ = scores

    # method 2, compute avg attribution
    if part_specific:
        part_ids = []
        for i in range(scores.shape[2]):
            if torch.max(scores_[:, :, i], dim=1).indices[0] == part_class: #or torch.max(scores[:, :, i], dim=1).indices[0] == 24:
                part_ids.append(i)
                target_part_scores.append((torch.max(scores_[:, :, i], dim=1).values[0],
                                           scores_[0, other_class, i]))
                # save point with large between-class confusion
                if nn.MSELoss()(target_part_scores[-1][0], target_part_scores[-1][1]) < 0.8:
                    large_diff_scores_target.append(i)
                    del part_ids[-1]

            elif torch.max(scores_[:, :, i], dim=1).indices[0] == other_class:
                other_part.append(i)
                other_part_scores.append((torch.max(scores_[:, :, i], dim=1).values[0],
                                          scores_[0, part_class, i]))
                if nn.MSELoss()(other_part_scores[-1][0], other_part_scores[-1][1]) < 0.9:
                    large_diff_scores_other.append(i)
                    #optional, delete points with large confusion
                    # del other_part[-1]
                    # del other_part_scores[-1]



        # print(scores[:, part_class, [part_ids[0], other_part[0]]])
        import matplotlib.pyplot as plt
        import numpy as np
        # score_diff = [nn.MSELoss()(x, y).detach().numpy() for (x, y) in target_part_scores]
        # colors = np.array([[1-s*s, 0, 0] for s in score_diff])
        # plt.scatter(np.linspace(0, len(score_diff), len(score_diff)), score_diff, color=colors)
        # plt.ylabel("squared distance of normalized scores")
        # plt.title("target = tabletop, other = legs")
        # plt.show()
        # score_diff = [nn.MSELoss()(x, y).detach().numpy() for (x, y) in other_part_scores]
        # colors = np.array([[1-s * s, 0, 0] for s in score_diff])
        # plt.scatter(np.linspace(0, len(score_diff), len(score_diff)), score_diff, color=colors)
        # plt.ylabel("squared distance of normalized scores")
        # plt.title("target = legs, other = tabletop")
        # plt.show()

        # for n in tqdm(range(len(large_diff_scores_target)), desc="computing gradients"): #original: part_ids
        #     pt_score = scores[:, part_class, large_diff_scores_target[n]]
        #     pt_score.backward(retain_graph=True)  # bottleneck
        #     target_feature = bwd_gradients[-1][0]
        #     gradients = target_feature.clone()
        #     grads = gradients.max(dim=1).values
        #     grads[grads < 0] = 0
        #     pooled_gradients.append(grads)
        #     bwd_gradients = []
        #
        # for n in tqdm(range(len(part_ids)), desc="computing gradients"):
        #     pt_score = scores[:, part_class, part_ids[n]]
        #     pt_score.backward(retain_graph=True)  # bottleneck
        #     target_feature = bwd_gradients[-1][0]
        #     gradients = target_feature.clone()
        #     grads = gradients.max(dim=1).values
        #     grads[grads < 0] = 0
        #     pooled_gradients.append(grads)
        #     bwd_gradients = []

        # ensemble target and outliers, PRECOMPUTE target score
        # for now only for one part
        for n in tqdm(range(len(part_ids)), desc="computing gradients"):
            pt_score = scores[:, part_class, part_ids[n]]
            pt_score.backward(retain_graph=True)  # bottleneck
            target_feature = bwd_gradients[-1][0]
            gradients = target_feature.clone()
            grads = gradients.max(dim=1).values
            grads[grads < 0] = 0
            pooled_gradients.append(grads)
            bwd_gradients = []
        l = sum(pooled_gradients) / len(pooled_gradients)
        g = torch.flatten(l)
        part_class_score = norm_unit(g)

        y_min, y_max, y_mean, y_std = [], [], [], []
        mags = []
        for id in range(0, 50):
            pooled_gradients = []
            if id == part_class:
                continue
            for n in tqdm(range(len(part_ids)), desc="computing gradients"):
                pt_score = scores[:, id, part_ids[n]]
                pt_score.backward(retain_graph=True)  # bottleneck
                target_feature = bwd_gradients[-1][0]
                gradients = target_feature.clone()
                grads = gradients.max(dim=1).values
                grads[grads < 0] = 0
                pooled_gradients.append(grads)
                bwd_gradients = []
            l = sum(pooled_gradients)/len(pooled_gradients)
            g = norm_unit(torch.flatten(l))
            mags.append(torch.norm(part_class_score-g))
        x = np.arange(1, 50)
        plt.bar(x, mags, color=[[1, 0.2, 0] if i in [0,1,2] else [0, 0.2, 1] for i in range(50)])
        plt.show()
        #     y_mean.append(torch.mean(g).numpy())
        #     y_min.append(torch.amin(g).numpy())
        #     y_max.append(torch.amax(g).numpy())
        #     y_std.append(torch.std(g).numpy())
        #     mags.append(torch.norm(l))
        # y_mean = np.array(y_mean)
        # y_std = np.array(y_std)
        # y_min = np.array(y_min)
        # y_max = np.array(y_max)
        #
        # x = np.arange(1, 51)
        # # plt.errorbar(x, y_mean, y_std, fmt='ok', lw=5)
        # # plt.errorbar(x, y_mean, [np.array(y_mean) - np.array(y_min), np.array(y_max) - np.array(y_mean)], fmt='.k',
        # #              ecolor='gray', lw=3)
        # plt.bar(x, mags)
        # plt.xlabel("part id")
        # plt.ylabel("gradient value (unnormalized)")
        # plt.show()



    else:
        num_points = scores.shape[2]
        for n in tqdm(range(num_points), desc="computing gradients..."):
            pt_score = scores[:, part_class, n]
            pt_score.backward(retain_graph=True)  # bottleneck
            target_feature = bwd_gradients[-1][0]
            gradients = target_feature.clone()
            gradients = gradients * torch.squeeze(fwd_maps[0][0], 0)  # HiResCAM modification
            grads = gradients.mean(dim=1)  # max(dim=1).values
            grads[grads < 0] = 0
            pooled_gradients.append(grads)
            bwd_gradients = []

    #  method 2, use global pooling layer
    print(sum(pooled_gradients))
    from functools import reduce
    #l = reduce(lambda a, b: torch.maximum(a, b), pooled_gradients)
    return sum(pooled_gradients)/len(pooled_gradients)


def misc():
    # plot gradient stats
    # plt.hist(inter_grads[-1].flatten().numpy(), bins=np.linspace(-0.5, 0.5, 101)) # all 50 * N
    # plt.show()
    # grad_mag = torch.norm(inter_grads[-1][:, :3, :], dim=1)
    # grad_mag = norm_p(grad_mag) # magnitude
    # plt.hist(grad_mag.flatten().numpy(), bins=np.linspace(0, 1, 101))

    y_min, y_max, y_mean, y_std = [], [], [], []
    n_layers = len(inter_grads)  # init needed
    for n in range(n_layers):
        ig = torch.flatten(inter_grads[n])
        y_mean.append(torch.mean(ig).numpy())
        y_min.append(torch.amin(ig).numpy())
        y_max.append(torch.amax(ig).numpy())
        y_std.append(torch.std(ig).numpy())
    y_mean = np.array(y_mean)
    y_std = np.array(y_std)
    y_min = np.array(y_min)
    y_max = np.array(y_max)

    x = np.arange(1, n_layers + 1)
    plt.errorbar(x, y_mean, y_std, fmt='ok', lw=5)
    plt.errorbar(x, y_mean, [np.array(y_mean) - np.array(y_min), np.array(y_max) - np.array(y_mean)], fmt='.k',
                 ecolor='gray', lw=3)
    # plt.ylim(-1, y_max.max())
    plt.xlabel("layers")
    plt.ylabel("gradient value")
    text = "max value: {:.4f}\n min value: {:.6f}\n mean: {:.6f}\n std: {:.6f}".format(y_max.max(), y_min.min(),
                                                                                       y_mean.mean(), y_std.mean())
    plt.annotate(text=text, xy=(1, 0.08))
    plt.show()


def layer_randomization_test():
    pass


def network_randomization_test():
    pass

