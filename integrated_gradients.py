import torch
from tqdm import tqdm


def get_baseline(target, mode='target'):
    assert target is not None
    if mode == 'target':
        baseline = target
    elif mode == 'random':
        baseline = torch.randn_like(target)
    elif mode == 'zero':
        baseline = torch.zeros_like(target)
    else:
        raise RuntimeError("unsupported mode: ", mode)
    return baseline


def ig_clf(F, x, x_, n=50, custom='max', **kwargs):
    # get gradients
    scaled_inputs = [torch.nn.Parameter(x_ + (float(i) / n) * (x - x_), requires_grad=True) for i in range(0, n + 1)]
    gradients = []

    # compute gradient wrt a particular class score
    for i in tqdm(range(len(scaled_inputs)), desc="computing gradients..."):
        inp = scaled_inputs[i]
        out_dict = F(inp)
        scores = out_dict['scores'] # [1, S]
        F.zero_grad()
        scores[0, kwargs['cat_id']].backward(retain_graph=True)
        gradient = inp.grad.clone()
        inp.grad.zero_()    # clear gradients
        gradients.append(gradient)
    gradients = sum(gradients) / len(gradients)

    # compute integrated gradients
    ig = (x - x_) * gradients
    #ig[ig<0]=0
    return ig


def ig_seg(F, x, x_, n=50, **kwargs):
    #TODO: add target label
    #TODO: per-entry score
    '''
    computes integrated gradients
    F: the model
    x: the input
    x_: the baseline
    n: number of steps for integration

    :param kwargs:
        -'elem_wise': computes the averged gradient flow of the entire input object and element-wise
                      multiplies it with IG (for all points or just once? TODO!)
    '''
    # get gradients
    scaled_inputs = [torch.nn.Parameter(x_ + (float(i) / n) * (x - x_), requires_grad=True) for i in range(0, n + 1)]
    gradients = []

    # optionally
    if kwargs['elem_wise']:
        obj_grads = []
        _, output = F(x)
        for c in range(output.shape[2]):
            output[0, kwargs['part_id'], c].backward(retain_graph=True)
            obj_grads.append(x.grad.clone())
            x.grad.zero_()
        # for all pts?

    obj_grad = sum(obj_grads)/len(obj_grads)
    F.zero_grad()



    # compute gradient wrt a particular part score
    for i in tqdm(range(len(scaled_inputs))):
        inp = scaled_inputs[i]
        _, output = F(inp)  # [1, 50, N]
        curr_gd = []


        for p in tqdm(range(output.shape[2])): # compute wrt all points
            output[0, kwargs['part_id'], p].backward(retain_graph=True)
            gradient = inp.grad.clone()
            gradient[gradient<0] = 0
            inp.grad.zero_()
            curr_gd.append(gradient)

        F.zero_grad()
        gradients.append(sum(curr_gd)/len(curr_gd))
    gradients = sum(gradients)

    # compute integrated gradients
    ig = (x - x_) * gradients * obj_grad if kwargs['elem_wise'] else 1
    return ig
