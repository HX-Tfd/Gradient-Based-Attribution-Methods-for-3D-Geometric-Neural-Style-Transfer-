# deep dream algorithm 2
def gradient_step_gbp(model, inp, layer_ids, part_ids, lr, iteration, channel_ids=None,
                        orig_inp=None,  target_part_scores=None, **kwargs):
    """
    :param feature0:
    :param inp: the input point cloud
    :param layer_ids: the ids of the layers to be used
    :param part_ids: the ids of the parts to be used
    :param kwargs:
            -'ascent': if True then performs gradient ascent, else gradient descent
    :return: the updated object {inp} and the GBP feature loss of the FIRST iteration
    """
    GBP = GuidedBackprop(model=model, eval=False, hook=True)
    GBP.generate_gradients_parts(inp,
                                  target_part_scores=target_part_scores,
                                  custom= None if target_part_scores is not None else 'id',
                                  id=None if target_part_scores is not None else part_ids)
    masked_features = GBP.masked_features
    masked_features = (masked_features[::-1])[4][:, channel_ids, :] #no/with global feature [layer][channel]
    print(masked_features)

    # use masked feature activations
    #TODO: fix masked feature during training? otherwise it keeps changing
    #TODO: seems that the gradients are not updating?
    #TODO: fix part of obj for opt.
    losses = []
    for masked_feature in masked_features:
        loss_item = torch.nn.MSELoss(reduction='mean')(masked_feature, torch.zeros_like((masked_feature)))
        losses.append(loss_item)
    loss = torch.sum(torch.stack(losses))
    model.zero_grad()
    print("deepdream loss:", loss)
    loss.backward()

    # normalize gradients
    grads = (inp.grad)[:, :3, :].detach().clone()
    grad_std = torch.std(grads)
    grad_mean = torch.mean(grads)
    grads = grads - grad_mean
    grads = grads / (grad_std + 1e-12)

    gradients = torch.zeros_like(inp)
    gradients[:, :3, :] = grads

    # inp_ = torch.zeros_like(inp)
    # for n in range(gradients.shape[2]):
    #     inp_[:, :, n] = inp[:, :, n] + lr * gradients[:, :, n] if \
    #         torch.argmax(part_scores[:, :, n]) in part_ids else inp[:, :, n]
    if kwargs['ascent']:
        inp = inp + lr * gradients
    else:
        inp = inp - lr * gradients
    if inp.grad is not None:
        inp.grad.zero_()

    # renormalize input points and subsample
    # if orig_inp is not None: # mixture
    #     orig_inp, inp_ = torch.squeeze(orig_inp), torch.squeeze(inp_)
    #     inp = torch.hstack((orig_inp, inp_))
    #     inp = nn.Parameter(inp.clone(), requires_grad=True)
    # else:
    inp = nn.Parameter(inp.clone(), requires_grad=True) # is this redundant?

    if (iteration+1) % visualize_every == 0:
        pcd = o3d.geometry.PointCloud()
        pts = (torch.squeeze(inp)).clone().transpose(0, 1)[:, :3].detach().cpu()
        pcd.points = o3d.utility.Vector3dVector((pts.numpy()))
        o3d.visualization.draw_geometries([pcd])

    # color attribution
    if (iteration+1) % visualize_every == 0:
        #pcd_color = color_attribution(pts=inp, mode='gradients', attribution=torch.squeeze(gradients, 0))
        pcd_color = color_attribution(pts=torch.squeeze(inp, 0), mode='displacement', attribution=torch.squeeze(orig_inp, 0))
        o3d.visualization.draw_geometries([pcd_color])

    return inp