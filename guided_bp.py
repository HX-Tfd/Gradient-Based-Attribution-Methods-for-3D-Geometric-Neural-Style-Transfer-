"""
Modified from:

@author: Utku Ozbulak - github.com/utkuozbulak

Tested with PointNet
"""
import torch
from torch.nn import ReLU, MSELoss
from tqdm import tqdm

import shapenet_constants


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image

       eval: internally puts model in evaluation model if set to true
       hook: update ReLUs if set to true
    """
    def __init__(self, model, eval=False, hook=True, hook_first=True):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # masked GBP features
        self.masked_features = []
        self.hooks = []

        # gradient of intermediate layers
        self.inter_gradients = []

        if eval:
            self.model.eval()
        if hook:
            self.update_relus()
        if hook_first:
            self.hook_first()

    # hook first layer
    def hook_first(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        m = self.model._modules.values()
        first_smlp = list(list(m)[0].modules())[0][0]
        first_conv = list(first_smlp.layers.named_children())[0][1]
        first_conv.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates all relu activation functions (backbone + classifier) so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function_features(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output (features are not to be copied)
            features = self.forward_relu_outputs[-1]
            corresponding_forward_output = features.clone().detach() # copy of features
            corresponding_forward_output[corresponding_forward_output > 0] = 1 # mask of positive fwd responses

            # keep non-negative gradient responses and non-negative forward outputs
            modified_grad_out = torch.nn.Parameter(corresponding_forward_output.clone() * torch.clamp(grad_in[0], min=0.0))
            modified_grad_out[modified_grad_out > 0] = 1 # mask of positive bwd responses
            self.masked_features.append(features*modified_grad_out)
            self.masked_features[-1].requires_grad = True

            del self.forward_relu_outputs[-1]
            return (modified_grad_out,)

        def relu_backward_hook_function_grad_only(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output (features are not to be copied)
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            self.inter_gradients.append(modified_grad_out)

            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass (features)
            """
            self.forward_relu_outputs.append(ten_out)

        # only hook the backbone
        for _, module in self.model._modules.items():
            for l in module.modules():
                if isinstance(l, ReLU):
                    self.hooks.append(l.register_full_backward_hook(relu_backward_hook_function_features))
                    self.hooks.append(l.register_forward_hook(relu_forward_hook_function))

    def generate_gradients(self, input_obj, target):
        # self.model.zero_grad() #??
        _, scores = self.model(input_obj)
        # Target class for backprop
        scores.backward(gradient=target)
        # of shape [3, num_samples]
        self.gradients = input_obj.grad[0, 0:3, :]
        return self.gradients

    def get_gradients(self):
        grads = self.gradients.clone()
        inter_grads = self.inter_gradients
        return grads, inter_grads[::-1]

    def generate_gradients_parts(self, input_obj, target_part_scores, content_part_id, custom=None, **kwargs):
        """
        masking can be one of the following:
            'max': uses the maximum value of the part scores for masking
            'max_n': uses the maximum n values of the part scores for masking, need to specify n
            'id': uses the part score(s) corresponding to its part id(s) for masking, need to specify id

        The input shape is [1, num_parts, num_points]
        """
        _, content_part_scores = self.model(input_obj)

        if target_part_scores is None:
            raise RuntimeError("target part scores should not be None!")

        else:
            if custom == 'max':
                val, id = torch.topk(target_part_scores, 1, dim=1)
                target = torch.zeros_like(target_part_scores)
                target = target.scatter(1, id, val)

            elif custom == 'max_n':
                val, id = torch.topk(target_part_scores, kwargs['n'], dim=1)
                target = torch.zeros_like(target_part_scores)
                target = target.scatter(1, id, val)

            elif custom == 'id':
                target_part_class = kwargs['id']
                if isinstance(target_part_class, int) or isinstance(target_part_class, torch.Tensor):
                    assert target_part_class in range(shapenet_constants.NUM_PART_CATEGORIES)
                # implementation for transferring one part to multiple parts, but not used for now
                elif isinstance(target_part_class, list):
                    for c in target_part_class:
                        assert c in range(shapenet_constants.NUM_PART_CATEGORIES)
                else:
                    raise TypeError("unsupported type for target part class: ", type(target_part_class))

                target_mask = torch.zeros_like(target_part_scores)
                target_mask[:, target_part_class, :] = 1
                target_part_scores = target_mask * target_part_scores
                target = target_part_scores

            else:
                target = target_part_scores

        # content_mask = torch.zeros_like(content_part_scores)
        # content_mask[:, content_part_id, :] = 1
        # content_part_scores = content_mask * content_part_scores

        loss = torch.nn.MSELoss()
        parts_loss = loss(content_part_scores,  target)
        print("part score loss:", parts_loss)
        parts_loss.backward()  # generates gbp-masked features

        for h in self.hooks:
            h.remove()

        return parts_loss