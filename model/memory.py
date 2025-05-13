import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import numpy as np

class MemoryLayer(Function):
    # def __init__(self, memory, alpha=0.01):
    #     super(MemoryLayer, self).__init__()
    #     self.memory = memory
    #     self.alpha = alpha

    @staticmethod
    def forward(self, inputs, targets, mem, alpha):
        alpha = np.array(alpha)
        alpha = torch.from_numpy(alpha).cuda()
        self.save_for_backward(inputs, targets, mem, alpha)
        # for x, y in zip(inputs, targets[0]):
        #     mem[y] = alpha * mem[y] + (1. - alpha) * x
        #     mem[y] /= mem[y].norm()
        outputs = inputs.mm(mem.t())
        return outputs

    @staticmethod
    def backward(self, grad_outputs):
        inputs, targets, mem, alpha = self.saved_tensors#
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(mem)
        for x, y in zip(inputs, targets[0]):
            mem[y] = alpha * mem[y] + (1. - alpha) * x
            mem[y] /= mem[y].norm()
        return grad_inputs, mem, None, None

class Memory(nn.Module):
    def __init__(self, num_features, num_classes, alpha=0.01):
        super(Memory, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = alpha

        self.mem = nn.Parameter(torch.zeros(num_classes, num_features), requires_grad=False)

    # @staticmethod
    def forward(self, inputs, targets, epoch=None):
        alpha = 0.5 * epoch / 60
        # alpha = 0.01
        mem = self.mem
        logits = MemoryLayer.apply(inputs, targets, mem, alpha)

        return logits


# class Exp(Function):
#     @staticmethod
#     def forward(ctx, i):
#         result = i
#         ctx.save_for_backward(result)
#         return result
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         result, = ctx.saved_tensors
#         return grad_output * result
#
#
# # Use it by calling the apply method:
# input = [2]
# input = torch.Tensor(input).long()
# output = Exp.apply(input)
# print(input)