import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxFusion(nn.Module):
    def __init__(self, num_inputs, weights=None, learnable=False):
        """
        Softmax 加权融合模块，支持张量数组输入 + 自定义初始权重。

        Args:
            num_inputs (int): 要融合的张量数量。
            weights (list or Tensor, optional): 初始权重向量（未归一化），长度应为 num_inputs。
            learnable (bool): 是否让权重可学习。
        """
        super(SoftmaxFusion, self).__init__()
        self.learnable = learnable

        if weights is not None:
            assert len(weights) == num_inputs, "weights 长度必须等于 num_inputs"
            initial_weights = torch.tensor(weights, dtype=torch.float32)
        else:
            initial_weights = torch.ones(num_inputs, dtype=torch.float32)

        if learnable:
            self.weights = nn.Parameter(initial_weights)
        else:
            self.register_buffer('weights', initial_weights)

    def forward(self, input_list):
        assert isinstance(input_list, (list, tuple)), "输入必须是张量列表"
        assert all(t.shape == input_list[0].shape for t in input_list), "所有张量形状必须一致"

        weights = F.softmax(self.weights, dim=0)
        output = sum(w * x for w, x in zip(weights, input_list))
        return output
