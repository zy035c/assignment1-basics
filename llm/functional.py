import torch


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    # apply softmax to the i-th dimension of the input
    # tensor. The output tensor should have the same shape as the input tensor, but its i-th dimension will
    # now have a normalized probability distribution. Use the trick of subtracting the maximum value in
    # the i-th dimension from all elements of the i-th dimension to avoid numerical stability issues.
    x2 = x - x.max(dim=i, keepdim=True).values
    x3 = torch.exp(x2)
    return x3 / x3.sum(dim=i, keepdim=True)
