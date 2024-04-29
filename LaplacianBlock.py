import torch
import torch.nn as nn

class LaplacianAdaptiveAttention(nn.Module):
    def __init__(self, norm_axis, num_heads, num_laplacians, padding_value, mean_offset_init=0, eps=1e-8):
        super().__init__()
        if not isinstance(norm_axis, int):
            raise ValueError("norm_axis must be an integer.")
        if num_heads <= 0 or not isinstance(num_heads, int):
            raise ValueError("num_heads must be a positive integer.")
        if num_laplacians <= 0 or not isinstance(num_laplacians, int):
            raise ValueError("num_laplacians must be a positive integer.")

        self.norm_axis = norm_axis
        self.eps = eps
        self.num_heads = num_heads
        self.padding_value = padding_value
        self.num_laplacians = num_laplacians

        self.mean_offsets = nn.Parameter(torch.zeros(num_laplacians, dtype=torch.float)) # offsets are initialized with 0.
        self.c = nn.Parameter(torch.randn(num_laplacians, dtype=torch.float))

    def forward(self, x, return_attention_details=False):
        if x.dim() < 2:
            raise ValueError(f"Input tensor must have at least 2 dimensions, got {x.dim()}.")
        if self.norm_axis >= x.dim() or self.norm_axis < -x.dim():
            raise ValueError(f"norm_axis {self.norm_axis} is out of bounds for input tensor with {x.dim()} dimensions.")

        mask = x != self.padding_value if self.padding_value is not None else None
        x_masked = torch.where(mask, x, torch.zeros_like(x)) if mask is not None else x

        median = x_masked.median(dim=self.norm_axis, keepdim=True) # ADAPTED
        b = torch.abs(x_masked - median).mean(dim=self.norm_axis, keepdim=True) + self.eps # ADAPTED - not sure whether it works.

        mixture = 1
        for i in range(self.num_laplacians):
            adjusted_median = median + self.mean_offsets[i] # ADAPTED (names)
            y_norm = (x - adjusted_median) / torch.sqrt(b) # ADAPTED (names) - how does normalization occur for Laplacian distributions?
            laplacian = torch.exp(-((y_norm ** 2) / (2.0 * (self.c[i] ** 2)))) / torch.sqrt(2 * torch.pi * (self.c[i] ** 2)) # equation (9), but second division term cannot be found in the paper.
            mixture *= laplacian

        mixture /= mixture.sum(dim=self.norm_axis, keepdim=True).clamp(min=self.eps)

        if return_attention_details:
            return torch.where(mask, x * mixture, x) if mask is not None else x * mixture, mixture.detach()
        else:
            return torch.where(mask, x * mixture, x) if mask is not None else x * mixture
            
            
class MultiHeadLaplacianAdaptiveAttention(nn.Module):
    def __init__(self, norm_axis, num_heads, num_laplacians, padding_value=None, eps=1e-8):
        super().__init__()
        self.norm_axis = norm_axis
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            LaplacianAdaptiveAttention(norm_axis, num_heads, num_laplacians, padding_value, eps)
            for _ in range(num_heads)
        ])

    def forward(self, x, return_attention_details=False):
        chunk_size = x.shape[self.norm_axis] // self.num_heads
        if chunk_size == 0:
            raise ValueError(f"Input tensor size along norm_axis ({self.norm_axis}) must be larger than the number of heads ({self.num_heads}).")

        outputs, attention_details_ = [], []
        for i in range(self.num_heads):
            start_index = i * chunk_size
            end_index = start_index + chunk_size if i < self.num_heads - 1 else x.shape[self.norm_axis]
            chunk = x.narrow(self.norm_axis, start_index, end_index - start_index)
            if return_attention_details:
                out, mixture = self.attention_heads[i](chunk, return_attention_details=True)
                outputs.append(out)
                attention_details_.append(mixture)
            else:
                outputs.append(self.attention_heads[i](chunk))

        if return_attention_details:
            return torch.cat(outputs, dim=self.norm_axis), torch.cat(attention_details_, dim=self.norm_axis)
        else:
            return torch.cat(outputs, dim=self.norm_axis)
            
            

class LaplacianBlock(nn.Module):
    def __init__(self, norm_axes, num_heads, num_laplacians, num_layers, padding_value=None, eps=1e-8):
        super().__init__()
        if len(norm_axes) != num_layers or len(num_heads) != num_layers or len(num_laplacians) != num_layers:
            raise ValueError("Lengths of norm_axes, num_heads, and num_laplacians must match num_layers.")

        self.layers = nn.ModuleList([
            MultiHeadLaplacianAdaptiveAttention(norm_axes[i], num_heads[i], num_laplacians[i], padding_value, eps)
            for i in range(num_layers)
        ])

    def forward(self, x, return_attention_details=False):
        attention_details_ = {}
        for idx, layer in enumerate(self.layers):
            if return_attention_details:
                x_, attention_details = layer(x, return_attention_details=True)
                attention_details_['layer_'+str(idx)] = attention_details
                x = x_ + x
            else:
                x = layer(x) + x

        if return_attention_details:
            return x, attention_details_
        return x