import random
import operator
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
import seaborn as sn
import torch
import torch.nn as nn 
from ViTHelper import MasterEncoder


class MultiheadEinsumOrig(nn.Module):
    def __init__(self, max_seq_length, embedding_size, num_atten_heads) -> None:
        super().__init__()

        self.max_seq_length = max_seq_length
        self.num_atten_heads = num_atten_heads
        self.embedding_size = embedding_size
        # Stacked matrices Wq, Wk, Wv. Used to obtain q, k, v values. We need to have multiple q, k, v vectors.
        # The input matrix is of shape (Nw, M) -- stacked emedding chunks on top of each other
        # output shape is (Nw, M * 3) -- we have q, k, v stacked horizontally and all chunks vertically
        self.wq_wk_wv = nn.Linear(embedding_size, embedding_size * 3, bias=False)
        # Output linedar layer     
        self.W = nn.Linear( embedding_size, embedding_size, bias=False)

    def forward(self, x):
        #Get q, k, v -- use combined matriices and then chunk the resluting tensor into 3 parts.
        # Treat each part as corresponding output for each head stacked together -- It doesn't matter in which order we learn the weights.
        # We get 3 tensors of shape (batch, max_seq_length, num_atten_heads * embedding_size)
        q, k, v = tuple(self.wq_wk_wv(x).view(-1, self.max_seq_length, self.embedding_size // self.num_atten_heads, 3, self.num_atten_heads) \
                              .permute(3, 0, 4, 1, 2))
        # Result (batch, heads, embedding (token), embedding size)
        # Einsum to calculate the dot product over the ebedding dimension for each token pair i j
        scaled_dot_product = torch.einsum('b h i d , b h j d -> b h i j', q, k) * (self.embedding_size//self.num_atten_heads) ** -0.5
        # Add softmax
        attention = torch.softmax(scaled_dot_product, dim=-1)
        # Multiply by v in the same manner ((token, token) * (token, embedding/n_heads))
        out = torch.einsum('b h i j , b h j d -> b h i d', attention, v)
        # Rearrange computations (batch, token, head * embedding per head):
        out = out.permute(0, 2, 1, 3).reshape(-1, self.max_seq_length, self.embedding_size)
        # apply final linear layer
        return self.W(out)

class MultiheadEinsum(nn.Module):
    def __init__(self, max_seq_length, embedding_size, num_atten_heads) -> None:
        super().__init__()

        self.max_seq_length = max_seq_length
        self.num_atten_heads = num_atten_heads
        self.embedding_size = embedding_size
        # Stacked matrices Wq, Wk, Wv. Used to obtain q, k, v values. We need to have multiple q, k, v vectors.
        self.wq_wk_wv = nn.ModuleList([nn.Linear(embedding_size // num_atten_heads, embedding_size // num_atten_heads * 3, bias=False)\
                                       for _ in range(num_atten_heads)])
        # Output linedar layer     
        self.W = nn.Linear( embedding_size, embedding_size, bias=False)

    def forward(self, x):
        #Get q, k, v -- use combined matriices and then chunk the resluting tensor into 3 parts.
        # Apply each linear model separately to each of the chunks
        heads_qkv = torch.stack([op(chunk) for op, chunk in zip(self.wq_wk_wv, torch.chunk(x, self.num_atten_heads, dim = -1))])\
            .view(self.num_atten_heads, -1, self.max_seq_length, 3, self.embedding_size // self.num_atten_heads).permute(3, 1, 0, 2, 4)
        print(heads_qkv.shape)
        # Get q, k, v as previously
        q, k, v = tuple(heads_qkv)
        # Result (batch, heads, embedding (token), embedding size)
        # Einsum to calculate the dot product over the ebedding dimension for each token pair i j
        scaled_dot_product = torch.einsum('b h i d , b h j d -> b h i j', q, k) * (self.embedding_size//self.num_atten_heads) ** -0.5
        # Add softmax
        attention = torch.softmax(scaled_dot_product, dim=-1)
        # Multiply by v in the same manner ((token, token) * (token, embedding/n_heads))
        out = torch.einsum('b h i j , b h j d -> b h i d', attention, v)
        # Rearrange computations (batch, token, head * embedding per head):
        out = out.permute(0, 2, 1, 3).reshape(-1, self.max_seq_length, self.embedding_size)
        # apply final linear layer
        return self.W(out)



class ViT(nn.Module):
    def __init__(self, im_size, patch_size, token_size, basic_encoders, num_heads, out_classes) -> None:
        super().__init__()
        self.im_size = im_size
        self.patch_size = patch_size
        self.n_patch = im_size // patch_size
        self.token_size = token_size
        # Create a linear mapping to generate patches (the number of output channels is essentially a token size)
        self.patch_transform = nn.Conv2d(3, token_size, patch_size, patch_size)
        # Create a learnable initial class token
        self.class_token = nn.Parameter(torch.rand(1, token_size))
        # Create learnable positional encoding
        self.posiional_encoding = nn.Parameter(torch.rand(self.n_patch**2 + 1, token_size))

        # Define a Transormer Encoder
        self.encoder = MasterEncoder(self.n_patch**2 + 1, token_size, basic_encoders, num_heads)
        # Define MLP:
        self.mlp = nn.Sequential(
            nn.Linear(token_size, out_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, input_image):
        batch = input_image.shape[0]
        # Get patch tokens using convoluation:
        patch_tokens = self.patch_transform(input_image)
        # Convert patches to acceptable shape:
        patch_tokens = patch_tokens.permute(0, 2, 3, 1)
        patch_tokens = patch_tokens.view(batch, -1, self.token_size)
        # Add class token:
        class_token = self.class_token.repeat(batch, 1, 1)
        patch_tokens = torch.cat([patch_tokens, class_token], dim = 1)
        # Add positional encoding
        pos_encoding = self.posiional_encoding.repeat(batch, 1, 1)
        patch_tokens += pos_encoding
        # Run encoder
        out = self.encoder(patch_tokens)
        # Get last element:
        out = out[:, -1, :]
        out = self.mlp(out)

        return out

if __name__=="__main__":

    # net = ViT(64, 16, 24, 10, 2, 5)
    # summary(net, input_size=(10, 3, 64, 64))
    net = MultiheadEinsum(15, 24, 4)
    input = torch.rand(7,15, 24)
    print(net(input).shape)
    summary(net, input_size=(7,15, 24))
    # net = MultiheadEinsum(4, 7, 16, 10)
    # input = torch.rand(13, 5, 7)
    # net(input)
 