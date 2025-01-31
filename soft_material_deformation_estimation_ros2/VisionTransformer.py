import torch
import torch.nn as nn
import numpy as np


class ViT(nn.Module):
    def __init__(self,
                 chw,
                 n_patches,
                 n_blocks,
                 hidden_dimension,
                 n_heads,
                 ):
        # Super constructor
        super(ViT, self).__init__()

        # Attributes
        self.chw = chw  # ( C , H , W )
        self.n_patches = int(n_patches)
        self.n_blocks = n_blocks  # number of encoder blocks
        self.n_heads = n_heads
        self.hidden_dimension = hidden_dimension  # linear mapping dimension is arbitrary -> parameter 'hidden dimension'
        self.mlp_ratio = 4

        # Input and patches sizes
        assert (chw[1] % n_patches == 0), "Input shape not entirely divisible by number of patches"
        assert (chw[2] % n_patches == 0), "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_dimension = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_dimension, self.hidden_dimension)

        # 2) Learnable classification token (linear embedding)
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_dimension))

        # 3) Positional embedding
        self.register_buffer("positional_embeddings",
                             get_positional_embeddings(n_patches ** 2 + 1, hidden_dimension),
                             persistent=False)

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(hidden_dimension, n_heads, self.mlp_ratio) for _ in range(n_blocks)])

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting the classification token only
        return  out[:, 0]


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 hidden_dimension,
                 n_heads,
                 mlp_ratio):
        super(TransformerEncoderBlock, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_dimension)
        self.mhsa = MhsaBlock(hidden_dimension, n_heads)
        self.norm2 = nn.LayerNorm(hidden_dimension)
        self.mlp = nn.Sequential(nn.Linear(hidden_dimension, mlp_ratio * hidden_dimension),
                                 nn.GELU(),
                                 nn.Linear(mlp_ratio * hidden_dimension, hidden_dimension),
                                 )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class MhsaBlock(nn.Module):
    # Multi-head Self Attention
    def __init__(self, hidden_dimension, n_heads):
        super(MhsaBlock, self).__init__()
        self.d = hidden_dimension
        self.n_heads = n_heads

        assert hidden_dimension % n_heads == 0, f"Can't divide dimension {hidden_dimension} into {n_heads} heads"

        d_head = int(hidden_dimension / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)
                # scaled dot product attention

                seq_result.append(self.softmax(q @ k.T / (self.d_head ** 0.5)) @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            if j % 2 == 0:
                result[i][j] = np.sin(i / (10000 ** (j / d)))
            else:
                result[i][j] = np.cos(i / (10000 ** ((j - 1) / d)))
    return result
