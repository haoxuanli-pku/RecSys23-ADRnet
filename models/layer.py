import numpy as np
import torch
import torch.nn.functional as F

import const
import utils


class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        # self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        # self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.fc = torch.nn.Linear(sum(field_dims), output_dim)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

        # self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # x = x + x.new_tensor(self.offsets).unsqueeze(0)

        # return torch.sum(self.fc(x), dim=1) + self.bias
        return self.fc(x.float())


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Linear(sum(field_dims), embed_dim)
        self.field_dims = field_dims
        # self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # x = x.long()
        x = x.float()
        out = self.embedding(x)
        return out


class FeaturesEmbeddingByFields(torch.nn.Module):
    """
    return tensor of (batch_size, self.embed_dim * self.num_fields)
    """
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_fields = len(field_dims)
        self.embedding = torch.nn.Sequential()
        for field_dim in field_dims:
            # self.embedding.append(torch.nn.Embedding(field_dim, embed_dim))
            self.embedding.append(torch.nn.Linear(field_dim, embed_dim))
        self.embeddings_index = utils.indices_array_generic_half(self.num_fields, self.num_fields)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        return : tensor of size ``(batch_size, num_fields * embed_dims)
        """
        x = x.float()
        batch_size = len(x)

        # 不同特征域分别embedding
        x_embeddings = torch.zeros((batch_size, self.embed_dim * self.num_fields)).cuda(const.CUDA_DEVICE)
        # x_embeddings = []

        cum_dims = np.array((0, *np.cumsum(self.field_dims)), dtype=np.long)
        for i in range(len(self.field_dims)):
            x_field = x[:, cum_dims[i]:cum_dims[i + 1]]
            embedding = self.embedding[i](x_field)
            # x_embeddings.append(torch.sum(embedding, dim=1))
            # x_embeddings[:, i * self.embed_dim: (i + 1) * self.embed_dim] = torch.sum(embedding, dim=1)
            x_embeddings[:, i * self.embed_dim: (i + 1) * self.embed_dim] = embedding
        return x_embeddings


class FieldAwareFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        # self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        #  索引偏移
        # offsets = []
        # for i in range(len(self.field_dims)):
        #     for j in range(self.field_dims[i]):
        #         if i == 0:
        #             offsets.append(0)
        #         else:
        #             offsets.append(self.field_dims[i])
        #
        # x = x + x.new_tensor(offsets).unsqueeze(0)

        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix


class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """

        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        # square_of_sum = torch.sum(x, dim=1) ** 2
        # sum_of_square = x ** 2
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiResidualUnits(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True, output_dim=1):
        super().__init__()
        layers = list()
        self.input_layer = torch.nn.Linear(input_dim, embed_dims[0])  # 把原embedding的维度转化为mru的维度
        self.output_layer = output_layer

        input_dim = embed_dims[0]
        for embed_dim in embed_dims[1:]:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            # layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            # layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, output_dim))

        self.mru = torch.nn.Sequential(*layers)

        self.num_layers = len(embed_dims)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        x = self.input_layer(x)
        for i in range(self.num_layers):
            x0 = x
            mru_layer = self.mru[i * 2: (i+1) * 2]
            mru_out = mru_layer(x)
            x = mru_out + x0

        if self.output_layer:
            x = self.mru[-1](x)

        return x


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True, output_dim=1):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            # layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.Sigmoid())
            # layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim

        if output_layer:
            layers.append(torch.nn.Linear(input_dim, output_dim))

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x.float())


class InnerProductNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        return torch.sum(x[:, row] * x[:, col], dim=2)


class OuterProductNetwork(torch.nn.Module):

    def __init__(self, num_fields, embed_dim, kernel_type='mat'):
        super().__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type
        self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))
        torch.nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        if self.kernel_type == 'mat':
            kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)
            return torch.sum(kp * q, -1)
        else:
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)


class CrossNetwork(torch.nn.Module):

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.rand((input_dim,)).unsqueeze(1)) for _ in range(num_layers)
        ])

        for w in self.w:
            torch.nn.init.xavier_uniform_(w.weight.data)

        for b in self.b:
            torch.nn.init.xavier_uniform_(b)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            xxw = x0 * xw
            bias = self.b[i].squeeze(1)
            x = x0 * xw + self.b[i].squeeze(1) + x
        return x


class AttentionalFactorizationMachine(torch.nn.Module):

    def __init__(self, embed_dim, attn_size, dropouts, scoring=True, num_ADR=1):
        super().__init__()
        self.attention = torch.nn.Linear(embed_dim * (embed_dim-1) // 2, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.dropouts = dropouts
        if scoring:
            self.fc = torch.nn.Linear(embed_dim * (embed_dim-1) // 2, 1)
        else:
            self.fc = torch.nn.Linear(embed_dim * (embed_dim-1) // 2, num_ADR)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        inner_product = p * q
        attn_scores = F.relu(self.attention(inner_product))
        attn_scores = F.softmax(self.projection(attn_scores), dim=1)
        attn_scores = F.dropout(attn_scores, p=self.dropouts[0], training=self.training)
        attn_output = attn_scores * inner_product
        attn_output = F.dropout(attn_output, p=self.dropouts[1], training=self.training)
        return self.fc(attn_output)


class CompressedInteractionNetwork(torch.nn.Module):

    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))


class AnovaKernel(torch.nn.Module):

    def __init__(self, order, reduce_sum=True):
        super().__init__()
        self.order = order
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        batch_size, num_fields, embed_dim = x.shape
        a_prev = torch.ones((batch_size, num_fields + 1, embed_dim), dtype=torch.float).to(x.device)
        for t in range(self.order):
            a = torch.zeros((batch_size, num_fields + 1, embed_dim), dtype=torch.float).to(x.device)
            a[:, t + 1:, :] += x[:, t:, :] * a_prev[:, t:-1, :]
            a = torch.cumsum(a, dim=1)
            a_prev = a
        if self.reduce_sum:
            return torch.sum(a[:, -1, :], dim=-1, keepdim=True)
        else:
            return a[:, -1, :]


