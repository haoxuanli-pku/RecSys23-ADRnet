import torch

import const
import models.layer as layer
import utils


class WideAndDeepModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.embeddingByFields = layer.FeaturesEmbeddingByFields(field_dims, embed_dim)
        self.mlp = layer.MultiLayerPerceptron(embed_dim * len(field_dims), mlp_dims, dropout)
        self.linear = layer.FeaturesLinear(field_dims)

        self.name = 'WideAndDeep'
        # self.embed_dim = embed_dim

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embeddingByFields(x)

        wide1 = self.linear(x)
        # wide2 = self.fm(embed_x_by_feature).unsqueeze(1)
        deep = self.mlp(embed_x)

        x = wide1 + deep
        return torch.sigmoid(x.squeeze(1))


class DrugWideAndDeepModel(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, num_ADR):
        super().__init__()
        # self.embeddingByFields = layer.FeaturesEmbeddingByFields(field_dims, embed_dim)
        # self.embeddingByFields = layer.FeaturesEmbedding(field_dims, embed_dim)
        # self.output_dim = embed_dim
        num_features = sum(field_dims)

        self.mlp = layer.MultiLayerPerceptron(num_features, mlp_dims, dropout, output_dim=num_ADR)

        self.linear = layer.FeaturesLinear(field_dims, num_ADR)

        self.name = 'WideAndDeep'
        # self.embed_dim = embed_dim

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # embed_x = self.embeddingByFields(x)
        x = x.float()
        wide1 = self.linear(x)
        # wide2 = self.fm(embed_x_by_feature).unsqueeze(1)
        deep = self.mlp(x)

        x = wide1 + deep
        return torch.sigmoid(x)


class DeepAndCrossModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.embeddingByFields = layer.FeaturesEmbeddingByFields(field_dims, embed_dim)
        # self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = layer.CrossNetwork(embed_dim * len(field_dims), num_layers)

        self.mlp = layer.MultiLayerPerceptron(embed_dim * len(field_dims), mlp_dims, dropout, output_layer=False)
        # self.mlp = layer.MultiLayerPerceptron(embed_dim * len(field_dims), mlp_dims, dropout, output_layer=True)
        self.linear = torch.nn.Linear(mlp_dims[-1] + embed_dim * len(field_dims), 1)
        # self.linear = torch.nn.Linear(mlp_dims[-1], 1)

        self.name = 'DeepAndCross'
        self.xent_func = torch.nn.BCELoss()

        torch.nn.init.xavier_uniform_(self.linear.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        embed_x = self.embeddingByFields(x)

        # DNN
        h_l2 = self.mlp(embed_x)

        # Cross Network Layer
        x_l1 = self.cn(embed_x)

        # Stacking Layer
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        # x_stack = h_l2

        p = self.linear(x_stack)
        # p = x_stack
        # return torch.sigmoid(p.squeeze(1))
        return torch.sigmoid(p.squeeze(1))


class DrugDeepAndCrossModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout, num_ADR):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_ADR = num_ADR
        self.embeddingByFields = layer.FeaturesEmbeddingByFields(field_dims, embed_dim)
        # self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = layer.CrossNetwork(embed_dim * len(field_dims), num_layers)

        self.mlp = layer.MultiLayerPerceptron(embed_dim * len(field_dims), mlp_dims, dropout, output_layer=False)
        # self.mlp = layer.MultiLayerPerceptron(embed_dim * len(field_dims), mlp_dims, dropout, output_layer=True)
        self.linear = torch.nn.Linear(mlp_dims[-1] + embed_dim * len(field_dims), num_ADR)
        # self.linear_out = torch.nn.Linear(1000, num_ADR)
        # self.linear = torch.nn.Linear(mlp_dims[-1], 1)

        self.name = 'DeepAndCross'
        self.xent_func = torch.nn.BCELoss()

        # torch.nn.init.xavier_uniform_(self.linear.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        embed_x = self.embeddingByFields(x)

        # DNN
        h_l2 = self.mlp(embed_x)

        # Cross Network Layer
        x_l1 = self.cn(embed_x)

        # Stacking Layer
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        # x_stack = h_l2

        out = self.linear(x_stack)
        # out = self.linear_out(out)
        # p = x_stack
        # return torch.sigmoid(p.squeeze(1))
        return torch.sigmoid(out)


class DNNModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mru_dims, dropout):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.embeddingByFields = layer.FeaturesEmbeddingByFields(field_dims, embed_dim)
        # self.linear = FeaturesLinear(field_dims, embed_dim)
        # self.embed_output_dim = num_fields * embed_dim
        self.mlp = layer.MultiLayerPerceptron(embed_dim * len(field_dims), mru_dims, dropout=dropout)
        self.name = 'DNN'

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_features)``
        """
        embed_x = self.embeddingByFields(x)  
        out = self.mlp(embed_x)
        return torch.sigmoid(out.squeeze(1))


class DrugDNNModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim

        self.embeddingByFields = layer.FeaturesEmbeddingByFields(field_dims, embed_dim)
        # self.embeddingByFields = layer.FeaturesEmbedding(field_dims, embed_dim)

        # self.linear = FeaturesLinear(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        # self.embed_output_dim = embed_dim

        self.mlp = layer.MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout=dropout, output_layer=False)
        self.name = 'DNN'

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_features)``
        """
        embed_x = self.embeddingByFields(x)
        out = self.mlp(embed_x)
        return torch.sigmoid(out)
    
    
class DeepCrossingModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mru_dims, dropout):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        # self.embeddingByFields = layer.FeaturesEmbeddingByFields(field_dims, embed_dim)
        self.embeddingByFields = layer.FeaturesEmbedding(field_dims, embed_dim)

        # self.linear = FeaturesLinear(field_dims, embed_dim)
        # self.embed_output_dim = len(field_dims) * embed_dim
        self.embed_output_dim = embed_dim

        self.mru = layer.MultiResidualUnits(self.embed_output_dim, mru_dims, dropout=dropout)
        self.name = 'DeepCrossing'

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_features)``
        """
        embed_x = self.embeddingByFields(x)
        out = self.mru(embed_x)
        return torch.sigmoid(out.squeeze(1))


class DrugDeepCrossingModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mru_dims, dropout, num_ADR):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.embeddingByFields = layer.FeaturesEmbeddingByFields(field_dims, embed_dim)
        # self.linear = FeaturesLinear(field_dims, embed_dim)
        # self.embed_output_dim = num_fields * embed_dim
        self.mru = layer.MultiResidualUnits(embed_dim * len(field_dims), mru_dims, dropout=dropout, output_layer=False)
        self.output_layer = torch.nn.Linear(mru_dims[-1], num_ADR)

        self.name = 'DeepCrossing'

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_features)``
        """
        embed_x = self.embeddingByFields(x)
        mru_out = self.mru(embed_x)
        out = self.output_layer(mru_out)

        return torch.sigmoid(out)


class PNNModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, method='inner'):
        super().__init__()
        num_fields = len(field_dims)
        if method == 'inner':
            self.pn = layer.InnerProductNetwork(num_fields)
        elif method == 'outer':
            self.pn = layer.OuterProductNetwork(num_fields, embed_dim)
        else:
            raise ValueError('unknown product type: ' + method)

        self.embeddingByFields = layer.FeaturesEmbeddingByFields(field_dims, embed_dim)
        # self.linear = FeaturesLinear(field_dims, embed_dim)

        self.embed_output_dim = num_fields * embed_dim
        self.mlp = layer.MultiLayerPerceptron(self.embed_output_dim + num_fields * (num_fields - 1) // 2, mlp_dims,
                                              dropout)
        # 存储每对特征域组合的index的list
        self.embeddings_index = utils.indices_array_generic_half(num_fields, num_fields)

        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.name = 'PNN'

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_features)``
        """
        # embed_x = self.embedding(x)  # 
        # embed_x = torch.sum(embed_x, dim=1)

        batch_size = len(x)

        x_embeddings = self.embeddingByFields(x)

        # 交叉项，size = （batch_size, num_pairs）
        num_pairs = len(self.embeddings_index)
        cross_term = torch.zeros((batch_size, num_pairs)).cuda(const.CUDA_DEVICE)

        pair_count = 0
        for pair in self.embeddings_index:
            emb_1 = x_embeddings[:, self.embed_dim * pair[0]: self.embed_dim * (pair[0] + 1)]
            emb_2 = x_embeddings[:, self.embed_dim * pair[1]: self.embed_dim * (pair[1] + 1)]
            element_product = emb_1 * emb_2
            inner_product = torch.sum(element_product, dim=1)
            cross_term[:, pair_count] = inner_product.unsqueeze(0)
            pair_count += 1

        # cross_term = self.pn(embed_x)
        x = torch.cat([x_embeddings, cross_term], dim=1)
        x = self.mlp(x)
        return torch.sigmoid(x)


class DrugPNNModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, num_ADR, method='inner'):
        super().__init__()
        num_fields = len(field_dims)
        if method == 'inner':
            self.pn = layer.InnerProductNetwork()
        elif method == 'outer':
            self.pn = layer.OuterProductNetwork(num_fields, embed_dim)
        else:
            raise ValueError('unknown product type: ' + method)

        self.embeddingByFields = layer.FeaturesEmbeddingByFields(field_dims, embed_dim)
        # self.linear = FeaturesLinear(field_dims, embed_dim)

        self.embed_output_dim = num_fields * embed_dim
        self.mlp = layer.MultiLayerPerceptron(self.embed_output_dim + num_fields * (num_fields - 1) // 2, mlp_dims,
                                              dropout, output_dim=num_ADR)
        # 存储每对特征域组合的index的list
        self.embeddings_index = utils.indices_array_generic_half(num_fields, num_fields)

        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.name = 'PNN'

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_features)``
        """
        # embed_x = self.embedding(x)  #
        # embed_x = torch.sum(embed_x, dim=1)

        batch_size = len(x)

        x_embeddings = self.embeddingByFields(x)

        # 交叉项，size = （batch_size, num_pairs）
        num_pairs = len(self.embeddings_index)
        cross_term = torch.zeros((batch_size, num_pairs)).cuda(const.CUDA_DEVICE)

        pair_count = 0
        for pair in self.embeddings_index:
            emb_1 = x_embeddings[:, self.embed_dim * pair[0]: self.embed_dim * (pair[0] + 1)]
            emb_2 = x_embeddings[:, self.embed_dim * pair[1]: self.embed_dim * (pair[1] + 1)]
            element_product = emb_1 * emb_2
            inner_product = torch.sum(element_product, dim=1)
            cross_term[:, pair_count] = inner_product.unsqueeze(0)
            pair_count += 1

        # cross_term = self.pn(embed_x)
        x = torch.cat([x_embeddings, cross_term], dim=1)
        x = self.mlp(x)
        return torch.sigmoid(x)


class FNNModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.embeddingByFields = layer.FeaturesEmbeddingByFields(field_dims, embed_dim)
        # self.embeddingByFields = layer.FeaturesEmbedding(field_dims, embed_dim)

        self.embed_output_dim = len(field_dims) * embed_dim
        # self.embed_output_dim = embed_dim

        self.mlp = layer.MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

        self.name = 'FNN'

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embeddingByFields(x)

        x = self.mlp(embed_x)
        return torch.sigmoid(x.squeeze(1))


class DrugFNNModel(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, num_ADR):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        # self.embeddingByFields = layer.FeaturesEmbeddingByFields(field_dims, embed_dim)
        self.embeddingByFields = layer.FeaturesEmbedding(field_dims, embed_dim)

        # self.embed_output_dim = len(field_dims) * embed_dim
        self.embed_output_dim = embed_dim

        self.mlp = layer.MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_dim=num_ADR)

        self.name = 'FNN'

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embeddingByFields(x)
        # embed_x = x.float()

        x = self.mlp(embed_x)
        return torch.sigmoid(x)


class NFMModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropouts):
        super().__init__()

        self.name = 'NFM'
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_fields = len(field_dims)
        self.embeddingByFields = layer.FeaturesEmbeddingByFields(self.field_dims, self.embed_dim)

        # self.embedding = torch.nn.Sequential()
        # for field_dim in field_dims:
        #     self.embedding.append(torch.nn.Embedding(field_dim, embed_dim))

        # 存储每对特征域组合的index的list
        self.embeddings_index = utils.indices_array_generic_half(self.num_fields, self.num_fields)

        self.linear = layer.FeaturesLinear(field_dims)

        self.mlp = layer.MultiLayerPerceptron(embed_dim, mlp_dims, dropouts[1])

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        batch_size = len(x)

        x_embeddings = self.embeddingByFields(x)

        # 交叉项，size = （batch_size, embedding size）
        cross_term = torch.zeros((batch_size, self.embed_dim)).cuda(const.CUDA_DEVICE)

        # 特征交叉池化层，对embedding两两求元素积，将所得的所有元素积加和
        pair_count = 0
        for pair in self.embeddings_index:
            emb_1 = x_embeddings[:, self.embed_dim * pair[0] : self.embed_dim * (pair[0]+1)]
            emb_2 = x_embeddings[:, self.embed_dim * pair[1] : self.embed_dim * (pair[1]+1)]
            element_product = emb_1 * emb_2
            cross_term += element_product
            pair_count += 1

        x = self.linear(x) + self.mlp(cross_term)
        return torch.sigmoid(x.squeeze(1))


class DrugNFMModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropouts, num_ADR):
        super().__init__()

        self.name = 'NFM'
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_fields = len(field_dims)
        self.embeddingByFields = layer.FeaturesEmbeddingByFields(self.field_dims, self.embed_dim)

        # self.embedding = torch.nn.Sequential()
        # for field_dim in field_dims:
        #     self.embedding.append(torch.nn.Embedding(field_dim, embed_dim))

        # 存储每对特征域组合的index的list
        self.embeddings_index = utils.indices_array_generic_half(self.num_fields, self.num_fields)

        self.linear = layer.FeaturesLinear(field_dims, num_ADR)

        self.mlp = layer.MultiLayerPerceptron(embed_dim, mlp_dims, dropouts[1], output_dim=num_ADR)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        batch_size = len(x)

        x_embeddings = self.embeddingByFields(x)

        # 交叉项，size = （batch_size, embedding size）
        cross_term = torch.zeros((batch_size, self.embed_dim)).cuda(const.CUDA_DEVICE)

        # 特征交叉池化层，对embedding两两求元素积，将所得的所有元素积加和
        pair_count = 0
        for pair in self.embeddings_index:
            emb_1 = x_embeddings[:, self.embed_dim * pair[0] : self.embed_dim * (pair[0]+1)]
            emb_2 = x_embeddings[:, self.embed_dim * pair[1] : self.embed_dim * (pair[1]+1)]
            element_product = emb_1 * emb_2
            cross_term += element_product
            pair_count += 1

        x = self.linear(x) + self.mlp(cross_term)
        return torch.sigmoid(x.squeeze(1))


class DeepFMModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = layer.FeaturesLinear(field_dims)
        self.fm = layer.FactorizationMachine(reduce_sum=False)
        self.embeddingByFields = layer.FeaturesEmbeddingByFields(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = layer.MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

        self.name = 'DeepFM'

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        embed_x = self.embeddingByFields(x)  # from (batch_size, 881) to (batch_size, 881, embed_dim)

        wide1 = self.linear(x)
        wide2 = self.fm(embed_x).unsqueeze(1)
        deep = self.mlp(embed_x)

        x = wide1 + wide2 + deep
        return torch.sigmoid(x.squeeze(1))


class DrugDeepFMModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, num_ADR):
        super().__init__()
        self.linear_1 = layer.FeaturesLinear(field_dims, num_ADR)
        self.linear_2 = torch.nn.Linear(embed_dim, num_ADR)
        self.fm = layer.FactorizationMachine(reduce_sum=False)
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.embed_output_dim = embed_dim
        self.mlp = layer.MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_dim=num_ADR)

        self.name = 'DeepFM'

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        embed_x = self.embedding(x)  # from (batch_size, 881) to (batch_size, 881, embed_dim)

        wide1 = self.linear_1(x)
        embed_fm = self.fm(embed_x)
        wide2 = self.linear_2(torch.sigmoid(embed_fm))
        deep = self.mlp(torch.sum(embed_x, dim=1))

        x = wide1 + wide2 + deep
        return torch.sigmoid(x.squeeze(1))


class AFMModel(torch.nn.Module):
    """
    A pytorch implementation of Attentional Factorization Machine.
    Reference:
        J Xiao, et al. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks, 2017.
    """

    def __init__(self, field_dims, embed_dim, attn_size, dropouts):
        super().__init__()
        self.embedding = layer.FeaturesEmbedding(field_dims, embed_dim)
        self.linear = layer.FeaturesLinear(field_dims)
        self.afm = layer.AttentionalFactorizationMachine(embed_dim, attn_size, dropouts)
        self.name = 'AFM'

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.afm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))


class DrugAFMModel(torch.nn.Module):
    """
    A pytorch implementation of Attentional Factorization Machine.
    Reference:
        J Xiao, et al. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks, 2017.
    """

    def __init__(self, field_dims, embed_dim, attn_size, dropouts, num_ADR):
        super().__init__()
        self.embedding = layer.FeaturesEmbedding(field_dims, embed_dim)
        self.linear = layer.FeaturesLinear(field_dims, num_ADR)
        self.afm = layer.AttentionalFactorizationMachine(embed_dim, attn_size, dropouts, scoring=False, num_ADR=num_ADR)
        self.name = 'AFM'

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.afm(self.embedding(x))
        return torch.sigmoid(x)
