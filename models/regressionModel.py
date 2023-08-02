import torch
import models.layer as layer
import const


class LRModel(torch.nn.Module):
    """
    A pytorch implementation of Logistic Regression.
    """

    def __init__(self, field_dims):
        super().__init__()
        self.linear = layer.FeaturesLinear(field_dims)
        # self.linear = torch.nn.Linear(sum(field_dims), 1)
        self.name = "LRModel"

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        output = self.linear(x).squeeze(1)
        return torch.sigmoid(output)


class POLY2Model(torch.nn.Module):

    def __init__(self, field_dims):
        super().__init__()
        # self.linear = torch.nn.Linear(1,1)
        num_features = sum(field_dims)
        cross_dims = int(num_features * (num_features - 1) // 2)
        self.w = torch.nn.Parameter(torch.rand(cross_dims).unsqueeze(1))  # xavier init needs 2D or more
        torch.nn.init.xavier_uniform_(self.w)
        self.name = "POLY2Model"

    def forward(self, x):
        # x: size (batch_size, num_fields)
        x = x.cpu().detach().numpy()
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures((2, 2), interaction_only=True, include_bias=False)
        x_poly = poly.fit_transform(x)
        # print(x_poly.shape)
        x_poly = torch.from_numpy(x_poly).cuda(const.CUDA_DEVICE)

        # x_poly_weighted = x_poly * self.w
        # out = torch.sum(x_poly_weighted, dim=1, keepdim=True).squeeze(1).float()

        x_poly = x_poly * self.w.squeeze(1)
        out = torch.sum(x_poly, dim=1, keepdim=True).squeeze(1).float()

        return out.sigmoid()


class FMModel(torch.nn.Module):

    def __init__(self, field_dims):
        super().__init__()
        self.linear = layer.FeaturesLinear(field_dims)
        self.fm = layer.FactorizationMachine(reduce_sum=False)

        self.w = torch.nn.Parameter(torch.rand(sum(field_dims)).unsqueeze(1))  # xavier init needs 2D or more
        torch.nn.init.xavier_normal_(self.w)


        self.name = "FMModel"

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # embed_x = self.embedding(x)

        # xw = x * (self.w.squeeze(1).cuda(const.CUDA_DEVICE))
        xw = x * self.w.squeeze(1)
        linear_x = self.linear(x)
        fm_x = self.fm(xw).unsqueeze(1)
        x = linear_x + fm_x
        return torch.sigmoid(x.squeeze(1))

# class FMModel(torch.nn.Module):
#     """
#     A pytorch implementation of Factorization Machine.
#     Reference:
#         S Rendle, Factorization Machines, 2010.
#     """
#
#     def __init__(self, field_dims, embed_dim):
#         super().__init__()
#         self.embedding = layer.FeaturesEmbedding(field_dims, embed_dim)
#         self.linear = layer.FeaturesLinear(field_dims)
#         self.fm = layer.FactorizationMachine(reduce_sum=True)
#
#     def forward(self, x):
#         """
#         :param x: Long tensor of size ``(batch_size, num_fields)``
#         """
#         x = self.linear(x) + self.fm(self.embedding(x))
#         return torch.sigmoid(x.squeeze(1))


class FFMModel(torch.nn.Module):
    """
    A pytorch implementation of Field-aware Factorization Machine.
    Reference:
        Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.linear = layer.FeaturesLinear(field_dims)
        self.ffm = layer.FieldAwareFactorizationMachine(field_dims, embed_dim)
        self.name = "FFMModel"

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        return torch.sigmoid(x.squeeze(1))


class LSPLMModel(torch.nn.Module):
    """
    A pytorch implementation of Logistic Regression.
    """

    def __init__(self, field_dims, m):
        super().__init__()
        # self.linear = layer.FeaturesLinear(field_dims)
        # self.field_dims = field_dims
        self.m = m
        self.feature_num = sum(field_dims)
        self.softmax = torch.nn.Sequential(torch.nn.Linear(self.feature_num, self.m).double(),
                                           torch.nn.Softmax(dim=1).double())
        self.logistic = torch.nn.Sequential(torch.nn.Linear(self.feature_num, self.m).double(),
                                            torch.nn.Sigmoid())
        self.name = "LSPLMModel"

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x.double()
        logistic_out = self.logistic(x)
        softmax_out = self.softmax(x)
        combine_out = logistic_out.mul(softmax_out)
        return combine_out.sum(1).float()
