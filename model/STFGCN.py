import torch.nn as nn
import torch
import math
import torch.nn.functional as F



abalation_pattern = "ETSO"

class CorrGraph(nn.Module):
    # input  history_x
    # return CorrMatrix
    #
    def __init__(self, length=12, dim=1):
        super(CorrGraph, self).__init__()
        self.d_k = 32
        # 特征数。1：仅考虑数据特征；3：加入时间特征
        self.dim = dim
        self.W_Q = nn.Linear(dim, self.d_k)
        self.W_K = nn.Linear(dim, self.d_k)
        # length用于区分x和x_backday的长度


        self.conv = nn.Conv2d(length, 1, stride=1, kernel_size=(1, 1))

    def forward(self, x):
        # x.shape = B,T,N,F
        if self.dim == 1:
            x = x[:, :, :, 0:1]

        Q = self.W_Q(x)
        K = self.W_K(x)
        # 归一化后score
        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        score = F.softmax(score, dim=-1)  # 按行

        score = self.conv(score)
        score = torch.squeeze(score, dim=1)
        return score


class Conv_withSpatialScore(nn.Module):  # in the sencond blocks;which focus on the change
    #
    def __init__(self, num_of_features, out_channels, device):  #
        super(Conv_withSpatialScore, self).__init__()
        self.device = device
        self.out_channels = out_channels
        self.relu = nn.ReLU()
        self.Theta = nn.Parameter(torch.FloatTensor(num_of_features, out_channels).to(self.device))
        self.k = nn.Parameter(torch.FloatTensor(3))

    def forward(self, x, score):
        batch_size, num_of_timesteps, num_of_vertices, num_of_features = x.shape
        outputs = []
        score = score.to(self.device)
        theta_k = self.Theta
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, time_step, :, :]  # (b, N, F_in)
            rhs = torch.einsum("BNM,BNN->BNM", graph_signal, score)
            output = (rhs.matmul(theta_k))
            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)
        result = self.relu(torch.cat(outputs, dim=-1))
        result = result.permute(0, 3, 1, 2)
        return result


class GraphFusion(nn.Module):
    def __init__(self, num_nodes):
        super(GraphFusion, self).__init__()
        self.num_nodes = num_nodes
        self.gate = nn.Sigmoid()
        self.relu = nn.ReLU()
        #self.Conv = nn.Conv2d(in_channels=,out_channels=,(1,1))
    def forward(self, graph, graph_):
        gate = self.gate(graph)

        period_graph = self.relu(torch.mul(gate, graph_))
        inter_graph = (graph + period_graph)/2

        return inter_graph


# TCN_patch
class TCN(nn.Module):
    def __init__(self, length, model_dim):
        super(TCN, self).__init__()
        self.length = length  # patch 长度
        self.t_length = length - 4 + 1
        #self.TempConv = nn.Conv2d(model_dim, model_dim, kernel_size=(1, 4), stride=(1, 1))

        self.TempConv1 = nn.Conv2d(model_dim, model_dim * 2, kernel_size=(1, 2), stride=(1, 1))
        self.Gelu = nn.GELU()
        self.TempConv2 = nn.Conv2d(model_dim * 2, model_dim, kernel_size=(1, 2), stride=(1, 1))
        self.normalize1 = nn.LayerNorm(model_dim)

        self.relu = nn.ReLU()
        self.TempConv3 = nn.Conv2d(model_dim, model_dim, kernel_size=(1, 6), stride=(1, 1))
        self.normalize2 = nn.LayerNorm(model_dim)

        #self.Final_Conv = nn.Conv2d(self.t_length * 2, self.t_length, stride=1, kernel_size=(1, 1))#原始为k1,k2,k3 = 234
        self.Final_Conv = nn.Conv2d(5, self.t_length, stride=1, kernel_size=(1, 1))

    def forward(self, x):
        # x.shape = B,T,N,F
        B, T, N, F = x.shape
        residual = x
        x = x.permute(0, 3, 2, 1)  # B,F,N,T
        x_single = x
        x_double = x
        x_double = self.TempConv1(x_double)
        x_double = self.Gelu(x_double)
        x_double = self.TempConv2(x_double)
        x_double = x_double.permute(0, 3, 2, 1)  # B,T,N,F
        y_double = self.normalize1(x_double)

        x_single = self.TempConv3(x_single)
        x_single = self.relu(x_single)
        x_single = x_single.permute(0, 3, 2, 1)  # B,T,N,F
        y_single = self.normalize2(x_single)

        y = torch.cat((y_single, y_double), dim=1)
        y = self.Final_Conv(y)

        return y


class TPC(nn.Module):
    def __init__(self, x_length, patch_length, patch_stride, model_dim):
        super(TPC, self).__init__()
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        N = (x_length - patch_length) / patch_stride + 1  # N: num of patch
        N = int(N)
        #

        self.TempConv = nn.ModuleList([
            TCN(length=patch_length, model_dim=model_dim)
            for _ in range(N)
        ])

    def forward(self, x):
        x_list = self.split(x)
        x_temp = []
        i = 0
        for t in self.TempConv:
            x_temp.append(t(x_list[i]))
            i = i + 1
        x_temp = torch.cat(x_temp, dim=1)
        return x_temp

    def split(self, x):
        timesteps = x.shape[1]
        patch_list = []
        for timestep in range(0, timesteps - self.patch_length + 1, self.patch_stride):
            patch_list.append(x[:, timestep:timestep + self.patch_length, :, :])
        # for timestep in range(self.length, timesteps, self.stride):
        #     patch_list.append(x[:, timestep - self.length:timestep, :, :])
        return patch_list


class stfgcn(nn.Module):
    def __init__(
            self,
            device,
            num_nodes,
            x_back_length=12,  # x_backday length
            in_steps=12,
            out_steps=12,
            steps_per_day=288,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=24,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=80,
            feed_forward_dim=256,
            num_heads=4,
            num_layers=3,
            use_mixed_proj=True,
            dropout=0.1,
            TPC_length=6,
            TPC_stride=2,
    ):

        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                + spatial_embedding_dim
                + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.feed_forward_dim = feed_forward_dim
        self.device = device
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.graph_Corr_backday = CorrGraph(x_back_length)  # graph_Corr.shape = B,T,N,N
        self.graph_Conv1 = Conv_withSpatialScore(self.model_dim, self.model_dim, self.device)

        self.graph_Corr2_x = CorrGraph(in_steps)
        self.graph_Conv2 = Conv_withSpatialScore(self.model_dim, self.model_dim, self.device)

        self.fusion_graph = GraphFusion(num_nodes)
        self.graph_Conv3 = Conv_withSpatialScore(self.model_dim, self.model_dim, self.device)

        self.tcn = TPC(in_steps, TPC_length, TPC_stride, self.model_dim)

    def forward(self, x, x_backday):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)

        batch_size = x.shape[0]
        origin = x # 保留包含 数据+天 周 数据
        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        ### 嵌入+高维映射
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        EmbeddingX = x

        if(abalation_pattern == "ETSO"):
            ###时间模块特征提取
            tcn_out = self.tcn(x)
            ###残差
            x = EmbeddingX + tcn_out
            ###时间特征提取完毕后 保留为 temporal_residual
            temporal_residual = x
            #图卷积模块
            graph_x = self.graph_Corr_backday(origin)
            graph_backday = self.graph_Corr2_x(x_backday)
            inter_graph = self.fusion_graph(graph_x, graph_backday)
            x_conv3 = self.graph_Conv3(temporal_residual, inter_graph)
            ###图卷积模块第三层残差
            x_graph_out = x_conv3 + temporal_residual
        if(abalation_pattern == "ETO"):
            tcn_out = self.tcn(x)
            ###残差
            x = EmbeddingX + tcn_out
            ###时间特征提取完毕后 保留为 temporal_residual
            temporal_residual = x
            x_graph_out = temporal_residual
        if(abalation_pattern == "ESO"):
            #图卷积模块
            graph_x = self.graph_Corr_backday(origin)
            graph_backday = self.graph_Corr2_x(x_backday)
            inter_graph = self.fusion_graph(graph_x, graph_backday)
            x_conv3 = self.graph_Conv3(EmbeddingX, inter_graph)
            ###图卷积模块第三层残差
            x_graph_out = x_conv3 + EmbeddingX
        if(abalation_pattern == "EO"):
            x_graph_out = EmbeddingX

        ###输出层输入命名为 x_graph_out
        if self.use_mixed_proj:
            out = x_graph_out.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x_graph_out.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out


if __name__ == "__main__":
    pass
#
