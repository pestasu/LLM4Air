import ipdb
import numpy as np
from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn.parameter import Parameter
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer


def check_device(device=None):
    if device is None:
        print("`device` is missing, try to train and evaluate the model on default device.")
        if torch.cuda.is_available():
            print("cuda device is available, place the model on the device.")
            return torch.device("cuda")
        else:
            print("cuda device is not available, place the model on cpu.")
            return torch.device("cpu")
    else:
        if isinstance(device, torch.device):
            return device
        else:
            return torch.device(device)


class Gagnn(nn.Module):
    def __init__(self, args):
        super(Gagnn, self).__init__()
        self.args = args
        self.mode = args.mode
        self.device = check_device()
        self.encoder = args.encoder
        self.w_init = args.w_init
        self.city_num = args.city_num
        self.group_num = args.group_num
        self.edge_h = args.edge_h
        self.gnn_layer = args.gnn_layer
        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.inpute_size = args.input_size
        if self.encoder == 'self':
            self.encoder_layer = TransformerEncoderLayer(self.inpute_size, nhead=2, dim_feedforward=128)
			# self.x_embed = Lin(self.inpute_size, x_em)
            self.x_embed = Lin(self.seq_len*self.inpute_size, args.x_em)
        elif self.encoder == 'lstm':
            self.input_LSTM = nn.LSTM(self.inpute_size,args.x_em,num_layers=1,batch_first=True)
        if self.w_init == 'rand':
            self.w = Parameter(torch.randn(args.city_num,args.group_num).to(self.device,non_blocking=True),requires_grad=True)
        elif self.w_init == 'group':
            self.w = Parameter(args.w,requires_grad=True)
        self.loc_embed = Lin(2, args.loc_em)
        self.u_embed1 = nn.Embedding(13, args.date_em) #month
        self.u_embed2 = nn.Embedding(7, args.date_em) #week
        self.u_embed3 = nn.Embedding(24, args.date_em) #hour
        self.edge_inf = Seq(Lin(args.x_em*2+args.date_em*3+args.loc_em*2, args.edge_h),ReLU(inplace=True))
        self.group_gnn = nn.ModuleList([NodeModel(args.x_em+args.loc_em,args.edge_h,args.gnn_h)])
        for i in range(self.gnn_layer-1):
            self.group_gnn.append(NodeModel(args.gnn_h,args.edge_h,args.gnn_h))
        self.global_gnn = nn.ModuleList([NodeModel(args.x_em+args.gnn_h,1,args.gnn_h)])
        for i in range(self.gnn_layer-1):
            self.global_gnn.append(NodeModel(args.gnn_h,1,args.gnn_h))
        if self.mode == 'ag':
            self.decoder = DecoderModule(args.x_em,args.edge_h,args.gnn_h,args.gnn_layer,args.city_num,args.group_num)
            self.predMLP = Seq(
                Lin(args.gnn_h,16),
                ReLU(inplace=True),
                Lin(16,1),
                # ReLU(inplace=True)
                )
        else:
            self.decoder = DecoderModule(args.x_em,args.edge_h,args.gnn_h,args.gnn_layer,args.city_num,args.group_num)
            self.predMLP = Seq(
                Lin(args.gnn_h,16),
                ReLU(inplace=True),
                Lin(16,self.pred_len),
                # ReLU(inplace=True)
                )	 #Lin(args.gnn_h,self.pred_len)
    def batchInput(self,x,edge_w,edge_index):
        sta_num = x.shape[1]
        x = x.reshape(-1,x.shape[-1])
        edge_w = edge_w.reshape(-1,edge_w.shape[-1])
        for i in range(edge_index.size(0)):
            edge_index[i,:] = torch.add(edge_index[i,:], i*sta_num)
        # print(edge_index.shape)
        edge_index = edge_index.transpose(0,1)
        # print(edge_index.shape)
        edge_index = edge_index.reshape(2,-1)
        return x, edge_w, edge_index

    def forward(self, x, edge_index, edge_w, loc): 
        edge_index, edge_w, loc = edge_index.to(x.device), edge_w.to(x.device), loc.to(x.device)
        edge_index = edge_index.repeat(x.shape[0], 1, 1)
        edge_w = edge_w.repeat(x.shape[0], 1, 1)
        loc = loc.repeat(x.shape[0], 1, 1)
        # ipdb.set_trace()
        # x = x.permute(0, 2, 1, 3) # B,N,L,C
        uu = torch.cat((x[:,0,-1,-4:-3], x[:,0,-1,-2:]), -1).long() #month,day,weekday,hour

        x = x.reshape(-1,x.shape[2],x.shape[3])
        x = x[..., :self.inpute_size]
        if self.encoder == 'self':
            # [S,B,E]
            # print(x.shape)
            # ipdb.set_trace()
            x = x.transpose(0,1)
            x = self.encoder_layer(x)
            x = x.transpose(0,1)
            # print(x.shape)
            x = x.reshape(-1,self.city_num,self.seq_len*x.shape[-1])
            x = self.x_embed(x)
            # x = x.reshape(-1,self.city_num,self.seq_len,x.shape[-1])
            # x = torch.max(x,dim=-2).values
            # print(x.shape)

        elif self.encoder == 'lstm':
            _,(x,_) = self.input_LSTM(x)
            x = x.reshape(-1,self.city_num,x.shape[-1])
            # print(x.shape)
        # print(x.shape)

        # graph pooling
        # print(self.w[10])
        # print("x:", x)
        # ipdb.set_trace()
        
        w = F.softmax(self.w.to(x.device))
        w1 = w.transpose(0,1)
        w1 = w1.unsqueeze(dim=0)
        w1 = w1.repeat_interleave(x.size(0), dim=0)
        # print(w.shape,x.shape)
        # print(loc.shape)
        loc = self.loc_embed(loc)
        x_loc = torch.cat([x,loc],dim=-1)

        g_x = torch.bmm(w1,x_loc)
        # print(g_x.shape)

        # group gnn
        u_em1 = self.u_embed1(uu[...,0])
        u_em2 = self.u_embed2(uu[...,1])
        u_em3 = self.u_embed3(uu[...,2])
        u_em = torch.cat([u_em1,u_em2,u_em3],dim=-1)
        # print("u_em:", u_em)
        # print(u_em.shape)
        for i in range(self.group_num):
            for j in range(self.group_num):
                if i == j: continue
                else:
                    g_edge_input = torch.cat([g_x[:,i],g_x[:,j],u_em],dim=-1)
                    tmp_g_edge_w = self.edge_inf(g_edge_input)
                    tmp_g_edge_w = tmp_g_edge_w.unsqueeze(dim=0)
                    tmp_g_edge_index = torch.tensor([i,j]).unsqueeze(dim=0).to(self.device,non_blocking=True)
                    if i == 0 and j == 1:
                        g_edge_w = tmp_g_edge_w
                        g_edge_index = tmp_g_edge_index
                    else:
                        g_edge_w = torch.cat([g_edge_w,tmp_g_edge_w],dim=0)
                        g_edge_index = torch.cat([g_edge_index,tmp_g_edge_index],dim=0)
        # print(g_edge_w.shape,g_edge_index.shape)
        g_edge_w = g_edge_w.transpose(0,1)
        g_edge_index = g_edge_index.unsqueeze(dim=0)
        g_edge_index = g_edge_index.repeat_interleave(u_em.shape[0],dim=0)
        g_edge_index = g_edge_index.transpose(1,2)
        # print(g_x.shape,g_edge_w.shape,g_edge_index.shape)
        g_x, g_edge_w, g_edge_index = self.batchInput(g_x, g_edge_w, g_edge_index)
        # print(g_x.shape,g_edge_w.shape,g_edge_index.shape)
        for i in range(self.gnn_layer):
            g_x = self.group_gnn[i](g_x,g_edge_index,g_edge_w)
        
        g_x = g_x.reshape(-1,self.group_num,g_x.shape[-1])
        # print(g_x.shape,self.w.shape)
        w2 = w.unsqueeze(dim=0)
        w2 = w2.repeat_interleave(g_x.size(0), dim=0)
        new_x = torch.bmm(w2,g_x)
        # print(new_x.shape,x.shape)
        new_x = torch.cat([x,new_x],dim=-1)
        edge_w = edge_w.unsqueeze(dim=-1)
        # print(new_x.shape,edge_w.shape,edge_index.shape)
        new_x, edge_w, edge_index = self.batchInput(new_x, edge_w, edge_index)
        # print(new_x.shape,edge_w.shape,edge_index.shape)
        for i in range(self.gnn_layer):
            new_x = self.global_gnn[i](new_x,edge_index,edge_w)
        # print(new_x.shape)
        if self.mode == 'ag':
            for i in range(self.pred_len):
                new_x = self.decoder(new_x,self.w,g_edge_index,g_edge_w,edge_index,edge_w)
                tmp_res = self.predMLP(new_x)
                tmp_res = tmp_res.reshape(-1,self.city_num)
                tmp_res = tmp_res.unsqueeze(dim=-1)
                if i == 0:
                    res = tmp_res
                else:
                    res = torch.cat([res,tmp_res],dim=-1)
                    
            res = res.reshape(-1,self.city_num,self.pred_len, 1)

        else:
            new_x = self.decoder(new_x,self.w,g_edge_index,g_edge_w,edge_index,edge_w)
            res = self.predMLP(new_x)
            res = res.reshape(-1,self.city_num,self.pred_len, 1)

        return res

class DecoderModule(nn.Module):
    def __init__(self,x_em,edge_h,gnn_h,gnn_layer,city_num,group_num):
        super(DecoderModule, self).__init__()
        self.city_num = city_num
        self.group_num = group_num
        self.gnn_layer = gnn_layer
        self.x_embed = Lin(gnn_h, x_em)
        self.group_gnn = nn.ModuleList([NodeModel(x_em,edge_h,gnn_h)])
        for i in range(self.gnn_layer-1):
            self.group_gnn.append(NodeModel(gnn_h,edge_h,gnn_h))
        self.global_gnn = nn.ModuleList([NodeModel(x_em+gnn_h,1,gnn_h)])
        for i in range(self.gnn_layer-1):
            self.global_gnn.append(NodeModel(gnn_h,1,gnn_h))

    def forward(self,x,trans_w,g_edge_index,g_edge_w,edge_index,edge_w):
        x = self.x_embed(x)
        x = x.reshape(-1,self.city_num,x.shape[-1])
        w = Parameter(trans_w,requires_grad=False).to(x.device,non_blocking=True)
        w1 = w.transpose(0,1)
        w1 = w1.unsqueeze(dim=0)
        w1 = w1.repeat_interleave(x.size(0), dim=0)
        g_x = torch.bmm(w1,x)
        g_x = g_x.reshape(-1,g_x.shape[-1])
        for i in range(self.gnn_layer):
            g_x = self.group_gnn[i](g_x,g_edge_index,g_edge_w)
        g_x = g_x.reshape(-1,self.group_num,g_x.shape[-1])
        w2 = w.unsqueeze(dim=0)
        w2 = w2.repeat_interleave(g_x.size(0), dim=0)
        new_x = torch.bmm(w2,g_x)
        new_x = torch.cat([x,new_x],dim=-1)
        new_x = new_x.reshape(-1,new_x.shape[-1])
        # print(new_x.shape,edge_w.shape,edge_index.shape)
        for i in range(self.gnn_layer):
            new_x = self.global_gnn[i](new_x,edge_index,edge_w)

        return new_x


class NodeModel(torch.nn.Module):
    def __init__(self,node_h,edge_h,gnn_h):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(node_h+edge_h,gnn_h), ReLU(inplace=True))
        self.node_mlp_2 = Seq(Lin(node_h+gnn_h,gnn_h), ReLU(inplace=True))

    def forward(self, x, edge_index, edge_attr):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        edge_index, edge_attr = edge_index.to(x.device), edge_attr.to(x.device)
        row, col = edge_index
        # ipdb.set_trace()
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)
