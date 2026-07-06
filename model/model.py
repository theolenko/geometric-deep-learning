import torch
import torch.nn.functional as F

from torch_geometric.nn import SplineConv

class ConvNet(torch.nn.Module): 
    def __init__(self, dim, n_responses, kernel_size=5):
        super().__init__()
        
        
        """ Claude proposal: 
        dim = 1  ->  kernel_size = 25   ->  25  control points
        dim = 2  ->  kernel_size = 5    ->  25  control points   (5²)
        dim = 3  ->  kernel_size = 3    ->  27  control points   (3³)"""
        
        self.dim = dim
        self.n_res = n_responses
        self.ks = kernel_size
        
        self.conv1 = SplineConv(in_channels=self.n_res,out_channels=8,dim=dim,kernel_size=self.ks)
        self.bn1 = torch.nn.BatchNorm1d(8)

        self.conv2 = SplineConv(in_channels=8,out_channels=16,dim=dim,kernel_size=self.ks)
        self.bn2 = torch.nn.BatchNorm1d(16)
        
        self.conv3 = SplineConv(in_channels=16, out_channels=32, dim=dim, kernel_size=self.ks)
        self.bn3 = torch.nn.BatchNorm1d(32)
        
        self.conv4 = SplineConv(in_channels=32, out_channels=32, dim=dim, kernel_size=self.ks)
        self.bn4 = torch.nn.BatchNorm1d(32)

        self.conv5 = SplineConv(in_channels=32, out_channels=32, dim=dim, kernel_size=self.ks)
        self.bn5 = torch.nn.BatchNorm1d(32)
        
        self.conv6 = SplineConv(in_channels=32, out_channels=16, dim=dim, kernel_size=self.ks)
        self.bn6 = torch.nn.BatchNorm1d(16)

        self.conv7 = SplineConv(in_channels=16, out_channels=8, dim=dim, kernel_size=self.ks)
        self.bn7 = torch.nn.BatchNorm1d(8)

        self.conv8 = SplineConv(in_channels=8, out_channels=2, dim=dim, kernel_size=self.ks)
        
        
    def forward(self, data):
        x, edge_index, pseudo = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, pseudo))
        x = self.bn1(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv2(x, edge_index, pseudo))
        x = self.bn2(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv3(x, edge_index, pseudo))
        x = self.bn3(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv4(x, edge_index, pseudo))
        x = self.bn4(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv5(x, edge_index, pseudo))
        x = self.bn5(x)
        x = F.dropout(x, p=.10, training=self.training)
        
        x = F.elu(self.conv6(x, edge_index, pseudo))
        x = self.bn6(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv7(x, edge_index, pseudo))
        x = self.bn7(x)
        x = F.dropout(x, p=.10, training=self.training)
        
        x = F.elu(self.conv8(x, edge_index, pseudo))
        return x
        
        

    
    