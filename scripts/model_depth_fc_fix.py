import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse

class GAT_TimeSeriesLayer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, obs_seq_len, pred_seq_len, num_heads):
        super(GAT_TimeSeriesLayer, self).__init__()
        self.pred_seq_len = pred_seq_len
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        # Embedding layer
        self.fc1 = nn.Linear(in_features, 32)
        
        # GAT layers
        self.gat1 = GAT_Layer(32, 32, num_heads)
        self.gat2 = GAT_Layer(32, 32, num_heads)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(32, 32, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(32, 32, num_layers=2, batch_first=True)
        
        # Fully connected output layers
        self.out1 = nn.Linear(in_features=32, out_features=hidden_features)
        self.out2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        
        # Conv1D for time sequence prediction
        self.conv1d = nn.Conv1d(in_channels=obs_seq_len, out_channels=pred_seq_len, kernel_size=3, padding=1)
        
        # Activation and dropout
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, adj_matrix):
        batch_size, seq_len, num_nodes, num_features = x.size()

        # Embedding
        x_emb = self.prelu(self.fc1(x.view(-1, num_features)))  # [B * S * N, 32]
        x_emb = x_emb.view(batch_size, seq_len, num_nodes, -1)  # [B, S, N, 32]

        # LSTM processing (no loop, batch process)
        x_emb = x_emb.permute(0, 2, 1, 3).contiguous()  # [B, N, S, F]
        x_emb = x_emb.view(batch_size * num_nodes, seq_len, -1)  # [B*N, S, F]
        lstm_out, _ = self.lstm1(x_emb)
        lstm_out = self.prelu(lstm_out)
        
        # Reshape for GAT layer input
        x_gru = lstm_out.view(batch_size, num_nodes, seq_len, -1).permute(0, 2, 1, 3)  # [B, S, N, F]

        # GAT layer
        x1 = self.gat1(x_gru, adj_matrix)
        x1 = self.dropout(self.prelu(x1)) + x_gru  # Skip connection

        x2 = self.gat2(x1, adj_matrix)
        x2 = self.dropout(self.prelu(x2)) + x1  # Skip connection

        # MLP and LSTM2
        x2 = x2.permute(0, 2, 1, 3).contiguous().view(batch_size * num_nodes, seq_len, -1)
        lstm_out, _ = self.lstm2(x2)
        lstm_out = self.prelu(lstm_out)
        
        # Conv1D for time series prediction
        x3 = lstm_out.view(batch_size, num_nodes, seq_len, -1).permute(0, 2, 1, 3).contiguous()
        x3 = x3.view(batch_size, seq_len, num_nodes * 32)
        x4 = self.conv1d(x3)
        x4 = x4.view(batch_size, self.pred_seq_len, num_nodes, 32)

        # Fully connected output layers
        x5 = self.out1(x4)
        x5 = self.prelu(x5)
        x6 = self.out2(x5)
        x6 = x6.view(batch_size, self.pred_seq_len, num_nodes, self.out_features)

        return x6

class GAT_Layer(nn.Module):
    def __init__(self, in_features, hidden_features, num_heads):
        super(GAT_Layer, self).__init__()
        self.hidden_features = hidden_features
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        assert hidden_features % num_heads == 0
        self.gat_features = hidden_features // num_heads

        # GAT Conv layer with consistent number of heads
        self.gat = GATConv(in_channels=in_features, out_channels=self.gat_features, heads=num_heads)

    def forward(self, x, adj_matrix):
        batch_size, seq_len, num_nodes, num_features = x.size()
        gat_output_reshaped = torch.empty(batch_size, seq_len, num_nodes, self.hidden_features).to(self.device)

        for time_step in range(seq_len):
            x_t = x[:, time_step, :, :].contiguous().view(-1, num_features)
            adj = adj_matrix[:, time_step, :, :]
            adj_t, _ = dense_to_sparse(adj)
            gat_output = self.gat(x_t, adj_t)
            gat_output_reshaped[:, time_step, :, :] = gat_output.view(batch_size, num_nodes, -1)

        return gat_output_reshaped