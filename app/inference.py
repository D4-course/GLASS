import functools
import impl.models as models
import pickle 
import torch 
import torch.nn as nn

def load_density_model():
  hidden_dim = 64
  conv_layer = 1
  dropout = 0
  jk_var = 1
  z_ratio = 0.95
  max_deg, output_channels = 4997, 3
  pool="size"
  aggr="sum"

  conv = models.EmbZGConv(hidden_dim,
                  hidden_dim,
                  conv_layer,
                  max_deg=max_deg,
                  activation=nn.ELU(inplace=True),
                  jk_var=1,
                  dropout=dropout,
                  conv=functools.partial(models.GLASSConv,
                                              aggr=aggr,
                                              z_ratio=z_ratio,
                                              dropout=dropout),
                  
                  gn=True)
  mlp = nn.Linear(hidden_dim * (conv_layer) if jk_var else hidden_dim,
              output_channels)
  pool_fn_fn = {
          "mean": models.MeanPool,
          "max": models.MaxPool,
          "sum": models.AddPool,
          "size": models.SizePool
      }
  if pool in pool_fn_fn:
      pool_fn1 = pool_fn_fn[pool]()
  else:
      raise NotImplementedError

  gnn = models.GLASS(conv, torch.nn.ModuleList([mlp]),
                      torch.nn.ModuleList([pool_fn1]))
  gnn.load_state_dict(torch.load('density.pt', map_location=torch.device('cpu')))
  return gnn 

def test_density_model(gnn, subgraph):
  graph = pickle.load(open('density_graph.pkl', 'rb'))
  output = gnn(*graph, torch.tensor([subgraph])) # subgraph is list of subgraph nodes e.g [1,2] 
  return output 

def load_ppi_model():
  hidden_dim = 64
  conv_layer = 2
  dropout = 0.5
  jk_var = 1
  z_ratio = 0.95
  max_deg, output_channels = 21520, 6
  pool="mean"
  aggr="sum"

  conv = models.EmbZGConv(hidden_dim,
                  hidden_dim,
                  conv_layer,
                  max_deg=max_deg,
                  activation=nn.ELU(inplace=True),
                  jk_var=1,
                  dropout=dropout,
                  conv=functools.partial(models.GLASSConv,
                                              aggr=aggr,
                                              z_ratio=z_ratio,
                                              dropout=dropout),
                  
                  gn=True)
  mlp = nn.Linear(hidden_dim * (conv_layer) if jk_var else hidden_dim,
              output_channels)
  pool_fn_fn = {
          "mean": models.MeanPool,
          "max": models.MaxPool,
          "sum": models.AddPool,
          "size": models.SizePool
      }
  if pool in pool_fn_fn:
      pool_fn1 = pool_fn_fn[pool]()
  else:
      raise NotImplementedError

  gnn = models.GLASS(conv, torch.nn.ModuleList([mlp]),
                      torch.nn.ModuleList([pool_fn1]))
  gnn.load_state_dict(torch.load('ppi_bp.pt', map_location=torch.device('cpu')))
  return gnn 

def test_ppi_bp_model(gnn, subgraph):
  graph = pickle.load(open('ppi_bp_graph.pkl', 'rb'))
  output = gnn(*graph, torch.tensor([subgraph])) # subgraph is list of subgraph nodes e.g [1,2] 
  return output 