# Introduction 
This repository contains the code to GLASS. We have made some improvements to align the code more closely with pep-8 standards and written tests for some basic functionalities. 

# Repository overview 

The root directory contains scripts for training models. ```GNNEmb.py``` is used to pretrain GNNs to obtain node embeddings. This can be done with 
```
python GNNEmb.py --use_nodeid --device $gpu_id --dataset $dataset --name $dataset
```
Following this, the model can be trained using ```GLASSTest.py``` by 
````
python GLASSTest.py --use_nodeid --use_seed --use_maxzeroone --repeat 10 --device $gpu_id --dataset $dataset
````

Common across both scripts are the functions ```split``` and ```buildModel``` which are used to split the dataset into different splits and initialize a GNN model given specific hyperparameters respectively.  
```GNNEmb.py``` additionally contains some functions to perform hyperparameter optimization using ```optune```

The implementation of the model itself as well as some other utility functions is contained in the ```impl``` directory. 

- ```config``` and ```metrics``` - Helper functions for setting the device for training and computing evaluation metrics respectively.  
- ```utils``` - Helper functions for padding input sequences and performing Max-Zero-One labelling (discussed in the paper to improve training speed)
- ```train``` - Helper functions for training and evaluation loops 
- ```SubGDataset``` - Dataset and Dataloader helper functions for loading subgraph labelled input sequences. Also contains dataset for Max-Zero-One labelling  
- ```models``` - Contains all model specific implementations. These can be broadly divided into three types - message passing model, GLASS model, pretraining model. A specific type of message passing network that transforms input features and mixes them with the original input features is used. The GLASS model uses this message passing network. The pretraining model is trained via link prediction task and is used to generate node embeddings. 