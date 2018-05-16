# NJNMF
Non-Linear Joint Weighted NMF for attributed graph

## Description
NJNMF is a clustering algorithm for attributed graph. It outperforms existing methods by using a non-linear function.

## How to use?
     $python implement.py -f FILE_NUMBER -m MODEL_NAME -k No.TOPIC1 -k2 No.TOPIC2

file_list = ["disney","amazon","enron","WebKB","citeseer","cora"]

program descriptions are below:

### implement.py
  
    This program is for implementation of NJNMF.
    Before you execute, type "python implement.py -h" and check options.
    
### njnmf.py (drop.py, slow.py)
  
    NJNMF core model is described in this program.
    
### jwnmf.py
  
    The existing model is described in this program(https://drive.google.com/file/d/0B3KyfUC4uNxtZUJJUEdQdmp1U2M/view).

### evaluate.py
  
    This program include these criteria (modularity, entropy, NMI).
     
### att_clustering.py
  
    kmeans algorithm is written in this program.
    
    ex. $python att_clustering.py -f FILE_NUMBER -k No.CLUSTERS -r REGULARIZATION
    
### draw_graph.py
    
    This program draws the graph from an adjacency matrix and a list of clusters which nodes belong to. It makes us easily understand the overview of the clustering result. 
