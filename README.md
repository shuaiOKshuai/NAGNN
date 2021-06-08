# NAGNN

# Neighbor-Anchoring Adversarial Graph Neural Networks

We present the datasets and code for our paper "Neighbor-Anchoring Adversarial Graph Neural Networks" (NAGNN for short), which is published in TKDE (2021).


## 1. Description for each file

	Folders:
		- NAGNN : the code of NAGNN model
		- datasets : the datasets
	
	Inside files of NAGNN folder:
		- paramsConfigPython : parameters setting file, the parameters can be modified in this file
		- mainGAN.py : the entry of training NAGNN
		- AdGCNTraining.py : the training body of NAGNN
		- generator.py : the generator model, in which we define the generator G
		- discriminator.py : the discriminator model, in which we define the discriminator D
		- processTools.py : some tool functions
	

## 2. Requirements (Environment)

	python-3.6.5
	tensorflow-1.13.1


## 3. How to run

- (1) First configure the parameters in file "paramsConfigPython";
- (2) Run "python3 mainGAN.py".
	
	remark : Command "python3 mainGAN.py" would call the pretrain (including D and G), train, validation and test together, and finally output the prediction results on the test data.


## 4. Datasets

The datasets could be downloaded from the links in the paper, and we also include the datasets in folder "datasets".
The readers can also prepare their own datasets. For the own data: five files should be prepared, including graph.node (describing the nodes information), graph.edge (describing the edges information), train_nodes, val_nodes and test_nodes, as follows,
- (1) node file ( graph.node )
	- The first row is the number of nodes + tab + the number of features
	- In the following rows, each row represents a node: the first column is the node_id, the second column is the label_id of current node, and the third to the last columns are the features of this node. All these columns should be split by tabs.
- (2) edge file ( graph.edge )
	- Each row is a directed edge, for example : 'a tab b' means a->b. We can add another line 'b tab a' to represent this is a bidirection edge. All these values should be split by tabs.
- (3) train_nodes
	- Including all the node ids for training, with each row corresponding to a node id.
- (4) val_nodes
	- Including all the node ids for validation, with each row corresponding to a node id.
- (3) test_nodes
	- Including all the node ids for test, with each row corresponding to a node id.


## 5. Note

We provide the code for NAGCN in folder "NAGNN" with base model GCN. For other GNN models, the readers can form the corresponding NA*** versions by replacing the current GNN model in Discriminator.


## 6. Cite

	@article{liu2021neighboranchoring,
		title={Neighbor-Anchoring Adversarial Graph Neural Networks},
		author={Liu, Zemin and Fang, Yuan and Liu, Yong and Zheng, Vincent W.},
		journal={IEEE Transactions on Knowledge and Data Engineering (TKDE)},
		year={2021},
		publisher={IEEE}
	}
			
