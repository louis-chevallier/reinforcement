
start :
	python v0.py --train
	python v0.py 

install :
	pip install "gymnasium[classic-control]" torchrl
	pip install "gymnasium[classic-control]"
	mamba install --quiet -y networkx
	 pip install graphviz

