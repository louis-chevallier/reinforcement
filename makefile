
start :
	python v0.py --load --train --start_episodes=10000 --episodes=30000

sss :v
	python v0.py --load --train
	python v0.py --load --eval

install :
	pip install "gymnasium[classic-control]" torchrl
	pip install "gymnasium[classic-control]"
	mamba install --quiet -y networkx
	 pip install graphviz

