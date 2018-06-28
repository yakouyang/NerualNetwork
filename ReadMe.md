# Nerual Network
Implementation of NerualNetwork with python from scratch (without use any ml libraries/framework)   
Detailed code comments in the source code

## Usage
In linux terminal, cd to source directory   
`python nerual_network.py`    

nerual_network.py mainly contains three different functions   
- Part 1: Gradient check
- Part 2: Retraining a network
- Part 3: Loading a trained nNetwork to estimate performance

Each part is separated by a comment character. By default, only Part3 has no comment, and it is convenient to quickly view the trained network effect. To show other sections, uncomment the corresponding section comment and comment out the remaining parts
## Default dataset
The hand-written digital set standard library **MNIST** is selected as the data set, the original data version in the .npz compression format is used, and the data set is derived from (the source code is stored)
https://s3.amazonaws.com/img-datasets/mnist.npz
## Requirements
python3    
numpy    
matplotlib    
datetime    
pickle    
