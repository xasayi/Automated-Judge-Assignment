# Who is a Better Matchmaker?

This repository contains code for the paper "Who is a Better Matchmaker? Human vs. Algorithmic Judge Assignment in a High-Stakes Startup Competition" by Sarina Xi, Orelia Pi, Miaomiao Zhang, Becca Xiong, Jacqueline Ng Lane, and Nihar B Shah, The structure of the directory is as follows:

1. `tutorial.ipynb` contains example usage of the code and algorithms. 
2. `similarity` folder contains similarity matrix calculations.
3. `evaluate.py` contains code for evaluating the similarity models.
4. `ensemble.py` contains code for creating an ensemble learner from similarity models.
5. `preprocess_data.py` contains code for preprocessing input data.
6. `constants.py` contains variables used while preprocessing data.
7. `example` and `wikipedia_files` are toy datasets that are used in the tutorial. 

## Requirements

To install requirements:
```
conda env create -f environment.yaml
```

## Usage

### Example Use
To understand the code functionality, go to tutorials.ipynb, it outlines example usage of similarity matrix calculations, evaluation, and ensemble optimization. 