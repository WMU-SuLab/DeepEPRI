# DeepEPRI
DeepEPRI: A  deep learning framework for identifying eRNA-auRNA interactions
## Framework
![image](https://github.com/WMU-SuLab/DeepEPRI/blob/main/images/workflow.jpg)
## Overview
* The folder __"data"__ contains the initial data (eRNA-auRNA pairing data) for seven cell lines in csv format.There is also an R script for data enhancement (`Data_Augmentation.R`,A tool of data augmentation provided by Mao et al. (2017). The details of the tool can be seen in https://github.com/wgmao/EPIANN.)\
* The folder __"best_weights"__
* The file __"model.py"__ is the code of the network architecture.
* The file __"train.py"__ is the code of the entire model.
* The file __"test.py"__ is the code of the entire model.
## Dependency
* Python 3.10.14\
* tensorflow 2.15.1\
* scikit-learn 1.5.1\
* scipy 1.14.0\
* numpy 1.26.4
## Usage
### Step 0. Setup environment
* First, in order to avoid conflicts between the project's packages and the user's commonly used environment, we recommend that users create a new conda virtual environment named dl through the following script:\
`conda create -n dl python=3.10`\
`conda activate dl`\
* Later, users can install all dependencies of the project by running the script:\
`pip install poetry`\
* Then download the pyproject.toml provided with this project and go to the folder where this file is stored to activate the environment:\
`poetry install`\
* Download the `bedtools` for subsequent data processing.
### Step 1. Prepare dataset
#### Users who have a need for data augmentation\
* Provide `a csv file` and use the script in the `data folder` to process the data, which will result in paired `fasta format files` for the training set and test set:\
`Rscript Data_Augmentation.R`\
* The data can then be converted into a format that the model can handle:\
`python sequence_processing.py`
#### Users who have no need for data augmentation\
* Provide two bed files and then use the bedtools tool to convert them to fasta format:\
`bedtools getfasta -fi hg19.fa -bed test.bed -name`\
* Then encode the data:\
`python sequence_processing.py`
### Step 3. Run DeepEPRI
* If you need to train the model from scratch:\
`python train.py`\
* Just need to test:\
`python test.py`
