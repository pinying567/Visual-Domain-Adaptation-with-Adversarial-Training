# Visual Domain Adaptation with Adversarial Training

**Author: Pin-Ying Wu**

**Table of contents**
- Overview
- Code
- Result Analysis

## Overview
### Task
Implement Domain Adaptation (from <span style="color: CornflowerBlue;">Source domain</span> to <span style="color: ForestGreen;">Target domain</span>) with three different datasets, USPS, MNIST-M, and SVHN in three following scenarios:

1) <span style="color: CornflowerBlue;">USPS</span> → <span style="color: ForestGreen;">MNIST-M</span>
2) <span style="color: CornflowerBlue;">MNIST-M</span> → <span style="color: ForestGreen;">SVHN</span>
3) <span style="color: CornflowerBlue;">SVHN</span> → <span style="color: ForestGreen;">USPS</span> <br>

- **Baseline Model (DANN)**:
<img src=asset/dann_arch.png width=80%> <br><br>

- **Improved Model (ADDA)**:

<img src=asset/adda_arch.png width=80%> <br>

- Reference : [Tzeng et al., “Adversarial Discriminative Domain Adaptation”, CVPR 2017](https://arxiv.org/pdf/1702.05464.pdf)
- For the improved model, we adopt the idea of Adversarial Domain Domain Adaptation (ADDA) but keep the same architecture of the feature extractor and the classifier as my baseline DANN model.
- The main idea of ADDA is to train a different feature extractor for the target domain and keep the feature extractor and the classifier of the source domain fixed.
- There are two training stages:
    1) Pre-training: Training the feature extractor and the classifier on the source domain, which is the same as the source-only model in baseline since I adopt the same architecture.
    2) Adversarial Adaptation: Finetuning the target domain feature extractor from the source domain pretrained model. In this stage, the source feature extractor and the classifier are fixed, and only the target feature extractor is trained with the discriminator.
- Different from DANN, ADDA does not use the negative of the discriminator loss but use the cross-entropy loss of the inverted label. This will separate the learning of feature extractor and discriminator, but also be able to encourage the feature extractor to generate source-like features. In evaluation time, the feature extractor for the target domain and the classifier trained on source domain are combined to classify samples in the target domain.


### Dataset
- **USPS**:
A digit dataset contains centered and normalized `16×16` pixel grayscale handwritten digits.
- **MNIST-M**:
A digit dataset contains `28x28` pixel digits combined from MNIST digits with the patches randomly extracted from color photos of BSDS500 as their backgrounds.
- **SVHN**:
A digit dataset contains centered `32x32` pixel RGB printed digits.


## Code
### Prerequisites
```
pip install -r requirements.txt
```

### Data Preparation
```
bash ./get_dataset.sh
```
The shell script will automatically download the dataset and store the data in a folder called `digit_data`.

### Training
1. Baseline model
```
bash ./train_baseline_adda.sh
```

2. Improved model
```
bash ./train_improved_adda.sh
```

### Checkpoints
|   baseline  |   improved  |
|:-----------:|:-----------:|
| [<span style="color: CornflowerBlue;">USPS</span> → <span style="color: ForestGreen;">MNIST-M</span>](https://www.dropbox.com/s/xut3hrd1l1p2dap/baseline_usps-mnistm.pkl?dl=1) | [<span style="color: CornflowerBlue;">USPS</span> → <span style="color: ForestGreen;">MNIST-M</span>](https://www.dropbox.com/s/lofx8agzlzq9bu3/improved_usps-mnistm.pkl?dl=1) |
| [<span style="color: CornflowerBlue;">MNIST-M</span> → <span style="color: ForestGreen;">SVHN</span>](https://www.dropbox.com/s/jrpzumh454iruw7/baseline_mnistm-svhn.pkl?dl=1) | [<span style="color: CornflowerBlue;">MNIST-M</span> → <span style="color: ForestGreen;">SVHN</span>](https://www.dropbox.com/s/kk452wtucvqzkuk/improved_mnistm-svhn.pkl?dl=1) |
|  [<span style="color: CornflowerBlue;">SVHN</span> → <span style="color: ForestGreen;">USPS</span>](https://www.dropbox.com/s/zenbd245ray8q9k/baseline_svhn-usps.pkl?dl=1)  |  [<span style="color: CornflowerBlue;">SVHN</span> → <span style="color: ForestGreen;">USPS</span>](https://www.dropbox.com/s/k3utue741nixtwg/improved_svhn-usps.pkl?dl=1)  |

### Evaluation
```
python3 eval.py <path_to_predicted_csv> <path_to_ground_truth>
```


## Result Analysis
### Baseline model (DANN)
<img src=asset/baseline.png width=80%> <br>

- From the tables above, we can observe that SVHN is the hardest dataset that has the lowest accuracy (93.50%) when the model is trained on itself. USPS is relatively easier, so the lower bound of this dataset is the highest (62.38%), and so does the accuracy after the adaptation (69.21%).

<!-- TSNE: usps-mnistm --> 
<img src=asset/baseline_usps-mnistm.png width=80%> <br>

<!-- TSNE: mnistm-svhn --> 
<img src=asset/baseline_mnistm-svhn.png width=80%> <br>

<!-- TSNE: svhn-usps --> 
<img src=asset/baseline_svhn-usps.png width=80%> <br>

- From the t-SNE plot of SVHN-USPS dataset with digit labels (a), we can observe that the data points from different classes are better separated from others in this scenario. However, the model of SVHN adapted from MNIST-M has the largest improvement than the lower bound (+16.17%). This can also be seen in the t-SNE plots of the features from two domains (b), where the features from two domains are better aligned in the scenario of MNISTM-SVHN.
- Similar to the training of GAN, we need to train the discriminator better first so that it can provide useful gradient to train the feature extractor. As a result, the initial weight for domain adversarial loss is set to zero. As the discriminator is trained, the weight is increased to provide more reverse gradient to the feature extractor. When training the model, we found that data augmentation is very useful. Before adopting data augmentation during training, it seemed hard to pass the baseline. The reason is that the number of training samples are limited, so it is easy for the model to overfit. Thus, we adopted rotation as the data augmentation, and it successfully improved the accuracy.

### Improved model (ADDA)
<img src=asset/improved.png width=80%> <br>

<!-- TSNE: usps-mnistm --> 
<img src=asset/improved_usps-mnistm.png width=80%> <br>

<!-- TSNE: mnistm-svhn --> 
<img src=asset/improved_mnistm-svhn.png width=80%> <br>

<!-- TSNE: svhn-usps --> 
<img src=asset/improved_svhn-usps.png width=80%> <br>

- We found it is difficult to tune the hyperparameters of ADDA model. For training ADDA, since the classifier is fixed, if the learning rate is too large, the feature extractor will make too huge step and the classifier will not be matched with the extracted feature. However, if the learning rate is too small, the model can hardly learn the feature adaptation. After several trials, we found that the learning rate of the discriminator needs to be larger so that the discriminator can provide informative gradient for training the feature extractor. In addition, for training the discriminator, we found that the big difference of quantity between the source domain and target domain data will make the training of discriminator unbalanced, so it tends to guess one of them directly. When implementing the model, it is better to use the same amount of source domain data and target domain data to train the discriminator.