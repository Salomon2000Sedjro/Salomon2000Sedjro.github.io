---
layout: post
title:  "Efficient Novelty Detection Methods for Early Warning of Potential Fatal Diseases:"
categories: Machine_Learning_Algorithms
permalink: /posts/early_warning
image: "https://github.com/Salomon2000Sedjro/Salomon2000Sedjro.github.io/blob/6c77cd7675370607b4604fa225feaf0d4cdb6bcc/photos/Early%20Warning/early_warning_head_1.png?raw=true"
---

![EarlyWarning](https://github.com/Salomon2000Sedjro/Salomon2000Sedjro.github.io/blob/6c77cd7675370607b4604fa225feaf0d4cdb6bcc/photos/Early%20Warning/early_warning_head_1.png?raw=true)

## Introduction
An early warning approach in healthcare is an algorithm that provides
information about upcoming risks to susceptible people before a Critical
Health Episode occurs. This enables action to be taken to reduce possible
harm and, in some situations, prevent it from occurring. Therefore, it is
of great importance to the reduction of mortality in healthcare.
Acute Hypotensive Episodes (AHE) and Tachycardia Episodes (TE)
are two of the most dangerous Critical Health Episodes in the Intensive
Care Units.

**Acute Hypotensive Episodes (AHE):** Any interval of 30 minutes
during which at least 90% of the Mean Arterial blood Pressure
(MAP) values are below 60 mmHg (millimeters of mercury)

**Tachycardia Episodes (TE):** Any interval of 30 minutes during
which at least 90% of Heart Rate (HR) measurements are over 100
bpm (beats per minute)

## Proposed Method
In general, existing warning systems for AHE and TE early prediction
typically have limited performance and high computational costs in
real-time alerting when dealing with large amounts of data. So, this work
presents MIG-LightGBM, a highly efficient warning system based on a
feature selection process with Mutual Information Gain (MIG) and an
episode classification approach with the predictive model Light Gradient
Boosting Machine (LightGBM)

![EarlyWarningSystem](https://github.com/Salomon2000Sedjro/Salomon2000Sedjro.github.io/blob/92e4a0ac4970008ae45fc2d0a50d92ee822cef3b/photos/Early%20Warning/EW_system.PNG?raw=true)

## Data compilation
The data used in this research is a subset of the Multi-parameter
Intelligent Monitoring for Critical Care (MIMIC) II database [1]. It
contains minute-by-minute time series of Heart Rate (HR), Systolic
Blood Pressure (SBP), Diastolic Blood Pressure (DBP), and Mean
Arterial blood Pressure (MAP) arranged into records, each of which
corresponds to an adult patient’s ICU stay.
The dataset was traversed by sub-sequences, each one being subdivided
into three windows:

**Target Window:** The period during which a Critical Health Event
can occur.

**Observation Window:** Observation period containing data to be
used to predict what will happen in the Target Window.

**Warning Window:** The gap between the Observation Window and
the Target Window.

![TwoSubsequences](https://github.com/Salomon2000Sedjro/Salomon2000Sedjro.github.io/blob/2abc66abbdd44706b6de58126a7929df1c4155f0/photos/Early%20Warning/conseq_subseq.PNG?raw=true)

## Feature extraction
Feature extraction, also known as feature engineering, is the process of
extracting features from raw data using domain expertise. The objective is
to utilize these extracted features to increase the quality of a machine
learning model outputs when compared to simply giving the raw data to
the machine learning model.
In this work, the feature extraction process was carried out according to
three techniques.

**Knowledge-based features.**

**Statistical features.**

**Cross-correlation features.**

**Wavelet features.**

The total number of features considered for the analysis was 111

## Feature Selection
Feature selection is an important phase in data cleaning. It not only
removes the redundant data but also aids in the discovery of the
most relevant ones, hence improving the model’s performance. Apart
from the feature importance ranking provided by the proposed predictive
model, the Mutual Information Gain approach was also used for the
feature selection process.

**Mutual Information Gain:**

Mutual Information Gain (MIG), as one of the most efficient feature
selection methods, is a measure of the "mutual dependency" of two
random variables. MIG creates a quantifiable link between a feature
and the target. Using MIG as a feature selector has two advantages:
It is model-neutral, which means it can be applied to a wide range of
machine learning models; and it is also fast.

Let xj be a feature and y the target variable. The Mutual Information
Gain for the two discrete random variables xj and y is given by

![MIG_formula](https://github.com/Salomon2000Sedjro/Salomon2000Sedjro.github.io/blob/012cd345f659d1d5da40c7d51f15105bcc4704f4/photos/Early%20Warning/MIG_formula.PNG?raw=true)

where px and py are the marginal probability and pxy is the joint
probability.

It is strongly related to the notion of entropy. This is due to the fact that
it may also be defined as the reduction of uncertainty of a random variable
if another is known. The definition of I(xj; y) can be rewritten as:

I(xj; y) = H(xj) − H(xj|y).

Given a set of features XS = {x1, ..., xn} and a single feature y, the Joint
Mutual Information between them is given by

![MIG_formula_2](https://github.com/Salomon2000Sedjro/Salomon2000Sedjro.github.io/blob/d874b3363eacb8c424dcef1c3e6bd87b67747ff3/photos/Early%20Warning/MIG_formula_2.PNG?raw=true)

**A process for selecting a subset of important features**

Let |S| = k. be the number of features to be selected.
The feature selection process should identify a subset of features
XSˆ = {xi1, ..., xik}, which maximizes the Joint Mutual Information
I(XS; y)) between the class label y and all possible feature subsets XS of
size k

![MIG_formula_3](https://github.com/Salomon2000Sedjro/Salomon2000Sedjro.github.io/blob/d874b3363eacb8c424dcef1c3e6bd87b67747ff3/photos/Early%20Warning/MIG_formula_3.PNG?raw=true)

**Greedy forward step-wise selection**

According to their Mutual Information with respect to the target y,
the features are ranked. The top feature is then selected.

Let XSt−1 = {xi1, ..., xit−1} denote the set of features that were
chosen at step t − 1.

Choose the next feature xit in such a way that the greatest
improvement in Joint Mutual Information is achieved by using XSt.
So,

![Greedy](https://github.com/Salomon2000Sedjro/Salomon2000Sedjro.github.io/blob/08a90ebf146343d9aa344e4e320e7a61650576c0/photos/Early%20Warning/Greedy.PNG?raw=true)


## RASSEL Algorithm

![algorithm_layout](https://github.com/Salomon2000Sedjro/Salomon2000Sedjro.github.io/blob/61dbeec442f92d0692c87fc34a2f8bcb22ea7217/photos/algo1.PNG?raw=true)

One of the most crucial ingredients in the above proposed algorithm is the dimension q of the subspace, because its value has a strong bearing on an important aspect of  the correlation among the base learners.
The book $Random Forests, Machine Learning, L. Breiman, 45 (2001), 532. recommends reasonable values;

### If p<<n
For classification, <img src="https://latex.codecogs.com/svg.image?q=[\sqrt{p}]" /> and for regression, <img src="https://latex.codecogs.com/svg.image?q=[p/3]" />
### If p>>n
For classification, <img src="https://latex.codecogs.com/svg.image?q=min([n/5],[\sqrt{p}])" /> and for regression, <img src="https://latex.codecogs.com/svg.image?q=min([n/5],[p/3])" />

### Variance of the ensemble prediction function
<img src="https://latex.codecogs.com/svg.image?\begin{align*}\mathbb{V}[\widehat{f}^{(L)}(.)]&space;&=&space;\mathbb{V}\biggl[\dfrac{1}{L}\sum_{l=1}^{L}\widehat{g}^{(l)}(.)\biggr]\\\&space;&=&space;\dfrac{1}{L^2}\biggl[\sum_{l=1}^{L}\mathbb{V}[\widehat{g}^{(l)}(.)]&plus;\sum_{l=1}^{L}\sum_{l'\ne&space;l}^{L}cov\bigg(\widehat{g}^{(l)}(.),\widehat{g}^{(l')}(.)\bigg)\biggr]\\\&space;&=&space;\dfrac{1}{L^2}\biggl[\sum_{l=1}^{L}\sigma^2&plus;\sum_{l=1}^{L}\sum_{l'\ne&space;l}^{L}\rho\sigma^2\biggr]\\\&space;&=&space;\dfrac{1}{L^2}\biggl[L\sigma^2&plus;L(L-1)\rho\sigma^2\biggr]\\\mathbb{V}[\widehat{f}^{(L)}(.)]&space;&space;&=&space;\dfrac{\sigma^2}{L}&plus;\dfrac{(L-1)}{L}\rho\sigma^2\\where\&space;\sigma^2&space;=&space;\mathbb{V}[\widehat{g}^{(L)}(.)]\&space;&and\&space;\rho\sigma^2&space;=&space;cov\bigg(\widehat{g}^{(l)}(.),\widehat{g}^{(l')}(.)\bigg)\&space;for\&space;l\ne&space;l'\end{align*}" />

## DATA-DRIVEN WEIGHTING SCHEME 			
### In regression
The proposed weighting scheme proposed in the RASSEL algorithm is generated by calculating the correlation <img src="https://latex.codecogs.com/svg.image?r_j" /> between each predictor <img src="https://latex.codecogs.com/svg.image?x_j" /> and the response variable y.
### In classification
We consider using the corresponding F statistic 

<img src="https://latex.codecogs.com/svg.image?F_j&space;=&space;\dfrac{(n-2)r_j^2}{r_j^2}" />
    
## EXTRACTING IMPORTANT FEATURES
### Algorithm for regression

![first_algorithm_layout](https://github.com/Salomon2000Sedjro/Salomon2000Sedjro.github.io/blob/9d2e5dc069d12c3109216d26240fe694be5fb5a6/photos/algo2.PNG?raw=true)

### Algorithm for classification

![second_algorithm_layout](https://github.com/Salomon2000Sedjro/Salomon2000Sedjro.github.io/blob/9d2e5dc069d12c3109216d26240fe694be5fb5a6/photos/algo3.PNG?raw=true)

## IMPLEMENTATION
### "Algorithm 2" in Python

```python

def EIV_regression(df, ypos):
    n, p = df.shape[0], df.shape[1]-1
    # Computing q 
    if df.shape[0] >= df.shape[1]:
        q = int(math.floor(p/3))
    else:
        q = int(np.min([math.floor(n/5), math.floor(p/3)]))
    # Generating the weighting schemes    
    r = []
    for j in range(p+1):
        if j != ypos-1:
            corr, _ = pearsonr(df.iloc[:,ypos-1], df.iloc[:,j])
            r.append(corr)
    vect_pi = [(i**2)/(la.norm(r)**2) for i in r]
    # Drawing q features  
    basis = []
    for i in range(q):
        basis.append(np.random.choice(p, 1, replace=False, p=vect_pi))
    return basis, vect_pi

```
### A representative simulation results for regression analysis on real dataset: "gifted.csv".
```python
df1 = pd.read_csv("gifted.csv")
plt.xlabel("Important variables extracted for regression")
plt.ylabel("Prior Feature Importance")

impR = EIV_regression(df1, 1)
plt.bar(range(1,df1.shape[1]), impR[1]);
label=["fatheriq","motheriq","speak","count","read","edutv","cartoons"]
plt.xticks(range(1,df1.shape[1]),label)
```

![regression](https://github.com/Salomon2000Sedjro/Salomon2000Sedjro.github.io/blob/56667934ab8fc4278c2b4777e152818057b6ae83/photos/regression.png)

### "Algorithm 3" in Python
### Extracting Important variables for classification

```python
def EIV_classification(df, ypos):
    n, p = df.shape[0], df.shape[1]-1
    # Computing q
    if df.shape[0] >= df.shape[1]:
        q = int(math.floor(np.sqrt(p)))
    else:
        q = int(np.min([math.floor(n/5), math.floor(np.sqrt(p))]))  
    # Generating the weighting schemes
    F = []
    for j in range(p+1):
        if j != ypos-1:
            y = df.iloc[:,ypos-1].values
            x = df.iloc[:,j].values
            grps = pd.unique(y)
            d_data = {grp:x[y == grp] for grp in grps}
            f, pval = stats.f_oneway(list(d_data.values())[0],list(d_data.values())[1])
            F.append(f)
    vect_pi = [(i)/(np.sum(F)) for i in F]
    # Drawing q features
    basis = []
    for i in range(q):
        basis.append(np.random.choice(p, 1, replace=False, p=vect_pi))
    return basis, vect_pi     

```
### A representative simulation results for classification analysis on real dataset: "prostate-cancer-1.csv".
```python
df = pd.read_csv("prostate-cancer-1.csv")
plt.xlabel("Important variables extracted for classification")
plt.ylabel("Prior Feature Importance")

imp = EIV_classification(df, 1)
plt.bar(range(1,df.shape[1]), imp[1]);
```

![classification](https://github.com/Salomon2000Sedjro/Salomon2000Sedjro.github.io/blob/36c05e4752ace1012811aa73118f9b01414abd32/photos/classification.png)

## COMPUTATIONAL DEMONSTRATION

![comparison](https://github.com/Salomon2000Sedjro/Salomon2000Sedjro.github.io/blob/bd1e63fb11d34d0eb8850d6d9def93ea564ecffe/photos/compare.png)

## CONCLUSION

This developed adaptive RASSEL algorithm outperforms many classifier
ensembles. It reaches the highest accuracy when the number of features
is large as well as the number of instances. In addition, it performs good
when there are redundant features on the dataset But it has limitations !
For instance, this method can not deal with dataset that has categorical
features. Instead it necessities to encode these features numerically. In
addition, the algorithm fails to select the optimal feature subsets, when
the number of features are very small.
To improve RASSEL, we can generalize it in such a way that all base
learners can be adapted easily
