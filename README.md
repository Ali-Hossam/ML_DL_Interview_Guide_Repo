# **Machine Learning and Deep Learning Interview Preparation**

Welcome to this Notebook dedicated to preparing for machine learning and deep learning interviews. Whether you're gearing up for a job interview or just looking to brush up on essential concepts, this notebook aims to guide you through key topics with enjoyable visualizations. The notebook is available in both .ipynb and .md formats.

# **Machine learning problems types**

Supervised learning: Samples have output value or label ex. (regression, classification)

Unsupervised learning: patterns are discovered in unlabeled data.
  * clustering: grouping similar datatpoints together.
  * Dimensionality Reduction: Reducing the number of features while retaining important information.
  * Self-supervised learning: generates labels from the input data. (ex. predict missing parts in an image or text)

Semi-Supervised learning: Train the model on a training set that contains both labeled and unlabeled data.

Reinforcement learning: An agent learning to make decisions by interacting with an environment.

Anomaly Detection: Focuses on identifying instances that deviate significantly from the norm in a dataset. It is commonly used for detecting unusual patterns or outliers.
  * Fraud detection
  * Network security
  * Fault detection


# **Machine learning pipeline**

<p align="center">
    <img src="https://static.designandreuse.com/img20/20230306c_1.jpg" width="400"/>
</p>

### Build dataset

1. Data collection
2. Data inspection: look for outliers, missing data or transform data
3. Summary statistics: Identify trends in the data, scale and shape of the data
4. Data visualization: Communicate project findings to stakeholders

---

# **Data Preprocessing**

### Unbalanced dataset

Classifiers will often perform poorly on unbalanced classes.

1. **Dominant class bias:** The majority class overwhelms the model, leading it to prioritize learning its patterns, neglecting the minority classes and resulting in poor performance for them.

2. **Metrics limitations:** Traditional accuracy can be misleading, reflecting good performance on the dominant class while ignoring misclassifications in minority classes.

3. **Learning challenges:** The model struggles to identify the rarer, nuanced patterns of minority classes due to insufficient data, often leading to overfitting or underfitting.

4. **Evaluation difficulties:** Assessing model performance across all classes becomes problematic, masking issues specifically affecting minority classes.

5. **Real-world impact:** Biased models perpetuate unfairness in applications like loan approvals or medical diagnoses, disproportionately impacting minority groups.

1. We can balance the size of samples in classes by either downsampling the larger class or upsampling the minor one.

>Upsampling
* Upsampling happens by duplicating minor data points

* Synthetic oversampling:
    1. start with a point in the minority class
    2. Choose one of KNN
    3. Create a random new point between them
    4. Repeat

>Downsampling
* Nearmiss: Keep points closest to nearby minority points
* Nearmiss II: Keep points that their average distance to the furthest samples of the other class is the smallest
* Nearmiss III: Find KNN of the majority class for each sample of the minority class, then select the neighbors having the large avg distance



>You must do Train test split without any over/under sampling

### High number of features 

To reduce high number of features we can use **Correlation** or **Mutual Information**.

**Correlation:**
- Correlation measures the linear relationship between two variables. It quantifies the extent to which one variable changes when the other variable changes, and it ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation).
- Correlation specifically measures linear associations and may not capture non-linear relationships.

**Mutual Information:**
- Mutual information measures the general dependence between two variables. It captures both linear and non-linear relationships and is more flexible in detecting any kind of association.
- Mutual information is based on information theory and doesn't assume a specific form of relationship between variables.

In the context of machine learning feature selection, mutual information can be used to identify and select relevant features. If two features have high mutual information, it suggests that they contain similar or redundant information. In such cases, you might choose to keep only one of the features to reduce dimensionality and improve computational efficiency.

### Creating Features

1. **Mathematical Transformations:**
   - Mathematical transformations involve operations such as addition, multiplication, logarithm, normalization, etc., on existing features. However, it's essential to strike a balance. Too complex transformations may introduce noise or make it harder for the model to learn.

2. **Counts:**
   - Count-based features involve tallying the occurrences of certain conditions, such as counting the number of data points with features greater than a specific number. This can be useful for capturing certain patterns or behaviors in the data.

3. **Building Up and Breaking Down Features:**
   - Creating new features by breaking down or building up existing ones can provide additional information. For example, splitting a string into individual components or combining two categorical features can reveal hidden patterns.

4. **Group Transformation:**
   - Group transformations involve aggregating or summarizing data based on certain groups or categories. This can help capture group-wise patterns or trends in the data.


## Terminologies
1. Hyperparameters: parameters that are not changing during training

2. **Bias Variance tradeoff**

    The bias-variance tradeoff refers to the balance between bias (underfitting) and variance (overfitting). Bias is related to the error introduced by approximating a real-world problem, which may be complex, by a too simple model. Variance, on the other hand, is the error introduced by too much complexity in the model.

    Finding the right balance is crucial to creating a model that generalizes well to new, unseen data. High bias can lead to oversimplification, while high variance can lead to overfitting. The goal is to minimize both bias and variance to achieve the best model performance.

---

# **Model Building**

## Classification

### Logistic regression

Predicts whether something is true or false, it fits an S-shape logistic function , and the curve goes from 0 to 1 which mean that the curve gives you the probability of being "some class" or not. It can be extended to multiclass as well and rather than having a line that seperates classes we would have a hyperplane.

In linear regression, we fit the line using least squares (find the line that minimizes the sum of the squares of these residuals).

In logistic regression, we use something called maximum likelihood.

<p align="center">
    <img src="https://facultystaff.richmond.edu/~tmattson/INFO303/images/logisticregressionanimatedgif.gif" width="400"/>
</p>



#### Maximum likelihood

Assume we want to classify whether an iamge is a dog or not, We pick a curve and compute likelihood of dog classes to be 1 and non dog classes to be 0 in the training dataset, then you multiply all likelihoods for all points and repeat the same step for another curve.

Finally we choose the curve with the maximum likelihhod.

### K-Nearest Neighbors

KNN can be used for both classification and regression. In classification, it uses the majority votes of it's K nearest neighbors. In regression, it takes the average value of it's K nearest neighbors.

#### **Distances**

**Euclidean distance (L2 norm):**

$$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

where:

* $\mathbf{x}$ and $\mathbf{y}$ are vectors representing data points.
* $n$ is the number of features in the data points.
* $x_i$ and $y_i$ are the values of the $i$th feature in $\mathbf{x}$ and $\mathbf{y}$, respectively.

**Manhattan distance (L1 norm):**

$$d(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n}|x_i - y_i|$$


<p align="center">
    <img src="https://media.geeksforgeeks.org/wp-content/uploads/20231207103856/KNN-Algorithm-(1).png" width="400"/>
</p>




#### **Pros**
1. Simple to implement 
2. No training phase

#### **Cons**
1. Choice of K
2. Computational complexity
3. Memory usage
4. Struggle with imbalanced dataset
5. In high-dimension spaces, the notion of distance becomes less meaningfull (curse of dimensionality)

### Support Vector Machine (SVM)

In logestic regression, if we have a point just to the right of our model line it can be misclassified and we want to avoid the sensitivity of our model.


SVM chooses the best hyperplane that maximizes the margin.

<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:1400/1*z_B0o4JbD0C6gpmcenUc4w.jpeg" width="400"/>
</p>


SVM is sensitive to outliers (tends to overfit), therefore we use reqularization.
In the context of SVM, the goal is to find a hyperplane that separates the classes while maximizing the margin between the classes. The optimization problem involves minimizing a loss function that includes both a term related to the margin and a regularization term. The regularization term helps control the complexity of the model and prevents overfitting.

#### The kernel trick

The kernel trick:
1. **Imagine data in "normal" space** where complex relationships are hard to learn.
2. **The trick:** Map the data to a **higher-dimensional space** using a "kernel function."
3. **In this new space, complex relationships become **linear and simpler** to learn.**
4. **Use regular linear models** (like SVMs) in this new space, but **without explicitly computing** the high-dimensional data points, thanks to the kernel function's magic!



<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:1400/1*mCwnu5kXot6buL7jeIafqQ.png" width="400"/>
</p>

### Decision Tree

Tends to split our dataset into two datasets in every iteration.
1. Select a feature and split into binary tree
2. Continue splitting with our available features
3. Continue untill leaf node is pure (only one class) ((max depth is reached))
 

<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:1400/1*XNRAIk1XTVCt7USKAKcVXw.gif" width="400"/>
</p>

#### **How to choose best number of splits?**
1. Plot learning curve: by plotting the performance (accuracy, mse, etc) on both train and validation set against different number of splits.

3. Prune the tree: train to the maximum depth then start removing some branches

#### **Pros**
1. Handles Non-Linearity
2. Handles any type of data
3. Scaling is not required

#### **Cons**
1. Small change in data greatly affect predictions (Instability)
2. Overfitting
3. Sensitive to imbalanced data

### Bagging : Bootstrap aggregation

an ensemble learning technique that involves training multiple instances of the same learning algorithm on different subsets of the training data. The goal is to improve the overall model's performance and robustness by reducing overfitting and variance. The main idea behind bagging is to introduce diversity among the models by training them on different subsets of the data.

For classification tasks, combine the predictions of individual models using techniques like majority voting. FOr regression takss, lithe predictions are often averaged.

**The primary benefits**

Reduced Overfitting: By training models on different subsets, the ensemble model is less likely to overfit to the noise or peculiarities of any specific subset.

Increased Stability: Bagging helps improve the stability and robustness of the model, particularly when the base model is sensitive to the specific training data.

#### Random Forest

<p align="center">
    <img src="https://1.bp.blogspot.com/-Ax59WK4DE8w/YK6o9bt_9jI/AAAAAAAAEQA/9KbBf9cdL6kOFkJnU39aUn4m8ydThPenwCLcBGAsYHQ/s0/Random%2BForest%2B03.gif" width="400"/>
</p>

### Boosting

Boosting is another ensemble learning technique, like bagging, but it focuses on sequentially training a series of weak learners to correct the errors of the previous ones. The main idea is to give more weight to the instances that were misclassified by earlier models, thus allowing subsequent models to focus on the more challenging examples.

Here's how boosting works:

1. **Train a Weak Learner:**
   - Start by training a weak learner (e.g., a shallow decision tree) on the original dataset.

2. **Assign Weights:**
   - Assign higher weights to the misclassified instances from the previous model, making them more influential in the next round.

3. **Train a New Weak Learner:**
   - Train another weak learner, giving more emphasis to the misclassified instances. This process is repeated iteratively.

4. **Combine Predictions:**
   - Combine the predictions of all weak learners, often using a weighted sum, to form the final boosted model.

Popular algorithms for boosting include AdaBoost (Adaptive Boosting) and Gradient Boosting.


<p align="center">
    <img src="https://media.geeksforgeeks.org/wp-content/uploads/20210707140911/Boosting.png" width="400"/>
</p>

#### **Loss functions**
1. (0 - 1) loss function: multiplies misclassifed points by one, and ignores correctly classified points. The problem is that it is not differentiable so it's difficult to optimize.

2. AdaBoost (adaptive boosting) : very sensitive to outliers

3. Graident Boosting : more robust to outliers than AdaBoost

<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:1037/1*cF2PsxVVPpVB25_cBtC1Kg.png" width="400"/>
</p>


The learning rate controls the contribution of each weak learner to the overall model. A lower learning rate requires more rounds but can lead to better generalization. Experiment with different learning rates.

### Stacking

Stacking, also known as Stacked Generalization, involves training multiple diverse models and combining their predictions using another model, often referred to as a meta-model or blender. Stacking aims to capture the strengths of different base models and improve overall predictive performance.

Here's how stacking works:

1. **Train Multiple Base Models:**
   - Train several diverse base models (e.g., decision trees, support vector machines, neural networks) on the original dataset.

2. **Make Predictions:**
   - Use each base model to make predictions on the validation set (or a subset of the training set that was not seen during training).

3. **Meta-Model Training:**
   - Train a meta-model on the predictions made by the base models. The meta-model takes the predictions of the base models as input features and learns to make the final predictions.

4. **Final Prediction:**
   - Combine the predictions of the base models and the meta-model to make the final prediction.

Stacking allows for more complex relationships between the base models and can potentially achieve better performance than individual models.

<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:809/0*e-na5r7mF8lVAfPK.png" width="400"/>
</p>

## Error metrics 

### 1. Confusion matrix

A matrix is used to evaluate the performance of a classification algorithm.

* True Positives (TP): The number of instances that were correctly predicted as positive (belonging to the target class).

* True Negatives (TN): The number of instances that were correctly predicted as negative (not belonging to the target class).

* False Positives (FP): The number of instances that were incorrectly predicted as positive (the model predicted the target class, but the actual class was negative).

* False Negatives (FN): The number of instances that were incorrectly predicted as negative (the model predicted a negative class, but the actual class was positive).

<p align="center">
    <img src="https://images.datacamp.com/image/upload/v1701364260/image_5baaeac4c0.png" width="400"/>
</p>




Precision and recall are two important metrics used to evaluate the performance of classification models, especially in situations where there is an imbalance between the classes. Here are definitions for both:

- **Precision** focuses on the accuracy of positive predictions among all instances predicted as positive.
- **Recall** how many positive predictions did we get correct from all positive predictions.

These metrics are often used together and can be combined into the **F1 Score**, which is the harmonic mean of precision and recall:


\begin{align*}
\text{Precision} &= \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} \\
\text{Recall} &= \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} \\
\text{F1 Score} &= \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\end{align*}



In multiclass problems we create a confusion matrix for each class.

Accuracy = Sum of True Positives for all classes / Total Number of Instances

<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:1400/1*DQBHIaYhnlefJbRCSNzvrg.gif" width="400"/>
</p>

## Threshold Choosing metrics
models generate a probability between 0 and 1 for classifing an object as "someclass" or not, and we want to find the threshold that once the prediction is above it it will be treated as class, otherwise it is not class.

### 1. Receiver operating characteristic (ROC)

The Receiver Operating Characteristic (ROC) curve is a graphical representation that illustrates the performance of a binary classification model. It plots the true positive rate (sensitivity or recall) against the false positive rate. In machine learning, ROC curves are often used to compare and evaluate the performance of different classification models.

* **True Positive Rate (TPR) of ROC :**

$$TPR = \frac{TP}{P}$$

where:

* **TP** (True Positives): The number of correctly classified positive cases.
* **P** (Total Positives): The total number of actual positive cases.

This equation represents the proportion of actual positive cases that the model correctly identified as positive. It is also known as the **recall**, **sensitivity**.


* **False Positive Rate (FPR):** $$FPR = \frac{FP}{N}$$

It represents the proportion of actual negative instances incorrectly classified as positive by the model.


<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*qW3Mobeew1xxnXJnBPy8LQ.jpeg" width="400"/>
</p>


### 2. Precision - Recall curve

Precision-Recall curves are graphical representations used to evaluate the performance of a classification model, particularly in scenarios where the data is imbalanced. These curves plot the trade-off between precision and recall at different probability thresholds for binary classification problems.

### What to choose ?
* ROC curve : better for data with balanced classes
* Precision-recall curve: better for data with imbalanced classes

## Unsupervised Learning

### **Clustering**
Clustering: Identify unknown structure in the data and can be used:
1. Classification : leverge patterns in unlabeled data, spam filter, groups of product reviews
2. Anomly detection: like fraud detection
3. Customer segmentation: Segment customers by one of the features
4. Improved supervised learning: Check a good model that you have trained on your entire dataset, and see how well that performed compared to a model trained on subsegments of your data.


#### K-Means
#### **Steps**
1. Pick n random points as clusters centroids
2. Compare distance between each centroid and all other datapoints, and assign each point to it's nearest cluster
3.  Adjust the centroids to the new mean of the cluster
4.  Repeat step 2 and 3
5.  Stop one centroids are no longer changing.

> different intializations yield different results

#### **Evaluating clustering performance**
1. Inertia: Sum of squared distance from each point x to its cluster
   $$\sum_{i=1}^n ||\boldsymbol{x}_i - \boldsymbol{c}_k||^2$$

   where:

   * $n$ is the number of data points.
   * $\boldsymbol{x}_i$ is the i th data point, represented as a vector.
   * $\boldsymbol{c}_k$ is the k th cluster center, represented as a vector.

   + smaller values correponds to tighter clusters
   + Sensitive to number of points in cluster
1. Distorition: averge of squared distance from each point (x) to its cluster c
   $$\frac{1}{n} \sum_{i=1}^n ||\boldsymbol{x}_i - \boldsymbol{c}_k||^2$$

   1. smaller values corresponds to tighter clusters
   2. Not sensitive to number of points

<p align="center">
    <img src="https://assets.blog.code-specialist.com/k_means_animation_6cdd31d106.gif" width="400"/>
</p>



#### Hierarchial Clustering
#### **Steps**
1. Start with each point as a cluster
2. Merge the pair of points that has minimum distance
3. If the closest pairs are two clusters, merge them
4. Keep iterating untill there is only one cluster


#### Distance metrics between clusters
1. Single linkage: Minimum paiwise distance between clusters
   1. **Pros**: Ensures a clear seperation of our clusters that had any points within a certain distances of one another.
   2. **Cons** : Very sensitive to outliers failing close to certain clusters.

2. Complete linkage: Maximum pariwise distance between clusters
   1. **Pros**: Much better in seperating out clusters if there's a bit of noise or overlapping points.
   2. **Cons** : Tends to break apart larger existing clusters

3. Average linkage : average pairwise distance between clusters

<p align="center">
    <img src="https://dashee87.github.io/images/hierarch.gif" width="400"/>
</p>


#### DBSCAN
A true clsutering algorithm that can have points that don't belong to any cluster (outliers). Points are clustered using density of local neighborhood.
#### **Steps**
1. Select a random point
2. Look at the radius epsiolon around that point
3. Check if there is enough points n-clu within the circle, make a cluster if yes
4. Process new random point


#### **Required inputs**
1. Distance metric
2. Epsiolon: radius of local neighborhood: how far away a point needs to be considered part of the cluster
3. N-clu: The minimum amount of points for a particular point to be considered a corepoint of a cluster

#### **Pros & Cons**
**Pros**
1. No need to specify number of clusters 
2. Allows for noise
3. Can handle any shape

**Cons**
1. Requires two paramters 
2. Doesn't do well with clusters with different density

<p align="center">
    <img src="https://dashee87.github.io/images/DBSCAN_search.gif" width="400"/>
</p>



#### Mean Shift Algorithm: Finding Density Peaks

The Mean Shift algorithm is a non-parametric unsupervised learning technique used for **clustering** and **density estimation**. It iteratively shifts each data point towards the "densest" region in its vicinity, essentially finding peaks in the data's density distribution.

**Steps:**

1. **Initialization:** Each data point is considered its own initial cluster center.
2. **Mean Shift Vector:** For each data point, calculate the "mean shift vector" - the direction pointing towards the higher density region within a defined neighborhood.
3. **Shifting:** Move each data point along the mean shift vector.
4. **Convergence:** Repeat steps 2-3 until points converge to stable locations, indicating local density peaks.
5. **Clusters:** Points converging to the same peak belong to the same cluster.

**Pros:**

* **No pre-defined number of clusters:** Adapts automatically based on data density.
* **Flexible for various shapes:** Can handle clusters of diverse shapes and sizes.
* **Robust to noise:** Less sensitive to outliers compared to some methods.

**Cons:**

* **Computationally expensive:** Can be slower than simpler algorithms like K-Means.
* **Parameter tuning:** Choosing the appropriate bandwidth for the neighborhood can affect results.
* **Limited interpretability:** Cluster boundaries might not be as clear as in distance-based algorithms.

<p align="center">
    <img src="https://dashee87.github.io/images/mean_shift_tutorial.gif" width="400"/>
</p>




### **Dimensionality Reduction**


Dimensionality reduction: Use structural characterstics to simplify data and can be used in image processing and image tracking.

**Curse of dimensionality**
1. Too many features lead to worse performance
2. Distance measures perform poorly
3. Incidence of outliers increases


#### PCA (Principal Component Analysis):

| Purpose/Goal                                                | Key Features                                                                                                    |
|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Dimensionality Reduction and Feature Extraction           | - Identifies principal components (linear combinations of features) that capture the most variance in the data.   |
|                                                             | - Reduces dimensionality by projecting data onto a lower-dimensional subspace while preserving variance.          |
|                                                             | - Orthogonal transformation, ensuring uncorrelated principal components.                                         |

<br>
<br>


> Outliers needs to be handled before applying PCA

<p align="center">
    <img src="http://www.billconnelly.net/wp-content/uploads/2021/05/PCA1-smaller-smaller.gif
    " width="400"/>
</p>

Previously, each individual was represented by their height and weight, i.e. 2 numbers. But now, each individual is represented by just 1 number: how far along the blue line they are. Because our original data has 2 dimensions, there is also a second principal component, which is perpendicular (or “orthogonal”) to the first principal component.


### Kernel PCA (Kernelized Principal Component Analysis):

| Purpose/Goal                                                | Key Features                                                                                                    |
|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Non-linear Dimensionality Reduction                        | - Extends PCA to handle non-linear relationships by using kernel functions.                                       |
|                                                             | - Applies a kernel trick to implicitly map data to a higher-dimensional space.                                    |
|                                                             | - Useful for capturing non-linear structures in the data.                                                        |



### Multidimensional Scaling (MDS):

| Purpose/Goal                                                | Key Features                                                                                                    |
|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Visualization of pairwise distances                        | - Preserves pairwise distances between data points in the low-dimensional space.                                  |
|                                                             | - Commonly used for visualizing similarities or dissimilarities in high-dimensional data.                        |
|                                                             | - Stress function minimization to optimize the representation in lower-dimensional space.                         |



### Non-Negative Matrix Factorization (NMF):

| Purpose/Goal                                                | Key Features                                                                                                    |
|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Parts-based representation, topic modeling            | - Decomposes a matrix into two non-negative matrices representing parts-based features.                           |
|                                                             | - Often used for extracting interpretable patterns in data, such as in topic modeling or image feature learning.  |
|                                                             | - Useful when the underlying factors are additive and non-negative.                                              |

These tables provide a concise overview of the purpose and key features of each technique.

---

# **Deep Learning**

## Feed forward Neural Network (FFNN)
Each neuron in a layer has weights to all neurons in the previous layer. 
<p align="center">
    <img src="https://afit-r.github.io/public/images/analytics/deep_learning/deep_nn.png" width="400"/>
</p>


### Neurons
Each neuron takes a group of weighted inputs, applies an activation function and returns an output. The neuron then applies an activation function to the sum of weighted inputs from each incoming syanpse.
<p align="center">
    <img src="https://ml-cheatsheet.readthedocs.io/en/latest/_images/neuron.png" width="400"/>
</p>

| Task                             | Neural Network Model         | Other Models/Approaches         |
| ---------------------------------| ----------------------------- | --------------------------------|
| Classification (Image)           | Convolutional Neural Network (CNN) | -                              |
| Classification (Text)            | Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM) | -                   |
| Regression (Numeric)             | Neural Network (Dense layers) | Support Vector Regression (SVR), Decision Trees, Random Forests |
| Time Series Prediction           | Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM) | Autoregressive Integrated Moving Average (ARIMA), Exponential Smoothing |
| Sentence Prediction (NLP)        | Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM) | Transformer (e.g., BERT, GPT), Attention-based models |
| Visual Data Representation       | Convolutional Neural Network (CNN) | Autoencoders, Transfer Learning (e.g., using pre-trained models like VGG16, ResNet) |
| Unsupervised Pretrained Networks | Generative Adversarial Networks (GANs), Dimensionality Reduction (e.g., Autoencoders) | -                              |



## Optimization and Gradient Desent

Start with a cost function (a function that measures the difference between the predicted values and the true values) then gradually move towards minimum.
1) start at a random point (parameters)
2) Computer the Gradient
3) Take the negative of the gradient
   * The Gradient is the direction of the biggest increase, so we take it's negative to point to the direction of the biggest decrease of the loss

4) Iterate till you reach global minimum

| Type of Gradient Descent | Batch Size Based Definition |
|---|---|
| Stochastic Gradient Descent (SGD) | Single example |
| Mini-batch Gradient Descent | Batch of examples |
| Batch Gradient Descent (BGD) or Vanilla Gradient Descent | Entire dataset |




| Optimizer | Advantages | Disadvantages | Best suited for | Difference from Gradient Descent |
|---|---|---|---|---|
| **Stochastic Gradient Descent (SGD)** | Simple, computationally efficient, works well with sparse data | Slow convergence, prone to local minima, requires careful learning rate tuning | Small datasets, convex problems | Updates parameters based on a single data point, can be noisy, might get stuck in local minima |
| **Mini-batch Gradient Descent** | Faster than SGD, less prone to local minima | Less accurate updates than SGD, requires choosing batch size | Medium-sized datasets, non-convex problems | Updates parameters based on a small batch of data points, reduces noise compared to SGD, requires choosing an appropriate batch size |
| **Momentum** | Faster convergence than SGD, smoother updates, helps overcome local minima | Requires tuning additional hyperparameter (momentum) | Non-convex problems, noisy gradients | Considers past gradients (momentum term) to smooth updates and avoid getting stuck in local minima |
| **Nesterov Momentum** | Variant of Momentum with improved convergence properties | Requires tuning multiple hyperparameters | More complex problems than vanilla Momentum | Similar to Momentum but calculates gradient based on an advanced position considering momentum, leading to faster convergence |
| **RMSprop** | Adaptive learning rate, good for sparse data, less sensitive to outliers than SGD | Can be slow to converge in some cases | Sparse datasets, noisy gradients | Adaptively adjusts learning rate for each parameter based on its past gradients, addressing issues with features of different scales |
| **Adagrad** | Adaptive learning rate, handles features with different scales well | Can decay learning rate too quickly, leading to slow convergence | Sparse datasets, features with large variations in scale | Similar to RMSprop but uses decaying average of squared gradients, can lead to overly aggressive learning rate decay |
| **Adam** | Combines Momentum and RMSprop, often fast and stable convergence, requires less tuning | More complex than SGD, may not be optimal for all problems | General-purpose optimizer, often a good default choice | Combines momentum, adaptive learning rates, and exponential moving averages for smoother updates and better convergence |

<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:1400/1*m7-otgfbxiAvSipInHliXw.gif" width="400"/>
</p>


## Activation functions 

| Activation Function | Advantages                                      | Disadvantages                                          |
|----------------------|-------------------------------------------------|--------------------------------------------------------|
| Sigmoid              | - Smooth gradient for optimization algorithms. | - Prone to vanishing gradient problem.                 |
|                      | - Output range suitable for binary classification. | - Not zero-centered, leading to slower convergence.    |
| Tanh                 | - Output range between -1 and 1.                | - Still suffers from vanishing gradient.               |
|                      | - Zero-centered, aiding optimization in some cases. |                                                        |
| ReLU                 | - Avoids vanishing gradient problem.             | - Outputs are unbounded, leading to exploding gradients.|
|                      | - Computational efficiency (simple thresholding).| - Not zero-centered, which can cause dead neurons .      |
|                      | - Converges faster than sigmoid and tanh.          |       |
| Leaky ReLU (LReLU)   | - Overcomes the "dead neurons" problem of ReLU.  | - Not zero-centered (although less problematic than ReLU).|
|                      | - Helps with the vanishing gradient issue to some extent. | - Introduces a new hyperparameter (slope of the leak).   |


<p align="center">
    <img src="https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/627d12431fbd5e61913b7423_60be4975a399c635d06ea853_hero_image_activation_func_dark.png" width="400"/>
</p>

> **In sigmoid function the output is between 0 and 1 so small inputs will lead to very small outputs near to zero therefore when computing gradient it will be very small (vanishing gradient) however, in tanh function, very small inputs leads to outputs near to -1 so the gradient will still have some value therefore the vanishing gradient is less than sigmoid**

## Batch and Layer Normalization
Batch Normalization is a technique applied to the outputs of activation functions. It helps maintain zero-centered activations by normalizing the inputs to each layer during training, this means that at each layer inputs have mean of zero and std of 1.

BatchNormalization calculates the mean and variance of each feature within a mini batch during training. This introduces dependency on the batch size, as the statistics computed from a small batch may not be accurately represent the entire dataset. In some cases, when the batch size is very small, batch normalization may lead to unstable or inaccurate results. 

An alternative would be Layer Normalization 

For each layer in a neural network, Batch Normalization calculates the mean and variance across all examples in a mini-batch. The steps for each layer are as follows:

1. **Compute Batch Statistics:**
   - For a given layer, during the forward pass, calculate the mean (\(\mu\)) and variance (\(\sigma^2\)) of the inputs across all examples in the current mini-batch. This is done independently for each channel or feature in the layer.

2. **Normalize Inputs:**
   - Normalize the inputs (\(x_i\)) in the batch using the calculated mean and variance. The normalization is performed element-wise for each feature in each channel of the layer.
   \[ \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \]
   where \(\epsilon\) is a small constant added to the denominator for numerical stability.

3. **Scale and Shift:**
   - Introduce learnable parameters (\(\gamma\), scale, and \(\beta\), shift) to scale and shift the normalized inputs. This step allows the network to learn the optimal scaling and shifting for each feature.
   \[ y_i = \gamma \hat{x}_i + \beta \]

4. **Update Parameters:**
   - During backpropagation, update the parameters (\(\gamma\) and \(\beta\)) of the layer using the gradients calculated with respect to the loss.

These steps are repeated for each mini-batch during training. The running mean and variance are also maintained during training, which are used for normalization during the inference (testing or prediction) phase. This ensures that the normalization behavior is consistent between training and inference.

For Layer Normalization (LN), the mean and standard deviation are calculated independently for each feature (or channel) and each example within a mini-batch. The normalization is then applied independently to each feature and each example. Here's a step-by-step explanation:

<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*gat8a-TUnopoYN_veGEi0w.png
    " width="400"/>
</p>

Batch: calculate mean and variance over the whole batch.<br>
layer: calculate mean and variance over all channels and pixels.


## Weight Intialization

| Technique | Activation Function Suitability | Advantages | Disadvantages |
|---|---|---|---|
| **Xavier Initialization** | Sigmoid, tanh | - Prevents vanishing/exploding gradients | - May not be optimal for ReLU |
| **He Initialization** | ReLU | - Prevents vanishing gradients | - Not suitable for sigmoid/tanh |
| **Normal Initialization** | Any | - Simple and computationally efficient | - No specific guarantees for gradient flow |
| **Uniform Initialization** | Any | - Can be faster than normal initialization | - Sensitive to initialization range |
| **Glorot Initialization** | Any | - Similar to uniform but adapted for different scales | - Sensitive to initialization range |
| **Zero Initialization** | ReLU | - Can promote sparsity | - Can lead to dead neurons |

## Loss Functions

### Regularization Techniques

In machine learning, **regularization** refers to a set of techniques used to **prevent overfitting** and improve the **generalizability** of a model.

**Overfitting** is a phenomenon where a model learns the training data too well, including the noise, and performs poorly on unseen data. This happens when the model is too complex and memorizes the specifics of the training data instead of learning the underlying patterns.

**Regularization aims to address this issue by penalizing complex models and encouraging them to be simpler.** 

- This can be achieved in various ways:<br>
    1. L1 Regularization (Lasso): This technique adds the absolute value of the magnitude of each parameter to the loss function. Parameters with small values become zero, effectively removing them from the model, leading to sparsity.

    2. L2 Regularization (Ridge): This technique adds the square of the magnitude of each parameter to the loss function. Unlike L1, all parameters remain present but are shrunk towards zero, reducing their individual impact.

    3. Elastic Net: This combines L1 and L2 regularization, offering benefits of both sparsity and parameter shrinkage.

    4. Early Stopping: This technique trains the model until its performance starts to decline on a validation set, preventing it from memorizing the training data further.

    5. Dropout: This technique randomly drops out a certain proportion of neurons in each layer during training, forcing the model to learn more robust features that are not dependent on specific neurons.
   
    6. Batch Norm: has the same effect as dropouts.
    7. Data augmentation
    8. Drop connect: rather than zeroing out the activations in the forward pass, instead we randomly zero out some values of the weight matrix.
    9. Fractional max pooling
    10. Stochastic Depth: randomly drop layers from training and in testing we use the whole networks. 


---

# **Computer Vision**

## Computer vision tasks

1. Image Classification: can be used in OCR and content filtering

2. Object detection: where in the image is the important objects

3. Segmentation: which pixel in the image represents our object of interest (classification for each pixel)

4. Activity recognition: Analyze videos to understand human actions

## Convolutional Neural Networks

Pixels are our features, and if we used FCNN it will require vast number of parameters. With this high number of parameters, variance would be increadibly high with a very high likelihood to overfitting.

Using Fully Connected Neural Network means that we will flatten our image into 1d array and we will images spatial information.

CNN uses the same filter across different locations in the image leading to fewer parameters compared to FCNN.

> Spatial information: Relation of features with each others.

### Terminologies
1. **Padding:** Adds a border of zeros around the image data, allowing filters to "see" beyond the edge and potentially retain information that would be cropped otherwise.
2. **Pooling:** Downsamples the data by summarizing local regions with a function like max or average, reducing spatial dimensions and computational cost.
3. **Stride:** Controls how much the filter moves across the image in each layer, affecting the output size and receptive field (area covered by a filter).


<p align="center">
    <img src="https://miro.medium.com/max/700/1*FHPUtGrVP6fRmVHDn3A7Rw.png
    " width="400"/>
</p>

## Transfer learning

Early layers in the CNN are meant to represent only the most primitive features such as edges, and later layers in the network are capturing features that are more particular to the specific image classification problem. These features in the later layers will build off of those earlier primitive layers..  

<p align="center">
    <img src="https://i.stack.imgur.com/bN2iA.png" width="400"/>
</p>


**Guidelines**

1. The more similar your data and problem are to the source data of the pre-trained model, the less fine tuning is necessary.
 
2. The more data you have, the more the network will benefit from longer and deeper fine-tuning.

3.  If you data is substantially different in nature than the data the source model was trained on, transfer learning may be useless.

### Popular CNN Architectures:

| Architecture | Advantages | Disadvantages | Typical Use Cases |
|---|---|---|---|
| **LeNet-5:** | Simple, pioneering architecture, good for small datasets | Shallow, limited complexity, might not handle large datasets well | Digit recognition, early computer vision tasks |
| **AlexNet:** | Deep architecture, achieved significant performance advancements, popularized CNNs | Large memory footprint, computationally expensive | Image classification, object detection |
| **VGGNet:** | Very deep architecture, explores stacking convolutional layers, good for large datasets | Even larger memory footprint than AlexNet, can be slow to train | Image classification, object detection, feature extraction |
| **ResNet:** | Introduces skip connections, alleviates vanishing gradient problem, enables deeper networks | Can be complex to implement, requires careful hyperparameter tuning | Image classification, object detection, segmentation |
| **Inception:** | Uses parallel filter sizes for multi-scale feature extraction, reduces parameters | Computationally expensive, memory intensive | Image classification, object detection |
| **DenseNet:** | Densely connects layers, encourages feature reuse, improves information flow | More complex than ResNet, requires more memory | Image classification, object detection, segmentation |
| **MobileNet:** | Designed for mobile/embedded devices, lightweight architecture, efficient use of parameters | Reduced accuracy compared to deeper models | Mobile applications, resource-constrained environments |
| **U-Net:** | Encoder-decoder architecture, specialized for segmentation tasks, preserves spatial information | Requires large amounts of labeled data, can be computationally expensive | Medical image segmentation, object segmentation |

### Visualization
Occlusion, Guided Backpropagation, Gradient Ascent, and Saliency Maps are all techniques for visualizing and interpreting Convolutional Neural Networks (CNNs). These techniques help reveal which parts of an input image contribute the most to the predictions made by the model. Here's an overview of each technique:

These visualization techniques are valuable for interpretability, allowing practitioners to understand how CNNs make decisions and identify the features or regions that influence the model's predictions. They can be useful for debugging, model improvement, and gaining insights into the decision-making process of neural networks.

1. **Occlusion Experiment:**
   - **Idea:** Involves systematically occluding (covering) different regions of the input image and observing the effect on the model's predictions.
   - **Procedure:** Slide a small patch or window across the image, occluding the image under the patch. Repeatedly classify the partially occluded images and observe changes in the model's confidence scores.
   - **Visualization:** Heatmap highlighting areas where occlusion has the most impact on the prediction.

<p align="center">
    <img src="https://blog.roboflow.com/content/images/2020/11/image-9.png" width="400"/>
</p>




2. **Guided Backpropagation:**
   - **Idea:** Visualizes the importance of each pixel in the input image by attributing the model's prediction to the pixels through the backpropagation process.
   - **Procedure:** Modify the backpropagation process to only propagate positive gradients, ignoring negative gradients during the backward pass.
   - **Visualization:** Results in a heatmap highlighting pixels with positive gradients, indicating their positive contribution to the prediction.

<p align="center">
    <img src="https://glassboxmedicine.files.wordpress.com/2019/10/featuredimage2.png?w=816" width="400"/>
</p>





3. **Gradient Ascent:**
   - **Idea:** Generates an image that maximizes the activation of a specific neuron or class by iteratively adjusting the input image.
   - **Procedure:** Start with a random image and iteratively update the image in the direction of maximizing the gradient of the target neuron or class with respect to the input image.
   - **Visualization:** Results in an image that activates the chosen neuron or class strongly.
<p align="center">
    <img src="https://glassboxmedicine.files.wordpress.com/2019/07/fig1a-1.png" width="400"/>
</p>




4. **Saliency Maps:**
   - **Idea:** Highlights important regions in the input image that contribute the most to the model's prediction.
   - **Procedure:** Compute the gradient of the predicted class with respect to the input image. Use the magnitude of the gradients as a measure of importance for each pixel.
   - **Visualization:** Generate a heatmap based on the gradient magnitudes, highlighting the salient regions.

<p align="center">
    <img src="https://media.geeksforgeeks.org/wp-content/cdn-uploads/20210722235025/Saliency-maps-generated-using-Grad-CAM-Grad-CAM-and-Smooth-Grad-CAM-respectively_W640.jpg" width="400"/>
</p>



### Guidelines
1. An important way to gain intution about how an algorithm works is to visulaize the mistakes that it makes (ex. misclassified images)

2. At the beginning of training the model, train on small portion of the data and make sure that you can overfit on these data and get near zero loss. 

3. Learning rate is the most important hyperparameter, and we want to adjust it first
   1. If it is too small, the loss is barely changing
   2. If it is too high, cost explode may happen, and you can get NaN

<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:459/0*C5kIkoBwht0fXRgs.jpeg" width="400"/>
</p>

4. Random search is generally better than grid search.
      * Random search explores different hyperparameter combinations randomly, while grid search exhaustively checks all combinations within a defined range. Both aim to find the best hyperparameters for a model, but random search is faster and less likely to get stuck in local optima, while grid search provides a more thorough exploration and guarantees finding the best setting within the defined ranges.

5. Avoid copying weights or data between CPU/GPU at each step (bottleneck problem)


6. **Coarse-to-fine approach for hyperparameter tuning**
      * Start by testing a large range of hyperparameters for just a few training iterations to find the combination of parameters that work well.
      * Once you find it, search more finely around these parameters

7. Model Ensembles
      * Train multiple independent models or use multiple snapshots of a single model during training.
      * Maybe instead of using actual parameter vector , keep a moving average of the paramer vector and uese that at test time (Polayk averaging)

8. In classification + localization techniques for detecting one object, we would   have two outputs instead of one, one for labels, and the other would be for rectangle parameters (x, y, H, W) 

9. Other technique in multiple object detection would be to use some image processing tools to get regions of the image that may have objects, and then apply a CNN network to classify these objects. (R-CNN) 
10. While training a model keep a constant number at the seed so that we get the same validation set every time we run it, this way, if we change the our model and retrain it, we know that any differences are due to changes in the model, not due to having a different random validation set. 

### Efficient methods to accelerate training

1. **Pruning weights:**
   - The process of removing unnecessary connections from the network. This can be done to reduce the size of the network, improve its performance, or make it more efficient.

2. **Trained Quantization:**
   - Weights don’t have to be the exact value; we can cluster every collection of neighbors into one relative number like [1.1, 1, 0.9, 1.14] to be 1. This reduces the precision of the weights, leading to potential memory and computational savings.

3. **Huffman coding:**
   - Huffman coding is based on a simple principle: the more frequent a symbol is, the less information is needed to represent it. In the context of neural networks, Huffman coding can be applied to compress representations and reduce memory requirements.

4. **Quantization:**
   - Works by reducing the precision of the weights and activations of the network, such as from 32-bit floating-point numbers to 8-bit integers. This can be done using a variety of methods, such as linear quantization, logarithmic quantization, and piecewise linear quantization. Quantization helps in minimizing memory footprint and accelerates inference.

5. **Winograd convolution:**
   - A specialized convolution algorithm that reduces the number of multiplications needed for convolution operations, leading to faster computation. Winograd convolution is particularly effective for small filter sizes and is used to accelerate the training of convolutional neural networks.


---

# **NLP**

## Recurrent Neural Netowrk (RNN)

Bag of Words : states how many times a wowrd appears in our document, however we want to do better.



## Outputs of an RNN:

An RNN typically outputs two main things at each time step:

* **Predictions:** The output of the network based on the current input and the hidden state. This can be used for making predictions about the sequence, like the next word in a sentence, or the sentiment of a review.
  
* **Hidden State:** The hidden state summarizes the information processed by the network up to the current time step. This information is then used to understand the context of the current input and make better predictions in subsequent steps.

## RNN Update Equation:

The update equation for the hidden state \(h_t\) in an RNN can be written as:

```
h_t = activation(W_hh * h_{t-1} + W_xh * x_t + b_h)
```

where:

* `W_hh`: Weight matrix for the recurrent connection, capturing the influence of the previous hidden state.
* `W_xh`: Weight matrix for the input, capturing the influence of the current input.
* `h_{t-1}`: Hidden state from the previous time step, carrying information about the past.
* `x_t`: Input at the current time step, the new information being processed.
* `b_h`: Bias term, a constant offset added to the weighted sum.
* `activation`: Activation function (e.g., `tanh` or `ReLU`) that introduces non-linearity and allows the network to learn complex relationships.

This equation shows how the RNN combines the current input with information from the past (stored in the hidden state) to update its internal representation and make predictions.

I hope this clarifies the information in markdown format! Let me know if you have any further questions.


<p align="center">
    <img src="https://media.geeksforgeeks.org/wp-content/uploads/20231204125839/What-is-Recurrent-Neural-Network-660.webp" width="400"/>
</p>



> Slight variation called backpropagation through time (BPTT) is used to train RNNs.

> In practice we still get a maximum length to our sequence: if the input is shorter than maximum, we pad it and if it is longer we truncate it.

> RNN often focuses on text applications, but it can be used for forecasting (customer sales, loss rates, network traffic), speech recognition (voice apps), manufacturing (sensor data), and Genome Sequences

> **In RNNs, it's hard to keep information from distant past in current memory.**

## Long Short Term Memory RNN (LSTM)

### LSTM Components:

1. **Memory Cell (Cell State):**
   - This is the long-term memory unit that can store information over long sequences.

2. **Input Gate:**
   - The input gate determines whether to let new information into the memory cell. It controls the update to the cell state.

3. **Forget Gate:**
   - The forget gate decides whether to remove or keep the existing information in the memory cell. It controls the extent to which the cell state should be forgotten.

4. **Output Gate:**
   - The output gate controls whether and how much information from the memory cell should be used in generating the output (hidden state) of the LSTM.

### Operations during Each Time Step:

- **Input at Time Step \(t\):**
  - The input at time step \(t\) is combined with the previous hidden state and the previous cell state.

- **Input Gate Operation:**
  - The input gate determines which values from the combined input should be updated and added to the memory cell.

- **Forget Gate Operation:**
  - The forget gate decides which values from the memory cell should be forgotten.

- **Update Cell State:**
  - The new cell state is updated based on the input and forget gate operations.

- **Output Gate Operation:**
  - The output gate determines the output (hidden state) of the LSTM based on the updated cell state.


<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:1156/1*laH0_xXEkFE0lKJu54gkFQ.png" width="400"/>
</p>

## Gated Recurrent Unit (GRU)

A Gated Recurrent Unit (GRU) is another type of recurrent neural network (RNN) architecture, similar to Long Short-Term Memory (LSTM), designed to address the challenges of capturing long-range dependencies in sequential data. GRUs were introduced as a simplified version of LSTMs with comparable performance but with fewer parameters, making them computationally more efficient.

The key idea behind GRUs, like LSTMs, is to introduce mechanisms to selectively update and use information over time. GRUs have two main gates:

1. **Update Gate (z):**
   - Controls the extent to which the previous memory should be combined with the candidate new memory.

2. **Reset Gate (r):**
   - Determines the extent to which the previous hidden state should be forgotten when computing the candidate new memory.

### Key Characteristics of GRUs:

- **Simpler Structure:** GRUs have a simpler structure compared to LSTMs, with fewer parameters.
- **Fewer Gates:** GRUs have two gates (reset and update), while LSTMs have three gates (input, forget, and output).
- **Similar Functionality:** Despite their simplicity, GRUs have been shown to be effective in capturing long-range dependencies in sequential data and are computationally more efficient than LSTMs.

GRUs and LSTMs are both popular choices for handling sequential data, and the choice between them often depends on the specific characteristics of the task and the available computational resources.

LSTMs are more complex, therefore they may be able to find more complicated patterns, on the other hand, GRU are faster to train.

<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:1400/1*yBXV9o5q7L_CvY7quJt3WQ.png" width="400"/>
</p>


## Seq2Seq models

Sequence-to-Sequence (Seq2Seq) models are a type of neural network architecture designed for sequence transduction tasks, where the goal is to convert one sequence into another. These models have been particularly successful in natural language processing tasks such as machine translation, text summarization, and speech recognition. The architecture was introduced to handle input and output sequences of variable lengths.

### Key Components of Seq2Seq Models:

1. **Encoder:**
   - The encoder takes the input sequence and converts it into a fixed-size context vector or hidden state. This context vector contains information about the entire input sequence.
   - Common encoder architectures include recurrent neural networks (RNNs), Long Short-Term Memory networks (LSTMs), or Gated Recurrent Units (GRUs).

2. **Decoder:**
   - The decoder generates the output sequence based on the context vector produced by the encoder. It processes the context vector along with its own internal hidden states to produce the output sequence.
   - Similar to the encoder, the decoder can use RNNs, LSTMs, or GRUs.

3. **Attention Mechanism (Optional):**
   - To address the challenge of capturing long-range dependencies in the input sequence, attention mechanisms are often incorporated. Attention allows the decoder to focus on different parts of the input sequence at each step of generating the output.
   - Attention mechanisms enhance the model's ability to align input and output sequences, especially in machine translation tasks.

### Workflow of Seq2Seq Models:

1. **Encoder Processing:**
   - The input sequence is fed into the encoder, and the encoder produces a context vector summarizing the input information.

2. **Context Vector:**
   - The context vector serves as the initial hidden state for the decoder. It contains a condensed representation of the input sequence.

3. **Decoder Generation:**
   - The decoder processes the context vector and generates the output sequence step by step. At each step, it considers the previous output and hidden state.

### Applications:

Seq2Seq models have been successfully applied to various tasks, including:

- **Machine Translation:** Translating sentences from one language to another.
- **Text Summarization:** Generating concise summaries of input text.
- **Speech Recognition:** Converting spoken language into written text.
- **Conversational Agents:** Generating responses in natural language.

<p align="center">
    <img src="https://miro.medium.com/max/942/1*KtWwvLK-jpGPSnj3tStg-Q.png" width="400"/>
</p>

## LSTM Applications
1. Forecasting
2. Speech Recognition
3. Machine translation
4. Image captioning
5. Question answering
6. Robotic Control

## AutoEncoder

Autoencoders are a type of neural network architecture designed for unsupervised learning, particularly in the field of feature learning and dimensionality reduction. The primary goal of autoencoders is to learn a compact representation of input data by training the network to encode the input into a lower-dimensional representation and then decode it back to the original input. The architecture consists of an encoder and a decoder, and the training process involves minimizing the reconstruction error.

### Key Components of Autoencoders:

1. **Encoder:**
   - The encoder takes the input data and maps it to a lower-dimensional representation, often referred to as the "encoding" or "latent space." This encoding is expected to capture important features of the input.

2. **Decoder:**
   - The decoder takes the encoding produced by the encoder and reconstructs the original input from it. The goal is to generate an output that closely matches the input.

3. **Latent Space:**
   - The latent space is the reduced-dimensional representation learned by the encoder. It serves as a compressed and informative representation of the input data.

### Applications
1. Dims Reduction
2. Information retrivel
3. Machine translation
4. Image related applications (generating , denoising, compression)
5. Drug discovery
6. Sound synthesis

> The decoder moddel can be used as generative model once trained.

> Auto encoders are often trained with a single layer each for the encoding and decoding step.

> PCA can be used for dims reduction, however, it has some limitations as that the learned features are linear combination of original features.

---

# **Reinforcement learning**
learn through consequences of actions in an environment to reach a goal (choose actions that maximize rewards)

### Model evaluation
Avg reward: graph represents the avg reward the agent earns during a training iteration


Avg percentage completion: the training graph represents the avg percentage of the track completed by the agent in all training eposides.

## Terminologies

| **Term**                     | **Definition**                                                                                           |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| Agent                    | The piece of software you are training to make decisions in an environment to reach a goal.            |
| State                    | The current position within the environment that is visible or known to the agent.                      |
| Episode                  | A period of trial and error when an agent makes decisions and receives feedback from its environment (from initial state to terminal state).|
| Action Space             | The set of all valid actions or choices available to an agent as it interacts within an environment.   |
| Discrete Action Space    | A finite set of possible actions.                                                                     |
| Continuous Action Space  | A range of possible actions.                                                                          |
| Exploration              | Wandering in a random direction to discover.                                                          |
| Exploitation             | Using experience to decide.                                                                           |


---

# **Generative AI**

If our NN learned a bunch of handwritten images, then it actually knows all the features that make up a handwritten image according to the NN, and therfore we can replicate that to produce something that according to the NN will be classified as a handwritten image.

There will be two models:
1. Generator model : creates fake output
2. Discriminator model : differentiates between real and fake output.

Generative Adversarial Networks (GANs) are a class of artificial intelligence models that were introduced by Ian Goodfellow and his colleagues in 2014. GANs consist of two neural networks, a generator, and a discriminator, that are trained simultaneously through adversarial training. The goal of GANs is to generate new data instances that resemble a given dataset.

### Key Components of GANs:

1. **Generator:**
   - The generator network takes random noise as input and transforms it into data samples. Its objective is to create realistic samples that are indistinguishable from the real data.

2. **Discriminator:**
   - The discriminator network evaluates whether a given sample is real (from the actual dataset) or fake (generated by the generator). Its goal is to correctly classify the origin of the samples.

3. **Adversarial Training:**
   - The generator and discriminator are trained in an adversarial manner. The generator aims to produce samples that are realistic enough to fool the discriminator, while the discriminator strives to become proficient at distinguishing between real and generated samples.

4. **Loss Function:**
   - GANs use a min-max game framework. The generator aims to minimize the probability of the discriminator correctly classifying its samples as fake, while the discriminator seeks to maximize this probability. This results in a loss function that is minimized by both networks during training.

<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:1400/1*jDPj5v3JKGRRRyDZmQTkpQ.gif" width="400"/>
</p>


## Terminologies

| Term          | Description                                                           |
|--------------------|-----------------------------------------------------------------------|
| Discriminator Loss | Evaluates how well the discriminator differentiates real and fake data|
| Generator Loss     | Measures the deviation of generated data from real data in the dataset|


---

# **Visualizations**
<!-- Data Exploration and Preprocessing Visualizations -->
<p align="center">
    <h3>Data Distribution Histogram</h3>
    <img src="https://api.www.labxchange.org/api/v1/xblocks/lb:LabXchange:10d3270e:html:1/storage/17__histogram-31626365193365-3b46e339f410f97cfae66fc8c127ea02.png" width="400"/>
</p>

<p align="center">
    <h3>Pair Plot</h3>
    <img src="https://seaborn.pydata.org/_images/pairplot_11_0.png" width="400"/>
</p>

<p align="center">
    <h3>Correlation Heatmap</h3>
    <img src="https://www.quanthub.com/wp-content/uploads/correlation_heatmap_food_health-1024x867.png" width="400"/>
</p>

<!-- Model Architecture Visualizations -->
<p align="center">
    <h3>Model Summary</h3>
    <img src="https://miro.medium.com/v2/resize:fit:1400/1*6c68EXZvK8ohMuIV_DVkng.png" width="400"/>
</p>

<p align="center">
    <h3>Graph Visualization</h3>
    <img src="https://d112y698adiu2z.cloudfront.net/photos/production/software_photos/001/199/089/datas/original.jpeg" width="400"/>
</p>

<!-- Training Progress Visualizations -->
<p align="center">
    <h3>Loss and Accuracy Plot</h3>
    <img src="https://discuss.pytorch.org/uploads/default/4572286a1ffae2ff1f9729c7dcc4584b03b97020" width="400"/>
</p>


<!-- Filter Visualizations -->
<p align="center">
    <h3>Filter Visualization</h3>
    <img src="https://srdas.github.io/DLBook/DL_images/cnn9.png" width="400"/>
</p>

<!-- Attention Maps -->
<p align="center">
    <h3>Attention Map</h3>
    <img src="https://www.mdpi.com/remotesensing/remotesensing-12-01366/article_deploy/html/images/remotesensing-12-01366-g002.png" width="400"/>
</p>

<!-- Embedding Visualizations -->
<p align="center">
    <h3>t-SNE Plot</h3>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/94/T-SNE_visualisation_of_word_embeddings_generated_using_19th_century_literature.png" width="400"/>
</p>


<!-- Decision Boundary Visualizations -->
<p align="center">
    <h3>Decision Boundary Plot</h3>
    <img src="https://i.stack.imgur.com/T3aMD.png" width="400"/>
</p>

<!-- Confusion Matrices and Classification Reports -->
<p align="center">
    <h3>Confusion Matrix</h3>
    <img src="https://media.geeksforgeeks.org/wp-content/uploads/20230316112623/download-(16).png" width="400"/>
</p>

<p align="center">
    <h3>Classification Report</h3>
    <img src="https://i.stack.imgur.com/LIWH1.jpg" width="400"/>
</p>


<!-- GAN Visualizations -->
<p align="center">
    <h3>Generated Samples</h3>
    <img src="https://machinelearningmastery.com/wp-content/uploads/2019/06/Example-of-Celebrity-Photographs-and-GAN-Generated-Emojis.png" width="400"/>
</p>

<p align="center">
    <h3>Latent Space Exploration</h3>
    <img src="https://miro.medium.com/v2/resize:fit:960/0*cYaaF2pFLECohCaI.gif" width="400"/>
</p>

<!-- Hyperparameter Search Visualizations -->
<p align="center">
    <h3>Hyperparameter Plot</h3>
    <img src="https://miro.medium.com/v2/resize:fit:1400/0*5N5USunsVC3h_jrx.png" width="400"/>
</p>

<!-- Explainability Visualizations -->
<p align="center">
    <h3>SHAP Values</h3>
    <img src="https://images.datacamp.com/image/upload/v1688055328/image_1268879d70.png" width="400"/>
</p>

---
