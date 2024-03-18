# AML Proposal: Exploring Probabilistic Attention Functions for Text Classification

March 18th, 2024

## 1 Overview

Recent research by Ioannides et al. (2024) has shown that the usage of probabilistic attention functions can enhance the capabilities of transformers in a variety of domains [1]. Specifically, Ioannides et al. (2024) have proposed a Gaussian adapative attention mechanism [GAAM] that has reached state of the art performance on text classification, image classification, and emotion recognition in speech. Thus, the usage of probabilitstic attention functions within transformers seems very promising, which is why we want to investigate it in more detail.

With regards to the paper by Ioannides et al., it caught our attention that the authors give no explicit reasons with regards to why a Gaussian distribution was chosen for the attention mechanism. Instead, they claim that when GAAM is applied independently to multiple attention heads, then they can jointly learn any possible probability distribution. The goal of our project is to investigate whether this claim holds and if adapting the attention function to another type of distribution potentially leads to an additional improvements in the results. Concretely, we intend to mimic the overall transformer architecture proposed by Ioannides et al., but adapt the attention function to a Laplacian attention mechanism in order to learn how the type of distribution chosen for the attention function affects the accuracy of the model.

## 2 Problem Setting

In the original transformer architecture, a simple multi-head dot-product attention function was used [2].

$$
\text{Attention}(Q, K, V)=\text{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V
$$

Subsequent research has explored if the usage of other types of attention functions can yield even better results, of which GAAM is a promising recent example [2]. As the name already suggests, GAAM closely resembles the formula of a Gaussian normal distribution. First, the sample mean $\bar \mu$ and the sample variance $\overline{\sigma^2}$ are calculated for every input feature vector $x$.

$$
\bar{\mu}=\frac{1}{N} \sum_{i=1}^N x_i, \quad \overline{\sigma^2}=\frac{1}{N} \sum_{i=1}^N x_i^2-(\mu)^2
$$

To approximate the population mean $\psi$ rather than the sample mean $\bar \mu$, GAAM also learns an offset $\delta$, such that $\psi = \bar \mu + \delta$. Afterwards, $x$ is normalized using $\psi$ and $\overline{\sigma^2}$. Then, GAAM can be applied to calculate the attention weights for each feature, where $\xi$ is the scaled variance.

$$
\text{GAAM}\left(x_i\right)=\exp \left(-\frac{x_{\text {norm }}{ }^2}{2 \xi}\right)
$$

In this project, we intend to explore how replacing the Gaussian attention function with an attention function that resembles a Laplacian distribution affects the performance of the model. First, the location parameter $\bar \mu_L$ and the scale parameter $\bar b$ will have to be calculated for each input feature $x$:

$$
\bar \mu_L=\text{med}(x), \quad \bar{b}=\frac{1}{n} \sum_{i=1}^n\left|x_i-\bar \mu_L\right|
$$

<!-- source: https://en.wikipedia.org/wiki/Laplace_distribution#Statistical_inference -->

As in GAAM, $\bar \mu _L$ will then be offset and normalized, after which a Laplacian adapative attention mechanism [LAAM], for example similar to the formula shown below, will be applied, where $\xi_L$ is a scaled version of the scale parameter $\bar b$.

$$
\text{LAAM}(x_i) = \exp \left(-\frac{|x_{\text{norm}}|}{\xi_L}\right)
$$

Naturally, GAAM and the Gaussian Adaptive Transfomer [GAT] will be the baseline we will compare our model trained on LAAM with, since our main reseach goal is exploring whether applying LAAM is able to match or even exceed the performance of GAAM.

## 3 Approach

Because of the limitations of the scope of this project, we will focus on one of the three modalities that GAAM was applied on in [1]: Text classification. Besides changing the attention mechanism, we will mimic the approach and architecture used by Ioannides et al. in order to obtain comparable results. Hence, we will use the pretrained model 'Llama 2' and train the new probabilistic attention module on the [AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) dataset, which contains a significant collection of web-based news articles, which the model classifies to belong either to the topic "world" (y=0), "sports" (y=1), "business" (y=2), or "sci/tech" (y=3).

## 4 Evaluation

Let $X$ be the set of input news articles and $Y$ be the set of corresponding topic categories. We aim to learn a model $M: X \rightarrow Y$ that maps each news article $x \in X$ to its correct category $y \in Y$ using a transformer that is based on a Laplacian attetion mechanism. Our goal is to minimize the classification error defined as:

$$
\min_{M} \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(M\left(x_{i}\right) \neq y_{i}\right)
$$

where $N$ is the total number of news articles in the dataset, $x_{i}$ is the $i$-th news article, and $y_{i}$ is its correct category.

## 5 Contribution

With this project, we aim to deepen the understanding of probabilistic attention functions by exploring what probabilistic attention functions can be devised other than Gaussian attention functions and how their application affects the performance of a text classification model.

## References

<!-- APA -->
[1] Ioannides, G., Chadha, A., & Elkins, A. (2024). Gaussian Adaptive Attention is All You Need: Robust Contextual Representations Across Multiple Modalities. <em>arXiv preprint arXiv:2401.11143</em>.

[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. <em>Advances in neural information processing systems, 30</em>.



