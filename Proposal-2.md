# AML Proposal: Exploring Probabilistic Attention Functions for Text Classification

mm20079

March 2024

## 1 Overview

Recent research by Ioannides et al. (2024) has shown that the usage of probabilistic attention functions can enhance the capabilities of transformers in a variety of domains [1]. Specifically, Ioannides et al. (2024) have proposed a Gaussian adapative attention mechanism [GAAM] that has reached state of the art performance on text classification, image classification, and emotion recognition in speech. Thus, the usage of probabilitstic attention functions within transformers seems very promising, which is why we want to investigate it in more detail.

With regards to the paper by Ioannides et al., it caught our attention that the authors give no explicit reasons with regards to why a Gaussian distribution was chosen for the attention mechanism (instead of e.g. a Laplacian or a $t$ distribution). Instead, they claim that when GAAM is applied independently to multiple attention heads, then they can jointly learn any possible probability distribution. The goal of our project is to investigate whether this claim holds and if adapting the attention function to another type of distribution potentially leads to an additional improvements in the results. Concretely, we intend to mimic the overall transformer architecture proposed by Ioannides et al., but adapt the attention function to different distributions in order to learn how the type of distribution chosen for the attention function affects the accuracy of the model. Alternative distributions we currently consider exploring are a Laplacian distribution, a chi-squared distribution, and a $t$-distribution.

## 2 Problem Setting

In the original transformer architecture, a simple multi-head dot-product attention function was used [2].

$$
\text{Attention}(Q, K, V)=\text{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V
$$

Subsequent research has explored if the usage of other types of attention functions can yield even better results, of which GAAM is a promising recent example [2]. As the name already suggests, GAAM closely resembles the formula of a Gaussian normal distribution

$$
G C T(x)=\exp \left(\frac{-(x-\mu)^{2}}{2 \sigma^{2}}\right)
$$

where $\mu$ is the ... and $\sigma$ is the ... . In this project, we intend to explore how replacing the Gaussian attention function with an attention function that resembles a Laplacian distribution affects the performance of the model. A Laplacian adaptive attention function could have a form similar to

$$
LCT(x) = \frac{1}{2 b} \exp -\left(\frac{|x-u|}{b}\right)
$$

where $b$ is the ... and $\sigma$ is the ... . Naturally, GAAM and the Gaussian Adaptive Transfomer [GAT] will be the baseline for our model - we are interested to see if a Laplacian attention function is able to match or even exceed the performance of a Gaussian attention function.

<!-- and $\phi$ which is a learnable offset of our sample mean $\hat{\mu}=\frac{1}{n} \sum i=1^{N} x_{i}$ giving the model the ability to approximate the population mean from the sample mean, $\mu_{\text {pop }}=\hat{\mu}+\phi$ --> 
<!-- In my opinion, this is too much detail for the time being - I'd just keep the math part as simple as possible right now, particulary because we don't know yet how we'll implement the different distributions etc. -->


## 3 Approach

Because of the limitations of the scope of this project, we will focus on one of the three modalities that GAAM was applied on: Text classification. Besides changing the attention function, we will mimic the approach and architecture used by Ioannides et al. [1] in order to obtain comparable results. Hence, we will use the pretrained model 'Llama 2' and train the new probabilistic attention module on the [AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) dataset, which contains a significant collection of web-based news articles, which the model classifies to belong either to the topic "world" (y=0), "sports" (y=1), "business" (y=2), or "sci/tech" (y=3).

## 4 Evaluation

Let $X$ be the set of input news articles and $Y$ be the set of corresponding topic categories. We aim to learn a model $M: X \rightarrow Y$ that maps each news article $x \in X$ to its correct category $y \in Y$ using a Laplacian Attention Mechanism. Our goal is to minimize the classification error defined as:

$$
\min_{M} \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(M\left(x_{i}\right) \neq y_{i}\right)
$$

where $N$ is the total number of news articles in the dataset, $x_{i}$ is the $i$-th news article, and $y_{i}$ is its correct category.

## 5 Contribution

With this project, we aim to deepen the understanding of probabilistic attention functions by exploring what probabilistic attention functions can be devised other than Gaussian attention functions and how their application affects the performance of a text classification model.

## Remarks

Issues: 
- Consistency wrt. different attention functions we consider using. Either we say that we'll try multiple ones (but then we have to mention them more formally), or we only talk about a Laplacian attention mechanism.
    - However, since we talked about trying multiple ones, I think we should mention them here already as well. 
- New structure - good like this?
- Some gaps still have to be filled (noted with "...")
- Ideally one additional sentence for the contribution

## References

[1] G. Ioannides, A. Chadha, and A. Elkins, "Gaussian adaptive attention is all you need: Robust contextual representations across multiple modalities," 2024 .

[2] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," 2023.



