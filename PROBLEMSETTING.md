# Proposal

## Theoretical Part

Recent research by Ioannides et al. (2024) has shown that the usage probabilistic attention functions can enhance the capabilities of transformers in a variety of domains. Specifically, Ioannides et al. (2024) have proposed a Gaussian adapative attention mechanism [GAAM] that has reached state of the art performance on emotion recognition in speech, text classification, and image classification. Thus, the usage of probabilitstic attention functions within transformers seems very promising, which is why we want to investigate it in more detail.

With regards to the paper by Ioannides et al., it caught our attention that the authors give no explicit reasons with regards to why a Gaussian distribution was chosen for the attention mechanism (instead of e.g. a Laplacian or a $t$-distribution). Instead, they claim that when GAAM is applied independently to multiple attention heads, then they can jointly learn any possible probability distribution. The goal of our project is to investigate whether this claim holds and if adapting the attention function to another type of distribution potentially leads to an additional improvements in the results. Concretely, we intend to mimic the overall transformer architecture proposed by Ioannides et al., but adapt the attention function to different distributions in order to learn how the type of distribution chosen for the attention function affects the accuracy of the model. Alternative distributions we currently consider exploring are a Laplacian distribution, a chi-squared distribution, and a $t$-distribution. 

## Technical Part

Because the scope of the semester project and our computational resources are limited, we thought about focusing on one of the three modalities covered in the paper: text/image/speech.


