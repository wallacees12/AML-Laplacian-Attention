# Advanced Machine Learning Proposal: Exploring Probabilistic Attention Functions for Text Classification

## Introduction

In the Advanced Machine Learning course at the University of Zurich, we worked on a semester project where we tried to replicate a [recent study](https://arxiv.org/abs/2401.11143) by Ioannides et al. (2024). A detailed description can be found in [Problemsetting](PROBLEMSETTING.md).

## Content

Jupyter notebooks [01](01-random-forest-classifier.ipynb) and [02](02-baseline-gpt2-model.ipynb) contain two different models that we used as a baseline. Notebooks [03](03-gpt2-with-GAAM-hyp-tuning.ipynb), [04](04-gpt2-with-GAAM.ipynb), [05](05-gpt2-with-LAAM.ipynb) use GPT2 and GAAM or LAAM based on the code blocks provided by the researchers.  In Jupyter notebooks [06](06-gpt2-with-normal-attention-and-GAAM.ipynb) and [07](08-two-model-architecture-with-proprietary-att-mechanism.ipynb) we follow a suggestion from the author of the paper, and in 08 we implemented our own GAAM and LAAM from scratch. And our project presentation can be found [here](project-presentation.pptx).


## Acknowledgements

At this point we would like to thank [Deborah Jakobi](https://github.com/theDebbister) and [David Reich](https://github.com/SiQube) for their help and great support with our project during the semester.  We would also like to thank [Georgios Ioannides](https://github.com/gioannides) who helped us with further insights and suggestions regarding their work. And of course a big thank you to [Lena A. Jäger](https://www.cl.uzh.ch/en/research-groups/digital-linguistics/people/group-leader/jaeger.html) for sharing her knowledge and experience in machine learning.