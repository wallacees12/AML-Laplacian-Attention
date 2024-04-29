# 28.04.2024 (Simon - gpt2_fine_tuning_gaussian_v2.ipynb)

- Tried new way of training the Gaussian model: First five epochs with frozen layers, then five epochs with unfrozen layers. Results were comparable (loss of appr. 1.4).
- Realization: Text and title are not separated in the dataset we use - title just comes first, then comes the text. Not a big problem in my opinion.
- Also tried training a GPT2-Model from scratch (no pre-trained weights), but this also yielded a loss of appr. 1.4.

# 29.04.2024 (Simon - gpt2_fine_tuning_gaussian_v2.ipynb)

- Did some research to check whether our implementation of the Gaussian attention mechanism is correct (mainly by going through the paper again)
- Tried to insert the gaussian block into the GPT2-architecture a bit differently (more closely to what is shown in the example provided by the authors), but that didn't work - gives an error (unexpected argument)
- I tried to implement a Gaussian block, but stumbled upon two problems:
    - not sure if my calculation of b is correct (for the formula, see the proposal)
    - how does normalization work for Laplacian distributions?

