# 28.04.2024 (Simon - gpt2_fine_tuning_gaussian_v2.ipynb)

- Tried new way of training the Gaussian model: First five epochs with frozen layers, then five epochs with unfrozen layers. Results were comparable (loss of appr. 1.4).
- Realization: Text and title are not separated in the dataset we use - title just comes first, then comes the text. Not a big problem in my opinion.
- Also tried training a GPT2-Model from scratch (no pre-trained weights), but this also yielded a loss of appr. 1.4.

# 29.04.2024 (Simon - gpt2_fine_tuning_gaussian_v2.ipynb)

- Did some research to check whether our implementation of the Gaussian attention mechanism is correct (mainly by going through the paper again)
- Tried to insert the gaussian block into the GPT2-architecture a bit differently (more closely to what is shown in the example provided by the authors), but that didn't work - gives an error (unexpected argument)
- I tried to implement a Laplacian block, but stumbled upon two problems:
    - not sure if my calculation of b is correct (for the formula, see the proposal)
    - how does normalization work for Laplacian distributions?

# 01.05.2024 (Simon - gpt2-model-gaussian-attention, gpt2-model-laplacian-attention, LaplacianBlock )

- I implemented a GPT2-Model with a Laplacian attention mechanism and ran it on the data. Performance: comparably bad.

    - I'm not entirely sure whether the implementation is correct (see first question for David below)

- Questions to ask David

    - Formulas in the paper don't perfectly match the formulas used in the implementation, which makes it hard to understand the mathematical reasoning behind the Gaussian attention mechanism exactly and to to map it into a Laplacian attention mechanism in a a correct way. How should we best deal with this?

        - Decided to not ask David this, since it would otherwise have been too much. Can anyone of you have a better look at this? Sam?

            - Have a look at the files `GaussianBlock` and `LaplacianBlock` in the repo.

    - The performance of our Gaussian and Laplacian models is really bad, even if we train it in different ways - e.g. (1) freezing the GPT2-layers for 5 epochs (only training the attention weights), then unfreezing them, or (2) using an untrained GPT2-model from the beginning.

        - What things might cause the terrible performance and how can we best address it?

# 07.05.2024 (Simon - gpt2_model_normal_and_gaussian_attention.ipynb)

- Implemented the approach. Only ran the first training cycle with all layers unfrozen.
- Problem I encountered: model cannot be saved because the weights "contain shared tensors".
    - As a result, I'm not sure whether the model was stored correctly (and whether the training process itself was correct).
    - The results (loss values logged during the training process), though, were comparably bad to when we replaced the GPT2 attention layer.
    - Also, I'm not sure whether the choice of hyperparameters was good (particularly the learning rate) or whether we should try training with frozen GPT2 layers.
    - Striking: The model outputted exactly 0.2500 (as it also did for the model with only Gaussian attention). Mistake in the calculation somewhere?