### To Do's

- **Scientific research**: What kind of distribution instead of Gaussian? Why? On which type of modality - speech, text, or image?
  - Quote abstract - does this mean that our type of research would be obsolete?
      "GAAM inte- grates learnable mean and variance into its attention mechanism, implemented in a Multi-Headed framework enabling it to collectively model             any Probability Distribution for dynamic recalibration of feature significance."
     - **Information from the introduction**: Because GAAM is used independently for each head, the combination of all the heads allows for other probability             distributions than a Gaussian normal distribution to be learned.
       - **Other idea**: We could test this hypothesis by changing the attention function to a Laplacian attention function and seeing whether that yields
         comparable results (proving that indeed, it doesn't matter which formula for the attention function is specifically used).
  - No specific reason is given for why a Gaussian distribution is used in the attention function - just the fact that for each feature, a mean and a variance      shall be learned.
    - Could be the basis for us arguing that it might be worth it to explore other probability distributions as well.
      - Other distributions
        - **Laplace distribution**: Probably the easiest to test, because it is very similar to the Gaussian normal distribution - adaptations to the   attention function likely to be fairly small & doable.
          - Could be the "minimum" our research looks at.
        - [**Chi-squared distribution**](https://en.wikipedia.org/wiki/Chi-squared_distribution): Mathematically more different from a Gaussian normal distribution: Doesn't have a mean and a variance, but instead its only parameter is the degrees of freedom $k$.
          - Probably more difficult to implemenent and compare with the original paper.
          - Same arguably holds for the [**t-distribution**](https://en.wikipedia.org/wiki/Student%27s_t-distribution)
            - What are your thoughts, Sam?
- **Feasibility**: Read through paper, ensure that training a model is feasible with our resources
- **Understanding of the implementation**: How roughly does the code work - how could we adapt it?

### Questions for T.A.

- **Scientific research**
  - Because the scope of the semester project and our computational resources are limited, we thought about focusing on one of the modalities covered in the paper: speech (emotion recognition), text (classification), or image (classification). For any of these modalities, where do you think that the assumption of a Gaussian distribution is most likely to be inaccurate?
- **Feasibility**:
  - No explicit numbers are stated in the paper indicating how long and costly training the model was. How can we obtain a better idea about how costly the training will be?
  - In case training turns out to be way to computationally expensive, how can we address that problem? Any other approaches than cutting down the size of the training dataset?

### Extension 
-Generate Distributions with data, and see if Gaussian can model this
