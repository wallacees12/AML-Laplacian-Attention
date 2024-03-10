- To Do's
  - Scientific research: What kind of distribution instead of Gaussian? Why? On which type of modality - speech, text, or image?
    - Quote abstract - does this mean that our type of research would be obsolete?
        "GAAM inte- grates learnable mean and variance into its attention mechanism, implemented in a Multi-Headed framework enabling it to collectively model             any Probability Distribution for dynamic recalibration of feature significance."
       - **Information from the introduction**: Because GAAM is used independently for each head, the combination of all the heads allows for other probability             distributions to be learned.
    - No specific reason is given for why a Gaussian distribution is used - just the fact that for each feature, a mean and a variance shall be learned.
  - Feasibility: Read through paper, ensure that training a model is feasible with our resources
  - Understanding of the implementation: How roughly does the code work - how could we adapt it?
