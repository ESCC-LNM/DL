# ESCC-LNM

ESCC-LNM is a PyTorch framework for predicting metastasis in individual regional and supraclavicular lymph nodes in esophageal squamous cell carcinoma (ESCC) using contrast-enhanced CT.

The central idea is to use radiomics to guide how the model interprets imaging evidence from each lymph node and its surrounding tissue. This lets the model weigh information from both regions when estimating the probability of metastasis for each node.

The retrospective study evaluated the framework across three centers, with one internal and two external test cohorts. In an exploratory analysis of nine patients with paired CT and transcriptomic data, the highest predicted nodal risk per patient was positively associated with a glutathione-related gene module and with *GSTM2* and *GSTM4* expression.

See the [data guide](docs/DATA.md) for the repository’s input requirements.
