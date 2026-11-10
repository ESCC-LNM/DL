# ESCC-LNM

## Background

Lymph node metastasis is an important factor in staging and treatment planning for esophageal squamous cell carcinoma (ESCC). A patient-level prediction estimates whether nodal metastasis is present. Estimating risk for individual nodes adds information about where suspected metastases are located and how widely they are distributed. On contrast-enhanced CT, radiologists commonly assess lymph nodes by their size, shape, and enhancement. Small metastatic nodes can be difficult to distinguish using these features alone.

## Study approach

ESCC-LNM is a PyTorch framework that estimates metastatic risk for individual regional and supraclavicular lymph nodes. It analyzes both the lymph node and its surrounding tissue. Radiomics features that describe nodal shape, intensity, and texture guide how the model interprets and weighs imaging evidence from these two regions.

The retrospective study evaluated the framework across three centers, with one internal and two external test cohorts. An exploratory radiogenomic analysis also investigated a potential link between model-predicted metastatic risk and glutathione-mediated redox signaling.

See the [data guide](docs/DATA.md) for the repository’s input requirements.
