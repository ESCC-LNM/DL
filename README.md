# ESCC-LNM

ESCC-LNM predicts metastasis in individual lymph nodes in esophageal squamous cell carcinoma (ESCC) from contrast-enhanced CT. ConvNeXt V2 Atto encodes 2.5D patches of each segmented node and its surrounding tissue. Radiomics features guide feature modulation and attention pooling.

## Performance

The retrospective study included 838 lymph nodes from 806 patients at three centers.

| Test cohort | Nodes | AUROC (95% CI) |
| --- | ---: | --- |
| Internal | 95 | 0.892 (0.824–0.960) |
| External 1 (regional) | 288 | 0.881 (0.837–0.925) |
| External 2 (supraclavicular) | 81 | 0.927 (0.864–0.990) |

In an exploratory analysis of nine patients with matched TCIA and TCGA data, the highest predicted nodal risk per patient correlated positively with *GSTM2* and *GSTM4* expression.

## Data access

Clinical data are available from the corresponding author upon reasonable request and institutional review board approval.

## Citation

> Radiomics-prior-guided deep learning for node-level lymph node metastasis prediction in esophageal squamous cell carcinoma: exploratory radiogenomic associations with glutathione-mediated redox signaling. *Study manuscript*.
