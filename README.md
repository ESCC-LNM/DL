\documentclass[11pt,a4paper]{article}
\usepackage[margin=25mm]{geometry}
\pagestyle{empty}

\begin{document}

\begin{center}
  {\Large\bfseries Radiomics-Guided Prediction of\\
  Lymph Node Metastasis in ESCC\par}
\end{center}

\section*{Overview}
This project predicts metastasis in individual regional and supraclavicular
lymph nodes in esophageal squamous cell carcinoma (ESCC) using contrast-enhanced
CT. A shared ConvNeXt V2 Atto encoder processes 2.5D patches of each segmented
node and its surrounding tissue. PyRadiomics features guide both deep feature
modulation and attention-based multi-instance pooling to estimate a metastatic
probability for each node.

\section*{Evaluation}
The retrospective study included 838 lymph nodes from 806 patients across
three centers. The PyTorch model used five-fold cross-validation on 374 training
nodes. The area under the ROC curve (AUC) in each held-out cohort was as follows.

\begin{center}
\begin{tabular}{lrr}
  \hline
  Test cohort & Nodes & AUC \\
  \hline
  Internal & 95 & 0.892 \\
  External 1 (regional) & 288 & 0.881 \\
  External 2 (supraclavicular) & 81 & 0.927 \\
  \hline
\end{tabular}
\end{center}

In an exploratory analysis of nine patients with matched TCIA CT and TCGA
transcriptomic data, the highest predicted nodal risk per patient correlated
positively with a glutathione-related co-expression module and with GSTM2 and
GSTM4 expression. These correlations were assessed without adjustment for
multiple testing.

\section*{Data access}
The clinical datasets are available from the corresponding author upon
reasonable request and institutional review board approval.

\section*{Associated manuscript}
\emph{Radiomics-prior-guided deep learning for node-level lymph node metastasis
prediction in esophageal squamous cell carcinoma: exploratory radiogenomic
associations with glutathione-mediated redox signaling.}

\end{document}
