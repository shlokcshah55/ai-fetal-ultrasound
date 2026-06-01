\begin{table}[h]
\centering
\caption{In-distribution classification performance (mean $\pm$ std over 3 folds).
AUPRC is primary given $\sim$85\% healthy prevalence.}
\label{tab:clf}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{AUPRC} $\uparrow$ & \textbf{AUROC} $\uparrow$ & \textbf{F1} $\uparrow$ & \textbf{Sens.} $\uparrow$ & \textbf{Spec.} $\uparrow$ \\
\midrule
Logistic Regression & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- \\
MLP                 & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- \\
MLP + MC Dropout    & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- \\
MLP + Energy        & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- \\
MLP + VoS           & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- \\
MLP + EDL           & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- \\
\bottomrule
\end{tabular}
\end{table}