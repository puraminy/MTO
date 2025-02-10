\documentclass[]{article}
\usepackage{booktabs} % for professional tables
\usepackage{makecell}
\usepackage{multirow,multicol}
\usepackage{graphicx} 
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{datetime}
\usepackage{tabularx}
\usepackage{adjustbox}
\usepackage[margin=1in, lmargin=1cm, rmargin=1cm]{geometry}

%opening
\begin{document}

                    \begin{table*}[h]
                        \label{table:1}
                        \caption{aa:SL}
                        \begin{adjustbox}{width=1\textwidth}
                        \begin{tabular}{|r|r|r|r|r|r|r|r|r|}
                        \hline
                        method &  \textbf{50-123} & \textbf{50-45} & \textbf{50-76} & \textbf{avg} & \textbf{500-123} & \textbf{500-45} & \textbf{500-76} & \textbf{avg} \\
\hline
0) \hyperref[fig:SIL]{SIL} &  $ 75.1 $ & $ 75.7 $ & $ 79.5 $ &\textcolor{blue}{ $ 76.8} $ & $ 82.1 $ & $ 80.8 $ & $ 82.9 $ &\textcolor{blue}{ $ 81.9} $ \\
\hline 
1) \hyperref[fig:SILP]{SILP} &  $ 77.2 $ & $ 78.9 $ & $ 60.5 $ &\textcolor{blue}{ $ 72.2} $ & $ 75.5 $ & $ 80.9 $ & $ 79.2 $ &\textcolor{blue}{ $ 78.5} $ \\
\hline 
2) \hyperref[fig:SILPI]{SILPI} &  $ 78.4 $ & $ 75.7 $ & $ 79.6 $ &\textcolor{blue}{ $ 77.9} $ & $ 83.8 $ & $ 82.3 $ & $ 83.2 $ &\textcolor{blue}{ $ 83.1} $ \\
\hline 
3) \hyperref[fig:P]{P} &  $ 58.3 $ & $ 58.2 $ & $ 72.1 $ &\textcolor{blue}{ $ 62.9} $ & $ 70.4 $ & $ 83.1 $ & $ 82.5 $ &\textcolor{blue}{ $ 78.7} $ \\
\hline 
4) \hyperref[fig:SLPI]{SLPI} &  $ 71.2 $ & $ 61.3 $ & $ 72.9 $ &\textcolor{blue}{ $ 68.5} $ & $ 83.9 $ & $ 84.1 $ & $ 83.7 $ &\textcolor{blue}{ $ 83.9} $ \\
\hline 
5) \hyperref[fig:SIP]{SIP} &  $ 66.5 $ & $ 53.5 $ & $ 65.2 $ &\textcolor{blue}{ $ 61.7} $ & $ 83.3 $ & $ 82.2 $ & $ 82.7 $ &\textcolor{blue}{ $ 82.7} $ \\
\hline 
6) \hyperref[fig:P2]{P2} &  $ 69.8 $ & $ 58.4 $ & $ 61.8 $ &\textcolor{blue}{ $ 63.3} $ & $ 83.1 $ & $ 83.5 $ & $ 76.3 $ &\textcolor{blue}{ $ 81.0} $ \\
\hline 
7) \hyperref[fig:SLP]{SLP} &  $ 46.6 $ & $ 49.6 $ & $ 62.6 $ &\textcolor{blue}{ $ 52.9} $ & $ 42.8 $ & $ 61.5 $ & $ 44.4 $ &\textcolor{blue}{ $ 49.6} $ \\
\hline 
8) \hyperref[fig:SL]{SL} &  $ 56.2 $ & $ 59.3 $ & $ 65.5 $ &\textcolor{blue}{ $ 60.3} $ & $ 75.0 $ & $ 76.3 $ & $ 78.6 $ &\textcolor{blue}{ $ 76.6} $ \\
\hline 
9) \hyperref[fig:PI]{PI} &  $ 76.2 $ & $ 77.0 $ & $ 65.4 $ &\textcolor{blue}{ $ 72.9} $ & $ 84.7 $ & $ 83.7 $ & $ 85.1 $ &\textcolor{blue}{ $ 84.5} $ \\
\hline 
\hline 

                        \end{tabular}
                        \end{adjustbox}
                    \end{table*}
                    


                    \begin{table*}[h]
                        \label{table:2}
                        \caption{bb:SL}
                        \begin{adjustbox}{width=1\textwidth}
                        \begin{tabular}{|r|r|r|r|r|r|r|r|r|}
                        \hline
                        method &  \textbf{50-123} & \textbf{50-45} & \textbf{50-76} & \textbf{avg} & \textbf{500-123} & \textbf{500-45} & \textbf{500-76} & \textbf{avg} \\
\hline
0) \hyperref[fig:SIL]{SIL} &  $ 79.5 $ & $ 78.3 $ & $ 78.1 $ &\textcolor{blue}{ $ 78.6} $ & $ 80.4 $ & $ 83.8 $ & $ 83.2 $ &\textcolor{blue}{ $ 82.5} $ \\
\hline 
1) \hyperref[fig:SILP]{SILP} &  $ 79.3 $ & $ 71.6 $ & $ 78.1 $ &\textcolor{blue}{ $ 76.3} $ & $ 82.7 $ & $ 82.0 $ & $ 82.2 $ &\textcolor{blue}{ $ 82.3} $ \\
\hline 
2) \hyperref[fig:SILPI]{SILPI} &  $ 79.2 $ & $ 75.5 $ & $ 76.7 $ &\textcolor{blue}{ $ 77.1} $ & $ 83.5 $ & $ 83.6 $ & $ 84.4 $ &\textcolor{blue}{ $ 83.8} $ \\
\hline 
3) \hyperref[fig:P2]{P2} &  $ 76.2 $ & $ 77.0 $ & $ 65.4 $ &\textcolor{blue}{ $ 72.9} $ & $ 84.7 $ & $ 83.7 $ & $ 85.1 $ &\textcolor{blue}{ $ 84.5} $ \\
\hline 
4) \hyperref[fig:PI]{PI} &  $ 76.2 $ & $ 77.0 $ & $ 65.4 $ &\textcolor{blue}{ $ 72.9} $ & $ 84.7 $ & $ 83.7 $ & $ 85.1 $ &\textcolor{blue}{ $ 84.5} $ \\
\hline 
5) \hyperref[fig:SLP]{SLP} &  $ 75.2 $ & $ 49.0 $ & $ 72.1 $ &\textcolor{blue}{ $ 65.4} $ & $ 82.3 $ & $ 67.2 $ & $ 81.4 $ &\textcolor{blue}{ $ 77.0} $ \\
\hline 
6) \hyperref[fig:SLPI]{SLPI} &  $ 73.7 $ & $ 77.2 $ & $ 71.5 $ &\textcolor{blue}{ $ 74.1} $ & $ 78.7 $ & $ 83.4 $ & $ 79.2 $ &\textcolor{blue}{ $ 80.4} $ \\
\hline 
7) \hyperref[fig:SIP]{SIP} &  $ 72.2 $ & $ 65.3 $ & $ 65.3 $ &\textcolor{blue}{ $ 67.6} $ & $ 71.3 $ & $ 82.2 $ & $ 79.9 $ &\textcolor{blue}{ $ 77.8} $ \\
\hline 
8) \hyperref[fig:P]{P} &  $ 71.2 $ & $ 67.4 $ & $ 67.1 $ &\textcolor{blue}{ $ 68.6} $ & $ 76.8 $ & $ 83.4 $ & $ 83.9 $ &\textcolor{blue}{ $ 81.4} $ \\
\hline 
9) \hyperref[fig:SL]{SL} &  $ 70.3 $ & $ 64.3 $ & $ 75.6 $ &\textcolor{blue}{ $ 70.1} $ & $ 45.5 $ & $ 82.1 $ & $ 82.1 $ &\textcolor{blue}{ $ 69.9} $ \\
\hline 
\hline 

                        \end{tabular}
                        \end{adjustbox}
                    \end{table*}
                    

mytable
myimage
\end{document}
