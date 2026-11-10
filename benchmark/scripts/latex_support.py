"""Minimal standalone preamble for optional generated-table compile checks."""
PREAMBLE = r'''\documentclass[10pt,letterpaper]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype}
\usepackage{amsmath,amssymb,mathtools,booktabs,array,tabularx}
\usepackage{graphicx,xcolor,enumitem,float,needspace}
\usepackage{hyperref}
\usepackage[capitalise,noabbrev]{cleveref}
\newcommand{\ind}{\mathbf 1}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}
'''
