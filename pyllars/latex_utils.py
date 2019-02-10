"""
Utilities for programmatically constructing a latex document
"""

import os
import pyllars.shell_utils as shell_utils

def vspace(out, vspace):
    out.write("\n\\vspace{")
    out.write(vspace)
    out.write("}\n\n")

def begin_numbered_list(out):
    out.write("\\begin{enumerate}\n")

def end_numbered_list(out):
    out.write("\\end{enumerate}\n")

def begin_bulleted_list(out):
    out.write("\\begin{itemize}\n")

def end_bulleted_list(out):
    out.write("\\end{itemize}\n")

def write_list_item(out, text):
    text = get_latex_safe_string(text)
    out.write("\\item ")
    out.write(text)
    out.write("\n")


def begin_table(out, columns:str, width:str="\\linewidth", caption:str=None, 
        label:str=None, include_table_tag:bool=False):

    if include_table_tag:
        out.write("\\begin{table}[]\n")
        out.write("\\centering\n")

    #out.write("\\begin{tabular}{")
    #out.write(columns)
    #out.write("}\n")

    if label is not None:
        write_label(out, label)

    out.write("\\begin{tabularx}")

    if width is not None:
        out.write("{")
        out.write(width)
        out.write("}")

    out.write("{")
    out.write(columns)
    out.write("}\n")

    if caption is not None:
        write_caption(out, caption)
        out.write("\\\\")

def write_hline(out):
    out.write("\\hline\n")

def end_table(out, include_bottom_rule:bool=True, include_table_tag:bool=False):
    #out.write("\\end{tabular}")

    if include_bottom_rule:
        out.write("\\bottomrule\n")

    out.write("\\end{tabularx}\n")

    if include_table_tag:
        out.write("\\end{table}\n")


def write_header(out, content:list, height:str=None):
    out.write("\\toprule\n")
    write_simple_row(out, content, height=height)
    out.write("\\midrule\n")
    out.write("\\endhead\n")

def begin_multirow(out, num_rows:int, width:str="*"):
    out.write("\\multirow{")
    out.write(str(num_rows))
    out.write("}{")
    out.write(width)
    out.write("}{")

def end_multirow(out):
    out.write("}")

def write_simple_row(out, content:list, initial_ampersand:bool=False, 
        final_new_row:bool=True, height:str=None):

    if initial_ampersand:
        out.write("&\t")

    for c in content[:-1]:
        out.write(str(c))
        out.write("\t&\t")

    out.write(str(content[-1]))
    
    if final_new_row:
        out.write("\t\\\\")

    if height is not None:
        out.write("[")
        out.write(height)
        out.write("]")
        
    out.write("\n")

def write_row_sep(out):
    out.write("\t\\\\\n")

def write_column_sep(out):
    out.write("\t&\t")

def begin_figure(out, placement_parameters="h"):
    out.write("\n\\begin{figure}[")
    out.write(placement_parameters)
    out.write("]\n")
    out.write("\t\\centering\n")

def write_caption(out, caption, label=None):
    caption = caption.replace("_", "-")
    
    out.write("\t\\caption{")
    out.write(caption)

    if label is not None:
        write_label(out, label)

    out.write("}\n")

def write_label(out, label, include_newline=False):
    out.write("\\label{")
    out.write(label)
    out.write("}")

    if include_newline:
        out.write("\n")


def end_figure(out):
    out.write("\\end{figure}\n")

def clearpage(out):
    out.write("\n\n")
    out.write("\\clearpage\n\n\n")

def newpage(out):
    out.write("\\newpage\n")

def section(out, title, label=None):
    out.write("\\section{")
    out.write(title)
    out.write("}\n")

    if label is not None:
        out.write("\\label{")
        out.write(label)
        out.write("}\n")
        
    out.write("\n")

def subsection(out, title, label=None):
    out.write("\\subsection{")
    out.write(title)
    out.write("}\n")

    if label is not None:
        out.write("\\label{")
        out.write(label)
        out.write("}\n")
        
    out.write("\n")
    
def write_graphics(out, image_file, height:str=None, width:str=None, 
        write_textwidth:bool=True, write_textheight:bool=True, 
        keepaspectratio:bool=True):

    image_file = get_latex_image_string(image_file)
    out.write("\t\\includegraphics[")
    
    if width is not None:
        out.write("width=")
        out.write(str(width))

        if write_textwidth:
            out.write("\\textwidth")

        if keepaspectratio:
            out.write(",")

    if height is not None:
        out.write("height=")
        out.write(str(height))

        if write_textheight:
            out.write("\\textheight")
        
        if keepaspectratio:
            out.write(",")
    
    if keepaspectratio:
        out.write("keepaspectratio")
    
    out.write("]{")
    out.write(image_file)
    out.write("}\n")

def float_barrier(out):
    """ Add a \FloatBarrier command here. """
    out.write("\\FloatBarrier\n")

def newline(out):
    """ Add a \newline command here. """
    out.write("\\newline\n")

def centering(out):
    """ Add a \centering command here. """
    out.write("\\centering\n")

def write(out, s, size=None):
    """ Make the text latex-safe and write it. """
    s = get_latex_safe_string(s)

    if size is not None:
        out.write("\\")
        out.write(size)
        out.write("{ ")

    out.write(s)

    if size is not None:
        out.write("}")

def bold(s):
    """ Make the text latex-safe and bold. """
    s = "\\textbf{" + get_latex_safe_string(s) + "}"
    return s

def get_latex_safe_string(s):
    """ This function replaces various special characters in the provided
        string to ensure that it renders correctly in latex.

        Current changes:
            "_" becomes "-"
    """
    s = s.replace("_", "-")
    return s

def get_latex_image_string(filename):
    """ This function removes the final dot (".") and extension from the
        filename, surrounds the remaining bits with braces ("{"), and sticks
        the dot and extension back on.
    """

    last_dot_index = filename.rfind(".")
    f = filename[:last_dot_index]
    extension = filename[last_dot_index+1:]
    latex_image_string = "{" + f + "}." + extension
    return latex_image_string

BASE_PACKAGES = [
    'amsmath',
    'graphicx',
    'float',
    'morefloats',
    'xspace',
    'multirow',
    'booktabs',
    'ltablex',
    'placeins'
]

def begin_document(out, title:str, abstract:str, packages:list=[], 
        commands:dict={}, base_packages:list=BASE_PACKAGES):

    header = get_header_text(title, abstract,
        packages=packages, 
        commands=commands, 
        base_packages=base_packages
    )

    out.write(header)

def end_document(out):
    footer = get_footer_text()
    out.write(footer)

def get_header_text(title:str, abstract:str, packages:list=[], 
        commands:dict={}, base_packages:list=BASE_PACKAGES):

    import string
    header = string.Template(r"""
\documentclass[a4paper,10pt]{article} % For LaTeX2e
\usepackage[utf8]{inputenc}
$base_packages
$packages

\pagestyle{empty}

\setlength{\parskip}{2mm}
\setlength{\parindent}{0cm}

\title{$title}

\newcolumntype{Y}{>{\centering\arraybackslash}X}

\def\|#1{\ensuremath{\mathtt{#1}}}
\def\!#1{\ensuremath{\mathbf{#1}}}
\def\*#1{\ensuremath{\mathcal{#1}}}

\newcommand*{\defeq}{\mathrel{\vcenter{\baselineskip0.5ex \lineskiplimit0pt
                     \hbox{\scriptsize.}\hbox{\scriptsize.}}}%
                     =}

\DeclareMathOperator*{\argmax}{arg\,max}

\newtheorem{definition}{Definition}
\newtheorem{theorem}{Theorem}

$commands

\begin{document}

\maketitle

\begin{abstract}
    $abstract
\end{abstract}
""")
    
    use_package_t = string.Template(r"\usepackage{$package}")

    base_packages_str = [use_package_t.substitute(package=p) for p in base_packages]
    base_packages_str = "\n".join(base_packages_str)

    packages_str = [use_package_t.substitute(package=p) for p in packages]
    packages_str = "\n".join(packages_str)

    command_t = string.Template(r"\newcommand{\$name}{$text}")
    commands_str = [command_t.substitute(name=k, text=v) for k,v in commands.items()]
    commands_str = "\n".join(commands_str)

    header = header.substitute(
        base_packages=base_packages_str,
        packages=packages_str,
        title=title,
        abstract=abstract,
        commands=commands_str
    )

    return header

def get_footer_text():
    footer = r"""

%\small
%\bibliographystyle{abbrv}
%\bibliography{library}

\end{document}
        """
    return footer

def compile(out_dir, doc_name):
    """ Compile the document by calling pdflatex twice. """

    os.chdir(out_dir)
    cmd = [
        "pdflatex",
        "-interaction",
        "nonstopmode",
        "-file-line-error",
        "-shell-escape",
        doc_name
    ]
    cmd = ' '.join(cmd)
    shell_utils.check_call(cmd, raise_on_error=False)

    # call again to fix references
    shell_utils.check_call(cmd, raise_on_error=False)

