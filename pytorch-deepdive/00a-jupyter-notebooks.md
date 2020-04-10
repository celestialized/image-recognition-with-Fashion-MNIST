# [Jupyter Notebooks](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html)
Comes out the box with anaconda. Google pops up with ipython in there. Ipython is the "computational engine" kernel that executes the notebook code. [Ipython extends REPL](http://ipython.org/ipython-doc/stable/overview.html#ipythonzmq). Testing code can be done also in Spyder- that looks like IDLE on a Pi. Main concern is that for any one of these we use the proper interpreter for the project we wish to test against. VSCode the interpreter can be clicked on and changed in the status bar. Then also know what the environment might trump out for you. What gets loaded by default (base) and how your .yml and requirements may also affect things during debugging.

## No config needed if...
We place your notebooks in your home folder or subfolders. Otherwise, you need to choose a Jupyter Notebook App start-up folder which will contain all the notebooks. Looks like upward recursive searching may fall into play here. 

## [Kernel](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html#id7)
When you open a Notebook document, the associated kernel is automatically launched. The default kernel is ipython. When the notebook is executed (either cell-by-cell or with menu Cell -> Run All), the kernel performs the computation and produces the results. Depending on the type of computations, the kernel may consume significant CPU and RAM. Note that the RAM is not released until the kernel is shut-down. Each kernel per notebook document is launched to run that document, and offers support for:
* [Python](https://github.com/ipython/ipython)
* [Julia] (https://github.com/JuliaLang/IJulia.jl)
* [R ](https://github.com/IRkernel/IRkernel)
* [Ruby] (https://github.com/minrk/iruby)
* [Haskell] (https://github.com/gibiansky/IHaskell)
* [Scala] (https://github.com/Bridgewater/scala-notebook)
* [node.js] (https://gist.github.com/Carreau/4279371)
* [Go] (https://github.com/takluyver/igo)

## Jupyter Notebook [App](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html#id6)
The Jupyter Notebook App is a server-client application that allows editing and running notebook documents via a web browser. The Jupyter Notebook App can be executed on a local desktop requiring no internet access (as described in this document) or can be installed on a remote server and accessed through the internet. If you wish to share a .ipynb file in html format without the person you are sharing with having to install jupyter notebooks see [nbviewer](https://nbviewer.jupyter.org)

## Notebook [Dashboard](https://jupyter-notebook.readthedocs.io/en/stable/)
The Notebook Dashboard is the component which is shown first when you launch Jupyter Notebook App. The Notebook Dashboard is mainly used to open notebook documents, and to manage the running kernels (visualize and shutdown).

The Notebook Dashboard has other features similar to a file manager, namely navigating folders and renaming/deleting files.

## Notebook Document
Self-contained documents that contain a representation of all content visible in the notebook web application, including inputs and outputs of the computations, narrative text, equations, images, and rich media representations of objects. Each notebook document has its own kernel.

## Jupytercon.com Use Python to Control Javascript widgets to display brain MEG stimuli
Python uses widgets to interact with javascript in real time. Linking widgets using traits. See bqplot on github for plotting. This is the point we ask ourselves how does the bus pass messages across languages? We will detail this later.
Widget | Use
|---|---|
ipywidgets|core ui controls
bqplot|2d plotting
pythreejs|3d plotting
ipyvolume|3d plotting
ipyleaflet|maps 

from bqplot import LinearScale, Scatter, Figure
from bqplot_extra.regression_lines import *

create a scatter and linear regression with an order one- basically a linear regression.
Every attribute of the plot is also a widget. program a slider to conrol the polynomial representation of the linear line

Principal Component Analysis PCA
* large dataset of correlated variables
* reduces dimensionality
* retain variance in the graph
* first few factors (3-5) usually enought to explain most of variance
Basic switch is to create a start and end, run the PCA on the set and map results

## Quick Reference Guide

Command | Explination | Use
|---|---|---|
jupyter kernelspec list | see what interpreters are available | when not finding ipython
jupyter notebook list | see the server(s) and port(s) | more than one port is in use, port conflicts
jupyter notebook stop <8888> | <port_number> instead of ctrl c 2x's | stop specific port since 5.0
??np.stack|anytime using ?? it inspects the numpy code