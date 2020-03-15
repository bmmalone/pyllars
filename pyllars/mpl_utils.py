"""
This module contains a number of helper functions for matplotlib.

For details about various arguments, such as allowed key word
arguments and how they will be interpreted, please consult the
appropriate parts of the matplotlib documentation:

* **Lines**: https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
* **Patches**: https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch
* **Scatter plots**: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter
* **Text**: https://matplotlib.org/api/text_api.html#matplotlib.text.Text

"""
import argparse
import itertools

import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy
import sklearn.metrics

import matplotlib_venn

import typing
from typing import Collection, Iterable, Mapping, Optional, Sequence, Tuple, Union

BarChartColorOptions = Union[
    matplotlib.colors.Colormap,
    Sequence,
    int,
    str
]
IntOrString = Union[int, str]
FigAx = Tuple[matplotlib.figure.Figure, plt.Axes]
MapOrSequence = Union[Mapping,Sequence]

import pyllars.utils as utils
import pyllars.validation_utils as validation_utils

import logging
logger = logging.getLogger(__name__)

###
# Constants
###

VALID_AXIS_VALUES = {'both', 'x', 'y'}
"""Valid `axis` values"""

VALID_WHICH_VALUES = {'major', 'minor', 'both'}
"""Valid `which` values"""

X_AXIS_VALUES = {'both', 'x'}
"""`axis` choices which affect the X axis"""

Y_AXIS_VALUES = {'both', 'y'}
"""`axis` choices which affect the Y axis"""

def _get_fig_ax(ax:Optional[plt.Axes]):
    """ Grab a figure and axis from `ax`, or create a new one
    """    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        
    return fig, ax
    
###
# Font helpers
###

def set_legend_title_fontsize(
        ax:plt.Axes, fontsize:IntOrString) -> None:
    """ Set the font size of the title of the legend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis

    fontsize : int, or a str recognized by matplotlib
        The size of the legend title

    Returns
    -------
    None, but the legend title fontsize is updated
    """
    legend = ax.legend_
    plt.setp(legend.get_title(),fontsize=fontsize)
    

def set_legend_fontsize(
        ax:plt.Axes, fontsize:IntOrString) -> None:
    """ Set the font size of the items of the legend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis

    fontsize : int, or a str recognized by matplotlib
        The size of the legend text

    Returns
    -------
    None, but the legend text fontsize is updated
    """
    legend = ax.legend_
    plt.setp(legend.get_texts(),fontsize=fontsize)

    
def set_title_fontsize(
        ax:plt.Axes, fontsize:IntOrString) -> None:
    """ Set the font size of the title of the axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis

    fontsize : int, or a str recognized by matplotlib
        The size of the title font
 
    Returns
    -------
    None, but the  title fontsize is updated
    """
    ax.title.set_fontsize(fontsize=fontsize)

def set_label_fontsize(
        ax:plt.Axes, fontsize:IntOrString, axis:str='both') -> None:
    """ Set the font size of the labels of the axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis

    fontsize : int, or a str recognized by matplotlib
        The size of the label font

    axis : str in {`both`, `x`, `y`}
        Which label(s) to update

    Returns
    -------
    None, but the respective label fontsizes are updated
    """
    validation_utils.validate_in_set(axis, VALID_AXIS_VALUES, "axis")
    
    if (axis == 'both') or (axis=='x'):
        l = ax.xaxis.label
        l.set_fontsize(fontsize)

    if (axis == 'both') or (axis=='y'):
        l = ax.yaxis.label
        l.set_fontsize(fontsize)
        

def set_ticklabels_fontsize(
        ax:plt.Axes,
        fontsize:IntOrString,
        axis:str='both',
        which:str='major'):
    """ Set the font size of the tick labels

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis

    fontsize : int, or a str recognized by matplotlib
        The size of the ticklabels

    {axis,which} : str
        Values passed to :meth:`matplotlib.axes.Axes.tick_params`. Please see
        the matplotlib documentation for more details.

    Returns
    -------
    None, but the ticklabel fontsizes are updated
    """
    validation_utils.validate_in_set(axis, VALID_AXIS_VALUES, "axis")
    validation_utils.validate_in_set(which, VALID_WHICH_VALUES, "which")
    
    ax.tick_params(axis=axis, which=which, labelsize=fontsize)

def set_ticklabel_rotation(
        ax:plt.Axes,
        rotation:IntOrString,
        axis:str='x',
        which:str='both'):
    """ Set the rotation of the tick labels

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis

    rotation : int, or a string matplotlib recognizes
        The rotation of the labels

    {axis,which} : str
        Values passed to :func:`matplotlib.pyplot.setp`. Please see
        the matplotlib documentation for more details.

    Returns
    -------
    None, but the ticklabels are rotated
    """
    validation_utils.validate_in_set(axis, VALID_AXIS_VALUES, "axis")
    validation_utils.validate_in_set(which, VALID_WHICH_VALUES, "which")

    adjust_xaxis = (axis == 'x') or (axis == 'both')
    adjust_yaxis = (axis == 'y') or (axis == 'both')

    adjust_major = (which == 'major') or (which == 'both')
    adjust_minor = (which == 'minor') or (which == 'both')

    if adjust_xaxis:
        xticklabels = []
        if adjust_major:
            xticklabels.extend(ax.xaxis.get_majorticklabels())
        if adjust_minor:
            xticklabels.extend(ax.xaxis.get_minorticklabels())
            
        plt.setp(xticklabels, rotation=rotation)
    
    if adjust_yaxis:
        yticklabels = []
        if adjust_major:
            yticklabels.extend(ax.yaxis.get_majorticklabels())
        if adjust_minor:
            yticklabels.extend(ax.yaxis.get_minorticklabels())
            
        plt.setp(yticklabels, rotation=rotation)

###
# Axes helpers
###
def center_splines(ax:plt.Axes) -> None:
    """ Places the splines of `ax` in the center of the plot.
    
    This is useful for things like scatter plots where (0,0) should be
    in the center of the plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis

    Returns
    -------
    None, but the splines are updated
    """
    
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    #ax.spines['left'].set_smart_bounds(True)
    #ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


    ax.xaxis.set_label_coords(0.5, 0)
    ax.yaxis.set_label_coords(-0.05, 0.5)
    

def hide_tick_labels(
        ax:plt.Axes,
        axis:str='both') -> None:
    """ Hide the tick labels on the specified axes.
    
    Optionally, some can be preserved.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis

    axis : str in {`both`, `x`, `y`}
        Axis of the tick labels to hide

    Returns
    -------
    None, but the tick labels of the axis are removed, as specified
    """
    hide_tick_labels_by_index(ax, axis=axis)

def hide_first_y_tick_label(ax:plt.Axes) -> None:
    """ Hide the first tick label on the y-axis

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis

    Returns
    -------
    None, but the tick label is hidden
    """
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)

def hide_tick_labels_by_text(
        ax:plt.Axes,
        to_remove_x:Collection=set(),
        to_remove_y:Collection=set()) -> None:
    """ Hide tick labels which match the given values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis

    to_remove_{x,y}: typing.Collection[str]
        The values to remove
        
    Returns
    -------
    None, but the specified tick labels are hidden
    """
    xticks = ax.xaxis.get_major_ticks()
    num_xticks = len(xticks)
    keep_x = [i for i in range(num_xticks) if xticks[i].label1.get_text() not in to_remove_x]
    
    
    yticks = ax.yaxis.get_major_ticks()
    num_yticks = len(yticks)
    keep_y = [i for i in range(num_yticks) if yticks[i].label1.get_text() not in to_remove_y]

    hide_tick_labels_by_index(ax, keep_x=keep_x, keep_y=keep_y)


def hide_tick_labels_by_index(
        ax:plt.Axes,
        keep_x:Collection=set(),
        keep_y:Collection=set(),
        axis:str='both') -> None:
    """ Hide the tick labels on both axes.
    
    Optionally, some can be preserved.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis

    keep_{x,y} : typing.Collection[int]
        The indices of any x-axis ticks to keep. The numbers are passed directly
        as indices to the "ticks" arrays.
        
    axis : str in {`both`, `x`, `y`}
        Axis of the tick labels to hide

    Returns
    -------
    None, but the tick labels of the axis are removed, as specified
    """
    
    validation_utils.validate_in_set(axis, VALID_AXIS_VALUES, "axis")

    if axis in X_AXIS_VALUES:
        xticks = ax.xaxis.get_major_ticks()
        for xtick in xticks:
            xtick.label1.set_visible(False)

        for x in keep_x:
            xticks[x].label1.set_visible(True)

    if axis in Y_AXIS_VALUES:
        yticks = ax.yaxis.get_major_ticks()
        for ytick in yticks:
            ytick.label1.set_visible(False)

        for y in keep_y:
            yticks[y].label1.set_visible(True)

            
###
# Standard, generic plot helpers
###

def plot_simple_bar_chart(
        bars:Sequence[Sequence[float]],
        ax:Optional[plt.Axes]=None,
        labels:Optional[Sequence[str]]=None,
        colors:BarChartColorOptions=plt.cm.Blues,
        xticklabels:Optional[Union[str,Sequence[str]]]='default',
        xticklabels_rotation:IntOrString='vertical',
        xlabel:Optional[str]=None,
        ylabel:Optional[str]=None,
        spacing:float=0,
        ymin:Optional[float]=None,
        ymax:Optional[float]=None,
        use_log_scale:bool=False,
        hide_first_ytick:bool=True,
        show_legend:bool=False,
        title:Optional[str]=None,
        tick_fontsize:int=12,
        label_fontsize:int=12,
        legend_fontsize:int=12,
        title_fontsize:int=12,
        tick_offset:float=0.5):
    """ Plot a simple bar chart based on the values in `bars`
    
    Parameters
    -----------
    bars : typing.Sequence[typing.Sequence[float]]
        The heights of each bar. The "outer" sequence corresponds to
        each clustered group of bars, while the "inner" sequence gives
        the heights of each bar within the group.
        
        As a data science example, the "outer" groups may correspond
        to different datasets, while the "inner" group corresponds to
        different methods.
    
    ax : typing.Optional[matplotlib.axes.Axes]
        The axis. If not given, then one will be created.
        
    labels : typing.Optional[typing.Sequence[str]]
        The label for each "outer" group in `bars`
        
    colors : BarChartColorOptions
        The colors of the bars for each "inner" group. The options and
        their interpretations are:
        
        * color map : the color of each bar will be taken as equi-distant colors sampled from the map. For example, if there are three bars in thei nner group, then the colors will be: `colors(0.0)`, `colors(0.5)`, and `colors(1.0)`.
            
        * sequence of colors : the color of each bar will be taken from the respective position in the sequence.
            
        * scalar (int or str) : all bars will use this color
        
    xticklabels : typing.Optional[typing.Union[str,typing.Sequence[str]]]
        The tick labels for the "outer" groups. The options and their
        interpretations are:
        
        * None : no tick labels will be shown
        * "default" : the tick labels will be the numeric tick positions
        * sequence of strings : the tick labels will be the respective strings
    
    xticklabels_rotation : typing.Union[str,int]
        The rotation for the `xticklabels`. If a string is given, it should be
        something which matplotlib can interpret as a rotation.
        
    {x,y}label : typing.Optional[str]
        Labels for the respective axes
        
    spacing : float
        The distance on the x axis between the "outer" groups.
        
    y{min,max} : typing.Optional[float]
        The min and max for the y axis. If not given, the default min is 0
        (or 1 if a logarithmic scale is used, see option below), and the default
        max is 2 times the height of the highest bar in any group.
        
    use_log_scale : bool
        Whether to use a normal or logarithmic scale for the y axis
        
    hide_first_ytick : bool
        Whether to hide the first tick mark and label on the y axis. Typically,
        the first tick mark is either 0 or 1 (depending on the scale of the y
        axis). This can be distracting to see, so the default is to hide it.
        
    show_legend : bool
        Whether to show the legend
        
    title : typing.Optional[str]
        A title for the axis
        
    {tick,label,legend,title}_fontsize : int
        The font size for the respective elements
        
    tick_offset : float
        The offset of the tick mark and label for the outer groups on the x axis
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure on which the bars were plotted
    
    ax : matplotlib.axes.Axes
        The axis on which the bars were plotted    
    """
    fig, ax = _get_fig_ax(ax)
    
    mpl_bars = []

    # first, handle the bars
    
    # TODO: check that the bar arrays are all the same length
    xticks = np.arange(len(bars[0]))
    
    width = 1 - 2*spacing
    width /= len(bars)
    
    if isinstance(colors, matplotlib.colors.Colormap):
        # then use "num_bars" equi-distant colors
        ls = np.linspace(0, 1, len(bars))
        color_vals = [colors(c) for c in ls]
        colors = color_vals
        
    elif validation_utils.validate_is_sequence(colors, raise_on_invalid=False):
        # make sure this is the correct size
        if len(colors) != len(bars):
            msg = ("The number of colors ({}) and the number of bars({}) does "
                "not match.".format(len(colors), len(bars)))
            raise ValueError(msg)
    else:
        # we assume color is a scalar, and we will use the same color
        # for all bars
        colors = [colors] * len(bars)
        
    if labels is None:
        labels = np.full(len(bars), "", dtype=object)

    for i, bar in enumerate(bars):
        xpos = xticks + i*width
        if len(bar) < len(xpos):
            xpos = xpos[:len(bar)]
        mpl_bar = ax.bar(xpos, bar, width=width, color=colors[i], label=labels[i])
        mpl_bars.append(mpl_bar)
        
    
    # now the x-axis
    if isinstance(xticklabels, str):
        if xticklabels == "default":
            xticklabels = xticks
       

    tick_offset = tick_offset - spacing

    if xticklabels is not None:
        ax.set_xticks(xticks+tick_offset)
        ax.set_xticklabels(
            xticklabels,
            fontsize=tick_fontsize, 
            rotation=xticklabels_rotation
        )
    else:
        ax.tick_params(
            axis='x', 
            which='both', 
            bottom='off',
            top='off',
            labelbottom='off'
        )

    ax.set_xlim((-width, len(xticks)+width/2))
    
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    
    # and the y-axis
    if use_log_scale:
        ax.set_yscale('log')

    if ymin is None:
        ymin = 0
        if use_log_scale:
            ymin=1

    if ymax is None:
        ymax = 2*max(max(x) for x in bars)
    
    ax.set_ylim((ymin, ymax))
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    
    if hide_first_ytick:
        yticks = ax.yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)

    # and the legend
    if show_legend:
        ax.legend(fontsize=legend_fontsize)

    # and the title
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    
    return fig, ax


def plot_simple_scatter(
        x:Sequence[float],
        y:Sequence[float],
        ax:Optional[plt.Axes]=None,
        equal_aspect:bool=True,
        set_lim:bool=True,
        show_y_x_line:bool=True,
        xy_line_kwargs:dict={},
        **kwargs)->FigAx:
    """ Plot a simple scatter plot of `x` vs. `y` on `ax`
    
    See the matplotlib documentation for more keyword arguments and details: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter
    
    Parameters
    ----------
    {x,y} : typing.Sequence[float]
        The values to plot
        
    ax : typing.Optional[matplotlib.axes.Axes]
        The axis. If not given, then one will be created.
        
    equal_aspect : bool
        Whether to set the aspect of the axis to `equal`
        
    set_lim : bool
        Whether to automatically set the min and max axis limits
        
    show_y_x_line : bool
        Whether to draw the y=x line. This will look weird if `set_lim` is False.
        
    xy_line_kwargs : typing.Mapping
        keyword arguments for plotting the y=x line, if it plotting
        
    **kwargs : <key>=<value> pairs
        Additional keyword arguments to pass to the scatter function. Some useful
        keyword arguments are:
        
        * `label` : the label for a legend
        * `marker` : https://matplotlib.org/examples/lines_bars_and_markers/marker_reference.html
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure on which the scatter points were plotted
    
    ax : matplotlib.axes.Axes
        The axis on which the scatter points were plotted
    """
    fig, ax = _get_fig_ax(ax)
        
    ax.scatter(x,y, **kwargs)

    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    lim = (min_val, max_val)
    
    if set_lim:
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        
    if show_y_x_line:
        ax.plot(lim, lim, **xy_line_kwargs)
    
    if equal_aspect:
        ax.set_aspect('equal')
        
    return fig, ax


def plot_stacked_bar_graph(
                   ax,                                 # axes to plot onto
                   data,                               # data to plot
                   colors=plt.cm.Blues,                # color map for each level or list of colors
                   x_tick_labels = None,                     # bar specific labels
                   stack_labels=None,                        # the text for the legend
                   y_ticks = None,                        # information used for making y ticks
                   y_tick_labels=None,
                   hide_first_ytick=True,
                   edge_colors=None,                      # colors for edges
                   showFirst=-1,                       # only plot the first <showFirst> bars
                   scale=False,                        # scale bars to same height
                   widths=None,                        # set widths for each bar
                   heights=None,                       # set heights for each bar
                   y_title=None,                          # label for x axis
                   x_title=None,                          # label for y axis
                   gap=0.,                             # gap between bars
                   end_gaps=False,                      # allow gaps at end of bar chart (only used if gaps != 0.)
                   show_legend=True,                   # whether to show the legend
                   legend_loc="best",                  # if using a legend, its location
                   legend_bbox_to_anchor=None,         # for the legend
                   legend_ncol=-1,                     # for the legend
                   log=False,                          # whether to use a log scale
                   font_size=8,                        # the font size to use for the tick labels
                   label_font_size=12,                 # the font size for the labels
                   legend_font_size=8
                   ):
    """ Create a stacked bar plot with the given characteristics. 
    
    This code is adapted from code by Michael Imelfort.
    """

#------------------------------------------------------------------------------
# data fixeratering

    # make sure this makes sense
    if showFirst != -1:
        showFirst = np.min([showFirst, np.shape(data)[0]])
        data_copy = np.copy(data[:showFirst]).transpose().astype('float')
        data_shape = np.shape(data_copy)
        if heights is not None:
            heights = heights[:showFirst]
        if widths is not None:
            widths = widths[:showFirst]
        showFirst = -1
    else:
        data_copy = np.copy(data).transpose()
    data_shape = np.shape(data_copy)

    # determine the number of bars and corresponding levels from the shape of the data
    num_bars = data_shape[1]
    levels = data_shape[0]

    if widths is None:
        widths = np.array([1] * num_bars)
        x = np.arange(num_bars)
    else:
        if not validation_utils.validate_is_sequence(widths, raise_on_invalid=False):
            widths = np.full(num_bars, widths)
            print("widths: ", widths)
        x = [0]
        for i in range(1, len(widths)):
            #x.append(x[i-1] + (widths[i-1] + widths[i])/2)
            x.append(x[i-1] + widths[i])

    # stack the data --
    # replace the value in each level by the cumulative sum of all preceding levels
    data_stack = np.reshape([float(i) for i in np.ravel(np.cumsum(data_copy, axis=0))], data_shape)

    # scale the data is needed
    if scale:
        data_copy /= data_stack[levels-1]
        data_stack /= data_stack[levels-1]
        if heights is not None:
            print("WARNING: setting scale and heights does not make sense.")
            heights = None
    elif heights is not None:
        data_copy /= data_stack[levels-1]
        data_stack /= data_stack[levels-1]
        for i in np.arange(num_bars):
            data_copy[:,i] *= heights[i]
            data_stack[:,i] *= heights[i]
# plot


    # if we were given a color map, convert it to a list of colors
    if isinstance(colors, matplotlib.colors.Colormap):
        colors = [ colors(i/levels) for i in range(levels)]
    
    if edge_colors is None:
        edge_colors = colors
    elif not validation_utils.validate_is_sequence(edge_colors, raise_on_invalid=False):
        edge_colors = np.full(levels, edge_colors, dtype=object)
    elif len(edge_colors) != len(levels):
        msg = "The number of edge_colors must match the number of stacks."
        raise ValueError(msg)

    # take cae of gaps
    gapd_widths = [i - gap for i in widths]

    if stack_labels is None:
        stack_labels = np.full(levels, '', dtype=object)

    # bars
    bars = []
    bar = ax.bar(x,
           data_stack[0],
           color=colors[0],
           edgecolor=edge_colors[0],
           width=gapd_widths,
           linewidth=0.5,
           align='center',
           label=stack_labels[0],
           log=log
           )
    bars.append(bar)

    for i in np.arange(1,levels):
        bar = ax.bar(x,
               data_copy[i],
               bottom=data_stack[i-1],
               color=colors[i],
               edgecolor=edge_colors[i],
               width=gapd_widths,
               linewidth=0.5,
               align='center',
               label=stack_labels[i],
               log=log
               )
        bars.append(bar)

    # borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    #ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # make ticks if necessary
    if y_ticks is not None:
        ax.set_yticks(y_ticks)

        if y_tick_labels is not None:
            ax.set_yticklabels(y_tick_labels, fontsize=font_size)
            
        if hide_first_ytick:
            yticks = ax.yaxis.get_major_ticks()
            yticks[0].label1.set_visible(False)

    else:
        ax.tick_params(
            axis='y',
            which='both',
            left='off',
            right='off',
            labelright='off',
            labelleft='off')

    if x_tick_labels is not None:
        ax.tick_params(axis='x', which='both', labelsize=font_size, direction="out")
        ax.xaxis.tick_bottom()
        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels, rotation='vertical')
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])

    # limits
    if end_gaps:
        ax.set_xlim(-1.*widths[0]/2. - gap/2., np.sum(widths)-widths[0]/2. + gap/2.)
    else:
        ax.set_xlim(-1.*widths[0]/2. + gap/2., np.sum(widths)-widths[0]/2. - gap/2.)
    
    ymin = 0
    if log:
        ymin = 1
    
    # labels
    if x_title is not None:
        ax.set_xlabel(x_title, fontsize=label_font_size)
    if y_title is not None:
        ax.set_ylabel(y_title, fontsize=label_font_size)

    # legend
    if show_legend:
        if legend_ncol < 1:
            legend_ncol = len(stack_labels)
        lgd = ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, ncol=legend_ncol,
                fontsize=legend_font_size)

    return bars

def plot_sorted_values(
        values:Sequence[float],
        ymin:Optional[float]=None,
        ymax:Optional[float]=None,
        ax:Optional[plt.Axes]=None,
        scale_x:bool=False,
        **kwargs) -> FigAx:
    """ Sort `values` and plot them

    Parameters
    ----------
    values : typing.Sequence[float]
        The values to plot

    y_{min,max} : float
        The min and max values for the y-axis. If not given, then these
        default to the minimum and maximum values in the list.
        
    scale_x : bool
        If True, then the `x` values will be equally-spaced between 0 and 1.
        Otherwise, they will be the values 0 to len(values)

    ax : typing.Optional[matplotlib.axes.Axes]
        An axis for plotting. If this is not given, then a figure and axis will
        be created.
        
    **kwargs : <key>=<value> pairs
        Additional keyword arguments to pass to the plot function. Some useful
        keyword arguments are:
        
        * `label` : the label for a legend
        * `lw` : the line width
        * `ls` : https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html
        * `marker` : https://matplotlib.org/examples/lines_bars_and_markers/marker_reference.html

    Returns
    -------
    fig :  matplotlib.figure.Figure
        The Figure associated with `ax`, or a new Figure

    ax : matplotlib.axes.Axes
        Either `ax` or a new Axis
    """
    fig, ax = _get_fig_ax(ax)
        
    y = np.sort(values)
    
    if scale_x:
        x = np.linspace(0,1, len(y))
    else:
        x = np.arange(len(y))
    
    ax.plot(x,y, **kwargs)

    if ymin is None:
        ymin = y[0]

    if ymax is None:
        ymax = y[-1]
    
    ax.set_ylim((ymin, ymax))
    ax.set_xlim((0, len(y)))
    
    return fig, ax 

###
# High-level, ML and statistics plotting helpers
###
def plot_binary_prediction_scores(
        y_scores:Sequence[float],
        y_true:Sequence[int],
        positive_label:int=1,
        positive_line_color='g',
        negative_line_color='r',
        line_kwargs:typing.Mapping={},
        positive_line_kwargs:typing.Mapping={},
        negative_line_kwargs:typing.Mapping={},
        title:Optional[str]=None,
        ylabel:Optional[str]="Score",
        xlabel:Optional[str]="Instance",
        title_font_size:int=20, 
        label_font_size:int=15,
        ticklabels_font_size:int=15,
        ax:Optional[plt.Axes]=None) -> FigAx:
    """ Plot separate lines for the scores of the positives and negatives
    
    Parameters
    ----------
    y_scores : typing.Sequence[float]
        The predicted scores of the positive class. For example, this may be
        found using something like: `y_scores = y_proba_pred[:,1]` for
        probabilistic predictions from most `sklearn` classifiers.
        
    y_true : typing.Sequence[int]
        The ground truth labels
        
    positive_label : int
        The value for the "positive" class
        
    {positive,negative}_line_color : color
        Values to use for the color of the respective lines. These can be
        anything which `matplotlib.plot` can interpret.
        
        These values have precedent over the other `kwargs` parameters.
        
    line_kwargs : typing.Mapping
        Other keyword arguments passed through to `plot` for both lines.
        
    {positive,negative}_line_kwargs : typing.Mapping
        Other keyword arguments pass through to `plot` for only the
        respective line.
        
        These values have precedent over `line_kwargs`.
    
    title : typing.Optional[str]
        If given, the title of the axis is set to this value
        
    {y,x}label : typing.Optional[str]
        Text for the respective labels
    
    {title,label,ticklabels}_font_size : int
        The font sizes for the respective elements.
    
    ax : typing.Optional[matplotlib.axes.Axes]
        The axis. If not given, then one will be created.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure on which the scores lines were plotted
    
    ax : matplotlib.axes.Axes
        The axis on which the score lines were plotted
    """
    
    fig, ax = _get_fig_ax(ax)

    # pull out the positivies
    m_positives = (y_true == positive_label)

    y_scores_positive = y_scores[m_positives]
    y_scores_negative = y_scores[~m_positives]

    positives_kwargs = {**line_kwargs, **positive_line_kwargs}
    positives_kwargs['color'] = positive_line_color
    plot_sorted_values(y_scores_positive, ax=ax, **positives_kwargs)
    
    negatives_kwargs = {**line_kwargs, **negative_line_kwargs}
    negatives_kwargs['color'] = negative_line_color
    plot_sorted_values(y_scores_negative, ax=ax, **negatives_kwargs)
    
    if title is not None:
        ax.set_title(title, fontsize=title_font_size)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_font_size)
        
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_font_size)
        
    set_ticklabels_fontsize(ax, ticklabels_font_size)
    
    return fig, ax

def plot_confusion_matrix(
        confusion_matrix:np.ndarray,
        ax:Optional[plt.Axes]=None,
        show_cell_labels:bool=True,
        show_colorbar:bool=True,
        title:Optional[str]="Confusion matrix", 
        cmap:matplotlib.colors.Colormap=plt.cm.Blues, 
        true_tick_labels:Optional[Sequence[str]]=None, 
        predicted_tick_labels:Optional[Sequence[str]]=None, 
        ylabel:Optional[str]="True labels", 
        xlabel:Optional[str]="Predicted labels", 
        title_font_size:int=20, 
        label_font_size:int=15,
        true_tick_rotation:Optional[IntOrString]=None,
        predicted_tick_rotation:Optional[IntOrString]=None,
        out:Optional[str]=None) -> FigAx:

    """ Plot the given confusion matrix
    
    Parameters
    -----------
    confusion_matrix : numpy.ndarray
        A 2-d array, presumably from :func:`sklearn.metrics.confusion_matrix`
        or something similar. The rows (Y axis) are the "true" classes while
        the columns (X axis) are the "predicted" classes.
    
    ax : typing.Optional[matplotlib.axes.Axes]
        The axis. If not given, then one will be created.
        
    show_cell_labels : bool
        Whether to show the values within each cell
        
    show_colorbar : bool
        Whether to show a color bar
        
    title : typing.Optional[str]
        If given, the title of the axis is set to this value
        
    cmap : matplotlib.colors.Colormap
        A colormap to determine the cell colors
    
    {true,predicted}_tick_labels : typing.Optional[typing.Sequence[str]]
        Text for the Y (true) and X (predicted) axis, respectively
        
    {y,x}label : typing.Optional[str]
        Text for the respective labels
    
    {title,label}_font_size : int
        The font sizes for the respective elements. The class labels (on the
        tick marks) use the `label_font_size`.
        
    {true,predicted}_tick_rotation : typing.Optional[IntOrString]
        The rotation arguments for the respective tick labels. Please see
        the matplotlib text documentation (https://matplotlib.org/api/text_api.html#matplotlib.text.Text)
        for more details.
        
    out : typing.Optional[str]
        If given, the plot will be saved to this file.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure on which the confusion matrix was plotted
    
    ax : matplotlib.axes.Axes
        The axis on which the confusion matrix was plotted
    """
    fig, ax = _get_fig_ax(ax)

    # a hack to give cmap a default without importing pyplot for arguments
    if cmap == None:
        cmap = plt.cm.Blues

    mappable = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    
    if show_colorbar:
        fig.colorbar(mappable)
    ax.grid(False)
    
    true_tick_marks = np.arange(confusion_matrix.shape[0])
    ax.set_yticks(true_tick_marks)
    
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_font_size)

    if true_tick_labels is None:
        true_tick_labels = list(true_tick_marks)

    ax.set_yticklabels(
        true_tick_labels,
        fontsize=label_font_size,
        rotation=true_tick_rotation
    )
    
    predicted_tick_marks = np.arange(confusion_matrix.shape[1])
    ax.set_xticks(predicted_tick_marks)
    
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_font_size)

    if predicted_tick_labels is None:
        predicted_tick_labels = list(predicted_tick_marks)

    ax.set_xticklabels(
        predicted_tick_labels,
        fontsize=label_font_size,
        rotation=predicted_tick_rotation
    )

    if show_cell_labels:
        # the choice of color is based on this SO thread:
        # https://stackoverflow.com/questions/2509443
        color_threshold = 125

        s = confusion_matrix.shape
        it = itertools.product(range(s[0]), range(s[1]))
        for i,j in it:
            
            val = confusion_matrix[i,j]
            cell_color = cmap(mappable.norm(val))

            # see the SO thread mentioned above
            color_intensity = (
                (255*cell_color[0] * 299) +
                (255*cell_color[1] * 587) +
                (255*cell_color[2] * 114)
            ) / 1000

            
            font_color = "white"
            if color_intensity > color_threshold:
                font_color = "black"
            text = val
            ax.text(j, i, text, ha='center', va='center', color=font_color,
                size=label_font_size)
    
    if title is not None:
        ax.set_title(title, fontsize=title_font_size)
        
    fig.tight_layout()

    if out is not None:
        plt.savefig(out, bbox_inches='tight')
        
    return fig, ax


def plot_mean_roc_curve(
        tprs:Sequence[Sequence[float]],
        fprs:Sequence[Sequence[float]],
        aucs:Optional[float]=None,
        label_note:Optional[str]=None,
        line_style:Mapping={'c':'b', 'lw':2, 'alpha':0.8},
        fill_style:Mapping={'color': 'grey', 'alpha':0.2},
        show_xy_line:bool=True,
        xy_line_kwargs:Mapping={'color': 'r', 'ls': '--', 'lw': 2},
        ax:Optional[plt.Axes]=None,
        title:Optional[str]=None,
        xlabel:Optional[str]="False positive rate",
        ylabel:Optional[str]="True positive rate",
        title_font_size:int=25,
        label_font_size:int=20,
        ticklabels_font_size:int=20) -> FigAx:
    """ Plot the mean plus/minus the standard deviation of the given ROC curves
    
    Parameters
    ----------
    tprs : typing.Sequence[typing.Sequence[float]]
        The true positive rate at each threshold
    
    fprs : typing.Sequence[typing.Sequence[float]]
        The false positive rate at each threshold
        
    aucs : typing.Optional[float]
        The calculated area under the ROC curve
        
    label_note : typing.Optional[str]
        A prefix for the label in the legend for this line.
        
    {line,fill}_style : typing.Mapping
        Keyword arguments for plotting the line and `fill_between`,
        respectively. Please see the mpl docs for more details.
        
    show_xy_line : bool
        Whether to draw the y=x line
        
    xy_line_kwargs : typing.Mapping
        Keyword arguments for plotting the x=y line.
    
    title : typing.Optional[str]
        If given, the title of the axis is set to this value
        
    {x,y}label : typing.Optional[str]
        Text for the respective labels
        
    {title,label,ticklabels}_font_size : int
        The font sizes for the respective elements
        
    ax : typing.Optional[matplotlib.axes.Axes]
        The axis. If not given, then one will be created.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure on which the ROC curves were plotted
    
    ax : matplotlib.axes.Axes
        The axis on which the ROC curves were plotted
    """
    fig, ax = _get_fig_ax(ax)
    
    # interpolate across the different curves so we have the same points
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs = []

    for tpr, fpr in zip(tprs, fprs):
        interp_tprs.append(scipy.interp(mean_fpr, fpr, tpr))
        interp_tprs[-1][0] = 0.0

    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    
    mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    label = "AUC: {:.2f} $\pm$ {:.2f}".format(mean_auc, std_auc)
    
    if label_note is not None:
        label = label_note + label
    
    ax.plot(mean_fpr, mean_tpr, label=label, **line_style)

    std_tpr = np.std(interp_tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, **fill_style)
    
    if show_xy_line:
        ax.plot([0,1], [0,1], label='Luck', **xy_line_kwargs)
        
    ax.set_aspect('equal')
    ax.set_xlim((-0.05, 1.05))
    ax.set_ylim((-0.05, 1.05))

    if title is not None and len(title) > 0:
        ax.set_title(title, fontsize=title_font_size)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_font_size)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_font_size)
        
    set_ticklabels_fontsize(ax, ticklabels_font_size)
    
    return fig, ax

def plot_roc_curve(
        tpr:Sequence[Sequence[float]],
        fpr:Sequence[Sequence[float]],
        auc:Optional[float]=None,
        show_points:bool=True,
        ax:Optional[plt.Axes]=None,
        method_names:Optional[Sequence[str]]=None,
        out:Optional[str]=None,
        line_colors:Optional[Sequence]=None,
        point_colors:Optional[Sequence]=None,
        alphas:Optional[Sequence[float]]=None,
        line_kwargs:Optional[Mapping]=None,
        point_kwargs:Optional[Mapping]=None,
        title:Optional[str]="Receiver operating characteristic curves",
        xlabel:Optional[str]="False positive rate",
        ylabel:Optional[str]="True positive rate",
        title_font_size:int=20,
        label_font_size:int=15,
        ticklabels_font_size:int=15) -> FigAx:
    """ Plot the ROC curve for the given `fpr` and `tpr` values
    
    Currently, this function plots multiple ROC curves.
    
    Optionally, add a note of the `auc`.
    
    Parameters
    ----------
    tpr : typing.Sequence[typing.Sequence[float]]
        The true positive rate at each threshold
    
    fpr : typing.Sequence[typing.Sequence[float]]
        The false positive rate at each threshold
        
    auc : typing.Optional[float]
        The calculated area under the ROC curve
        
    show_points : bool
        Whether to plot points at each threshold
    
    ax : typing.Optional[matplotlib.axes.Axes]
        The axis. If not given, then one will be created.
        
    method_names : typing.Optional[typing.Sequence[str]]
        The name of each method
        
    out : typing.Optional[str]
        If given, the plot will be saved to this file.
        
    line_colors : typing.Optional[typing.Sequence[color]]
        The color of each ROC line
        
    point_colors : typing.Optional[typing.Sequence[color]]
        The color of the points on each each ROC line
    
    alphas : typing.Optional[typing.Sequence[float]]
        An alpha value for each method
        
    {line,point}_kwargs : typing.Optional[typing.Mapping]
        Additional keyword arguments for the respective elements
        
    title : typing.Optional[str]
        If given, the title of the axis is set to this value
        
    {x,y}label : typing.Optional[str]
        Text for the respective labels
        
    {title,label,ticklabels}_font_size : int
        The font sizes for the respective elements
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure on which the ROC curves were plotted
    
    ax : matplotlib.axes.Axes
        The axis on which the ROC curves were plotted
    """
    fig, ax = _get_fig_ax(ax)
    
    if alphas is None:
        alphas = [1.0] * len(tpr)
    elif len(alphas) != len(tpr):
        msg = "The ROC curve must have the same number of alpha values as methods"
        raise ValueError(msg)
        
    if line_colors is None:
        line_colors = ['k'] * len(tpr)
    elif len(line_colors) != len(tpr):
        msg = "The ROC curve must have the same number of line colors as methods"
        raise ValueError(msg)
    
    if point_colors is None:
        point_colors = ['k'] * len(tpr)
    elif len(point_colors) != len(tpr):
        msg = "The ROC curve must have the same number of point colors as methods"
        raise ValueError(msg)

    for i in range(len(tpr)):
        l = ""
        if method_names is not None:
            l += str(method_names[i])

        if auc is not None:
            l += " "
            l += "AUC: {:.2f}".format(auc[i])
           
        
        if show_points:
            for j in range(1, len(fpr[i])):
                points_y = [tpr[i][j-1], tpr[i][j]]
                points_x = [fpr[i][j-1], fpr[i][j]]
                # this plots the lines connecting each point
                ax.plot(points_x, points_y, color=line_colors[i], zorder=1, alpha=alphas[i], **line_kwargs)
                
            ax.scatter(fpr[i], tpr[i], label=l, c=point_colors[i], alpha=alphas[i], zorder=2, **point_kwargs)
        else:
            ax.plot(fpr[i], tpr[i], alpha=alphas[i], c=line_colors[i], label=l, **line_kwargs)

    # plot 
    ax.plot([0,1], [0,1], label='Luck', color='r', ls='--', lw=2)
    ax.set_aspect('equal')
    ax.set_xlim((-0.05, 1.05))
    ax.set_ylim((-0.05, 1.05))

    if title is not None and len(title) > 0:
        ax.set_title(title, fontsize=title_font_size)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_font_size)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_font_size)
        
    set_ticklabels_fontsize(ax, ticklabels_font_size)
        
    if out is not None:
        fig.savefig(out, bbox_inches='tight')
        
    return fig, ax
    
def plot_trend_line(
        x:Sequence[float],
        intercept:float,
        slope:float,
        power:float,
        ax:Optional[plt.Axes]=None,
        **kwargs) -> FigAx:
    """ Draw the trend line implied by the given coefficients.

    Parameters
    ----------
    x : typing.Sequence[float]
        The points at which the function will be evaluated and where 
        the line will be drawn

    {intercept,slope,power} : float
        The coefficients of the trend line. Presumably, these come from
        :func:`pyllars.stats_utils.fit_with_least_squares` or something
        similar.
    
    ax : typing.Optional[matplotlib.axes.Axes]
        The axis. If not given, then one will be created.

    **kwargs : <key>=<value> pairs
        Keyword arguments to pass to the ax.plot function (color, etc.). Please
        consult the matplotlib documentation for more details: https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure on which the trend line was plotted
    
    ax : matplotlib.axes.Axes
        The axis on which the trend line was plotted
    """
    fig, ax = _get_fig_ax(ax)
    
    x = np.sort(x)
    y = power * x ** 2 + slope * x + intercept

    #Plot trendline
    ax.plot(x, y, **kwargs)
    
    return fig, ax

def plot_venn_diagram(
        sets:MapOrSequence,
        ax:Optional[plt.Axes]=None,
        set_labels:Optional[Sequence[str]]=None,
        weighted:bool=False,
        use_sci_notation:bool=False,
        sci_notation_limit:float=999,
        labels_fontsize:int=14,
        counts_fontsize:int=12) -> matplotlib_venn._common.VennDiagram:
    """ Wrap the matplotlib_venn package.
    
    Please consult the package documentation for more details: https://github.com/konstantint/matplotlib-venn
    
    **N.B.** Unlike most of the other high-level plotting helpers,
    this function returns the venn diagram object rather than the
    figure and axis objects.
    
    Parameters
    -----------
    set : typing.Union[typing.Mapping,typing.Sequence]
        If a dictionary, it must follow the conventions of 
        `matplotlib_venn`. If a dictionary is given, the number of sets
        will be guessed based on the length of one of the entries.
        
        If a sequence is given, then it must be of length two or three.
        
        The type of venn diagram will be based on the number of sets.
    
    ax : typing.Optional[matplotlib.axes.Axes]
        The axis. If not given, then one will be created.
        
    set_labels : typing.Optional[typing.Sequence[str]]
        The label for each set. The order of the labels must match the
        order of the sets.
    
    weighted : bool
        Whether the diagram is weighted (in which the size of the circles
        in the venn diagram are based on the number of elements) or
        unweighted (in which all circles are the same size)
        
    use_sci_notation : bool
        Whether to convert numbers to scientific notation
        
    sci_notation_limit : float
        The maximum number to show before switching to scientific
        notation
        
    {labels,counts}_fontsize : int
        The respective font sizes
        
    Returns
    ---------
    venn_diagram : matplotlib_venn._common.VennDiagram
        The venn diagram
    """
    key_len = 0
    if isinstance(sets, dict):
        random_key = list(sets.keys())[0]
        key_len = len(random_key)
    
    if (len(sets) == 2) or (key_len == 2):
        if weighted:
            v = matplotlib_venn.venn2(sets, ax=ax, set_labels=set_labels)
        else:
            v = matplotlib_venn.venn2_unweighted(sets, ax=ax, set_labels=set_labels)
            
    elif (len(sets) == 3) or (key_len == 3):
        if weighted:
            v = matplotlib_venn.venn3(sets, ax=ax, set_labels=set_labels)
        else:
            v = matplotlib_venn.venn3_unweighted(sets, ax=ax, set_labels=set_labels)
    else:
        msg = "Only two or three sets are supported"
        raise ValueError(msg)
    
    
    for l in v.set_labels:
        if l is not None:
            l.set_fontsize(labels_fontsize)
        
    for l in v.subset_labels:
        if l is None:
            continue

        l.set_fontsize(counts_fontsize)
        
        if use_sci_notation:
            val = int(l.get_text())
            if val > sci_notation_limit:
                val = "{:.0E}".format(val)
                l.set_text(val)

    return v

###
# Other helpers
###
def add_fontsizes_to_args(
        args:argparse.Namespace,
        legend_title_fontsize:int=12,
        legend_fontsize:int=10,
        title_fontsize:int=20,
        label_fontsize:int=15,
        ticklabels_fontsize:int=10):
    """ Add reasonable default fontsize values to `args`
    """
    args.legend_title_fontsize = legend_title_fontsize
    args.legend_fontsize = legend_fontsize
    args.title_fontsize = title_fontsize
    args.label_fontsize = label_fontsize
    args.ticklabels_fontsize = ticklabels_fontsize

def draw_rectangle(
        ax:plt.Axes,
        base_x:float,
        base_y:float,
        width:float,
        height:float,
        center_x:bool=False, 
        center_y:bool=False,
        **kwargs) -> FigAx:
    """ Draw a rectangle at the given x and y coordinates.
    
    Optionally, these can be adjusted such that they are the respective
    centers rather than edge values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which the rectangle will be drawn

    base_{x,y} : float
        The base x and y coordinates

    {width,height} : float
        The width (change in x) and height (change in y) of the rectangle

    center_{x,y}: bool
        Whether to adjust the x and y coordinates such that they become the
        center rather than lower left. In particular, if `center_x` is `True`,
        then `base_x` will be shifted left by `width/2`; likewise, if `center_y`
        is `True`, then `base_y` will be shifted down by `height/2`.

    **kwargs : key=value pairs
        Additional keywords are passed to the patches.Rectangle constructor.
        Please see the matplotlib documentation for more details: https://matplotlib.org/api/_as_gen/matplotlib.patches.Rectangle.html
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure on which the rectangle was drawn
    
    ax : matplotlib.axes.Axes
        The axis on which the rectangle was drawn
    """
    fig, ax = _get_fig_ax(ax)
    
    y_offset = 0
    if center_y:
        y_offset = height/2
        
    x_offset = 0
    if center_x:
        x_offset = width/2
        
    y = base_y - y_offset
    x = base_x - x_offset
    ax.add_patch(patches.Rectangle((x,y), width, height, **kwargs))  
    
    return fig, ax



def get_diff_counts(data_np):
    """ This function extracts the differential counts necessary for visualization
        with stacked_bar_graph. It assumes the counts for each bar are given as a
        separate row in the numpy 2-d array. Within the rows, the counts are ordered
        in ascending order. That is, the first column contains the smallest count, the
        second column contains the next-smallest count, etc.
        
        For example, if the columns represnt some sort of filtering approach, then the
        last column would contain the unfiltered count, the next-to-last column 
        would give the count after the first round of filtering, etc.
    """
    
    # add an extra column so the diff counts will work
    zeros = np.zeros((data_np.shape[0], 1))
    data_np = np.append(zeros, data_np, axis=1)
    
    # get the diffs so the stacks work correctly
    diff = np.diff(data_np)
    return diff