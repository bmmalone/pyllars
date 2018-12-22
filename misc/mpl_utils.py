"""
This module contains a number of helper functions for matplotlib.
"""

import itertools

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import misc.utils as utils
import misc.validation_utils as validation_utils

import logging
logger = logging.getLogger(__name__)

_VALID_AXIS_VALUES = {
    'both',
    'x',
    'y'
}

_X_AXIS_VALUES = {
    'both',
    'x'
}

_Y_AXIS_VALUES = {
    'both',
    'y'
}

def add_fontsizes_to_args(args,
        legend_title_fontsize=12,
        legend_fontsize=10,
        title_fontsize=20,
        label_fontsize=15,
        ticklabels_fontsize=10):
    """ Add reasonable default fontsize values to the arguments
    """
    args.legend_title_fontsize = legend_title_fontsize
    args.legend_fontsize = legend_fontsize
    args.title_fontsize = title_fontsize
    args.label_fontsize = label_fontsize
    args.ticklabels_fontsize = ticklabels_fontsize



def set_legend_title_fontsize(ax, fontsize):
    """ Set the font size of the title of the legend.

    Parameters
    ----------
    ax: mpl.Axis
        The axis

    fontsize: int, or string mpl recognizes
        The size of the legend title

    Returns
    -------
    None, but the legend title fontsize is updated
    """
    legend = ax.legend_
    plt.setp(legend.get_title(),fontsize=fontsize)

def set_legend_fontsize(ax, fontsize):
    """ Set the font size of the items of the legend.

    Parameters
    ----------
    ax: mpl.Axis
        The axis

    fontsize: int, or string mpl recognizes
        The size of the legend text

    Returns
    -------
    None, but the legend text fontsize is updated
    """
    legend = ax.legend_
    plt.setp(legend.get_texts(),fontsize=fontsize)

    
def set_title_fontsize(ax, fontsize):
    """ Set the font size of the title of the axis.

    Parameters
    ----------
    ax: mpl.Axis
        The axis

    fontsize: int, or string mpl recognizes
        The size of the title font
 
    Returns
    -------
    None, but the  title fontsize is updated
    """
    ax.title.set_fontsize(fontsize=fontsize)

def set_label_fontsize(ax, fontsize, axis='both'):
    """ Set the font size of the label of the axis.

    Parameters
    ----------
    ax: mpl.Axis
        The axis

    fontsize: int, or string mpl recognizes
        The size of the title font

    which: string
        Should be 'both', 'x', or 'y'

    Returns
    -------
    None, but the respective label fontsizes are updated
    """
    if (axis == 'both') or (axis=='x'):
        l = ax.xaxis.label
        l.set_fontsize(fontsize)

    if (axis == 'both') or (axis=='y'):
        l = ax.yaxis.label
        l.set_fontsize(fontsize)





def center_splines(ax):
    """ This function places the splines of the given axis in the center of the
        plot. This is useful for things like scatter plots where (0,0) should be
        in the center of the plot.

        Parameters
        ----------
        ax : mpl.Axis
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

def hide_first_y_tick_label(ax):
    """ Hide the first tick label on the y-axis.

    Parameters
    ----------
    ax: mpl.Axis
        The axis

    Returns
    -------
    None, but the tick label is hidden
    """
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)

def hide_tick_labels_by_text(ax, to_remove_x=[], to_remove_y=[]):
    """ Hide tick labels which match the given values.

    Parameters
    ----------
    ax: mpl.Axis
        The axis

    to_remove_{x,y}: list-like of strings
        The values to remove
    """
    xticks = ax.xaxis.get_major_ticks()
    num_xticks = len(xticks)
    keep_x = [i for i in range(num_xticks) if xticks[i].label1.get_text() not in to_remove_x]
    
    
    yticks = ax.yaxis.get_major_ticks()
    num_yticks = len(yticks)
    keep_y = [i for i in range(num_yticks) if yticks[i].label1.get_text() not in to_remove_y]

    hide_tick_labels(ax, keep_x=keep_x, keep_y=keep_y)


def hide_tick_labels(ax, keep_x=[], keep_y=[], axis='both'):
    """ Hide the tick labels on both axes. Optionally, some can be preserved.

    Parameters
    ----------
    ax : mp.Axis
        The axis

    keep_{x,y} : list-like of ints
        The indices of any x-axis ticks to keep. The numbers are passed directly
        as indices to the xticks array.
        
    axis : string in {'both', 'x', 'y'}
        Axis of the tick labels to hide

    Returns
    -------
    None, but the tick labels of the axis are removed, as specified
    """
    
    validation_utils.validate_in_set(axis, _VALID_AXIS_VALUES, "axis")

    if axis in _X_AXIS_VALUES:
        xticks = ax.xaxis.get_major_ticks()
        for xtick in xticks:
            xtick.label1.set_visible(False)

        for x in keep_x:
            xticks[x].label1.set_visible(True)

    if axis in _Y_AXIS_VALUES:
        yticks = ax.yaxis.get_major_ticks()
        for ytick in yticks:
            ytick.label1.set_visible(False)

        for y in keep_y:
            yticks[y].label1.set_visible(True)

def set_ticklabels_fontsize(ax, fontsize, axis='both', which='major'):
    """ Set the font size of the tick labels.

    Parameters
    ----------
    ax: mpl.Axis
        The axis

    fontsize: int, or string mpl recognizes
        The size of the ticklabels

    axis, which: strings
        Values passed to ax.tick_params. Please see the mpl documentation for
        more details.

    Returns
    -------
    None, but the ticklabel fontsizes are updated
    """
    ax.tick_params(axis=axis, which=which, labelsize=fontsize)

VALID_AXIS_VALUES = {'x', 'y', 'both'}
VALID_WHICH_VALUES = {'major', 'minor', 'both'}
def set_ticklabel_rotation(ax, rotation, axis='x', which='both'):
    """ Set the rotation of the tick labels.

    Parameters
    ----------
    ax: mpl.Axis
        The axis

    rotation: int, or a string mpl recognizes
        The rotation of the labels

    axis: 'x', 'y', 'both'
        The axis whose tick labels will be rotated

    which: 'major', 'minor', 'both'
        Which of the tick labels to affect

    Returns
    -------
    None, but the ticklabels are rotated
    """
    if axis not in VALID_AXIS_VALUES:
        msg = "{} is not a valid axis value".format(axis)
        raise ValueError(msg)

    if which not in VALID_WHICH_VALUES:
        msg = "{} is not a valid which value".format(which)
        raise ValueError(msg)

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



def remove_top_and_right_splines(ax):
    """ This function removes the spines on the top and right of the axis.

        Parameters
        ----------
        ax : mpl.Axis
            The axis

        Returns
        -------
        None, but the splines and ticks of the axis are updated
    """

    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def plot_roc_curve(tpr, fpr, auc=None, field_names=None, out=None, cmaps=None, alphas=None, 
    title="Receiver operating characteristic curves", font_size=20, legend_font_size=15, 
    top_adjustment=0.9, xlabel="False positive rate", ylabel="True positive rate"):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors

    fig, ax = plt.subplots()

    if alphas is None:
        alphas = [np.ones(len(tpr[0]))] * len(tpr)

    if cmaps is None:
        cmaps = [plt.cm.Blues] * len(alphas)
    elif len(cmaps) != len(alphas):
        msg = "The ROC curve must have the same number of cmaps as alphas"
        raise ValueError(msg)

    for i in range(len(tpr)):
        l = ""
        if field_names is not None:
            l += field_names[i]

        if auc is not None:
            l += " "
            l += "AUC: {:.2f}".format(auc[i])
                
        color = 'k' # cmap(i/len(tpr))
        for j in range(1, len(fpr[i])):
            points_y = [tpr[i][j-1], tpr[i][j]]
            points_x = [fpr[i][j-1], fpr[i][j]]
            # this plots the lines connecting each point
            ax.plot( points_x, points_y, color=color, zorder=1 )


        ax.scatter(fpr[i], tpr[i], label=l, linewidths=0.1, c=alphas[i], cmap=cmaps[i], zorder=2)

    ax.plot([0,1], [0,1])
    ax.set_aspect('equal')
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))

    ax.legend(loc='lower right', fontsize=legend_font_size)

    if title != None and len(title) > 0:
        fig.suptitle(title, fontsize=font_size)

    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    fig.tight_layout()
    fig.subplots_adjust(top=top_adjustment)

    if out is not None:
        plt.savefig(out, bbox_inches='tight')

def plot_confusion_matrix(
        confusion_matrix,
        ax=None,
        show_cell_labels=True,
        show_colorbar=True,
        title="Confusion matrix", 
        cmap=None, 
        true_tick_labels = None, 
        predicted_tick_labels = None, 
        ylabel="True labels", 
        xlabel="Predicted labels", 
        title_font_size=20, 
        label_font_size=15,
        true_tick_rotation=None,
        predicted_tick_rotation=None,
        out=None):

    """ Plot the given confusion matrix
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # a hack to give cmap a default without importing pyplot for arguments
    if cmap == None:
        cmap = plt.cm.Blues

    mappable = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    
    if show_colorbar:
        fig.colorbar(mappable)
    ax.grid(False)
    
    true_tick_marks = np.arange(confusion_matrix.shape[0])
    ax.set_ylabel(ylabel, fontsize=label_font_size)
    ax.set_yticks(true_tick_marks)

    if true_tick_labels is None:
        true_tick_labels = list(true_tick_marks)

    ax.set_yticklabels(
        true_tick_labels,
        fontsize=label_font_size,
        rotation=true_tick_rotation
    )
    
    predicted_tick_marks = np.arange(confusion_matrix.shape[1])
    ax.set_xlabel(xlabel, fontsize=label_font_size)
    ax.set_xticks(predicted_tick_marks)

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
    
    ax.set_title(title, fontsize=title_font_size)
    fig.tight_layout()

    if out is not None:
        plt.savefig(out, bbox_inches='tight')

def plot_venn_diagram(sets, ax=None, set_labels=None, weighted=False, use_sci_notation=False,
                               labels_fontsize=14, counts_fontsize=12, sci_notation_limit=999):

    """ This function is a wrapper around matplotlib_venn. It most just makes
        setting the fonts and and label formatting a bit easier.

        Args:
            sets: either a dictionary, a list-like of two sets or a list-like of
                three sets. If a dictionary, it must follow the conventions of
                matplotlib_venn. If a dictionary is given, the number of sets
                will be guessed based on the length of a random key.

            ax (mpl.axis): an axis for drawing

            set_labels (list of strings): the label for each set. The order
                of the labels must match the order of the sets

            weighted (bool): whether to draw a weighted or unweighted diagram

            use_sci_notation (bool): whether to convert numbers to scientific
                notation

            sci_notation_limit (float): the maximum number to show before
                switching to scientific notation

            labels_fontsize, counts_fontsize (int): the respective fontsizes

        Returns:
            matplotlib_venn.VennDiagram: the diagram

        Imports:
            matplotlib_venn

    """


    import matplotlib_venn

    
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

def create_simple_bar_chart(ax,
                            bars,
                            labels=None,
                            colors=None, # this will not accept an (rgba list-like specification)
                            xticklabels='default',
                            xticklabels_rotation='vertical',
                            xlabel=None,
                            spacing=0,
                            ymin=None,
                            ymax=None,
                            ylabel=None,
                            use_log_scale=False,
                            hide_first_ytick=True,
                            show_legend=False,
                            title=None,
                            fontsize=12,
                            label_fontsize=12,
                            legend_fontsize=12,
                            title_fontsize=12,
                            tick_offset=0.5
                           ):

    import numpy as np
    import matplotlib.colors
    import matplotlib.pyplot as plt
    import misc.utils as utils
    
    mpl_bars = []

    # first, handle the bars
    
    # TODO: check that the bar arrays are all the same length
    xticks = np.arange(len(bars[0]))
    
    width = 1 - 2*spacing
    width /= len(bars)
    
    # figure out what to do with "colors"
    if colors is None:
        colors = plt.cm.Blues

    if isinstance(colors, matplotlib.colors.Colormap):
        # then use "num_bars" equi-distant colors
        ls = np.linspace(0, 1, len(bars))
        color_vals = [colors(c) for c in ls]
        colors = color_vals
        
    elif utils.is_sequence(colors):
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
        ax.set_xticklabels(xticklabels, fontsize=fontsize, 
                        rotation=xticklabels_rotation)
    else:
        ax.tick_params(axis='x', 
                        which='both', 
                        bottom='off',
                        top='off',
                        labelbottom='off')

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
    
    return ax


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


def create_stacked_bar_graph(
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

    import misc.utils as utils

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
        if not utils.is_sequence(widths):
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
    elif not utils.is_sequence(edge_colors):
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

def plot_simple_scatter(
        x, y,
        ax=None,
        equal_aspect=True,
        set_lim=True,
        show_y_x_line=True,
        xy_line_kwargs={},
        **kwargs):
    """ Plot a simple scatter plot of x vs. y on `ax`
    
    If `fig` and `ax` are not given, then will be created.
    
    See the matplotlib documentation for more keyword arguments and details:
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    
    Parameters
    ----------
    x,y : array-like of numbers
        The values to plot
        
    ax : mpl.Axis
        An axis for plotting. If this is not given, then a figure and axis will
        be created.
        
    equal_aspect : bool
        Whether to set the aspect of the axis to `equal`
        
    set_lim : bool
        Whether to automatically set the min and max axis limits
        
    show_y_x_line : bool
        Whether to draw the y=x line. This will look weird if `set_lim` is False.
        
    xy_line_kwargs : dict
        keyword arguments for plotting the y=x line, if it plotting
        
    **kwargs : <key>=<value> pairs
        Additional keyword arguments to pass to the plot function. Some useful
        keyword arguments are:
        
        * `label` : the label for a legend
        * `marker` : https://matplotlib.org/examples/lines_bars_and_markers/marker_reference.html
        
    Returns
    -------
    fig, ax : mpl.Figure and mpl.Axis
        The figure and axis on which the scatter points were plotted
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
        
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
    
def plot_trend_line(ax, x, intercept, slope, power, **kwargs):
    """ Draw the trend line implied by the given coefficients.

    Parameters
    ----------
    ax : mpl.Axis
        The axis on which the line will be drawn

    x : list of floats
        The points at which the line will be drawn

    intercept, slope, power : floats
        The coefficients of the trend line

    **kwargs : <key>=<value> pairs
        Keyword arguments to pass to the ax.plot function (color, etc.)

    Returns
    -------
    None, but the line will be drawn on the axis
    """
    x = np.sort(x)
    y = power * x ** 2 + slope * x + intercept

    #Plot trendline
    ax.plot(x, y, **kwargs)

def draw_rectangle(ax, base_x, base_y, width, height, center_x=False, 
        center_y=False, **kwargs):
    """ Draw a rectangle at the given x and y coordinates. Optionally, these
    can be adjusted such that they are the respective centers rather than edge
    values.

    Parameters
    ----------
    ax: mpl.Axis
        The axis on which the rectangle will be drawn

    base_{x,y}: number
        The base x and y coordinates

    width, height: number
        The width (change in x) and height (change in y) of the rectangle

    center_{x,y}: bool
        Whether to adjust the x and y coordinates such that they become the
        center rather than lower left. In particular, if center_x is True, then
        base_x will be shifted left by width/2; likewise, if center_y is True,
        then base_y will be shifted down by height/2.

    kwargs: key=value pairs
        Additional keywords are passed to the patches.Rectangle constructor

    base
    """
    y_offset = 0
    if center_y:
        y_offset = height/2
        
    x_offset = 0
    if center_x:
        x_offset = width/2
        
    y = base_y - y_offset
    x = base_x - x_offset
    ax.add_patch(patches.Rectangle((x,y), width, height, **kwargs))   

def plot_sorted_values(values, ymin=None, ymax=None, ax=None, scale_x=False, **kwargs):
    """ Sort `values` and plot them

    Parameters
    ----------
    values : list-like of numbers
        The values to plot

    y{min,max} : floats
        The min and max values for the y-axis. If not given, then these
        default to the minimum and maximum values in the list.
        
    scale_x : bool
        If True, then the `x` values will be equally-spaced between 0 and 1.
        Otherwise, they will be the values 0 to len(values)

    ax : mpl.Axis
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
    fig : mpl.Figure
        The Figure associated with `ax`, or a new Figure

    ax : mpl.Axis
        Either `ax` or a new Axis
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
        
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
