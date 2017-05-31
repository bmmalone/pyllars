#! /usr/bin/env python3

import matplotlib
matplotlib.use('agg')

import argparse

import numpy as np
import pandas as pd
import sklearn.metrics

import matplotlib.pyplot as plt

import misc.utils as utils

default_prediction_field = 'bf'

default_n_gtp_field = 'n_gtp'
default_n_gtn_field = 'n_gtn'
default_tp_field = 'tp'
default_fp_field = 'fp'

default_title = "Receiver operating characteristic curve"
default_label = 'Predictions'

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This script plots an ROC curve based on predictions. In particular, "
        "each row in a data frame is taken to be a prediction. They are sorted by a given "
        "field (such as a Bayes' factor or p-value). Additionally, each prediction must "
        "already include the number of true positives and false positives. Furthermore, the "
        "data frame must include columns which collectively give the total number of ground "
        "truth positives and negatives.")

    parser.add_argument('predictions', help="The predictions data frame")
    parser.add_argument('out', help="The output (image) file")
    
    parser.add_argument('-l', '--label', help="The label that will be used for the "
        "series in the plot", default=default_label)
    parser.add_argument('-t', '--title', help="The title of the plot", 
        default=default_title)
    
    parser.add_argument('--prediction-field', help="The field by which the predictions will "
        "be sorted. Presumably, this gives some notion of the confidence of the predictions "
        "(e.g., Bayes' factors).", default=default_prediction_field)

    parser.add_argument('--use-predictions-as-alphas', help="If this flag is present, then "
        "the prediction values will be linearly scaled in the range (0,1). Those values "
        "will be used as alpha values in the plot.", action='store_true')

    parser.add_argument('--n-gtp-field', help="The field whose sum gives the total number "
        "of ground truth positives in the dataset", default=default_n_gtp_field)
    parser.add_argument('--n-gtn-field', help="The field whose sum gives the total number "
        "of ground truth negatives in the dataset", default=default_n_gtn_field)
    parser.add_argument('--tp-field', help="The field which gives the number of true "
        "positives associated with each prediction", default=default_tp_field)
    parser.add_argument('--fp-field', help="The field which gives the number of false "
        "positives associated with each prediction", default=default_fp_field)
    args = parser.parse_args()

    res = pd.read_csv(args.predictions)
    res_sorted = res.sort_values(args.prediction_field, ascending=False)

    # we MUST count the ground truth numbers before filtering out any of the ORFs
    gtp = res_sorted[args.n_gtp_field].sum()
    gtn = res_sorted[args.n_gtn_field].sum()

    tp = res_sorted[args.tp_field].cumsum()
    fp = res_sorted[args.fp_field].cumsum()

    tpr = tp / gtp
    fpr = fp / gtn

    # add dummy "all predictions" to the end
    tpr = tpr.append(pd.Series([1]))
    fpr = fpr.append(pd.Series([1]))

    tpr = np.array(tpr)
    fpr = np.array(fpr)

    auc = sklearn.metrics.auc(fpr, tpr)

    alphas = None
    if args.use_predictions_as_alphas:
        alphas = res_sorted[args.prediction_field]
        #alphas = (alphas - alphas.mean()) / alphas.std()
        #alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())

        alphas = alphas.append(pd.Series([1]))
        alphas = np.array(alphas)

    utils.plot_roc_curve([tpr], [fpr], [auc], [args.label], args.out, cmaps=[plt.cm.Blues], 
        alphas=[alphas], title=args.title)

if __name__ == '__main__':
    main()

