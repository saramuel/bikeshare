import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import sem
from sklearn.model_selection import train_test_split

from bikeshare.data import load_data
from bikeshare.preprocessing import TrendRemover

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams["legend.handlelength"] = 1
mpl.rcParams["legend.labelspacing"] = 0.1


def plot_prediction(model, X_train, y_train, X_test, y_test, plot_output_dir):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # plot correlation between ground truth and prediction
    cmp_train = pd.concat([y_train['cnt'], y_train_pred], axis=1, keys=['true', 'predicted'])
    cmp_test = pd.concat([y_test['cnt'], y_test_pred], axis=1, keys=['true', 'predicted'])
    cmp_train['data set'] = 'training'
    cmp_test['data set'] = 'test'
    cmp_all = pd.concat([cmp_train, cmp_test])
    sns_plot = sns.lmplot(x='true', y='predicted', hue='data set', data=cmp_all, legend=False)
    x_ls = np.linspace(0, 1100, 1000)
    plt.plot(x_ls, x_ls, color='grey')
    lgnd = plt.gca().legend(loc=2, frameon=False)
    for handle in lgnd.legendHandles:
        handle.set_sizes([10])
    sns_plot.savefig(os.path.join(plot_output_dir, f'true_predict_corr.pdf'), bbox_inches='tight')

    # scatter plot training and test set with predictions
    fig, ax = plt.subplots(1, 1)
    y_train['cnt'].plot(ax=ax, style='.', color='black', label='training set')
    y_train_pred.plot(ax=ax, style='.', color='grey', label='pred. training set')
    y_test['cnt'].plot(ax=ax, style='.', color='steelblue', label='test set')
    y_test_pred.plot(ax=ax, style='.', color='darkorange', label='pred. test set')
    ax.set_xlabel("Date")
    ax.set_ylabel("No. of rentals per hour")
    lgnd = ax.legend(frameon=False)
    for handle in lgnd.legendHandles:
        handle._legmarker.set_markersize(10)
    fig.savefig(os.path.join(plot_output_dir, f'train_predict_scatter.pdf'), bbox_inches='tight')


def plot_grouped_usage(best_clf_casual, best_clf_registered, data_dir, plot_output_dir):
    X, y = load_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    y_pred_casual = best_clf_casual.predict(X_test)
    y_pred_registered = best_clf_registered.predict(X_test)
    y_pred = y_test.copy()
    y_pred['registered'] = np.array(y_pred_registered)
    y_pred['casual'] = np.array(y_pred_casual)

    for dataset, y_plots, y_names in zip(['alldata', 'predicttest'], [[y], [y_pred, y_test]],
                                         [['all'], ['prediction', 'test set']]):
        for time_interval in ['H', 'D', 'M']:
            fig, ax = plt.subplots(1, 1)
            for y_plot, y_name in zip(y_plots, y_names):
                for users, c in zip(['registered', 'casual'], ['steelblue', 'crimson']):
                    y_cnt = y_plot[users]
                    if time_interval == 'H':
                        y_grouped = y_cnt.groupby(y_cnt.index.hour)
                        x_grouped = np.unique(y_cnt.index.hour)
                    elif time_interval == 'D':
                        y_grouped = y_cnt.groupby(y_cnt.index.weekday)
                        x_grouped = np.unique(y_cnt.index.weekday)
                    elif time_interval == 'M':
                        y_grouped = y_cnt.groupby(y_cnt.index.month)
                        x_grouped = np.unique(y_cnt.index.month)
                    y_mean = y_grouped.mean()
                    y_sem = y_grouped.aggregate(lambda g: sem(g, axis=None))

                    if y_name == 'prediction':
                        ls = 'dashed'
                        label = f"pred. {users}"
                    else:
                        ls = 'solid'
                        label = users
                    ax.plot(x_grouped, y_mean, lw=2, color=c, linestyle=ls, label=label)
                    ax.fill_between(x_grouped, y_mean-y_sem, y_mean+y_sem, color=c,
                                    lw=0, alpha=0.2, label=None)

                    xticks = list(np.arange(x_grouped.min(), x_grouped.max() + 1, 1))
                    if time_interval == 'H':
                        xticklabels = [f"${xt}$" if i % 2 == 0 else ""
                                       for i, xt in enumerate(xticks)]
                    else:
                        xticklabels = [f"${xt}$" for xt in xticks]
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                    ax.set_xlabel(time_interval)
                    ax.set_ylabel("Mean No. of rentals per hour")
                    ax.legend(frameon=False)
            fig.savefig(os.path.join(plot_output_dir, f'mean_usage_{dataset}_{time_interval}.pdf'),
                        bbox_inches='tight')


def plot_grouped_usage_bias(best_clf_casual, best_clf_registered, data_dir, plot_output_dir):
    X, y = load_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    y_pred_casual = best_clf_casual.predict(X_test)
    y_pred_registered = best_clf_registered.predict(X_test)
    y_pred = y_test.copy()
    y_pred['registered'] = np.array(y_pred_registered)
    y_pred['casual'] = np.array(y_pred_casual)

    for time_interval in ['H', 'D', 'M']:
        fig, ax = plt.subplots(1, 1)
        for user, c in zip(['registered', 'casual'], ['steelblue', 'crimson']):
            dy = y_pred[user] - y_test[user]
            for evl, ls in zip(['bias', 'mad'], ['dashed', 'solid']):
                if evl == 'mad':
                    dy = dy.abs()
                if time_interval == 'H':
                    dy_gr = dy.groupby(dy.index.hour)
                    x_gr = np.unique(dy.index.hour)
                elif time_interval == 'D':
                    dy_gr = dy.groupby(dy.index.weekday)
                    x_gr = np.unique(dy.index.weekday)
                elif time_interval == 'M':
                    dy_gr = dy.groupby(dy.index.month)
                    x_gr = np.unique(dy.index.month)

                dy_mean = dy_gr.mean()
                dy_sem = dy_gr.aggregate(lambda g: sem(g, axis=None))
                ax.plot(x_gr, dy_mean, lw=2, color=c, linestyle=ls, label=f"{evl} {user}")
                ax.fill_between(x_gr, dy_mean-dy_sem, dy_mean+dy_sem,
                                color=c, lw=0, alpha=0.2, label=None)

        xticks = list(np.arange(x_gr.min(), x_gr.max() + 1, 1))
        if time_interval == 'H':
            xticklabels = [f"${xt}$" if i % 2 == 0 else "" for i, xt in enumerate(xticks)]
        else:
            xticklabels = [f"${xt}$" for xt in xticks]
        ax.axhline(y=0, c='k', ls=':', lw=1.5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(time_interval)
        ax.set_ylabel("Mean bias / mad per hour")
        ax.set_ylim(-40, 100)
        ax.legend(loc=2, frameon=False)
        fig.savefig(os.path.join(plot_output_dir, f'bias_mad_{time_interval}.pdf'),
                    bbox_inches='tight')


def plot_trendremover_real(X, y, plot_output_dir):
    trend_remover = TrendRemover(remove_trend=True)
    y_trans = y.copy()
    y_trans['cnt'] = np.array(trend_remover.fit_transform(y['cnt']))
    trend = trend_remover.trend_model_.predict(
        trend_remover._datetime_to_numpy(X.index).reshape(-1, 1)).ravel()

    fig, ax = plt.subplots(1, 1)
    y['cnt'].plot(ax=ax, style='.', color='grey', label='original data')
    y_trans['cnt'].plot(ax=ax, style='.', color='darkorange', label='transformed data')
    ax.plot(X.index, trend, color='black', label='time trend')
    ax.set_xlabel("Date")
    ax.set_ylabel("No. of rentals per hour")
    lgnd = ax.legend(frameon=False)
    for handle in lgnd.legendHandles:
        handle._legmarker.set_markersize(10)
    fig.savefig(os.path.join(plot_output_dir, f'trendremoval.pdf'), bbox_inches='tight')
