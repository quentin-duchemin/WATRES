from .models import lightning_interface
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import pyreadr
import torch.nn.functional as F
import pickle
import os
import seaborn as sns
from matplotlib.ticker import ScalarFormatter


def get_names_algos(results, algo2name):
    if algo2name is None:
        algo2name = {}
        for site in results.keys():
            for algo in results[site].keys():
                algo2name[algo] = algo
    return algo2name


def show_global_cum_ttd(results, algo2name=None):
    x_abs = np.arange(6900)/(24*30)
    count_fig = 0
    for site, res_site in results.items():
        for algo, value in res_site.items():
            plt.figure(count_fig)
            plt.plot(x_abs, results[site][algo]['global_PQhat'][:6900], label='Prediction')
            plt.plot(x_abs, results[site][algo]['global_PQtrue'][:6900], label='Ground truth')
            plt.xlabel('Age (in Months)', fontsize=13)
            plt.ylabel('Global cumulative TTD', fontsize=14)
            plt.legend()
            plt.title(site+'  '+algo2name[algo], fontsize=14)
            plt.show()
            count_fig += 1



def show_Cout(settings_algos, n_start=0, n_end=-1, algo2name=None):
    algo2name = get_names_algos(settings_algos, algo2name)
    for i in range(len(settings_algos)):
        site = settings_algos[i]['site']
        algo = settings_algos[i]['algo']
        with open(settings_algos[i]['path_results'], "rb") as input_file:
            result = pickle.load(input_file)  
        fig, ax = plt.subplots()
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
        lst_test = result['timeyear_test'][:n_end][n_start:]
        plt.scatter(lst_test, result['Chat'][:n_end][n_start:], c='red')
        plt.plot(lst_test, result['Chat'][:n_end][n_start:],  c='red', linestyle='--', label='Predicted')
        plt.scatter(lst_test, result['Cout'][:n_end][n_start:],  c='orange')
        plt.plot(lst_test, result['Cout'][:n_end][n_start:],  c='orange', linestyle='--', label='Ground truth')
        plt.title(site+' '+algo2name[algo])
        plt.ylabel('Output tracer', fontsize=14)
        plt.legend()
        plt.show()


def gather_results_all_sites(settings_algos):
    results = {}
    for i in range(len(settings_algos)):
        site = settings_algos[i]['site']
        algo = settings_algos[i]['algo']
        if not(site in results):
            results[site] = {}
        with open(settings_algos[i]['path_results'], "rb") as input_file:
            result = pickle.load(input_file)
            results[site][algo] = result
    return results
        
def show_errors(results, algo2name=None, err='ERROR_Cout'):
    # ls_errors = ['ERROR_Cout', 'ERROR_global_PQ']
    algo2name = get_names_algos(results, algo2name)
    errors = [err]
    count_fig = 0
    plt.figure()
    markers = ['*','+', 'x', '.', 'd']
    colors = ['red', 'orange', 'blue', 'brown', 'green']
    mark = {algo:markers[i] for i,algo in enumerate(list(algo2name.keys()))}
    color = {algo:colors[i] for i,algo in enumerate(list(algo2name.keys()))}
    tickslabel = []
    for site, res_site in results.items():
        count_fig += 1
        for algo, value in res_site[site].items():
            if count_fig==1:
                plt.scatter([count_fig], [value[error] for error in errors], label=algo2name[algo], marker=mark[algo], c=color[algo])
            else:
                plt.scatter([count_fig], [value[error] for error in errors], marker=mark[algo], c=color[algo])
        tickslabel.append(site)
    plt.xticks(list(range(1,count_fig+1)), tickslabel, rotation = 90, fontsize=13)
    plt.legend(fontsize=14)
    if errors==['ERROR_Cout']:
        plt.title('Error on the output concentration', fontsize=14)
    else:
        plt.title('Error on the global age distribution', fontsize=14)
    #plt.savefig('/data/quentin/code/final_models/BERT/BERT_multi_MODEL_standard_improved/results_paper/figures/'+err+'.png', dpi=300, bbox_inches="tight")
    plt.show()
    
def show_errors_Cout(settings_algos):
    show_errors(settings_algos, err='ERROR_Cout')
    
def show_errors_global_ttd(settings_algos):
    show_errors(settings_algos, err='ERROR_global_PQ')

def show_violin_ywf(results, algo2name=None):
    algo2name = get_names_algos(results, algo2name)
    idxalgo = 0
    for algo, name_algo in algo2name.items():
        idxalgo += 1
        dataset = []
        lsywf = []
        real_or_esti = []
        tickslabel = []
        plt.figure(idxalgo)
    
        for site, res_site in results.items():
            esti_ywf = res_site[algo]['ywfhat'][:,4]
            dataset += [site]*2*len(esti_ywf)
            lsywf += list(esti_ywf)
            true_ywf = res_site[algo]['ywf_true'][:,4]
            lsywf += list(true_ywf)
            real_or_esti += ['Predicted']*len(esti_ywf)
            real_or_esti += ['Ground truth']*len(esti_ywf)
            tickslabel.append(site)
    
        df = {'dataset': dataset, 'ywf': lsywf, 'real_or_esti':real_or_esti}
    
        sns.violinplot(data=df, x="dataset", y="ywf", hue="real_or_esti", split=True)
        plt.xticks(list(range(0,len(tickslabel))), tickslabel, rotation = 90, fontsize=13)
        plt.legend(fontsize=14,ncol=2)
        plt.title('Model: '+name_algo, fontsize=14)
                #plt.savefig('/data/quentin/code/final_models/BERT/BERT_multi_MODEL_standard_improved/results_paper3_guided/figures/violin_ywf'+algo_ref+'_'+label_concen+'.png', dpi=300, bbox_inches="tight")
        plt.show()



import matplotlib.cm as cm
import colorsys

def get_colors_rainbow(K = 10):
    colors = cm.rainbow(np.linspace(0, 1, K))
    # Reduce brightness by converting RGB to HSV and adjusting Value (V)
    adjusted_colors = []
    for color in colors:
        r, g, b, _ = color
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        v = v * 0.8  # Reduce brightness to 70% of original
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        adjusted_colors.append((r, g, b))
    return adjusted_colors

def show_quantile_ywf_single_figure(results, nb_groups = 10, months_ywf=3, algo2name=None, site2name=None):
    from matplotlib.lines import Line2D
    if site2name is None:
        site2name = list(results.keys())
    count_site = -1
    algo2name = get_names_algos(results, algo2name)
    colors = np.flip(get_colors_rainbow(K=nb_groups))
    markers = ["o", "s", "D", "*", "X", "v", "P"]

    min_val_glob, max_val_glob = 1, 0
    for site, res_site in results.items():
        count_site += 1
        for algo, value in res_site.items():
            Q_test = value['Q_test']
            # Get quantile bins (10 quantiles)
            quantile_bins = np.quantile(Q_test, np.linspace(0, 1, nb_groups+1))
            # Assign each value to a quantile bin
            quantile_indexes = np.digitize(Q_test, quantile_bins, right=True)
            # Create a dictionary to hold the indexes for each group
            grouped_indexes = {i: np.where(quantile_indexes == i)[0] for i in range(1, nb_groups+1)}

            data_pred = []
            data_true = []
            
            for idx_group, idxs in grouped_indexes.items():
                if len(idxs) > 0:
                    data_pred.append(np.mean(value['ywfhat'][:, months_ywf-1][idxs]))
                    data_true.append(np.mean(value['ywf_true'][:, months_ywf-1][idxs]))

            # Find the axis limits based on the data
            all_data = np.concatenate((data_pred, data_true))
            min_val, max_val = np.min(all_data), np.max(all_data)
            max_val_glob = max([max_val_glob, max_val])
            min_val_glob = min([min_val_glob, min_val])

            
            # Plot with different colors for each group and create a legend entry
            for i in range(1, nb_groups+1):
                if i in grouped_indexes and len(grouped_indexes[i]) > 0:
                    plt.scatter(
                        np.mean(value['ywfhat'][:, months_ywf-1][grouped_indexes[i]]),
                        np.mean(value['ywf_true'][:, months_ywf-1][grouped_indexes[i]]),
                        color=colors[i-1],
                        marker=markers[count_site]
                    )
    
                    
    # Set the same limits for x and y axes
    plt.xlim(min_val_glob, max_val_glob)
    plt.ylim(min_val_glob, max_val_glob)
    plt.plot([min_val_glob, max_val_glob], [min_val_glob, max_val_glob], linestyle='--', color='black')
    plt.xlabel('Predicted', fontsize=13)
    plt.ylabel('Ground truth', fontsize=14)
    # Custom legend for colors (Streamflow quantiles)
    color_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i-1], markersize=10)
                    for i in range(1, nb_groups+1)]
    quantile_legend = plt.legend(color_legend, [f'Q{i}' for i in range(1, nb_groups+1)], 
                                 title="Streamflow quantiles",
                                 loc=5, bbox_to_anchor=(0.93, 0.2), ncol=2)
    
    # Add the color legend to the plot
    plt.gca().add_artist(quantile_legend)

    # Custom legend for markers (Sites)
    marker_legend = [Line2D([0], [0], marker=markers[i], color='w', markerfacecolor='grey', markersize=10)
                     for i in range(len(site2name))]
    plt.legend(marker_legend, site2name, title="Sites", loc=1, bbox_to_anchor=(0.8, 1.1), ncol=2)

    plt.tight_layout()
    plt.show()


def show_quantile_ywf(results, nb_groups = 10, months_ywf=3, algo2name=None, site2name=None):
    if site2name is None:
        site2name = list(results.keys())
    count = 0
    algo2name = get_names_algos(results, algo2name)
    colors = np.flip(get_colors_rainbow(K=nb_groups))
    markers = ["o", "s", "D", "*", "X", "+", "x", "^", "v", "P"]

    for site, res_site in results.items():
        for algo, value in res_site.items():
            Q_test = value['Q_test']
            # Get quantile bins (10 quantiles)
            quantile_bins = np.quantile(Q_test, np.linspace(0, 1, nb_groups+1))
            # Assign each value to a quantile bin
            quantile_indexes = np.digitize(Q_test, quantile_bins, right=True)
            # Create a dictionary to hold the indexes for each group
            grouped_indexes = {i: np.where(quantile_indexes == i)[0] for i in range(1, nb_groups+1)}

            data_pred = []
            data_true = []
            
            for idx_group, idxs in grouped_indexes.items():
                if len(idxs) > 0:
                    data_pred.append(np.mean(value['ywfhat'][:, months_ywf-1][idxs]))
                    data_true.append(np.mean(value['ywf_true'][:, months_ywf-1][idxs]))

            # Find the axis limits based on the data
            all_data = np.concatenate((data_pred, data_true))
            min_val, max_val = np.min(all_data), np.max(all_data)

            
            plt.figure(count)
            # Plot with different colors for each group and create a legend entry
            for i in range(1, nb_groups+1):
                if i in grouped_indexes and len(grouped_indexes[i]) > 0:
                    plt.scatter(
                        np.mean(value['ywfhat'][:, months_ywf-1][grouped_indexes[i]]),
                        np.mean(value['ywf_true'][:, months_ywf-1][grouped_indexes[i]]),
                        color=colors[i-1],
                        marker=markers[i-1],
                        label=f'{i}'
                    )
            # Set the same limits for x and y axes
            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)
            plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black')
            plt.xlabel('Predicted', fontsize=13)
            plt.ylabel('Ground truth', fontsize=14)
            plt.legend(title='Streamflow quantiles')
            plt.title(f'{site2name[site]}', fontsize=14)
            plt.show()
            count += 1
    

def show_TS_ywf(results, algo2name=None):
    count = 0
    algo2name = get_names_algos(results, algo2name)
    for site, res_site in results.items():
        for algo, value in res_site.items():
            plt.figure(count)
            plt.plot(results[site][algo]['timeyear_test'], results[site][algo]['ywf_true'][:,2], label="Ground truth")
            plt.plot(results[site][algo]['timeyear_test'], results[site][algo]['ywfhat'][:,2], linestyle='--', label="Predicted")
            plt.xlabel('Date', fontsize=13)
            plt.ylabel('Young Water Fraction (3 months)', fontsize=14)
            plt.legend()
            plt.title(f'{site} | Model: {algo2name[algo]}', fontsize=14)
            plt.show()
            count += 1


def show_ywf(results, algo2name=None, site2name=None):
    size_marker = 100
    algo2name = get_names_algos(results, algo2name)
    if site2name is None:
        site2name = list(results.keys())
    markers = ['*','+', 'x', '.', 'd','|']
    colors = ['red', 'orange', 'blue', 'brown', 'green', 'purple']
    mark = {algo:markers[i] for i,algo in enumerate(list(algo2name.keys()))}
    color = {algo:colors[i] for i,algo in enumerate(list(algo2name.keys()))}
    tickslabel = []
    count_fig = 0
    for site, res_site in results.items():
        count_fig += 1
        count_algo = 0
        for algo, value in res_site.items():
            mean_ywf = np.mean(res_site[algo]['ywfhat'][:, 2])
            if count_fig==1:
                plt.scatter([count_fig], mean_ywf, label=algo2name[algo], marker=mark[algo], c=color[algo], s=size_marker, alpha=0.6)
            else:
                plt.scatter([count_fig], mean_ywf, marker=mark[algo], c=color[algo], s=size_marker, alpha=0.6)
            count_algo += 1
            if count_algo==1:
                mean_ywf_true = np.mean(res_site[algo]['ywf_true'][:,2])
                if count_fig==1:
                    plt.scatter([count_fig], mean_ywf_true, label='Ground truth', marker='_', c='black', s=size_marker)
                else:
                    plt.scatter([count_fig], mean_ywf_true, marker='_', c='black', s=size_marker)

        tickslabel.append(site2name[site])
        

    plt.xticks(list(range(1,count_fig+1)), tickslabel, rotation = 90, fontsize=13)
    plt.legend(
        fontsize=12,
        loc='center',         # Position legend to the left center
        bbox_to_anchor=(0.5, 1.12),   # Move it outside the plot, centered vertically
        ncol=2                     # Set the number of columns
    )

    plt.ylabel('Young water fraction', fontsize=14)
    #plt.savefig('/data/quentin/code/final_models/BERT/BERT_multi_MODEL_standard_improved/results_paper3_guided/figures/ywf_values'+'_'+label_concen+'.png', dpi=300, bbox_inches="tight")
    plt.show()
