# Sage Santomenna 2024, 2025

import itertools
import sys, os
import psutil
from os import makedirs
from os.path import join, exists, abspath, expanduser
import logging
from typing import List, Optional
from datetime import datetime
from pytz import UTC as dtUTC
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import corner
from contextlib import contextmanager
import uuid
import enum
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from argparse import ArgumentParser

import xarray as xr


from .config import Config, config_path

class NodeShape(enum.Enum):
    """Enum for node shapes in the pipeline visualization."""
    CIRCLE = "o"
    SQUARE = "s"
    RECT = "s"
    TRIANGLE = "^"
    DIAMOND = "d"
    PENTAGON = "p"
    HEXAGON = "h"
    
class NodeSpec:
    """Specification for a node in the pipeline visualization."""
    def __init__(self, shape: NodeShape, label: str, color:str, ID:str, edge_weight:float=1.0):
        self.shape = shape
        self.label = label
        self.color = color
        self.ID = ID
        self.edge_weight = edge_weight
    
    def __repr__(self) -> str:
        return str((self.label,self.color,self.ID))
        
        
def visualize_graph(nodes, title:str|None,fig:Figure|None=None,ax:Axes|None=None) -> tuple[Figure, Axes]:
    from networkx.drawing.nx_agraph import graphviz_layout
    import networkx as nx
    
    G = nx.DiGraph()
    def add_nodes_and_edges(node, edges):
        G.add_node(node)
        for edge, sub_edges in edges.items():
            G.add_edge(node, edge)
            if sub_edges:
                add_nodes_and_edges(edge, sub_edges)

    for node, edges in nodes.items():
        add_nodes_and_edges(node, edges)

    for node in G.nodes:
        G.nodes[node]['color'] = node.color
        G.nodes[node]['shape'] = "plaintext"
        label = node.label[0]
        for i in range(1, len(node.label)):
            c = node.label[i]
            if c.isupper() and i < len(node.label) - 1 and node.label[i + 1].islower():
                label += "\n" + c
            else:
                label += c
        node.label= label
        G.nodes[node]['label'] =node.label        
        
    # top_node = list(nodes.keys())[0]
    # initial_positions = {n:(0,-n.depth) for n in G.nodes}
    
    # next_layer = list(nodes[top_node].keys())
    # for i, node in enumerate(next_layer):
    #     initial_positions[node] = (-1, (-0.5+i/len(next_layer))*len(next_layer))
    
    pos = graphviz_layout(G, prog="dot", args="-Grankdir=RL")
    
    if fig is None or ax is None:
        fig, ax = plt.subplots()
        
    nx.draw_networkx_edges(
        G,
        pos=pos,
        ax=ax,
        arrows=True,
        arrowstyle="-",
        node_shape="s",
    )

    tr_figure = ax.transData.transform
    tr_axes = fig.transFigure.inverted().transform

    icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05
    icon_center = icon_size / 2.0

    for n in G.nodes:
        xf, yf = tr_figure(pos[n])
        xa, ya = tr_axes((xf, yf))
        a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        a.text(
            0.5,0.5, G.nodes[n]["label"], fontsize=10, ha='center', va='center', transform=a.transAxes, bbox=dict(facecolor=G.nodes[n]["color"], alpha=1)
        )
        a.axis("off")
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=16)
    return fig, ax


@contextmanager
def suppress_stdout(enabled=True):
    """Context manager to suppress print output."""
    if enabled:
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        if enabled:
            sys.stdout.close()
            sys.stdout = original_stdout

def current_dt_utc():
    return datetime.now(tz=dtUTC)

def file_timestamp():
    return current_dt_utc().strftime("%Y%m%d_%H_%M")

# parse an input path into a full path
def parse_path(path):
    return abspath(expanduser(path))

def savefig(plot_name,config,outdir=None):
    if outdir is None:
        outdir = join(parse_path(config["filestore"]),"plots")
    makedirs(outdir,exist_ok=True)
    plt.savefig(join(outdir,plot_name),bbox_inches="tight",dpi=300)
    
def write_out(*args, level=logging.INFO, logger=None):
    """ Writes out a message to the console and optionally to a logger."""
    msg = " ".join(str(arg) for arg in args)
    if logger is not None:
        logger.log(level, msg)
    else:
        print(msg)

# def write_crash_report(config,report_dir,e,tb=None):
#     BASE_CRASH_DIR = join(parse_path(config['filestore']),config['BASE_CRASH_DIR'])
#     report_dir = join(BASE_CRASH_DIR, report_dir)
#     os.makedirs(report_dir,exist_ok=True)
#     timestamp = datetime.datetime.now(tz=pytz.UTC).strftime('%Y_%m_%d_%H_%M_%S')
#     fname = join(report_dir, f'{timestamp}.txt')
#     write_out(f"CRASH! Writing crash report to {fname}")
    
#     with open(fname, 'w') as f:
#         f.write("Crash report: \n")
#         f.write('Exception: ' + str(e) + '\n')
#         f.write('Traceback: ' + '\n')
#         if tb:
#             traceback.print_tb(tb, file=f)
#         else:
#             traceback.print_tb(sys.exc_info()[2], file=f)

# for saving the best fit parameters from a sampler like dynesty to a dictionary 
def save_bestfit_dict(best_fit_dict, outfile):
    import pickle
    with open(outfile, "wb") as f:
        pickle.dump(best_fit_dict, f)

    return best_fit_dict


# adapted from jerry xuan
def plot_corner(plot_chain, labels_all, baryrv=0, overplot_vals=[], unitless_titles=None,
                overplot_samp=np.array([]), overplot_samp3=np.array([]), overplot_samp4=np.array([]),
                overplot_samp5=np.array([]), overplot_samp6=np.array([]),  overplot_samp7=np.array([]), range_list=None, show_title = True, quantiles=(0.16,0.5,0.84),
                plot_calc_m=False, mode='gpi', weights_list=[], plot_labels=[], fs=27.5, fs2=25, levels = (1-np.exp(-0.5),),
                c_list=['xkcd:cerulean', 'xkcd:tomato', 'xkcd:purple', 'xkcd:goldenrod', 'gray', 'teal', 'green'], max_n_ticks=5, plot_datapoints=False):
    
    if len(overplot_vals) > 0:
        overplot = overplot_vals
    else:
        overplot = None

    # number of samples to overplot
    num_samples = len(plot_labels)
    smooth = True
    fig = corner.corner(plot_chain, range=range_list,
                figsize=(10,10), bins=30, density=True, truths=overplot, truth_color='black',
                quantiles=quantiles, labels=labels_all, show_titles=show_title, titles=unitless_titles,
                smooth=smooth, color=c_list[0], levels=levels, contour_kwargs={'linestyles': 'dashed'},
                    title_kwargs={"fontsize": fs}, label_kwargs=dict(fontsize=fs2), max_n_ticks=max_n_ticks,
                    title_fmt=".2f", plot_datapoints=plot_datapoints)
    
    # Assuming fig is your corner plot figure
    # for ax in fig.get_axes():
    #     # Loop through all texts in each ax
    #     for text in ax.texts:
    #         print('text?')
    #         print(text)
    #         # You can adjust 0.05 to more or less to move the text up or down
    #         text.set_y(text.get_position()[1] + 0.5)
            
    # for control of labelsize of x,y-ticks:
    for ax in fig.get_axes(): 
        ax.tick_params(axis='both', labelsize=fs2)
        ax.xaxis.set_label_coords(0.5, -0.36)  # Adjusts the X-axis label position
        ax.yaxis.set_label_coords(-0.36, 0.5)  # Adjusts the Y-axis label position
    
    # corner.overplot_points(fig, overplot)

    if overplot_samp.shape[0] > 0:
        if len(weights_list) == 0:
            this_weight = None
        else:
            this_weight = weights_list[0]
        corner.corner(overplot_samp, color=c_list[1], smooth=smooth, bins=30, density=True, contour_kwargs={'linestyles': 'dashed'},
                    fig=fig, levels=(1-np.exp(-0.5),), weights=this_weight, range=range_list, plot_datapoints=plot_datapoints)
    if overplot_samp3.shape[0] > 0:
        if len(weights_list) == 0:
            this_weight = None
        else:
            this_weight = weights_list[1]
        corner.corner(overplot_samp3, color=c_list[2], smooth=smooth, bins=30, density=True, contour_kwargs={'linestyles': 'dashed'},
                    fig=fig, levels=(1 - np.exp(-0.5),), weights=this_weight, range=range_list, plot_datapoints=plot_datapoints)

    if overplot_samp4.shape[0] > 0:
        if len(weights_list) == 0:
            this_weight = None
        else:
            this_weight = weights_list[2]
        corner.corner(overplot_samp4, color=c_list[3], bins=30, density=True, contour_kwargs={'linestyles': 'dashed'},
                    alpha=0.8, smooth=smooth, weights=this_weight, range=range_list, plot_datapoints=plot_datapoints,
                    fig=fig, levels=(1 - np.exp(-0.5),))

    if overplot_samp5.shape[0] > 0:
        if len(weights_list) == 0:
            this_weight = None
        else:
            this_weight = weights_list[3]
        corner.corner(overplot_samp5, color=c_list[4], bins=30, density=True, contour_kwargs={'linestyles': 'dashed'},
                    alpha=0.8, smooth=smooth, weights=this_weight, range=range_list,
                    fig=fig, levels=(1 - np.exp(-0.5),), plot_datapoints=plot_datapoints)


    if overplot_samp6.shape[0] > 0:
        if len(weights_list) == 0:
            this_weight = None
        else:
            this_weight = weights_list[4]
        corner.corner(overplot_samp6, color=c_list[5], bins=30, density=True, contour_kwargs={'linestyles': 'dashed'},
                    alpha=0.8, smooth=smooth, weights=this_weight, range=range_list,
                    fig=fig, levels=(1 - np.exp(-0.5),), plot_datapoints=plot_datapoints)

    if overplot_samp7.shape[0] > 0:
        if len(weights_list) == 0:
            this_weight = None
        else:
            this_weight = weights_list[5]
        corner.corner(overplot_samp7, color=c_list[6], bins=30, density=True, contour_kwargs={'linestyles': 'dashed'},
                    alpha=0.8, smooth=smooth, weights=this_weight, range=range_list,
                    fig=fig, levels=(1 - np.exp(-0.5),), plot_datapoints=plot_datapoints)


    if len(plot_labels) > 0:
        all_lines = []
        for s in range(num_samples):
            this_line = mlines.Line2D([], [], color=c_list[s], label=plot_labels[s])
            all_lines.append(this_line)

        if len(labels_all) <= 3:
            fig.legend(handles=all_lines, bbox_to_anchor=(1, 1), loc=1, borderaxespad=0, fontsize=fs2-2.5)
        else:
            fig.legend(handles=all_lines, bbox_to_anchor=(1, 1), loc=1, borderaxespad=0, fontsize=fs2-1)
            
def get_pool(config):
    kwargs = {}
    parallel_cfg = config["parallel"]
    do_mpi = parallel_cfg["mpi"]  # bool
    processes = parallel_cfg["processes"]
    if do_mpi:
        kwargs = {"use_dill":True}
    import schwimmbad
    pool = schwimmbad.choose_pool(mpi=do_mpi,processes=processes, **kwargs)
    if do_mpi:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    return pool

def get_sampler_pool(config):
    sampler = config[config['target']]['sampler_type']
    if sampler=="pymultinest":
        return None
    return get_pool(config)

def get_new_run_id():
    return str(uuid.uuid4())

def archive_config(config: Config):
    checksum = config.checksum()
    archive_dir = parse_path(config["config_archive"])
    outpath = join(archive_dir, f"{checksum}.toml")
    if not exists(outpath):
        makedirs(archive_dir, exist_ok=True)
        config.write(outpath)
    return outpath

def groupby(group_by: List[str], ds:xr.Dataset, flatten=False):
    dims = list(ds.coords)
    variables = list(ds.data_vars)
    other_params = [p for p in variables if p not in group_by]

    # print(f"Grouping dataset by {group_by} with other parameters {other_params}")
    # print(f"Dims: {dims}, Vars: {variables}")

    if not dims:
        subgroup = {p: ds[p].values for p in group_by}
        vals = {o: ds[o].values for o in other_params}
        if flatten:
            coords = [{}]
            yield subgroup, coords, [vals]
            return
        else:
            remaining = ds[other_params]
            yield subgroup, remaining
            return

    flat = ds.stack(flat_index=dims)
    df = flat.to_dataframe()
    grouped = df.groupby(group_by)
    # print("grouped:",grouped)
    for group_vals, group_df in grouped:
        # print("----")
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        # print("group_vals:",group_vals)
        sg = dict(zip(group_by, group_vals))
        # print("sg:",sg)
        # print("group_df:",group_df)
        if flatten:
            coords = []
            vals = []        
            for _, row in group_df.iterrows():
                coords.append(dict(zip(group_df.index.names, row.name)))
                # coords.append({d: row[d] for d in ds.dims})
                vals.append({p: row[p] for p in other_params})
            yield sg, coords, vals
        else:
            if dims:
                index_vals = group_df.index
                subset = flat.sel({"flat_index": index_vals})[other_params]
                if "flat_index" not in subset.dims:  # the remaining dataset is flat
                    yield sg, subset
                else:
                    unstacked = subset.unstack("flat_index")
                    yield sg, unstacked
            else:
                subset = ds.sel({var: val for var, val in zip(group_by, group_vals)})
                yield sg, subset[other_params]


# this wants to be with the hypothetical NDimData class
def broadcast(*args):
    broadcasted = xr.broadcast(*args)    
    all_coords = {c: np.array(v) for c, v in broadcasted[0].coords.variables.items()}
    coord_keys = list(all_coords.keys())
    coord_vals = list(all_coords.values())
    all_sel_values = list(itertools.product(*coord_vals))
    selectors = [dict(zip(coord_keys, s)) for s in all_sel_values]
    return broadcasted, selectors

def joint_iter(*args):
    broadcasted, selectors = broadcast(*args)
    for selector in selectors:
        yield selector, tuple(b.sel(selector).item() for b in broadcasted)
        
def spanning_dims(*args) -> List[str]:
    return list(set([v for x in args for v in list(x.coords.variables)]))


def get_mpi_info():
    """Get the rank of the current process and the size of the MPI pool

    :return: rank, size
    :rtype: int, int
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return rank, size
    
    
def get_process_info():
    """Get the PID of the calling process and its memory usage

    :return: PID, memory (GB)
    :rtype: int, float
    """
    mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
    pid = os.getpid()
    return pid, mem

def compare_evidence_pymn(dir1, dir2):
    from pymultinest.analyse import Analyzer
    import pickle
    from os.path import join, basename
    import glob

    # determine the number of params in each fit
    n_params_1 = np.genfromtxt(join(dir1,"plot_chain.npy")).shape[1]
    n_params_2 = np.genfromtxt(join(dir2,"plot_chain.npy")).shape[1]

    # find the basename of each fit
    summary_file_1 = glob.glob(join(dir1,"*summary.txt"))[0]
    base1 = basename(summary_file_1).replace("summary.txt",'')
    basepath_1 = join(dir1,base1)

    summary_file_2 = glob.glob(join(dir2,"*summary.txt"))[0]
    base2 = basename(summary_file_2).replace("summary.txt",'')
    basepath_2 = join(dir2,base2)

    analyzer1 = Analyzer(n_params=n_params_1, outputfiles_basename=basepath_1)
    analyzer2 = Analyzer(n_params=n_params_2, outputfiles_basename=basepath_2)

    logZ1 = analyzer1.get_stats()['global evidence']
    logZerr1 = analyzer1.get_stats()['global evidence error']

    logZ2 = analyzer2.get_stats()['global evidence']
    logZerr2 = analyzer2.get_stats()['global evidence error']

    logB21 = logZ2 - logZ1
    logB21_err = (logZerr1**2 + logZerr2**2)**0.5
    B21 = np.exp(logB21)
    B21_err = np.exp(logB21_err)

    print(f"logZ1 = {logZ1:.2f} ± {logZerr1:.2f}")
    print(f"logZ2 = {logZ2:.2f} ± {logZerr2:.2f}")
    print(f"log Bayes factor (Model 2 vs 1): {logB21:.2f} ± {logB21_err:.2f}")
    print(f"Bayes factor (Model 2 vs 1): {B21:.2f} ± {B21_err:.2f}")

    if B21 < 1:
        favored, disfavored = 1,2
    else:
        favored, disfavored = 2,1
    print(f"Model {favored} is favored over model {disfavored}.")

    # log likes
    # Load posterior samples with weights and log-likelihoods
    samples1 = analyzer1.get_equal_weighted_posterior()
    samples2 = analyzer2.get_equal_weighted_posterior()

    # The log-likelihood is typically the last column
    loglike1 = samples1[:, -1]
    loglike2 = samples2[:, -1]

    # Get the maximum log-likelihood (best fit)
    max_loglike1 = np.max(loglike1)
    max_loglike2 = np.max(loglike2)

    print(f"Max log-likelihood (Model 1): {max_loglike1:.2f}")
    print(f"Max log-likelihood (Model 2): {max_loglike2:.2f}")
    
    

def compare_evidence(filename1, filename2):
    '''
    Computes evidence of fit2 / evidence of fit1
    If this value is >> 1, fit 2 is statistically favored

    Returns: 
    dz (float): ratio of evidences

    '''
    from dynesty import NestedSampler
    import pickle

    # old dynesty (v 1.1)
    try: 
        file = pickle.load( open(filename1, 'rb') )
        file2 = pickle.load( open(filename2, 'rb') )
        last_logz = file['logz'][-1]

    except Exception as e:
        print('new dynesty live file...')
        file = NestedSampler.restore(filename1).results
        file2 = NestedSampler.restore(filename2).results
    
    # baseline file to compare against
    last_logz = file['logz'][-1]
    last_logz_last100 = np.median(file['logz'][-100:])

    # print(file.keys())
    last_logz2 = file2['logz'][-1]
    last_logz2_last100 = np.median(file2['logz'][-100:])

    print('number of iterations for 1 vs 2:')
    print(file['niter'], file2['niter'])

    print('final loglike for 1 vs 2:')
    print(file['logl'][-1], file2['logl'][-1])

    dz = np.exp(last_logz2 - last_logz)
    print('logz for 1 vs 2: ')
    print(last_logz, last_logz2)
    last_100_dz = np.exp(last_logz2_last100 - last_logz_last100)

    return dz, last_100_dz

def sika_argparser(description:str="Run this model", default_cfg_path:Optional[str]=None):
    if default_cfg_path is None:
        default_cfg_path = config_path
    parser = ArgumentParser(description=description)
    parser.add_argument("run_name", type=str, help="Name of this run")
    parser.add_argument("--config", type=str, default=default_cfg_path, help="Path to config file")
    parser.add_argument(
        "--restore-from",
        type=str,
        default=None,
        help="Path to a sampler save file to resume from",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Path to output directory. will be created if does not exist. If not specified, will be created from model_out/<model_name>/<run_name>_<timestamp>",
    )
    
    return parser

def parse_sika_args(parser:ArgumentParser):
    args = parser.parse_args()
    config_path = parse_path(args.config)
    restore_from = args.restore_from
    run_name = args.run_name
    if restore_from is not None:
        restore_from = parse_path(restore_from)
        
    outdir = args.outdir
    if outdir is not None:
        outdir = parse_path(outdir)
    
    return run_name, config_path, restore_from, outdir, args


def format_selector_string(selector:dict,filename=True):
    s = f"{selector}".replace("{",'').replace("}",'').replace('\'','')
    if filename:
        return s.replace(": ","_")
    return s
