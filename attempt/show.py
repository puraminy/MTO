# v 10
import scipy.stats as stats
# get ANOVA table as R like output
import statsmodels.api as sm
from statsmodels.formula.api import ols
# from pandas.table.plotting import table # EDIT: see deprecation warnings below
from pandas.plotting import table
# import dataframe_image as dfi
# from metrics.metrics import do_score
#import pyperclip
#from metrics.metrics import do_score

from distutils.dir_util import copy_tree, remove_tree
import subprocess
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib
from curses import wrapper
from tabulate import tabulate
import click
import warnings
import itertools
import numpy as np
import statistics as stat
from glob import glob
import six
import debugpy
import os, shutil
import re
import seaborn as sns
from pathlib import Path
import pandas as pd
from attempt.win import *
from mylogs import * 
from datetime import datetime, timedelta
import time
import json
from tqdm import tqdm
# from comet.utils.myutils import *
#from attempt.utils.utils import combine_x,combine_y,add_margin
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageChops
#import sklearn
#import sklearn.metrics
#import attempt.metrics.metrics as mets

def trim_white_borders(image):
    # Convert the image to RGB (if not already in that mode)
    image = image.convert('RGB')

    # Create a new image with the same size and a white background
    bg = Image.new('RGB', image.size, (255, 255, 255))

    # Find the difference between the image and the white background
    diff = ImageChops.difference(image, bg)
    
    # Find the bounding box of the non-zero regions in the difference image
    bbox = diff.getbbox()
    
    if bbox:
        # Crop the image to the bounding box
        cropped_image = image.crop(bbox)
        # Save the cropped image
        return cropped_image


def combine_x(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height),(255, 255, 255))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    return new_im

def combine_y(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = max(widths)
    max_height = sum(heights)

    new_im = Image.new('RGB', (total_width, max_height),(255, 255, 255) )

    y_offset = 0
    for im in images:
      new_im.paste(im, (0, y_offset))
      y_offset += im.size[1]

    return new_im

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result



warnings.simplefilter(action='ignore', category=Warning)
file_id = "name"

table_mid_template = """
    \\begin{{table*}}
        \centering
        \\begin{{tabular}}{{{}}}
        \hline
        {}
        \end{{tabular}}
        \caption{{{}}}
        \label{{{}}}
    \end{{table*}}
    """
table_fp_template = """
    \\begin{{table*}}
        \\begin{{adjustbox}}{{width=1\\textwidth}}
        \\begin{{tabular}}{{{}}}
        \hline
        {}
        \end{{tabular}}
        \caption{{{}}}
        \label{{{}}}
        \end{{adjustbox}}
    \end{{table*}}
    """
table_env_template = """
    \\begin{{table*}}[h!]
        minipage
        \caption{{{}}}
        \label{{{}}}
    \end{{table*}}
    """
table_hm_template = """\\begin{{minipage}}{{.4\\linewidth}}
    \centering
    \label{{{}}}
    \pgfplotstabletypeset[
    color cells={{min={},max={}}},
    col sep=&,	% specify the column separation character
    row sep=\\\\,	% specify the row separation character
    columns/N/.style={{reset styles,string type}},
    /pgfplots/colormap={{whiteblue}}{{rgb255(0cm)=(255,255,255); rgb255(1cm)=(0,200,200)}},
    ]{{{}}}
    \end{{minipage}}"""
def latex_table(rep, rname, mdf, all_exps, sel_col, category, caption=""):
    maxval = {}
    for ii, exp in enumerate(all_exps): 
        exp = exp.replace("_","-")
        exp = _exp = exp.split("-")[0]
        if not sel_col in rep:
            continue
        if not exp in rep[sel_col]:
            continue
        for rel in mdf['prefix'].unique(): 
            if not rel in rep[sel_col][exp]:
                continue
            val = rep[sel_col][exp][rel]
            if type(val) == list:
                assert val, rel + "|"+ sel_col + "|"+ exp
                val = stat.mean(val)
            if not rel in maxval or val > maxval[rel]:
                maxval[rel] = val

    table_cont2=""
    table_cont2 += "method & "
    head2 = "|r|"
    for rel in mdf['prefix'].unique(): 
        table_cont2 += "\\textbf{" + rel + "} &"
        head2 += "r|"
    table_cont2 += " avg. " 
    head2 += "r|"
    table_cont2 = table_cont2.strip("&")
    table_cont2 += "\\\\\n"
    table_cont2 += "\\hline\n"
    for ii, exp in enumerate(all_exps): 
        exp = exp.replace("_","-")
        exp = _exp = exp.split("-")[0]
        if not sel_col in rep:
            continue
        if not exp in rep[sel_col]:
            continue

        table_cont2 += " \hyperref[fig:" + category + _exp + "]{" + _exp + "} &"
        for rel in mdf['prefix'].unique(): 
            if not rel in rep[sel_col][exp]:
                continue
            val = rep[sel_col][exp][rel]
            if type(val) == list:
                val = stat.mean(val)
            if val == maxval[rel]:
                table_cont2 += "\\textcolor{teal}{" +  f" $ {val:.1f} $ " + "} &"
            else:
                table_cont2 += f" $ {val:.1f} $ &"
        if "avg" in rep[sel_col][exp]:
            avg = rep[sel_col][exp]["avg"]
            if type(avg) == list and avg:
                avg = stat.mean(avg)
            if avg:
                avg = "{:.1f}".format(avg)
            table_cont2 += f" $ \\textcolor{{blue}}{{{avg}}} $ &"
        table_cont2 = table_cont2.strip("&")
        table_cont2 += "\\\\\n"
    table_cont2 += "\\hline \n"
    for head, cont in zip([head2],
            [table_cont2]):
        label = "table:" + rname + sel_col.replace("_","-") 
        capt = caption
        if not capt:
           capt = category + " \hyperref[table:show]{ Main Table } | " + label
        table = """
            \\begin{{table*}}[h!]
                \centering
                \\begin{{tabular}}{{{}}}
                \hline
                {}
                \end{{tabular}}
                \caption{{{}}}
                \label{{{}}}
            \end{{table*}}
            """
        table = table.format(head, cont, capt, label)
    return table


def calculate_precision_recall(row):
    TP = row  # True Positives (matches with corresponding target column)
    FP = row.sum() - TP  # False Positives (sum of all matches - TP)
    FN = ct.sum(axis=1)[row.name] - TP  # False Negatives (sum of matches for corresponding prediction - TP)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    return pd.Series({'precision': precision, 'recall': recall})

def recalc_rouge2(df, spath):
   def match_text(row):
      return 1. if str(row['vpred']) == str(row['target_text']) else 0. 
   df["rouge_score"] = df.apply(match_text, axis=1)
   df.to_csv(spath, index=False, sep="\t")
   return df["rouge_score"].mean()

def recalc_rouge3(df, spath):
   def match_text(row):
      return 1. if str(row['pred_text1']) == str(row['target_text']) else 0. 
   df["rouge_score"] = df.apply(match_text, axis=1)
   df.to_csv(spath, index=False, sep="\t")
   return df["rouge_score"].mean()



def recalc_rouge(df, spath):
   def match_text(row):
      cleaned_vpred = remove_choice(str(row['vpred']))
      return 1. if cleaned_vpred in str(row['vtarget']) else 0. 
   df["rouge_score"] = df.apply(match_text, axis=1)
   df.to_csv(spath, index=False, sep="\t")
   return df["rouge_score"].mean()

def preserve_choices(text):
    pattern = r'choice\d+'
    matches = re.findall(pattern, text)
    preserved_text = ' '.join(matches)
    return preserved_text.strip() 

def remove_choice(text):
    # Regular expression to match 'Choice' followed by any number of digits
    pattern = re.compile(r'\bchoice\d*\b', re.IGNORECASE)
    # Substitute the matched patterns with an empty string
    cleaned_text = pattern.sub('', text)
    # Remove any extra whitespace that may result from the removal
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def recalc_v3(df, spath):
   def match_text(row):
       cleaned_vpred = remove_choice(str(row['vpred']))
       cleaned_vtarget = remove_choice(str(row['vtarget']))
       cond = cleaned_vtarget in cleaned_vpred and len(cleaned_vpred) < len(cleaned_vtarget) + 10
       return 1. if cond else 0.

   def match_text1_tt(row):
        pred_choice = preserve_choices(row["pred_text1"])
        cleaned_vpred = remove_choice(str(row['vpred']))
        cleaned_vtarget = remove_choice(str(row['vtarget']))
        match_choice = pred_choice == row['target_text'].strip()
        match_text = cleaned_vpred.strip() != '' and cleaned_vpred in cleaned_vtarget
        match_text = match_text and len(cleaned_vpred) < len(cleaned_vtarget) + 10
        other_choice_text = not match_text and cleaned_vpred in row["input_text"]
        match_choice = match_choice and not other_choice_text
        return row['target_text'] if (match_choice or matche_text) else 'mistake'


   def match_text1(row):
       pred_choice = preserve_choices(row["pred_text1"])
       cleaned_vpred = remove_choice(str(row['vpred']))
       cleaned_vtarget = remove_choice(str(row['vtarget']))
       cond1 = pred_choice == row['target_text'].strip()
       if len(cleaned_vpred.split()) > 2:
           cond2 = cleaned_vpred in cleaned_vtarget
       else:
           cond2 = cleaned_vtarget in cleaned_vpred
       cond3 = not cond2 and cleaned_vpred in row["input_text"]
       return 1. if cond1 and not cond3 else 0.

   def match_text2(row):
       pred_choice = preserve_choices(row["pred_text1"])
       return pred_choice

   if "MaskedChoicePrompting" in df["template"].iloc[0]:
       df["rouge_score"] = df.apply(match_text1, axis=1)
   else:
       df["rouge_score"] = df.apply(match_text, axis=1)
   df.to_csv(spath, index=False, sep="\t")
   return df["rouge_score"].mean()

def vpredin(df, spath):
    def match_text(row):
        # Remove ChoiceN from vpred and input_text
        cleaned_vpred = remove_choice(str(row['vpred']))
        cleaned_input_text = remove_choice(str(row['input_text']))
        return 0.0 if cleaned_vpred in cleaned_input_text else 1.0

    df["vpredin_score"] = df.apply(match_text, axis=1) * 100
    df.to_csv(spath, index=False, sep="\t")
    mean = df["vpredin_score"].mean()
    return mean

function_map = {
        "recalc_rouge": recalc_rouge,
        "recalc_rouge2": recalc_rouge2,
        "recalc_rouge3": recalc_rouge3,
        "recalc_v3": recalc_v3,
        "vpredin": vpredin,
        }

def create_label2(row):
    label = 'PT'
    if not "compose_method" in row:
        return label
    elif row['compose_method'] == "mcat":
        label = 'MCAT'
        #nsp = int(row["num_source_prompts"])
        #numt = int(row["num_prompt_tokens"])
        #if numt == nsp*15:
        #    label += "15"
    elif row['compose_method'] == "mwavg":
        label = 'MSUM'
    elif row['compose_method'] == "wavg":
        if "use_source_prompts" in row and  row["use_source_prompts"]:
            label = 'SSUM'
        else:
            label = 'MPT'
    return label

def create_label(row):
    label = ''
    if row['add_target']:
        label += 'A'
    if not "use_source_prompts" in row or row['use_source_prompts']:
        label += 'S'
        if row['load_source_prompts']:
            label += 'I'
        if row['learn_source_prompts']:
            label += 'L'
    if row['use_private_prompts']:
        label += 'P'
        if row['load_private_prompts']:
            label += 'I'
    return label

def save_df_as_image(df, path):
    # Set background to white
    norm = matplotlib.colors.Normalize(-1,1)
    colors = [[norm(-1.0), "white"],
            [norm( 1.0), "white"]]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    # Make plot
    plot = sns.heatmap(df, annot=True, cmap=cmap, cbar=False)
    fig = plot.get_figure()
    fig.savefig(path)

def correct_path(s, d, path):
    with open(s, 'r') as f1:
        _dict = json.load(f1)
    _dict["output_dir"] = path 
    _dict["save_path"] = str(Path(path).parent)
    with open(d, 'w') as f2:
        json.dump(_dict, f2, indent=3)


def plot_bar(rep, folder, sel_col):
    methods = list(rep[sel_col + "@m_score"].keys()) 
    bar_width = 0.25
    r = np.arange(9)
    ii = -1
    color = ["red","green","blue"]
    for key in [train_num + "@m_score","500@m_score"]: 
        ii += 1
        column = [float(rep[key][met]) for met in methods]
        r = [x + bar_width for x in r]
        plt.bar(r, column, color=color[ii], width=bar_width, edgecolor='white', label=key)

# Add xticks on the middle of the group bars
    plt.xlabel('Methods', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(column))], methods)

# Add y axis label and title
    plt.ylabel('Performance', fontweight='bold')
    plt.title('Performance of Methods')

# Add legend and show the plot
    plt.legend()
    pname = os.path.join(folder, "bar.png")
    plt.savefig(pname)
    return pname

def score_colors(df,row,col, default=None):
    try:
        df[col] = df[col].astype(float)
    except:
        return default
    max_val = df[col].max()
    if df.iloc[row][col] == max_val:
        return 136
    return 247

def pivot_colors(df,row,col, default=None):
    rel_col = "n-" + col
    try:
        df[col] = df[col].astype(float)
    except:
        return default
    max_val = df[col].max()
    if rel_col in df:
        if df.iloc[row][rel_col] <= 1:
            return 91
        if df.iloc[row][col] == max_val:
            return 136
    return default if default is not None else TEXT_COLOR 

def index_colors(df,row,col, default=None):
    index = df.iloc[row][col]
    return index

def pred_colors(df,row,col, default=None):
    pred = df.iloc[row][col]
    pred = str(pred)
    target = df.iloc[row]["target_text"]
    terget = str(target)
    vpred = df.iloc[row]["vpred"] if "vpred" in df.iloc[row] else ""
    vtarget = df.iloc[row]["vtarget"] if "vtarget" in df.iloc[row] else ""
    vpred = str(vpred)
    vtarget= str(vtarget)
    if pred in target:
        if vtarget in vpred:
            return WARNING_COLOR
        else:
            return MSG_COLOR
    return 81

def cat_colors(df,row,col, default=None):
    cat = df.iloc[row][col]
    cat = str(cat)
    if "research" in cat or "question" in cat:
        return WARNING_COLOR
    elif "done" in cat:
        return MSG_COLOR
    return default

def time_colors(df,row,col, default=None):
    rel_col = "time" 
    last_hour = datetime.now() - timedelta(hours = 1)
    last_10_minutes = datetime.now() - timedelta(minutes = 10)
    if rel_col in df:
        time = str(datetime.now().year) + "-" + df.iloc[row][rel_col]
        try:
            time = datetime.strptime(time, '%Y-%m-%d-%H:%M')
            if time > last_10_minutes:
                return WARNING_COLOR
            elif time > last_hour:
                return 81
        except:
            pass
    return default if default is not None else TEXT_COLOR 


def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    sd = superitems(data)
    fname = Path(path).stem
    if fname == "results":
        main_df = pd.DataFrame(sd, columns=["exp","model","lang", "wrap","frozen","epochs","stype", "date", "dir", "score"])
    else:
        main_df = pd.DataFrame(sd, columns=["tid","exp","model","lang", "wrap","frozen","epochs","date", "field", "text"])

    out = f"{fname}.tsv"
    df = main_df.pivot(index=list(main_df.columns[~main_df.columns.isin(['field', 'text'])]), columns='field').reset_index()

    #df.columns = list(map("".join, df.columns))
    df.columns = [('_'.join(str(s).strip() for s in col if s)).replace("text_","") for col in df.columns]
    df.to_csv(path.replace("json", "tsv"), sep="\t", index = False)
    return df

def remove_uniques(df, sel_cols, tag_cols = [], keep_cols = []):
    _info_cols = []
    _tag_cols = tag_cols
    _sel_cols = []
    _df = df.nunique()
    items = {k:c for k,c in _df.items()}
    df.columns = df.columns.get_level_values(0)
    for c in sel_cols:
        if not c in items:
            continue
        _count = items[c]
        if c in keep_cols:
            _sel_cols.append(c)
        elif _count > 1: 
           _sel_cols.append(c)
        else:
           _info_cols.append(c) 
    if _sel_cols:
        for _col in tag_cols:
            if not _col in _sel_cols:
                _sel_cols.append(_col)

    _sel_cols = list(dict.fromkeys(_sel_cols))
    return _sel_cols, _info_cols, tag_cols

def list_dfs(df, main_df, s_rows, FID):
    dfs_items = [] 
    dfs = []
    ii = 0
    dfs_val = {}
    for s_row in s_rows:
        exp=df.iloc[s_row]["fid"]
        prefix=df.iloc[s_row]["prefix"]
        dfs_val["exp" + str(ii)] = exp
        mlog.info("%s == %s", FID, exp)
        cond = f"(main_df['{FID}'] == '{exp}') & (main_df['prefix'] == '{prefix}')"
        tdf = main_df[(main_df[FID] == exp) & (main_df['prefix'] == prefix)]
        tdf = tdf[["pred_text1", "exp_name", "id","hscore", "out_score", "bert_score","query", "resp", "resp_template", "template", "rouge_score", "fid","prefix", "input_text","target_text", "sel"]]
        tdf = tdf.sort_values(by="rouge_score", ascending=False)
        sort = "rouge_score"
        dfs.append(tdf)
    return dfs


def get_main_vars(df):
    main_vars = []
    if "main_vars" in df:
        mvars = []
        for var in df["main_vars"].unique():
            dvar = json.loads(var)
            lvar = list(dvar.keys())
            mvars.extend([e for e in lvar 
                    if not e in ['max_train_samples', 'source_prompts',
                        'task_name', 'num_train_epochs']])
        main_vars = list(dict.fromkeys(mvars))
    return main_vars

def summarize(df, rep_cols=None, score_col=None, rename =True):
    mdf = df #main_df
    pivot_cols = ["prefix"]
    if not rep_cols:
        file_dir = Path(__file__).parent
        with open(os.path.join(file_dir, 'cols.json'),'r') as f:
            all_cols = json.load(f)
        if 'sel_cols' in all_cols:
            sel_cols = all_cols['sel_cols'] 
            rep_cols = all_cols['rep_cols'] if "rep_cols" in all_cols else sel_cols
            extra_cols = all_cols['extra_cols'] if "extra_cols" in all_cols else []
            exclude_cols = all_cols['ex_cols'] if "ex_cols" in all_cols else []
        if "compose_method" in df:
            rep_cols = rep_cols + extra_cols
        if not "compose_method" in df:
            df["compose_method"] = "PT"

    rels = df["prefix"].unique()
    if score_col is not None: 
        score_cols = [score_col,'num_preds'] 
    else:
        #score_cols = ['m_score', 'num_preds'] 
        score_cols = ['rouge_score', 'num_preds'] 

    main_vars = get_main_vars(df)
    rep_cols = rep_cols + score_cols + main_vars + sel_cols 
    rep_cols = list(dict.fromkeys(rep_cols))
    _agg = {}
    _rep_cols = []
    for c in rep_cols: 
        if not c in mdf:
            continue
        if c in score_cols: # or c in ["d_seed", "max_train_samples"]: 
            _agg[c] = "mean"
        elif c.endswith("score"): 
            score_cols.append(c)
            _agg[c] = "mean"
        elif c not in pivot_cols:
            _rep_cols.append(c)
            _agg[c] = "first"
    gcol = _rep_cols
    if not "eid" in rep_cols:
        gcol += ["eid"] 
    #gcol.remove("eid")
    #gcol.remove("folder")
    mdf[gcol] = mdf[gcol].fillna('none')
    pdf = mdf.pivot_table(index=gcol, columns=pivot_cols, 
            values=score_cols, aggfunc='mean', margins=True)
    columns = pdf.columns.to_flat_index()
    # pdf["avg"] = pdf.mean(axis=1, skipna=True)
    #pdf['fid'] = mdf.groupby(gcol)['fid'].first()
    # pdf['eid'] = mdf.groupby(gcol)['eid'].first()
    #pdf['cat'] = mdf.groupby(gcol)['cat'].first()
    pdf.reset_index(inplace=True)
    pdf.columns = [col[1] if col[0] == score_cols[0] and rename 
            else col[0][0:2] + "-" + col[1] if col[0] in score_cols else col[0]
            for col in pdf.columns]
    # pdf['cat'] = pdf['cat'].apply(lambda x: x.split('-')[0]) 
    # if not "label" in pdf:
    #    pdf['label'] = pdf.apply(create_label, axis=1)
    #pdf['ref'] = pdf.apply(
    #        lambda row: f" \\ref{{{'fig:' + str(row['eid'])}}}", axis=1)
    pdf = pdf.round(2)
    df = pdf.iloc[:-1]
    return df

def add_scores(df):
    score_col = "rouge_score"
    # backit(df, sel_cols)
    if "depth_score" in df:
        df["depth_min_score"] = df.groupby(['fid','prefix','input_text'])["depth_score"].transform("min")
    # df["perp_min_score"] = df.groupby(['fid','prefix','input_text'])["perp_score"].transform("min")
    df["rouge_score"] = df.groupby(['fid','prefix','input_text'])["rouge_score"].transform("max")
    df["bert_score"] = df.groupby(['fid','prefix','input_text'])["bert_score"].transform("max")
    df["hscore"] = df.groupby(['fid','prefix','input_text'])["hscore"].transform("max")
    df["pred_freq"] = df.groupby(['fid','prefix','pred_text1'],
                     sort=False)["pred_text1"].transform("count")
    cols = ['fid', 'prefix']
    tdf = df.groupby(["fid","input_text","prefix"]).first().reset_index()
    df = df.merge(tdf[cols+['pred_text1']]
         .value_counts().groupby(cols).head(1)
         .reset_index(name='pred_max_num').rename(columns={'pred_text1': 'pred_max'})
       )

    #temp = (pd
    #       .get_dummies(df, columns = ['pred_text1'], prefix="",prefix_sep="")
    #       .groupby(['fid','prefix'])
    #       .transform('sum'))
    #df = (df
    #.assign(pred_max_num=temp.max(1), pred_max = temp.idxmax(1))
    #)
    return df


def grouping(df, FID='fid'):
    col = [FID, "prefix"]
    _agg = {}
    for c in df.columns:
        if c.endswith("score"):
            _agg[c] = ['mean', 'std']  # Aggregate both mean and standard deviation
        else:
            _agg[c] = ["first", "nunique"]
    #df = df.groupby(col).agg(_agg).reset_index(drop=True)
    gb = df.groupby(col)
    counts = gb.size().to_frame(name='group_records')
    counts.columns = counts.columns.to_flat_index()
    gbdf = gb.agg(_agg)
    gbdf.columns = gbdf.columns.to_flat_index()
    df = (counts.join(gbdf))
    df = df.reset_index(drop=True)
    scols = [c for c in df.columns if type(c) != tuple]
    tcols = [c for c in df.columns if type(c) == tuple]
    df.columns = scols + ['_'.join(str(i) for i in col) for col in tcols]
    avg_len = 1 #(df.groupby(col)["pred_text1"]
                #   .apply(lambda x: np.mean(x.str.len()).round(2)))
    ren = {
            "target_text_nunique":"num_targets",
            "pred_text1_nunique":"num_preds",
            "input_text_nunique":"num_inps",
            }
    for c in df.columns:
        #if c == FID + "_first":
        #    ren[c] = "fid"
        if c.endswith("_mean"):
            ren[c] = c.replace("_mean","")
        elif c.endswith("_first"):
            ren[c] = c.replace("_first","")
    df = df.rename(columns=ren)
    df["avg_len"] = avg_len
    df = df.sort_values(by = ["rouge_score"], ascending=False)
    return df

def add_cols(df):
    for col in df.columns:
        if col.endswith("score"):
            df[col] = pd.to_numeric(df[col])

    df = df.drop(columns = [c for c in df.columns if c.startswith("c-")])
    df['id']=df.index
    df = df.reset_index(drop=True)
    if not "tag" in df:
        df["tag"] = np.NaN 
    if not "hscore" in df:
        df["hscore"] = np.NaN 

    if not "pid" in df:
        df["pid"] = 0
    if "gen_route_methods" in df:
        df["gen_norm_method"] = df["gen_route_methods"]
        df["norm_method"] = df["apply_softmax_to"]

    if "gen_norm_methods" in df:
        df["gen_norm_method"] = df["gen_norm_methods"]

    if not "query" in df:
        df["query"] = df["input_text"]
    if not "learning_rate" in df:
        df["learning_rate"] = 1

    if not "prefixed" in df:
        df["prefixed"] = False

    if not "sel" in df:
       df["sel"] = False

    if not "template" in df:
       df["template"] = ""

    if not "bert_score" in df:
       df["bert_score"] = 0

    df["model_temp"] = df["model_name_or_path"].str.split("-").str[2]
    df["model_base"] = df["model_name_or_path"].str.split("-").str[1]
    #if "fid" in df:
    #    df = df.rename(columns={"fid":"expid"})

    template_mapping = {
         'sup': 'Mapping', 
         'unsup': 'MaskedMapping',
         'sup-nat': 'Prompting', 
         'unsup-nat': 'MaskedPrompting',
         'vnat-v3': 'MaskedChoicePrompting',
         'vnat-v33': 'MaskedChoicePrompting33',
         'vnat_1-vs2': "ChoicePrompting",
         'vnat_0-v4': "MaskedAnswerPrompting",
         'vnat_0-vs2': "AnswerPrompting",
         '0-ptar-sup': 'PostPT', 
         "ptar-sup":"PreSup", 
         'ptar-unsup-nat':'MaskedPrePT',
         'ptar-sup-nat': 'PrePT', 
         '0-ptar-unsup': 'MaskedPostPT',
         '0-ptar-vnat-v3': 'MaskedAnswerPT', 
         '0-ptar-vnat_1-vs1': 'AnswerPT',
         'ptar-vnat_0-v4':'PreMaskedAnswerPT',
         '0-ptar-vnat_0-v4':'PostMaskedAnswerPT',
         'ptar-vnat_0-vs2':'PreAnswerPT',
         '0-ptar-vnat_0-vs2':'PostAnswerPT',
         # 'vnat_1-vs2': "Predict Choice Number",
        }
    templates = df["template"].unique()
    if any("sup" in str(x) or "vnat" in str(x) for x in templates):
        df['template'] = df['template'].map(template_mapping)
    model_temp_mapping = {
        'none': '---', 
        'sup': 'LM', 
        'unsup': 'Denoising',
        "mixed": "Mixed"
        }
    model_temps = df["model_temp"].unique()
    if any("sup" in str(x) for x in model_temps):
        df['model_temp'] = df['model_temp'].map(model_temp_mapping)
    if "input_text" in df:
        df['input_text'] = df['input_text'].str.replace('##','')
        df['input_text'] = df['input_text'].str.split('>>').str[0]
        df['input_text'] = df['input_text'].str.strip()

    if not "plen" in df:
        df["plen"] = 8
    if not "blank" in df:
        df["blank"] = "blank"
    if not "opt_type" in df:
        df["opt_type"] = "na"
    if not "rouge_score" in df:
        df["rouge_score"] = 0
    if not "bert_score" in df:
        df["bert_score"] = 0

    if "mask_type" in df:
        df["cur_masking"] = (df["mask_type"].str.split("-").str[1] + "-" 
                + df["mask_type"].str.split("-").str[2]) 
    if "use_masked_attn" in df:
        df["use_masked_attn"] = df["use_masked_attn"].astype(str)

    if False:
        # df.loc[df["rouge_score"] < 1, ["rouge_score", "bert_score"]] *= 100
        df["rouge_score"] = df["rouge_score"]*100 
        df["bert_score"] = df["bert_score"]*100 
    if True: #"compose_method" in df:
        df["expid"] = df["exp_name"].str.split("-").str[1]
        df["expname"] = df["exp_name"].str.split("-").str[1]
        df["ftag"] = df["folder"].str.split("/").str[-1]
        df["ftag"] = df["ftag"].str.split("_").str[0]

        #df["model_base"] = df["model_name_or_path"].apply(lambda x: '-'.join(x.split('-')[1:2] + x.split('-')[-2:]))

    if False: #"expid" in df:
        df["fexpid"] = df["expid"]
        df["expname"] = df["expid"].str.split("-").str[0]
        df["expname"] = df["expname"].str.split("_").str[0]
        df["expid"] = df["expid"].str.split("-").str[1]
        df["expid"] = df["expid"].str.split(".").str[0]
    return df

def find_common(df, main_df, on_col_list, s_rows, FID, char, tag_cols):
    dfs_items = [] 
    dfs = []
    ii = 0
    dfs_val = {}
    for s_row in s_rows:
        if s_row > len(df) - 1:
            continue
        exp=df.iloc[s_row]["fid"]
        prefix=df.iloc[s_row]["prefix"]
        dfs_val["exp" + str(ii)] = exp
        # mlog.info("%s == %s", FID, exp)
        cond = f"(main_df['{FID}'] == '{exp}') & (main_df['prefix'] == '{prefix}')"
        tdf = main_df[(main_df[FID] == exp) & (main_df['prefix'] == prefix)]
        _cols = tag_cols + ["pred_text1", "top_pred", "top", "exp_name", "id","hscore", "bert_score", "out_score","query", "resp","resp_template", "template", "rouge_score", "fid","prefix", "input_text","target_text", "sel"]
        _cols = list(dict.fromkeys(_cols))
        _cols2 = [] 
        for col in _cols:
            if col in main_df:
                _cols2.append(col)
        tdf = tdf[_cols2]
        tdf = tdf.sort_values(by="rouge_score", ascending=False)
        sort = "rouge_score"
        if len(tdf) > 1:
            tdf = tdf.groupby(on_col_list).first()
            tdf = tdf.reset_index()
            for on_col in on_col_list:
                tdf[on_col] = tdf[on_col].astype(str).str.strip()
            #tdf = tdf.set_index(on_col_list)
        dfs.append(tdf) #.copy())
        ii += 1
    if char == "i":
        return df, exp, dfs
    if ii > 1:
        intersect = reduce(lambda  left,right: pd.merge(left,right,on=on_col_list,
                                    how='inner'), dfs)
        if char == "k":
            union = reduce(lambda  left,right: pd.merge(left,right,on=on_col_list,
                                    how='outer'), dfs)
            dfs_val["union"] = str(len(union))
            dfs_val["int"] = str(len(intersect))
            dfs_items.append(dfs_val)
            df = pd.DataFrame(dfs_items)
        else:
            df = intersect
    else:
       df = tdf
       df["sum_fid"] = df["id"].sum()
    return df, exp, dfs

def calc_metrics(main_df):
    infos = []
    all_exps = main_df['eid'].unique()
    for exp in all_exps:
        for task in main_df["prefix"].unique():
            cond = ((main_df['eid'] == exp) & (main_df["prefix"] == task))
            tdf = main_df[cond]
            preds = tdf["pred_text1"]
            preds = preds.fillna(0)
            if len(preds) == 0:
                continue
            golds = tdf["target_text"]
            task_metric = mets.TASK_TO_METRICS[task] if task in mets.TASK_TO_METRICS else ["rouge"]
            metrics_list = []
            for mstr in task_metric:
                metric = getattr(mets, mstr)
                met = metric(preds, golds)
                metrics_list.append(met)
            if met: 
                v = list(met.values())[0]
                main_df.loc[cond, "m_score"] = round(float(v),1)
            #for met in metrics_list:
            #    for k,v in met.items():
            #        infos.append(exp + ":" + task + ":" + str(k) + ":" + str(v))
            #        infos.append("---------------------------------------------")
    return infos


class MyDF:
    context = ""
    df = None
    sel_row = 0
    sel_rows = []
    sel_cols = []
    selected_cols = []
    cur_row = 0
    cur_col = -1
    left = 0
    info_cols = []
    sort = ""
    group_col = ""
    is_filtered = False
    def __init__(self, df, context, sel_cols, cur_col,info_cols, 
            sel_rows, sel_row, cur_row, 
            left, group_col, selected_cols, sort, is_filtered, **kwargs):
        self.df = df
        self.context = context
        self.sel_cols = sel_cols
        self.cur_col = cur_col
        self.info_cols = info_cols
        self.sel_rows = sel_rows
        self.cur_row = cur_row
        self.sel_row = sel_row
        self.left = left
        self.group_col = group_col
        self.selected_cols = selected_cols
        self.sort = sort
        self.is_filtered = is_filtered


def show_df(df, summary=False):
    global dfname, hotkey, global_cmd

    hk = hotkey
    cmd = global_cmd 
    ms_cmd = False
    sel_row = 0
    cur_col = -1
    cur_row = 0
    std.refresh()
    ROWS, COLS = std.getmaxyx()
    ch = 1
    left = 0
    #text_win = cur.newpad(ROWS * 10, COLS * 10)
    #text_win.bkgd(' ', cur.color_pair(TEXT_COLOR)) # | cur.A_REVERSE)
    max_row, max_col= text_win.getmaxyx()
    width = 15
    top = 10
    height = 10
    cond = ""
    sort = ""
    asc = False
    info_cols = load_obj("info_cols", dfname, []) 
    info_cols_back = []
    sel_vals = []
    col_widths = load_obj("widths", "")
    def refresh():
        text_win.refresh(0, left, 0, 0, ROWS-1, COLS-1)
    def fill_text_win(rows):
        text_win.erase()
        for row in rows:
            mprint(row, text_win)
        refresh()

    def save_df(df): 
        return
        s_rows = range(len(df))
        show_msg("Saving ...")
        for s_row in s_rows:
            exp=df.iloc[s_row]["fid"]
            tdf = main_df[main_df["fid"] == exp]
            spath = tdf.iloc[0]["path"]
            tdf.to_csv(spath, sep="\t", index=False)


    if not col_widths:
        col_widths = {"query":50, "model":30, "pred_text1":30, "epochs":30, "date":30, "rouge_score":7, "bert_score":7, "out_score":7, "input_text":50}


    main_vars = get_main_vars(df)
    #if not "word_score" in df:
    #    df['word_score'] = df['pred_text1'].str.split().str.len()


    #if not "l1_decoder" in df:
    #    df["l1_decoder"] ="" 
    #    df["l1_encoder"] ="" 
    #    df["cossim_decoder"] ="" 
    #    df["cossim_encoder"] ="" 

    #if not "m_score" in df:
    #    calc_metrics(df)
    #if "test_f1" in df:
    #    df["m_score"] = df["test_f1"]
    if not summary:
        df = add_cols(df)

    main_df = df
    edit_col = ""
    count_col = ""
    extra = {"filter":[], "inp":""}
    save_obj(dfname, "dfname", "")
    sel_cols = list(df.columns)
    fav_path = os.path.join(base_dir, dfname + "_fav.tsv")
    if Path(fav_path).exists():
        fav_df = pd.read_table(fav_path)
    else:
        fav_df = pd.DataFrame(columns = df.columns)
    sel_path = os.path.join(home, "atomic2020", "new_test.tsv")
    if Path(sel_path).exists():
        sel_df = pd.read_table(sel_path)
        if not "sel" in sel_df:
            sel_df["sel"] = False
    else:
        sel_df = pd.DataFrame(columns = ["prefix","input_text","target_text", "sel"])
        sel_df.to_csv(sel_path, sep="\t", index=False)

    back = []
    cur_df = None
    context = "main"
    is_filtered = False

    def backit(df, sel_cols, cur_df = None):
        if len(sel_cols) < 2:
            mbeep()
            return
        if not cur_df:
            cur_df = MyDF(df, context, sel_cols, cur_col,info_cols, 
                sel_rows, sel_row, cur_row, left, group_col, selected_cols, sort, is_filtered)
        back.append(cur_df)
        general_keys["b"] = "back"

    filter_df = main_df
    tag_cols = []
    if False: #"taginfo" in df:
        tags = df.loc[0, "ftag"]
        tags = tags.replace("'", "\"")
        tags = json.loads(tags)
        tag_cols = list(tags.keys())

    if "expid" in tag_cols:
        tag_cols.remove("expid")

    #df.loc[df.expid == 'P2-1', 'expid'] = "PI" 
    #tag_cols.insert(1, "expid")
    #if "m_score" in df:
    #    df["m_score"] = np.where((df['m_score']<=0), 0.50, df['m_score'])

    orig_tag_cols = tag_cols.copy()
    src_path = ""
    if "src_path" in df:
        src_path = df.loc[0, "src_path"]
        if not src_path.startswith("/"):
            src_path = os.path.join(home, src_path)
    if "pred_text1" in df:
        br_col = df.loc[: , "bert_score":"rouge_score"]
        df['nr_score'] = df['rouge_score']
        df['nr_score'] = np.where((df['bert_score'] > 0.3) & (df['nr_score'] < 0.1), df['bert_score'], df['rouge_score'])


    df['prefix'] = df['prefix'].str.replace("yelp_polarity", "yelp-polarity")
    pattern = r'(^.+)_(\1$)'
    # Use str.replace() with regex to replace matching patterns
    df['prefix'] = df['prefix'].str.replace(pattern, r'\1', regex=True)

    #wwwwwwwwww
    colors = ['blue','teal','orange', 'red', 'purple', 'brown', 'pink','gray','olive','cyan']
    context_map = {"g":"main", "G":"main", "X":"view", "r":"main"}
    general_keys = {"l":"latex df"}
    shortkeys = {"main":{"r":"pivot table"}}
    ax = None
    fig = None
    if "Z" in hotkey:
        df["m_score"] = df["rouge_score"]
    context = dfname
    font = ImageFont.truetype("/usr/share/vlc/skins2/fonts/FreeSans.ttf", 30)
    seq = ""
    reset = False
    prev_idea = ""
    pcols = [] #pivot unique cols
    cond_colors = {} # a dictionary of functions
    back_sel_cols = []
    all_sel_cols = []
    main_sel_cols = []
    register = {}
    visual_mode = False
    search = ""
    si = 0
    mode = "main"
    sort = "rouge_score"
    on_col_list = []
    keep_cols = []
    unique_cols = []
    group_sel_cols = []
    group_df = None
    pivot_df = None
    rep_cols = []
    index_cols = []
    dfs = []
    pivot_cols = ['prefix']
    exclude_cols = []
    experiment_images = {}
    cond_set = {}

    #sel_cols =  load_obj("sel_cols", context, [])
    #info_cols = load_obj("info_cols", context, [])
    all_cols = {}
    file_dir = Path(__file__).parent
    with open(os.path.join(file_dir, 'cols.json'),'r') as f:
        all_cols = json.load(f)

    if 'sel_cols' in all_cols:
        if not summary:
            sel_cols = all_cols['sel_cols'] 
        info_cols = all_cols['info_cols'] 
        rep_cols = all_cols['rep_cols'] if "rep_cols" in all_cols else sel_cols
        index_cols = all_cols['index_cols']
        extra_cols = all_cols['extra_cols'] if "extra_cols" in all_cols else []
        exclude_cols = all_cols['ex_cols'] if "ex_cols" in all_cols else []
    if "compose_method" in df:
        rep_cols = rep_cols + extra_cols
    #if not "compose_method" in df:
    #    df["compose_method"] = "PT"
    if "label" in df:
        rep_cols = rep_cols + ["label"]

    main_sel_cols = sel_cols.copy()
    settings = load_obj("settings", "gtasks", {})

    rels = [] # df["prefix"].unique()
    use_rouge = True # settings["use_rouge"] if "use_rouge" in settings else False 
    if use_rouge:
        score_cols = ['rouge_score','depth_score','perp_score','num_preds'] 
    else:
        score_cols = ['m_score','num_preds'] 
        # score_cols = ['bert_score', 'num_preds'] 

    rep_cols = main_vars + sel_cols + rep_cols + score_cols
    rep_cols = list(dict.fromkeys(rep_cols))
    back_sel_cols = sel_cols.copy()

    if 'init_col' in all_cols:
        init_col = all_cols['init_col']
        if not init_col in sel_cols and init_col in df:
            sel_cols.insert(1, init_col) 
        cur_col = sel_cols.index(init_col) - 1

    sel_fid = "" 
    df_cond = True
    df_conds = []
    open_dfnames = [dfname]
    dot_cols = {}
    selected_cols = []
    rep_cmp = load_obj("rep_cmp", "gtasks", {})
    capt_pos = settings["capt_pos"] if "capt_pos" in settings else "" 
    rname = settings.setdefault("rname", "rpp")
    task = ""
    if "prefix" in df:
        task = df["prefix"][0]
    #if not "learning_rate" in df:
    #    df[['fid_no_lr', 'learning_rate']] = df['fid'].str.split('_lr_', 1, expand=True)
    prev_cahr = ""
    FID = "fid"
    sel_exp = ""
    infos = []
    back_rows = []
    back_infos = []
    sel_rows = []
    prev_cmd = ""
    do_wrap = True
    sel_group = 0
    group_col = ""
    keep_uniques = False
    group_rows = []

    def row_print(df, col_widths ={}, _print=False):
        nonlocal group_rows, sel_row
        infos = []
        group_mode = group_col and group_col in sel_cols 
        margin = min(len(df), 5) # if not group_mode else 0
        sel_dict = {}
        g_row = ""
        g = 0
        g_start = -1
        row_color = TEXT_COLOR
        sel_col_color = 102 # HL_COLOR #TITLE_COLOR
        selected_col_color = SEL_COLOR
        cross_color = SEL_COLOR # WARNING_COLOR # HL_COLOR   
        sel_row_color = HL_COLOR if not group_mode else row_color 
        g_color = row_color
        _cur_row = cur_row #-1 if group_mode else sel_row 
        ii = 0 
        gg = 0 # count rows in each group
        pp = 0 # count printed rows
        for idx, row in df.iterrows():
           text = "{:<5}".format(ii)
           _sels = []
           _infs = []
           if (group_mode and group_col in row and row[group_col] != g_row):
               g_row = row[group_col]
               gg = 0
               if not keep_uniques and _print and _cur_row >= 0 and ii >= _cur_row - margin:
                   g_text = "{:^{}}".format(g_row, COLS)
                   # mprint("\n", text_win, color = HL_COLOR) 
                   mprint(g_text, text_win, color = HL_COLOR) 
                   # mprint("\n", text_win, color = HL_COLOR) 
               if g_start >= 0:
                   group_rows = range(g_start, ii)
                   g_start = -1
               if g % 2 == 0:
                  row_color = TEXT_COLOR #INFO_COLOR 
                  sel_col_color = ITEM_COLOR 
                  g_color = row_color
               else:
                  row_color = TEXT_COLOR
                  sel_col_color = TITLE_COLOR
                  g_color = row_color
               if g == sel_row: # sel_group:
                  #_cur_row = ii
                  #row_color = SEL_COLOR
                  #g_color = WARNING_COLOR
                  g_start = ii
               g+=1
           if _cur_row < 0 or ii < _cur_row - margin:
               ii += 1
               pp += 1
               continue
           if group_mode and keep_uniques and gg > 0:
               ii += 1
               continue

           # if group_mode: cross_color = sel_col_color
           _color = row_color
           if cur_col < 0:
               if ii == sel_row:
                  _color = HL_COLOR
               else:
                  _color = sel_col_color
           if pp == _cur_row:
               sel_row = ii
           if pp in sel_rows:
               _color = MSG_COLOR
           if pp == _cur_row and not group_mode:
                _color = cross_color if cur_col < 0 else SEL_COLOR 
           if _print:
               mprint(text, text_win, color = _color, end="") 
           if _print:
               _cols = sel_cols + info_cols
           else:
               _cols = sel_cols
           for sel_col in _cols: 
               if  sel_col in _sels:
                   continue
               if not sel_col in row: 
                   if sel_col in sel_cols:
                       sel_cols.remove(sel_col)
                   continue
               content = str(row[sel_col])
               content = content.strip()
               orig_content = content
               content = "{:<4}".format(content) # min length
               if sel_col in wraps and do_wrap:
                   content = content[:wraps[sel_col]] + ".."
               if "score" in sel_col:
                   try:
                       content = "{:.2f}".format(float(content))
                   except:
                       pass
               _info = sel_col + ":" + orig_content
               if sel_col in info_cols:
                   if pp == _cur_row and not sel_col in _infs:
                      infos.append(_info)
                      _infs.append(sel_col)
               if pp == _cur_row:
                   sel_dict[sel_col] = row[sel_col]
               if not sel_col in col_widths:
                   col_widths[sel_col] = len(content) + 2
               if len(content) > col_widths[sel_col]:
                   col_widths[sel_col] = len(content) + 2
               col_title = map_cols[sel_col] if sel_col in map_cols else sel_col
               min_width = max(5, len(col_title) + 1)
               max_width = 100
               if len(sel_cols) > 2:
                   max_width = int(settings["max_width"]) if "max_width" in settings else 36
               _width = max(col_widths[sel_col], min_width)
               _width = min(_width, max_width)
               col_widths[sel_col] = _width 
               _w = col_widths[sel_col] 
               if sel_col in sel_cols:
                   if (cur_col >=0 and cur_col < len(sel_cols) 
                          and sel_col == sel_cols[cur_col]):
                       if pp == _cur_row: 
                          cell_color = cross_color 
                       #elif sel_col in cond_colors:
                       #    cell_color = cond_colors[sel_col](df, ii, sel_col, 
                       #            default = sel_col_color)
                       elif sel_col in selected_cols:
                          cell_color = selected_col_color
                       else:
                          cell_color = sel_col_color
                   else:
                       if ii in sel_rows:
                          cell_color = MSG_COLOR
                       elif sel_col in selected_cols:
                          cell_color = selected_col_color
                       elif pp == _cur_row:
                          cell_color = sel_row_color
                       elif sel_col in cond_colors:
                           cell_color = cond_colors[sel_col](df, ii, sel_col, 
                                   default = row_color)
                       elif sel_col == group_col:
                          cell_color = g_color
                       else:
                          cell_color = row_color
                   content = textwrap.shorten(content, width=max_width, placeholder="...")
                   text = "{:<{x}}".format(content, x= _w)
                   if _print:
                       mprint(text, text_win, color = cell_color, end="") 
                   _sels.append(sel_col)

           _end = "\n"
           if _print:
               mprint("", text_win, color = _color, end="\n") 
           ii += 1
           gg += 1
           pp += 1
           if pp > _cur_row + ROWS:
               break
        return infos, col_widths

    def get_sel_dfs(df, row_id="fid", col="eid"):
        dfs = []
        s_rows = sel_rows
        if not s_rows:
            s_rows = [sel_row]
        
        # Iterate through each selected row
        for s_row in s_rows:
            # If row_id is present in df, use it to filter main_df
            if row_id in df.columns:
                row_value = df.iloc[s_row][row_id]
                # Filter main_df using row_id and append result to dfs
                tdf = main_df[main_df[row_id] == row_value]
                dfs.append(tdf)
        
        # Concatenate the filtered DataFrames if we have any
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        
        # If no DataFrame filtering was done, return None 
        return None 


    def get_sel_rows(df, row_id="eid", col="eid", from_main=True):
        values = []
        s_rows = sel_rows
        exprs = []
        if not s_rows:
            s_rows = [sel_row]
        for s_row in s_rows:
            if row_id is None:
                val=df.iloc[s_row][col]
                values.append(val)
            elif row_id in df:
                row=df.iloc[s_row][row_id]
                if from_main:
                    tdf = main_df[main_df[row_id] == row]
                else:
                    tdf = df[df[row_id] == row]
                val=tdf.iloc[0][col]
                values.append(val)
                exp=tdf.iloc[0][row_id]
                exprs.append(exp)
        return exprs, values 


    for _col in ["input_text","pred_text1","target_text"]:
        if _col in df:
            df[_col] = df[_col].astype(str)

    map_cols =  load_obj("map_cols", "atomic", {})
    def get_images(df, exps, fid='eid', merge="vert", image_keys="all", crop=False):
        imgs = {}
        dest = ""
        start = "pred"
        fnames = []
        for exp in exps:
            cond = f"(main_df['{fid}'] == '{exp}')"
            tdf = main_df[main_df[fid] == exp]
            if tdf.empty:
                return imgs, fnames
            path=tdf.iloc[0]["path"]
            path = Path(path)
            #_selpath = os.path.join(path.parent, "pred_sel" + path.name) 
            #shutil.copy(path, _selpath)
            # grm = tdf.iloc[0]["gen_route_methods"]
            runid = tdf.iloc[0]["eid"]
            #run = "wandb/offline*" + str(runid) + f"/files/media/images/{start}*.png"
            run = "img_logs/{start}*.png"
            paths = glob(str(path.parent) +"/img_logs/*.png")
            # paths = glob(run)
            spath = "images/exp-" + str(runid)
            if Path(spath).exists():
                shutil.rmtree(spath)
            Path(spath).mkdir(parents=True, exist_ok=True)
            images = []
            kk = 1
            key = exp # "single"
            ii = 0
            for img in paths: 
                fname = Path(img).stem
                # if fname in fnames:
                #    continue
                fnames.append(fname) #.split("_")[0])
                parts = fname.split("_")
                ftype = fname.split("@")[1]
                if kk < 0:
                    _, key = list_values(parts)
                    kk = parts.index(key)
                    key = parts[kk]
                dest = os.path.join(spath, fname + ".png") 
                # if not fname.startswith("pred_sel"):
                #    selimg = str(Path(img).parent) + "/pred_sel" +  fname + ".png"
                #    os.rename(img, selimg)
                #    img = selimg
                shutil.copyfile(img, dest)
                _image = Image.open(dest)
                if key == "single": key = str(ii)
                if not key in imgs:
                    imgs[key] = {} # [_image]
                imgs[key][ftype] = _image
                images.append({"image": dest})
        if imgs:
            fnames = []
            c_imgs = {}
            if Path("temp").exists():
                shutil.rmtree("temp")
            Path("temp").mkdir(parents=True, exist_ok=True)
            for key, img_dict in imgs.items():
                #sorted_keys = (sorted(img_dict.keys()))
                if not image_keys:
                  key_list = ["sim", "rsim", "score", 
                                "effect", "init_router", "router", "mask"] 
                  # image_keys = ["score", "router"] 
                else:
                  key_list = []
                  for prefix in image_keys:
                      key_list.extend([k for k in img_dict.keys() if k.startswith(prefix)])
                # TODO fixed
                img_list = [img_dict[k] for k in key_list if k in img_dict] 
                max_width = 0
                if crop:
                    for i, img in enumerate(img_list):
                        img_list[i] = trim_white_borders(img)

                if len(img_list) > 0:
                    if len(img_list) > 1 and merge == "vert":
                        new_im = combine_y(img_list)
                    else:
                        new_im = combine_x(img_list)
                    name = str(key) 
                    dest = os.path.join("temp", name.strip("-") + ".png")
                    new_im.save(dest)
                    c_imgs[key] = [new_im] 
                    if new_im.width > max_width:
                        max_width = new_im.width
                    fnames.append(dest)
            for key, img_list in c_imgs.items():
                for i, img in enumerate(img_list):
                    if crop:
                        #imageBox = img.getbbox()
                        #_image = img.crop(imageBox)
                        _image = trim_white_borders(img)
                        # _image = add_margin(_image, 10, 0, 0, 0, (255, 255, 255))
                        c_imgs[key][i] = _image
                    elif img.width < max_width:
                        dif = max_width - img.width // 2
                        _image = add_margin(img, 0, dif, 0, dif, (255, 255, 255))
                        c_imgs[key][i] = _image
            imgs = c_imgs
        return imgs, fnames

    if not map_cols:
        map_cols = {
            "epochs_num":"epn",
            "exp_trial":"exp",
            "pred_text1":"pred",
            "target_text":"tgt",
            "template":"tn",
            "pred_max_num":"pnm",
            "attn_learning_rate":"alr",
            "attn_method":"am",
            "attend_source":"att_src",
            "attend_target":"att_tg",
            "attend_input":"att_inp",
            "add_target":"add_tg",
            "rouge_score":"rg",
            "out_score":"out",
            "bert_score":"bt",
            }
    wraps = {
            "tag":20,
            }
    adjust = True
    show_consts = False
    show_extra = False
    show_infos = False
    show_rest = False
    consts = {}
    extra = {"filter":[]}
    orig_df = main_df.copy()
    prev_char = ""
    new_df = True
    cur_col_name = ""
    while prev_char != "q":
        group_rows = []
        left = min(left, max_col  - width)
        left = max(left, 0)
        top = min(top, max_row  - height)
        top = max(top, 0)
        sel_row = min(sel_row, len(df) - 1)
        sel_row = max(sel_row, 0)
        cur_row = max(cur_row, 0)
        cur_row = min(cur_row, len(df) - 1)
        sel_rows = sorted(sel_rows)
        sel_rows = list(dict.fromkeys(sel_rows))
        sel_cols = list(dict.fromkeys(sel_cols))
        sel_group = max(sel_group, 0)
        #sel_group = min(sel_row, sel_group)
        cur_col = min(cur_col, len(sel_cols) - 1)
        cur_col = max(cur_col, -1)
        if (cur_col_name and sel_cols[cur_col] != cur_col_name 
                and cur_col_name in sel_cols 
                and not ch in [LEFT, RIGHT]):
            cur_col = sel_cols.index(cur_col_name)
        cur_col_name = sel_cols[cur_col]
        back_df = back[-1].df if len(back) > 0 else df
        if not hotkey:
            # std.clear()
            text_win.erase()
            #text_win.clear()
            if adjust:
                _, col_widths = row_print(df, col_widths={})
            text = "{:<5}".format(sel_row)
            for i, sel_col in enumerate(sel_cols):
               if not sel_col in df:
                   sel_cols.remove(sel_col)
                   continue
               head = sel_col if not sel_col in map_cols else map_cols[sel_col] 
               #head = textwrap.shorten(f"{i} {head}" , width=15, placeholder=".")
               if not sel_col in col_widths and not adjust:
                    _, col_widths = row_print(df, col_widths={})
                    adjust = True
               if sel_col in col_widths and len(head) > col_widths[sel_col]:
                   col_widths[sel_col] = len(head) 
               _w = col_widths[sel_col] if sel_col in col_widths else 5
               if i == cur_col:
                  #head = inline_colors.INFO2 + head + inline_colors.ENDC 
                  head = head + "*"
                  text += "{:<{x}}".format(head, x=_w) 
               else:
                  text += "{:<{x}}".format(head, x=_w) 
            mprint(text, text_win) 
            #fffff
            infos,_ = row_print(df, col_widths, True)
            # refresh()
            text_win.noutrefresh(0, left, 0, 0, ROWS-1, COLS -1)
            # text_win.refresh(0, left, 0, 0, ROWS-1, COLS-2)
            # cur.doupdate()
            # cur.flushinp()
        if cur_col < len(sel_cols) and len(sel_cols) > 0:
            _sel_col = sel_cols[cur_col]
            if not df.empty:
                _sel_val = df.iloc[sel_row][_sel_col]
                infos.append("{},{}:{}".format(sel_row, _sel_col, _sel_val))
                show_rest = type(_sel_val) == str and len(_sel_val) > 50
                if show_infos and not show_rest:
                    if "query" in df.iloc[sel_row]:
                        _sel_val = df.iloc[sel_row]["query"]
                    else:
                        _, _sel_val = get_sel_rows(df, col="query", from_main=True) 
                    infos.append("query:{}".format(_sel_val))
                    infos.append("-------------------------")
                    if "resp" in df.iloc[sel_row]:
                        _sel_val = df.iloc[sel_row]["resp"]
                    else:
                        _, _sel_val = get_sel_rows(df, col="resp", from_main=True) 
                    infos.append("resp:{}".format(_sel_val))
        for c in sel_cols:
            if not c in df:
                continue
            if "score" in c:
                mean = df[c].mean()
                _info = f"Mean {c}:" + "{:.2f}".format(mean)
                infos.append(_info)
        infos.append("-------------------------")
        if show_consts:
            consts["len"] = str(len(df))
            consts["root"] =str([Path(root_path).stem]*5)
            consts["context"] = context
            consts["keys"] = general_keys
            if context in shortkeys:
                consts["keys"] = {**general_keys,**shortkeys[context]}
            for key,val in consts.items():
                if type(val) == list:
                    val = "-".join(val)
                infos.append("{:<5}:{}".format(key,val))
        if show_extra:
            show_extra = False
            for key,val in extra.items():
                if type(val) == list:
                    val = "-".join(val)
                infos.append("{:<5}:{}".format(key,val))
        if show_infos or show_rest or show_consts or show_extra:
            info_lines = change_info(infos)

        prev_char = chr(ch)
        prev_cmd = cmd
        if not ms_cmd:
            if global_cmd and not hotkey or hotkey == "q":
                cmd = global_cmd
                global_cmd = ""
            else:
                cmd = ""
            if hotkey == "":
                cur.doupdate()
                if new_df:
                    cur.flushinp()
                    cur.flash()
                ch = std.getch()
            else:
                ch, hotkey = ord(hotkey[0]), hotkey[1:]
            char = chr(ch)
        else:
            ch = 0
            char = ""
        if char != "q" and prev_char == "q": 
            consts["exit"] = ""
            prev_char = ""
        extra["inp"] = char

        seq += char
        vals = []
        get_cmd = False
        adjust = True
        visual_mode = visual_mode and ch in [LEFT, RIGHT, DOWN]
        #if char in context_map:
        #    context = contexts_map[char] 
        if ch == cur.KEY_NPAGE:
            left += 20
            adjust = False
            cur_col += 5
            ch = RIGHT
        if ch == cur.KEY_PPAGE:
            left -= 20
            adjust = False
            cur_col -= 5
            ch = LEFT
        if ch == SDOWN:
            info_cols_back = info_cols.copy()
            info_cols = []
        if context == "details": # or context == "notes":
            old_search = search
            pattern = re.compile("[A-Za-z0-9]+")
            if ch == cur.KEY_BACKSPACE:
                if search:
                   search = search[:-1]
                   char == ""
                   ch = 0
                else:
                   context = ""
                if not search:
                   mbeep()
            elif pattern.fullmatch(char) is not None:
                search += char 
            if search and search != old_search:
                col = sel_cols[cur_col]
                df = search_df[search_df[col].astype(str).str.contains(search, na=False)]
                # .first_valid_index()
                # si = min(si, len(mask) - 1)
                # sel_row = df.loc[mask.any(axis=1)].index[si]
                char = ""
                ch = 0
            consts["search"] = "/" + search
        if char == ";":
            # info_cols = info_cols_back.copy()
            backit(df, sel_cols)
            context = "details"
            max_width = 100
            consts["search"] = "/"
            infos = []
            for c in df.columns:
                value = df.iloc[sel_row][c]
                _info = {"col":c, "val":value}
                infos.append(_info)
            df = pd.DataFrame(data=infos)
            df = df.sort_values(by="col", ascending=True)
            search_df = df
            sel_cols = ["col","val"]
        if ch == LEFT:
            if visual_mode:
                _col_name = sel_cols[cur_col]
                if not _col_name in selected_cols:
                    selected_cols.append(_col_name)
            cur_col -= 1
            cur_col = max(-1, cur_col)
            #if cur_col < 15 and all_sel_cols:
            #    sel_cols = all_sel_cols[:20]
            cur_col = min(len(sel_cols)-1, cur_col)
            cur_col_name = sel_cols[cur_col]
            width = len(cur_col_name) + 2
            if cur_col_name in col_widths:
                width = col_widths[cur_col_name]
            _sw = sum([col_widths[x] if x in col_widths else len(x) + 2 
                for x in sel_cols[:cur_col]])
            if _sw < left:
                left = _sw - width - 10 
            adjust = False
        if ch == RIGHT:
            if visual_mode:
                _col_name = sel_cols[cur_col]
                if not _col_name in selected_cols:
                    selected_cols.append(_col_name)
            cur_col += 1
            #if cur_col > 15 and len(all_sel_cols) > 10:
            #    sel_cols = all_sel_cols[10:30]
            cur_col = min(len(sel_cols)-1, cur_col)
            cur_col = max(0,cur_col)
            cur_col_name = sel_cols[cur_col]
            width = len(cur_col_name) + 2
            if cur_col_name in col_widths:
                width = col_widths[cur_col_name]
            _sw = sum([col_widths[x] if x in col_widths else len(x) + 2 
                for x in sel_cols[:cur_col]])
            if _sw >= left + COLS - 10:
                left = _sw - 10 
            adjust = False
        if char in ["+","-","*","/"] and prev_char == "x":
            _inp=df.iloc[sel_row]["input_text"]
            _prefix=df.iloc[sel_row]["prefix"]
            _pred_text=df.iloc[sel_row]["pred_text1"]
            _fid=df.iloc[sel_row]["fid"]
            cond = ((main_df["fid"] == _fid) & (main_df["input_text"] == _inp) &
                    (main_df["prefix"] == _prefix) & (main_df["pred_text1"] == _pred_text))
            if char == "+": _score = 1.
            if char == "-": _score = 0.
            if char == "/": _score = 0.4
            if char == "*": _score = 0.7

            main_df.loc[cond, "hscore"] = _score 
            sel_exp = _fid
            sel_row += 1
            adjust = False
        if ch == DOWN:
            if context == "inp":
                back_rows[-1] += 1
                hotkey = "bp"
            elif False: #TODO group_col and group_col in sel_cols:
                sel_group +=1
            else:
                if visual_mode:
                    if not cur_row in sel_rows:
                        sel_rows.append(cur_row)
                cur_row += 1
            adjust = False
        elif ch == UP: 
            if context == "inp":
                back_rows[-1] -= 1
                hotkey = "bp"
            elif False: #TODO group_col and group_col in sel_cols:
                sel_group -=1
            else:
                cur_row -= 1
            adjust = False
        elif ch == cur.KEY_SRIGHT:
            cur_row += ROWS - 4
        elif ch == cur.KEY_HOME:
            cur_row = 0 
            sel_group = 0
        elif ch == cur.KEY_SHOME:
            left = 0 
        elif ch == cur.KEY_END:
            # cur_row = len(df) -1
            show_infos = not show_infos
        elif ch == cur.KEY_SLEFT:
            cur_row -= ROWS - 4
        elif char == "l" and prev_char == "l":
            seq = ""
        elif char == "s":
            if selected_cols:
                df = df.sort_values(by=selected_cols, ascending=asc)
            else:
                col = sel_cols[cur_col]
                df = df.sort_values(by=col, ascending=asc)
            asc = not asc
        elif char in ["\""] or ch == cur.KEY_NPAGE:
            col = sel_cols[cur_col]
            if col in selected_cols:
                selected_cols.remove(col)
            else:
                selected_cols.append(col)
            consts["selected_cols"] = selected_cols
            cur_col += 1
            mbeep()
        elif char in ["+"]:
            col = sel_cols[cur_col]
            if col in score_cols:
                score_cols.remove(col)
            else:
                score_cols.append(col)
            consts["score_cols"] = score_cols 
            cur_col += 1
            mbeep()
        elif char == "-":
            backit(df, sel_cols)
            col = sel_cols[cur_col]
            val=df.iloc[sel_row][col]
            cond = True
            for o in ["gen_norm_method","norm_method"]:
                vo=df.iloc[sel_row][o]
                cond = cond & (df[o] == vo)
            df = df[cond]
        elif char == ".":
            col = sel_cols[cur_col]
            val=df.iloc[sel_row][col]
            dot_cols[col] = val
            if "sel" in consts:
                consts["sel"] += " " + col + "='" + str(val) + "'"
            else:
                consts["sel"] = col + "='" + str(val) + "'"
        elif char == "=":
            col = sel_cols[cur_col]
            val=df.iloc[sel_row][col]
            if col == "fid": col = FID
            if "filter" in consts:
                consts["filter"] += " " + col + "='" + str(val) + "'"
            else:
                consts["filter"] = col + "='" + str(val) + "'"
            df_conds.append((col, df[col] == val))
        elif char == "=" and prev_char == "x":
            col = info_cols[-1]
            sel_cols.insert(cur_col, col)
        elif char == ">":
            col = info_cols.pop()
            sel_cols.insert(cur_col, col)
        elif char in "01234" and prev_char == "#":
            canceled, col, val = list_df_values(df, get_val=False)
            if not canceled:
                sel_cols = order(sel_cols, [col],int(char))
        elif char in ["E"]:
            if not edit_col or char == "E":
                canceled, col, val = list_df_values(df, get_val=False)
                if not canceled:
                    edit_col = col
                    extra["edit col"] = edit_col
                    refresh()
            if edit_col:
                new_val = rowinput()
                if new_val:
                    df.at[sel_row, edit_col] = new_val
                    char = "SS"
        elif char in ["%"]:
            canceled, col, val = list_df_values(df, get_val=False)
            if not canceled:
                if not col in sel_cols:
                    sel_cols.insert(0, col)
                    save_obj(sel_cols, "sel_cols", context)
        elif char in ["W"] and prev_char == "x":
            save_df(df)
        elif char == "B":
            scorers = settings.get("scorer","bert")
            _score = scorers + "_score"
            if not _score in df:
                df[_score] = 0
            #_score = "rouge_score"
            #scorers = "rouge"
            exprs, scores = get_sel_rows(df, row_id="fid", col=_score, from_main=False) 
            #if _score > 0:
            #    continue
            for exp, score in zip(exprs, scores):
                tdf = main_df[main_df['fid'] == exp]
                spath = tdf.iloc[0]["path"]
                # spath = str(Path(spath).parent)
                # if str(score) != "nan" and score > 0:
                #    continue
                # tdf = tdf.head(5)
                scores = do_score(tdf, scorers, spath, reval=True) 
                # main_df.loc[main_df.fid == exp, "bert_score"] = tdf["bert_score"]
            # df = main_df
            # hotkey = hk
        if char == "O":
            sel_exp=df.iloc[sel_row]["eid"]
            tdf = main_df[main_df['eid'] == sel_exp]
            spath = tdf.iloc[0]["path"]
            subprocess.Popen(["nautilus", spath])
        if char in ["o","y","k", "p"]:
            tdf = df #pivot_df if pivot_df is not None and context == "pivot" else df
            images = []
            exprs, _ = get_sel_rows(tdf)
            merge = "vert"
            image_keys = "" 
            if char == "o" and "images" in settings:
                image_keys = settings["images"].split("@")
            elif char == "y":
                image_keys = ["effect"]
                merge = "vert"
            elif char == "k" or char == "p":
                image_keys = ["score","sim"]
                merge = "horiz"

            experiment_images, fnames = get_images(tdf, exprs, 'eid', 
                    merge = merge, image_keys = image_keys, crop = char == "k")
            dif_cols = ["expid"]
            for col in sel_cols:
                if col in pcols or col in ["eid"]:
                    continue
                vals = []
                for exp in exprs:
                    if col in tdf:
                        v = tdf.loc[tdf.eid == exp, col].iloc[0]
                        vals.append(v)
                if all(str(x).strip() == str(vals[0]).strip() for x in vals):
                    continue
                else:
                    dif_cols.append(col)

            capt_pos = settings["capt_pos"] if "capt_pos" in settings else "" 
            pic = None
            for key, img_list in experiment_images.items(): 
                im = img_list[0]
                images.append(im)

            if images:
                pic = combine_x(images) if char =="o" else combine_y(images)
                if len(images) > 1:
                    font = ImageFont.truetype("/usr/share/vlc/skins2/fonts/FreeSans.ttf", 30)
                else:
                    font = ImageFont.truetype("/usr/share/vlc/skins2/fonts/FreeSans.ttf", 20)
                im = pic
                if capt_pos and capt_pos != "none" and char != "k":
                    width, height = im.size
                    gap = 150*len(exprs) + 50
                    if capt_pos == "top":
                        _image = add_margin(im, gap, 5, 0, 5, (255, 255, 255))
                        xx = 10
                        yy = 30 
                    elif capt_pos == "below":
                        _image = add_margin(im, 0, 5, gap, 5, (255, 255, 255))
                        xx = 10
                        yy = height + 50 
                    elif capt_pos == "left":
                        _image = add_margin(im, 0, 5, 5, 700, (255, 255, 255))
                        xx = 10
                        yy = 10
                    draw = ImageDraw.Draw(_image)
                    for col_set in [dif_cols, pcols]:
                        for key in exprs:
                            caption_dict = {}
                            if context == "pivot" and not tdf.loc[tdf['eid'] == key].empty:
                                caption_dict = tdf.loc[tdf['eid'] == key, 
                                        col_set].iloc[0].to_dict()
                            for cc, value in caption_dict.items(): 
                                if cc in pcols:
                                    cc = cc.split("_")[0]
                                if cc.endswith("score"):
                                    mm = map_cols[cc] if cc in map_cols else cc
                                    mm = "{}:".format(mm)
                                    draw.text((xx, yy),mm,(150,150,150),font=font)
                                    tw, th = draw.textsize(mm, font)
                                    mm = "{:.2f}".format(value)
                                    xx += tw + 10
                                    draw.text((xx, yy),mm,(250,5,5),font=font)
                                    tw, th = draw.textsize(mm, font)
                                else:
                                    mm = map_cols[cc] if cc in map_cols else cc
                                    mm = "{}:".format(mm)
                                    draw.text((xx, yy),mm,(150,150,150),font=font)
                                    tw, th = draw.textsize(mm, font)
                                    mm = "{}".format(value)
                                    xx += tw + 10
                                    draw.text((xx, yy),mm,(20,25,255),font=font)
                                    tw, th = draw.textsize(mm, font)
                                if capt_pos == "left":
                                    xx = 10
                                    yy += 60
                                else:
                                    xx += tw + 10
                            yy += 40
                            xx = 10
                        yy += 40
                        xx = 10
                    pic = _image
            if pic is not None:
                dest = os.path.join("routers.png")
                if char == "p":
                    # fname = rowinput("prefix:", default="image")
                    ptemp = os.path.join(home, "pictemp", "image.png")
                    if Path(ptemp).is_file():
                        pic2 = Image.open(ptemp)
                        pic = combine_x([pic, pic2])
                    else:
                        pic.save(ptemp)
                pic.save(dest)
                #pname=tdf.iloc[sel_row]["image"]
                subprocess.Popen(["eog", dest])
        elif char == "L" and False:
            label = rowinput("label:")
            s_rows = sel_rows
            if not s_rows: s_rows = [sel_row]
            for s_row in s_rows:
                df.iloc[s_row, df.columns.get_loc('label')] = label
                exp=df.iloc[s_row]["eid"]
                cond = f"(main_df['eid'] == '{exp}')"
                tdf = main_df[main_df.eid == exp]
                path=tdf.iloc[0]["path"]
                if Path(path).is_file():
                    tdf = pd.read_table(path)
                    tdf["label"] = label
                    tdf.to_csv(path, sep="\t", index=False)
        elif char == "L" and False:
            if len(register) == 0:
                show_msg("Nothing in register!")
                mbeep()
            else:
                if not "key" in sel_cols:
                    sel_cols.insert(0, "key")
                group_col = "key"
                df = pd.concat([d for k,d in register])
                df = df.sort_values(by = "key")
        elif char == "L" and False:
            s_rows = sel_rows
            if not sel_rows:
                s_rows = group_rows
                if not group_rows:
                    s_rows = [sel_row]
            all_rows = range(len(df))
            Path("temp").mkdir(parents=True, exist_ok=True)
            imgs = []
            for s_row in all_rows:
                exp=df.iloc[s_row]["fid"]
                cond = f"(main_df['{FID}'] == '{exp}')"
                tdf = main_df[main_df[FID] == exp]
                path=tdf.iloc[0]["path"]
                folder = str(Path(path).parent)
                path = os.path.join(folder, "last_attn*.png")
                images = glob(path)
                tdf = pd.DataFrame(data = images, columns = ["image"])
                tdf = tdf.sort_values(by="image", ascending=False)
                pname=tdf.iloc[0]["image"]
                dest = os.path.join("temp", str(s_row) + ".png")
                shutil.copyfile(pname, dest)
                if s_row in s_rows:
                    _image = Image.open(pname)
                    imgs.append(_image)
            if imgs:
                new_im = combine_y(imgs)
                name = "-".join([str(x) for x in s_rows]) 
                pname = os.path.join("temp", name.strip("-") + ".png")
                new_im.save(pname)
            subprocess.run(["eog", pname])
        elif char == "l" and prev_char == "p":
            exp=df.iloc[sel_row]["fid"]
            cond = f"(main_df['{FID}'] == '{exp}')"
            tdf = main_df[main_df[FID] == exp]
            path=tdf.iloc[0]["path"]
            conf = os.path.join(str(Path(path).parent), "exp.json")
            with open(conf,"r") as f:
                infos = f.readlines()
            subwin(infos)
        elif char == "l" and prev_char == "x" and False:
            exp=df.iloc[sel_row]["eid"]
            exp = str(exp)
            logs = glob(str(exp) + "*.log")
            if logs:
                log = logs[0]
                with open(log,"r") as f:
                    infos = f.readlines()
                subwin(infos)

        elif char == "<":
            col = sel_cols[cur_col]
            sel_cols.remove(col)
            info_cols.append(col)
            save_obj(sel_cols, "sel_cols", context)
            save_obj(info_cols, "info_cols", context)
        elif char == "N" and prev_char == "x":
            backit(df,sel_cols)
            sel_cols=["pred_max_num","pred_max", "tag","prefix","rouge_score", "num_preds","bert_score"]
        elif (char == "j" and not prev_char == "x" and hk=="G"):
            backit(df,sel_cols)
            exp=df.iloc[sel_row]["fid"]
            cond = f"(main_df['{FID}'] == '{exp}')"
            df = main_df[main_df[FID] == exp]
            sel_cols=tag_cols + ["bert_score", "out_score", "pred_text1","target_text","input_text","rouge_score","prefix"]
            sel_cols, info_cols, tag_cols = remove_uniques(df, sel_cols, orig_tag_cols)
            unique_cols = info_cols.copy()
            df = df[sel_cols]
            df = df.sort_values(by="input_text", ascending=False)
        elif char == "I" or char == "#" or char == "3":
            canceled, col, val = list_df_values(df, get_val=False, extra=["All"])
            if not canceled:
                if col == "All":
                    sel_cols = list(df.columns)
                else:
                    if col in sel_cols: 
                        sel_cols.remove(col)
                    if col in info_cols:
                        info_cols.remove(col)
                    if char == "#" or char == "3": 
                        sel_cols.insert(cur_col, col)
                    else:
                        info_cols.append(col)
                    orig_tag_cols.append(col)
            save_obj(sel_cols, "sel_cols", context)
            save_obj(info_cols, "info_cols", context)
        elif char in ["o","O"] and prev_char == "x":
            inp = df.loc[df.index[sel_row],["prefix", "input_text"]]
            df = df[(df.prefix != inp.prefix) | 
                    (df.input_text != inp.input_text)] 
            mbeep()
            sel_df = df.sort_values(by=["prefix","input_text","target_text"]).drop_duplicates()
            sel_df.to_csv(sel_path, sep="\t", index=False)
        elif char in ["w","W"]:
            sel_cols, info_cols, tag_cols = remove_uniques(df, sel_cols, orig_tag_cols)
        elif char in ["w","W"] and False:
            inp = df.loc[df.index[sel_row],["prefix", "input_text","pred_text1"]]
            df.loc[(df.prefix == inp.prefix) & 
                    (df.input_text == inp.input_text),["sel"]] = True
            _rows = main_df.loc[(main_df.prefix == inp.prefix) & 
                    (main_df.input_text == inp.input_text), 
                    ["prefix","input_text", "target_text","sel"]]
            row = df.loc[(df.prefix == inp.prefix) & 
                    (df.input_text == inp.input_text),:]
            sel_df = sel_df.append(_rows,ignore_index=True)
            #df = df.sort_values(by="sel", ascending=False).reset_index(drop=True)
            #sel_row = row.index[0]
            if char == "W":
                new_row = {"prefix":inp.prefix,
                           "input_text":inp.input_text,
                           "target_text":inp.pred_text1, "sel":False}
                sel_df = sel_df.append(new_row, ignore_index=True)
            consts["sel_path"] = sel_path + "|"+ str(len(sel_df)) + "|" + str(sel_df["input_text"].nunique())
            mbeep()
            sel_df = sel_df.sort_values(by=["prefix","input_text","target_text"]).drop_duplicates()
            sel_df.to_csv(sel_path, sep="\t", index=False)
        elif char == "h" and False:
            backit(df, sel_cols)
            sel_cols = ["prefix", "input_text", "target_text", "sel"]
            df = sel_df
        elif char in ["h","v"] and prev_char == "x":
            _cols = ["template", "model", "prefix"]
            _types = ["l1_decoder", "l1_encoder", "cossim_decoder", "cossim_encoder"]
            canceled, col = list_values(_cols)
            folder = "/home/ahmad/share/comp/"
            if Path(folder).exists():
                shutil.rmtree(folder)
            Path(folder).mkdir(parents=True, exist_ok=True)
            files = []
            for _type in _types:
                g_list = ["template", "model", "prefix"]
                mm = main_df.groupby(g_list, as_index=False).first()
                g_list.remove(col)
                # mlog.info("g_list: %s", g_list)
                g_df = mm.groupby(g_list, as_index=False)
                sel_cols = [_type, "template", "model", "prefix"]
                #_agg = {}
                #for _g in g_list:
                #    _agg[_g] ="first"
                #_agg[col] = "count"
                #df = g_df.agg(_agg)
                if True:
                    gg = 1
                    total = len(g_df)
                    for g_name, _nn in g_df:
                        img = []
                        images = []
                        for i, row in _nn.iterrows():
                            if row[_type] is None:
                                continue
                            f_path = row[_type] 
                            if not Path(f_path).is_file(): 
                                continue
                            img.append(row[_type])
                            _image = Image.open(f_path)
                            draw = ImageDraw.Draw(_image)
                            draw.text((0, 0),str(i) +" "+ row[col] ,(20,25,255),font=font)
                            draw.text((0, 70),str(i) +" "+ g_name[0],(20,25,255),font=font)
                            draw.text((0, 140),str(i) +" "+ g_name[1],(20,25,255),font=font)
                            draw.text((250, 0),str(gg) + " of " + str(total),
                                    (20,25,255),font=font)
                            images.append(_image)
                        gg += 1
                        if images:
                            if char == "h":
                                new_im = combine_x(images)
                            else:
                                new_im = combine_y(images)
                            name = _type + "_".join(g_name) + "_" + row[col]
                            pname = os.path.join(folder, name + ".png")
                            new_im.save(pname)
                            files.append({"pname":pname, "name":name})
                if files:
                    df = pd.DataFrame(files, columns=["pname","name"])
                    sel_cols = ["name"]
                else:
                    show_msg("No select")
        elif char == "x" and prev_char == "b" and context == "":
            backit(df, sel_cols)
            df = sel_df
        # png files
        elif char == "l" and False: 
            df = main_df.groupby(["l1_decoder", "template", "model", "prefix"], as_index=False).first()
            sel_cols = ["l1_decoder", "template", "model", "prefix"]
        elif char == "z" and False: 
            fav_df = fav_df.append(df.iloc[sel_row])
            mbeep()
            fav_df.to_csv(fav_path, sep="\t", index=False)
        elif char == "Z" and prev_char == "x":
            main_df["m_score"] = main_df["rouge_score"]
            df = main_df
            hotkey = "CGR"
            #backit(df, sel_cols)
            #df = fav_df
        elif char == "j" and False:
            canceled, col = list_values(info_cols)
            if not canceled:
                pos = rowinput("pos:","")
                if pos:
                    info_cols.remove(col)
                    if int(pos) > 0:
                        info_cols.insert(int(pos), col)
                    else:
                        sel_cols.insert(0, col)
                    save_obj(info_cols, "info_cols", dfname)
                    save_obj(sel_cols, "sel_cols", dfname)
        elif char in "56789" and prev_char == "\\":
            cmd = "top@" + str(int(char)/10)
        elif char == "BB": 
            sel_rows = []
            for i in range(len(df)):
                sel_rows.append(i)
        elif char == "==": 
            col = sel_cols[cur_col]
            exp=df.iloc[sel_row][col]
            if col == "fid": col = FID
            if col == "fid":
                sel_fid = exp
            # mlog.info("%s == %s", col, exp)
            df = main_df[main_df[col] == exp]
            filter_df = df
            hotkey = hk
        elif char  == "a" and prev_char == "a": 
            col = sel_cols[cur_col]
            FID = col 
            extra["FID"] = FID
            df = filter_df
            hotkey=hk
        elif char == "A" and prev_char == "g":
            col = sel_cols[cur_col]
            FID = col 
            extra["FID"] = FID
            df = main_df
            hotkey=hk
        elif char == "AA":
            gdf = filter_df.groupby("input_text")
            rows = []
            for group_name, df_group in gdf:
                for row_index, row in df_group.iterrows():
                    pass
            arr = ["prefix","fid","query","input_text","template"]
            canceled, col = list_values(arr)
            if not canceled:
                FID = col 
                extra["FID"] = FID
                df = filter_df
                hotkey=hk
        elif is_enter(ch) and prev_char == "s": 
            sort = selected_cols + [col] 
            df = df.sort_values(by=sort, ascending=asc)
            selected_cols = []
            asc = not asc
        if char in ["m","s"] and context == "grouping":
            if not selected_cols:
                selected_cols = ["label","max_train_samples"]
            if char == "m":
                df = back_df
                df = df.groupby(selected_cols).mean(numeric_only=True).reset_index()
            elif char == "s":
                df = back_df
                df = df.groupby(selected_cols).std(numeric_only=True).reset_index()
            df = df.round(2)
        elif char in ["a"] and context != "grouping":
            backit(df, sel_cols)
            context = "grouping"
            shortkeys["grouping"] = {"m":"show mean","s":"show std"}
            cols = selected_cols.copy()
            scol = sel_cols[cur_col]
            if not selected_cols:
                cols = ["label", "max_train_samples"]
            if len(cols) > 0:
                if "All" in df:
                    df = df.groupby(cols, as_index=False)["All"].agg(['mean', 'std']).reset_index()
                else:
                    df = df.groupby(cols, as_index=False)["m_score"].agg(['mean', 'std']).reset_index()
                # df = df.groupby(cols, as_index=False).mean(numeric_only=True).reset_index()
                # df.iloc[:, -2:] = df.iloc[:, -2:] * 100
                df = df.round(2)
                sel_cols = list(df.columns)
                cond_colors["mean"] = score_colors
            selected_cols = []
            left = 0
        elif char in ["g", "u"]:
            context = "group_mode"
            if cur_col < len(sel_cols):
                col = sel_cols[cur_col]
                keep_uniques = char == "u"
                if col == group_col:
                    group_col = ""
                    cur_row = 0
                    sel_group = 0
                else:
                    group_col = col
                    cur_row = 0
                    sel_group = 0
                    if sort in df:
                        df = df.sort_values(by=[group_col, sort], ascending=[True, False])
        elif char == "A": 
            consts["options"] = "b: back"
            backit(df, sel_cols)
            if cur_col < 0:
                show_msg("Please select a column")
            else:
                col = sel_cols[cur_col]
                # df = df.groupby(col, as_index=False).mean(numeric_only=True).reset_index()
                if "m_score" in df:
                    df = df.groupby(col)["m_score"].agg(['mean', 'std']).reset_index()
                    df.columns = ['_'.join(col).strip() 
                            if isinstance(col, tuple) else col for col in df.columns]
                    # df = df.sort_values(by=["m_score"], ascending=False)
                elif "All" in df:
                    df = df.groupby(col)["All"].agg(['mean', 'std']).reset_index()
                    df.columns = ['_'.join(col).strip() 
                        if isinstance(col, tuple) else col for col in df.columns]
                    df = df.sort_values(by=["All"], ascending=False)
                df = df.round(2)
                sel_cols = list(df.columns)
                left = 0
                cond_colors["mean"] = score_colors
        elif char == "a" and False: 
            consts["options"] = "b: back"
            backit(df, sel_cols)
            col = sel_cols[cur_col]
            df = df.groupby([col]).mean(numeric_only=True).reset_index()
            df = df.round(2)
            if "m_score" in df:
                df = df.sort_values(by=["m_score"], ascending=False)
                sort = "m_score"
            elif "avg" in df:
                df = df.sort_values(by=["avg"], ascending=False)
                sort = "avg"
        elif char == "u": # and False:
            infos = calc_metrics(main_df)
            subwin(infos)
        elif char == "U" and prev_char == "x": 
            if sel_col:
                df = df[sel_col].value_counts(ascending=False).reset_index()
                sel_cols = list(df.columns)
                col_widths["index"]=50
                info_cols = []
        elif char == "C": 
            df = add_scores(df)
            extra["filter"].append("group predictions")
        elif char == " ":
            if sel_row in sel_rows:
                sel_rows.remove(sel_row)
            else:
                sel_rows.append(sel_row)
            sel_rows = sorted(sel_rows)
            adjust = False
        elif char == "#" and prev_char == "x": 
            if not sel_rows:
                tinfo=df.iloc[sel_row]["ftag"]
                infos = tinfo.split(",")
                infos.append(main_df.loc[0, "path"])
                subwin(infos)
            else:
                s1 = sel_rows[0]
                s2 = sel_rows[1]
                f1 = df.iloc[s1]["eid"]
                f2 = df.iloc[s2]["eid"]
                infos = []
                for col in main_df.columns:
                    if (
                        col == "ftag"
                        or col == "extra_fields"
                        or col == "path"
                        or col == "folder"
                        or col == "exp_name"
                        or col == "fid"
                        or col == "id"
                        or col == "eid"
                        or col == "full_tag"
                        or col.startswith("test_")
                        or "output_dir" in col
                    ):
                        continue

                    values_f1 = main_df[main_df["eid"] == f1][col]
                    values_f2 = main_df[main_df["eid"] == f2][col]

                    if (pd.notna(values_f1.iloc[0]) and pd.notna(values_f2.iloc[0]) 
                            and values_f1.iloc[0] != values_f2.iloc[0]):
                        if values_f1.iloc[0] != values_f2.iloc[0]:
                            infos.append(f"{col}: {values_f1.iloc[0]}")
                            infos.append(f"{col}: {values_f2.iloc[0]}")

                subwin(infos)                
        elif char == "z" and prev_char == "z":
            consts["context"] = context
            sel_cols =  load_obj("sel_cols", context, [])
            info_cols = load_obj("info_cols", context, [])
        elif char == "G":
            # backit(df, sel_cols)
            # ggggggggggggggggggggggg
            context = "main"
            if FID == "input_text":
                context = "inp2"
            left = 0
            if False: #reset:
                info_cols = ["bert_score", "num_preds"]
            if False: #col == "fid":
                sel_cols = ["eid", "rouge_score"] + tag_cols + ["method", "trial", "prefix","num_preds", "bert_score", "out_score", "pred_max_num","pred_max", "steps","max_acc","best_step", "st_score", "learning_rate", "vpredin",  "num_targets", "num_inps", "train_records", "train_records_nunique", "group_records", "wrap", "frozen", "prefixed"] 
                sel_cols = list(dict.fromkeys(sel_cols))
            reset = False
            group_sel_cols = sel_cols.copy()
            if "folder" in group_sel_cols:
                group_sel_cols.remove("folder")
            if "prefix" in group_sel_cols:
                group_sel_cols.remove("prefix")
                group_sel_cols.insert(0, "prefix")

            df = grouping(df, FID)
            # sel_cols = ["expname","eid","prefix","rouge_score","num_preds"]
            if not "num_preds" in sel_cols:
                sel_cols.append("num_preds")
            group_sel_cols = sel_cols.copy()
            group_df = df.copy()
            exp_num = df["folder"].nunique()
            consts["Number of experiments"] = str(exp_num)
            sort = "rouge_score"
        elif char == "z":
            backit(df, sel_cols)
            exprs, _ = get_sel_rows(df)
            sel_rows = []
            df = df[df['eid'].isin(exprs)]
        elif char == "z" and prev_char == "a": 
            if len(df) > 1:
                sel_cols, info_cols, tag_cols = remove_uniques(df, sel_cols, 
                        orig_tag_cols, keep_cols)
                unique_cols = info_cols.copy()
            info_cols_back = info_cols.copy()
            info_cols = []
            save_obj(sel_cols, "sel_cols", context)
            save_obj(info_cols, "info_cols", context)
        elif char == "M" and False:
            exp=df.iloc[sel_row]["eid"]
            cond = f"(main_df['eid'] == '{exp}')"
            tdf = main_df[main_df.eid == exp]
            path=tdf.iloc[0]["output_dir"]
            js = os.path.join(path, "exp.json")
            meld = ["meld", js]
            if "conf" in tdf:
                conf = tdf.iloc[0]["conf"]
                if not "/" in conf:
                    conf = os.path.join(home, "results", conf + ".json")
                meld.append(conf)
            subprocess.Popen(meld)
        elif char == "B":
            if "cfg" in df:
                _,files = get_sel_rows(df, row_id="cfg", col="cfg", from_main=False)
                files = [os.path.join(home, "results", c + ".json") for c in files]
                consts["base"] = files[0]
            else:
                _,dirs = get_sel_rows(df, col="output_dir")
                out_dir = dirs[0]
                exp_files = [os.path.join(d, "exp.json") for d in dirs]
                exp_file = exp_files[0]
                if "base" in consts:
                    base_file = consts["base"]
                    src = os.path.join(Path(base_file).parent, Path(base_file).stem)
                    dst = os.path.join(Path(exp_file).parent.parent, 
                            Path(base_file).stem + "_base")
                    shutil.copytree(src, dst)
                    mbeep()
                    #arr = ["meld", base_file, exp_file]
                    #subprocess.run(arr)
        elif char == "t":
            backit(df, sel_cols)
            mode = "cfg"
            files = glob(os.path.join(home,"results","*.json"))
            #for f in files:
            # correct_path(f)
            fnames = [Path(f).stem for f in files]
            rows = []
            for fname,_file in zip(fnames, files):
                ts = os.path.getmtime(_file)
                ctime = datetime.utcfromtimestamp(ts).strftime('%m-%d %H:%M:%S')
                parts = fname.split("@")
                rest = parts
                score = ""
                if len(parts) > 1:
                    rest = parts[0]
                    score = parts[1][:4]
                score = score.replace(".json","")
                score = float(score)
                method, cmm, ep, tn = rest.split("_")
                rows.append({"cfg":fname, "score":score, 
                    "method": method, "at":ctime, "cmm":cmm, "ep":ep, "tn":tn})
            df = pd.DataFrame(data=rows)
            sel_cols = df.columns
            df = df.sort_values("at", ascending=False)
            #with open(, 'r') as f:
            #    lines = f.readlines()
            #subwin(lines)                
        elif char == "M":
            s_rows = sel_rows
            if not sel_rows:
                s_rows = [sel_row]
            t_path = ""
            pfx = ""
            for s_row in s_rows:
                exp=df.iloc[s_row]["eid"]
                cond = f"(main_df['eid'] == '{exp}')"
                tdf = main_df[main_df.eid == exp]
                path=tdf.iloc[0]["output_dir"]
                if not t_path:
                    t_path = Path(path).parent
                folder_name = pfx +  Path(path).stem
                defpath = os.path.join(t_path, folder_name)
                new_path = rowinput("Copy To:", default=defpath)
                t_path = Path(new_path).parent
                new_folder_name = Path(new_path).stem
                pfx = new_folder_name.replace(Path(path).stem,"")
                if new_path:
                    parent = Path(path).parent
                    new_parent = Path(new_path).parent
                    pname = Path(path).parent.name
                    expid = Path(path).name
                    folders = glob(os.path.join(str(parent), "Eval-"+ str(expid) + "*"))
                    for folder in folders:
                        new_folder = Path(folder).stem
                        new_folder = new_folder.replace(str(expid),new_folder_name)
                        if Path(folder).exists():
                            copy_tree(folder, os.path.join(str(new_parent), new_folder))
                            # remove_tree(folder)
                    if Path(path).exists():
                        copy_tree(path, new_path)
                        # remove_tree(path)
                    mbeep()
            sel_rows = []
        elif char == "d":
            s_rows = sel_rows
            if not sel_rows:
                s_rows = [sel_row]
            df.drop(df.index[s_rows], inplace=True)
            sel_rows = []
        elif char == "D":
            ans = ""
            s_rows = sel_rows
            if not sel_rows:
                s_rows = [sel_row]
            irange = []
            for s_row in s_rows:
                exp=df.iloc[s_row]["eid"]
                cond = f"(main_df['eid'] == '{exp}')"
                tdf = main_df[main_df.eid == exp]
                path=tdf.iloc[0]["folder"]
                if ans != "a":
                    ans = rowinput("Delete? y) yes, a) all (" + path[:20] + "..." + path[-20:] + "):", default="y")
                if False: 
                    parent = Path(path).parent
                    pname = Path(path).parent.name
                    expid = Path(path).name
                    folders = glob(os.path.join(str(parent), "Eval-"+ str(expid) + "*"))
                    for folder in folders:
                        try:
                            shutil.rmtree(folder)
                        except:
                            show_msg("not exist")
                if not Path(path).exists():
                    show_msg("not exist")
                elif ans == "y" or ans == "a":
                    main_df.drop(main_df[main_df.folder == path].index, inplace=True)
                    irange.append(s_row)
                    remove_tree(path)
                    mbeep()
            df.drop(df.index[irange], inplace=True)
            sel_rows = []
            #df = df.loc[np.isin(np.arange(len(df)), irange)]
        elif char == "T" or char == "U" or char == "Y":
            s_rows = sel_rows
            if not sel_rows:
                s_rows = [sel_row]
            pfix = ""
            ignore_fname = False if char == "T" else True
            dest_folder = "comp"
            if char == "Y":
                # dest_folder = rowinput("Dest:")
                dest_folder = "comp2"
            temp_path = "/home/ahmad/temp/"
            comp_path = temp_path + dest_folder
            Path(comp_path).mkdir(parents=True, exist_ok=True)
            if char in ["U", "Y"]:
                if dest_folder and Path(comp_path).exists():
                    shutil.rmtree(comp_path)
                    cmd = f"sshpass -p 'a' ssh -t ahmad@10.42.0.210 'rm /home/ahmad/temp/{dest_folder}/*'"
                    os.system(cmd)
                cmd = f"sshpass -p 'a' ssh -t ahmad@10.42.0.210 'mkdir -p /home/ahmad/temp/{dest_folder}'"
                os.system(cmd)
                Path(comp_path).mkdir(parents=True, exist_ok=True)
            adrfile = os.path.join(temp_path, "address.txt")
            adr = open(adrfile,"w")
            for s_row in s_rows:
                exp=df.iloc[s_row]["eid"]
                score = ""
                if "rouge_score" in df:
                    score=df.iloc[s_row]["rouge_score"]
                elif "All" in df:
                    score=df.iloc[s_row]["All"]
                cond = f"(main_df['eid'] == '{exp}')"
                tdf = main_df[main_df.eid == exp]
                prefix=tdf.iloc[0]["expname"]
                expid=tdf.iloc[0]["expid"]
                # path=tdf.iloc[0]["output_dir"]
                rpath=tdf.iloc[0]["folder"]
                print(rpath, file=adr)
                if not str(expid).isnumeric():
                    expid=Path(rpath).stem
                path = str(Path(rpath).parent) # + "/" + str(expid)
                js = os.path.join(path,"conf_" + expid + ".json")
                fname = "exp"
                if char == "Y":
                    compose=tdf.iloc[0]["compose_method"]
                    epc=tdf.iloc[0]["num_train_epochs"]
                    tn=tdf.iloc[0]["max_train_samples"]
                    score = str(round(score,2)) if score else "noscore" 
                    fname = prefix + "-" + compose + "-" + str(epc) \
                        + "-" + str(tn) + "@" + score + "@.json"
                if not ignore_fname:
                    fname = rowinput("prefix:", default=fname)
                if fname:
                    parent = path 
                    pname = Path(path).name
                    expid = Path(path).name
                    if char in ["U","Y"]:
                        dest = os.path.join(comp_path, "exp_" + str(prefix) + ".json")
                    elif "conf" in fname:
                        dest = os.path.join(home, "confs", fname)
                    elif "reval" in fname:
                        dest = os.path.join(home, "reval", fname)
                    else:
                        folders = glob(os.path.join(str(parent), "Eval-"+ str(expid) + "*"))
                        results_folder = os.path.join(home,"results",
                                fname.replace(".json",""))
                        for folder in folders:
                            try:
                                shutil.copytree(folder, 
                                        results_folder + "/" + Path(folder).name)
                            except FileExistsError:
                                pass
                        dest = os.path.join(home, "results", fname)

                    if Path(js).is_file():
                        shutil.copyfile(js, dest)
                        correct_path(js, dest, path)
                        mbeep()
                        if char in ["U", "Y"]:
                            to = "ahmad@10.42.0.210:" + dest 
                            cmd = f'sshpass -p "a" rsync -P -ae "ssh" -zarv "{js}" "{to}"'
                            # cmd = f"sshpass -p 'a' ssh -t ahmad@10.42.0.210 'cp {js} {dest}"
                            os.system(cmd)
                    # subprocess.run(cmd.split())
            adr.close()
            if char in ["U", "Y"]:
                to = "ahmad@10.42.0.210:" + temp_path 
                cmd = f'sshpass -p "a" rsync -P -ae "ssh" -zarv "{adrfile}" "{to}"'
                os.system(cmd)
            std.clear()
            hotkey = "]"
        elif char == "p" and False:
            pivot_cols = sel_cols[cur_col]
            consts["pivot col"] = pivot_cols
        elif char == "K":
            folder_path = "/home/ahmad/pictemp"
            files = os.listdir(folder_path)
            for file in files:
                file_path = os.path.join(folder_path, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    else:
                        show_msg(f"Skipping {file_path} as it is not a file.")
                except Exception as e:
                    show_msg(f"Error while deleting {file_path}: {e}") 
            show_msg("Done!")
            mbeep()
        elif char == "U" and False:
            left = 0
            backit(df, sel_cols)

            s_rows = sel_rows
            if not sel_rows:
                s_rows = [sel_row]
            cond = ""
            for s_row in s_rows:
                exp=df.iloc[s_row]["fid"]
                cond += f"| (main_df['{FID}'] == '{exp}') "
            cond = cond.strip("|")
            filter_df = main_df[eval(cond)]
            df = filter_df.copy()
            sel_rows = []
            FID = "input_text"
            hotkey = hk
        elif char == "e" and context != "notes":
            if sort != "time":
                df = df.sort_values(by="time", ascending=False)
                sort = "time"
            elif "All" in df:
                df = df.sort_values(by="All", ascending=False)
                sort = "All"
        elif (char == "i" or char == "j") and context == "pivot": 
            backit(df, sel_cols)
            context = "prefix"
            if char == "j":
                col = sel_cols[cur_col]
            s_rows = sel_rows
            if not sel_rows: s_rows = [sel_row]
            dfs = []
            for s_row in s_rows:
                sel_exp=df.iloc[s_row]["eid"]
                if char == "j":
                    tdf = group_df[(group_df['eid'] == sel_exp) 
                            & (group_df["prefix"] == col)]
                else:
                    tdf = group_df[(group_df['eid'] == sel_exp)] 
                dfs.append(tdf)
            df = pd.concat(dfs, ignore_index=True)
            sel_cols = index_cols + ["fid"] + group_sel_cols.copy()
            df = df.sort_values(by=["prefix",score_cols[0]], ascending=False)
            left = 0
            # sel_rows = range(len(df))
            char = ""
            if char == "j":
                char = "i"
        if char in ["n", "i"] and "fid" in df: # and prev_cahr != "x" and hk == "gG":
            backit(df, sel_cols)
            left = 0
            context= "comp"
            cur_col = -1
            sel_group = 0
            show_infos = False
            s_rows = sel_rows
            if not sel_rows:
                s_rows = group_rows
                if not group_rows:
                    s_rows = [sel_row]
            sel_rows = sorted(sel_rows)
            if sel_rows:
                sel_row = sel_rows[-1]
            sel_rows = []
            on_col_list = ["pred_text1"]
            other_col = "target_text"
            if char =="i": 
                pass
            group_col = "input_text"
            on_col_list = ["input_text"] 
            other_col = "pred_text1"
            if char =="t": 
                on_col_list = ["target_text"] 
                other_col = "pred_text1"
            on_col_list.extend(["prefix"])
            g_cols = []
            _rows = s_rows
            if char == "n":
                dfs = []
                all_rows = range(len(df))
                for r1 in all_rows:
                    for r2 in all_rows:
                        if r2 > r1:
                            _rows = [r1, r2]
                            _df, sel_exp, _dfs = find_common(df, filter_df, on_col_list, _rows, FID, char, tag_cols = index_cols)
                            dfs.append(_df)
                df = pd.concat(dfs,ignore_index=True)
                #df = df.sort_values(by="int", ascending=False)
            elif len(s_rows) > 1:
                sel_cols=orig_tag_cols + ["num_preds","eid","bert_score", "out_score","pred_text1","target_text","input_text","vpred","vtarget","rouge_score","prefix"]
                sel_cols, info_cols, tag_cols = remove_uniques(df, sel_cols, orig_tag_cols)
                unique_cols = info_cols.copy()
                sel_cols = list(dict.fromkeys(sel_cols))
                _cols = tag_cols + index_cols + sel_cols + rep_cols + info_cols
                _cols = list(dict.fromkeys(_cols))
                df, sel_exp, dfs = find_common(df, main_df, on_col_list, _rows, 
                                               FID, char, _cols)
                df = pd.concat(dfs).sort_index(kind='mergesort')
                _all = len(df)
                cdf=df.sort_values(by='input_text').drop_duplicates(subset=['input_text', 'pred_text1'], keep="first")
                _common = _all - len(cdf)
                consts["Common"] = str(_common) + "| {:.2f}".format(_common / _all)
            else:
                #path = df.iloc[sel_row]["path"]
                #path = Path(path)
                #_selpath = os.path.join(path.parent, "sel_" + path.name) 
                #shutil.copyfile(path, _selpath)
                exp=df.iloc[sel_row]["fid"]
                sel_exp = exp
                #FID="expid"
                cond = f"(main_df['{FID}'] == '{exp}')"
                df = main_df[main_df[FID] == exp]
                if "prefix" in df:
                    task = df.iloc[0]["prefix"]
                sel_cols=orig_tag_cols + ["num_preds","prefix","bert_score", "out_score","pred_text1","top_pred", "vpred", "query", "resp", "vtarget", "top", "target_text","input_text","rouge_score","prefix"]
                sel_cols, info_cols, tag_cols = remove_uniques(df, sel_cols, 
                        main_vars, keep_cols=["pred_text1"])
                #unique_cols = info_cols.copy()
                sel_cols = list(dict.fromkeys(sel_cols))
                # df = df[sel_cols]
                df = df.sort_values(by="input_text", ascending=False)
                sort = "input_text"
                info_cols = ["query", "resp", "prefix"]
                df = df.reset_index()
            if len(df) > 1:
                sel_cols=orig_tag_cols + ["eid","prefix", "bert_score","pred_text1", "target_text", "top_pred", "input_text", "vpred", "vtarget", "rouge_score"]
                ii = 0
                for col in index_cols:
                    if col in sel_cols:
                        sel_cols.remove(col)
                    sel_cols.insert(ii, col)
                    ii += 1
                sel_cols, info_cols, tag_cols = remove_uniques(df, sel_cols, 
                        main_vars, keep_cols=["fid", "prefix", "pred_text1"])
                if "pred_text1" in sel_cols:
                    sel_cols.remove("pred_text1")
                ii = 0
                _sort = []
                #for col in index_cols:
                #    if col in sel_cols:
                #       _sort.append(col)
                #        ii += 1
                sel_cols.insert(ii, "prefix")
                sel_cols.insert(ii + 1, "pred_text1")
                sel_cols = list(dict.fromkeys(sel_cols))
                df = df.sort_values(by=["input_text","prefix"]+_sort, ascending=False)
                unique_cols = info_cols.copy()
                cond_colors["pred_text1"] = pred_colors
                info_cols_back = info_cols.copy()
                info_cols = []

        elif char == "M" and prev_char == "l":
            left = 0
            if sel_exp and on_col_list:
                backit(df, sel_cols)
                _col = on_col_list[0]
                _item=df.iloc[sel_row][_col]
                sel_row = 0
                if sel_fid:
                    df = main_df[(main_df["fid"] == sel_fid) & (main_df[FID] == sel_exp) & (main_df[_col] == _item)]
                else:
                    df = main_df[(main_df[FID] == sel_exp) & (main_df[_col] == _item)]
                sel_cols = ["fid","input_text","pred_text1","target_text","bert_score", "hscore", "rouge_score", "prefix"]
                df = df[sel_cols]
                df = df.sort_values(by="bert_score", ascending=False)
        elif char == "D" and False: 
            s_rows = sel_rows
            if FID == "fid":
                mdf = main_df.groupby("fid", as_index=False).first()
                mdf = mdf.copy()
                _sels = df["fid"]
                for s_row, row in mdf.iterrows():
                    exp=row["fid"]
                    if char == "d":
                        cond = main_df['fid'].isin(_sels) 
                    else:
                        cond = ~main_df['fid'].isin(_sels) 
                    tdf = main_df[cond]
                    if  ch == cur.KEY_SDC:
                        spath = row["path"]
                        os.remove(spath)
                    main_df = main_df.drop(main_df[cond].index)
                df = main_df
                filter_df = main_df
                sel_rows = []
                hotkey = hk
        elif char == "D" and prev_char == "x":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                del main_df[col]
                char = "SS"
                if col in df:
                    del df[col]
        elif char == "o" and prev_char == "x":
            if "pname" in df:
                pname = df.iloc[sel_row]["pname"]
            elif "l1_encoder" in df:
                if not sel_rows: sel_rows = [sel_row]
                sel_rows = sorted(sel_rows)
                pnames = []
                for s_row in sel_rows:
                    pname1 = df.iloc[s_row]["l1_encoder"]
                    pname2 = df.iloc[s_row]["l1_decoder"]
                    pname3 = df.iloc[s_row]["cossim_encoder"]
                    pname4 = df.iloc[s_row]["cossim_decoder"]
                    images = [Image.open(_f) for _f in [pname1, pname2,pname3, pname4]]
                    new_im = combine_y(images)
                    name = "temp_" + str(s_row) 
                    folder = os.path.join(base_dir, "images")
                    Path(folder).mkdir(exist_ok=True, parents=True)
                    pname = os.path.join(folder, name + ".png")
                    draw = ImageDraw.Draw(new_im)
                    draw.text((0, 0), str(s_row) + "  " + df.iloc[s_row]["template"] +  
                                     " " + df.iloc[s_row]["model"] ,(20,25,255),font=font)
                    new_im.save(pname)
                    pnames.append(pname)
                if len(pnames) == 1:
                    pname = pnames[0]
                    sel_rows = []
                else:
                    images = [Image.open(_f) for _f in pnames]
                    new_im = combine_x(images)
                    name = "temp" 
                    folder = os.path.join(base_dir, "images")
                    Path(folder).mkdir(exist_ok=True, parents=True)
                    pname = os.path.join(folder, name + ".png")
                    new_im.save(pname)
            if "ahmad" in home:
                subprocess.run(["eog", pname])
        elif char in ["o","O"] and prev_char=="x":
            files = [Path(f).stem for f in glob(base_dir+"/*.tsv")]
            for i,f in enumerate(files):
                if f in open_dfnames:
                    files[i] = "** " + f

            canceled, _file = list_values(files)
            if not canceled:
                open_dfnames.append(_file)
                _file = os.path.join(base_dir, _file + ".tsv")
                extra["files"] = open_dfnames
                new_df = pd.read_table(_file)
                if char == "o":
                    df = pd.concat([df, new_df], ignore_index=True)
                else:
                    main_df = pd.concat([main_df, new_df], ignore_index=True)
        elif char == "t" and prev_char=="x":
            cols = get_cols(df,5)
            if cols:
                tdf = df[cols].round(2)
                tdf = tdf.pivot(index=cols[0], columns=cols[1], values =cols[2]) 
                fname = rowinput("Table name:", "table_")
                if fname:
                    if char == "t":
                        tname = os.path.join(base_dir, "plots", fname + ".png")
                        wrate = [col_widths[c] for c in cols]
                        tax = render_mpl_table(tdf, wrate = wrate, col_width=4.0)
                        fig = tax.get_figure()
                        fig.savefig(tname)
                    else:
                        latex = tdf.to_latex(index=False)
                        tname = os.path.join(base_dir, "latex", fname + ".tex")
                        with open(tname, "w") as f:
                            f.write(latex)

        elif char == "P":
            fig, ax = plt.subplots()
            #cols = selected_cols 
            #if cols:
            #    df = df.sort_values(cols[1])
            #    x = cols[0]
            #    y = cols[1]
                #ax = df.plot.scatter(ax=ax, x=x, y=y)
            #    ax = sns.regplot(df[x],df[y])
        elif (is_enter(ch) or char == "x") and prev_char == ".":
            backit(df, sel_cols)
            if char == "x":
                consts["sel"] += " MAX"
                score_agg = "max"
            else:
                consts["sel"] += " MEAN"
                score_agg = "mean"
            _agg = {}
            for c in df.columns:
                if c.endswith("score"):
                    _agg[c] = score_agg
                else:
                    _agg[c] = "first"
            df = df.groupby(list(dot_cols.keys())).agg(_agg).reset_index(drop=True)
        elif is_enter(ch) and prev_char == "=":
           backit(df, sel_cols)
           df_conds.sort(key=lambda tup: tup[0])
           df_cond = True
           prev_col = -1
           for col, cond in df_conds:
               if col == prev_col: 
                   df_cond = df_cond | cond
               else:
                   df_cond = df_cond & cond
               prev_col = col
           df = df[df_cond]
           df_conds = [] 
           group_col = ""
           keep_uniques = False
        elif is_enter(ch) or char in ["f", "F"]:
            if (is_enter(ch) or char == "f") and is_filtered:
               df = back_df
            else:
                backit(df, sel_cols)
                cond_set = {}
            is_filtered = True
            col = sel_cols[cur_col]
            if col == "fid": col = FID
            canceled, col, val = list_df_values(df, col, get_val=True)
            if not canceled:
               if type(val) == str:
                  if not col in cond_set or is_enter(ch):
                      cond_set[col] = f"(df['{col}'] == '{val}')"
                  else:
                      cond_set[col] += f" | (df['{col}'] == '{val}')"
               else:
                  if not col in cond_set or is_enter(ch):
                      cond_set[col] = f"(df['{col}'] == {val})"
                  else:
                      cond_set[col] += f" | (df['{col}'] == {val})"
            cond = ""
            for c,v in cond_set.items():
                cond += "(" + v + ") & "
            cond = cond.strip(" & ")
            # mlog.info("cond %s, ", cond)
            if cond:
               df = df[eval(cond)]
               #df = df.reset_index()
               filter_df = df
               if not "filter" in extra:
                  extra["filter"] = []
               extra["filter"].append(cond)
               sel_row = 0
               keep_cols.append(col)
            if len(df) > 1:
                sel_cols, info_cols_back, tag_cols = remove_uniques(df, sel_cols, 
                        keep_cols=keep_cols + pivot_cols + info_cols + pcols)
        if char == "V":
            start = sel_rows[-1] if sel_rows else 0
            end = sel_row if sel_rows else len(df) - 1
            sel_rows = range(start, end + 1)
        if char == "V" and False:
            backit(df, sel_cols)
            sel_col = sel_cols[cur_col]
            cond = True 
            for col in orig_tag_cols:
                if not col == sel_col and col in main_df:
                    val=df.iloc[sel_row][col]
                    cond = cond & (main_df[col] == val)
            filter_df = main_df
            df = main_df[cond]
            hotkey = hk
        elif char == "r" and prev_char == "x":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                new_name = rowinput(f"Rename {col}:")
                main_df = main_df.rename(columns={col:new_name})
                char = "SS"
                if col in df:
                    df = df.rename(columns={col:new_name})



        elif char in ["d"] and prev_char == "x":
            canceled, col, val = list_df_values(main_df)
            if not canceled:
                main_df = main_df.drop(main_df[main_df[col] == val].index)
                char = "SS"
                info_cols = []
                if col in df:
                    df = df.drop(df[df[col] == val].index)
        elif (ch == cur.KEY_DC and context != "notes"): 
            col = sel_cols[cur_col]
            if col in orig_tag_cols:
                orig_tag_cols.remove(col)
            if col in tag_cols:
                tag_cols.remove(col)
            sel_cols.remove(col)
            save_obj(sel_cols, "sel_cols", context)
        elif ch == cur.KEY_DC and context == "notes":
            df = df.drop(df.iloc[sel_row].name)
            doc_dir = "/home/ahmad/findings" #os.getcwd() 
            note_file = os.path.join(doc_dir, "notes.csv")
            df.to_csv(note_file, index=False)
        elif ch == cur.KEY_SDC:
            #col = sel_cols[cur_col]
            #sel_cols.remove(col)
            show_infos = not show_infos 
            if info_cols:
                col = info_cols.pop()
            #save_obj(sel_cols, "sel_cols", context)
            save_obj(info_cols, "info_cols", context)
        elif ch == cur.KEY_SDC and prev_char == 'x':
            col = sel_cols[0]
            val = sel_dict[col]
            cmd = rowinput("Are you sure you want to delete {} == {} ".format(col,val))
            if cmd == "y":
                main_df = main_df.drop(main_df[main_df[col] == val].index)
                char = "SS"
                info_cols = []
                if col in df:
                    df = df.drop(df[df[col] == val].index)
        elif char == "x":
            sel_rows = []
            selected_cols = []
            dot_cols = {}
            keep_cols = []
            consts = {}
            visual_mode = False
        elif char == "v":
            _col_name = sel_cols[cur_col]
            if not _col_name in selected_cols:
                selected_cols.append(_col_name)
                visual_mode = True
            else:
                visual_mode = False
            if prev_char == "x":
                info_cols = ["bert_score", "num_preds"]
            if prev_char == "x": 
                sel_cols = ["eid", "rouge_score"] + tag_cols + ["method", "trial", "prefix","num_preds", "bert_score", "pred_max_num","pred_max", "steps","max_acc","best_step", "st_score", "learning_rate",  "num_targets", "num_inps", "train_records", "train_records_nunique", "group_records", "wrap", "frozen", "prefixed"] 
                save_obj(sel_cols, "sel_cols", context)
        elif char == "M" and prev_char == "x":
            info_cols = []
            for col in df.columns:
                info_cols.append(col)
        elif char == "m" and "cfg" in df:
            char = ""
            _,files = get_sel_rows(df, row_id="cfg", col="cfg", from_main=False)
            files = [os.path.join(home, "results", c + ".json") for c in files]
            files.insert(0, "meld")
            subprocess.Popen(files)
        elif char == "m":
            exprs,dirs = get_sel_rows(df, row_id="expid", col="folder")
            files = []
            for expid, rpath in zip(exprs, dirs):
                path = str(Path(rpath).parent) + "/" + str(expid)
                js = os.path.join(path, "exp.json")
                files.append(js)
            files.insert(0, "meld")
            subprocess.Popen(files)
        elif char == "m" and prev_char == "x":
            info_cols = []
            sel_cols = []
            cond = get_cond(df, "model", 2)
            df = main_df[eval(cond)]
            if df.duplicated(['qid','model']).any():
                show_err("There is duplicated rows for qid and model")
                char = "r"
            else:
                df = df.set_index(['qid','model'])[['pred_text1', 'input_text','prefix']].unstack()
                df.columns = list(map("_".join, df.columns))
        elif is_enter(ch) and prev_char == "x":
            col = sel_cols[0]
            val = sel_dict[col]
            if not "filter" in extra:
                extra["filter"] = []
            extra["filter"].append("{} == {}".format(col,val))
            df = filter_df[filter_df[col] == val]
            df = df.reset_index()
            if char == "F":
                sel_cols = order(sel_cols, [col])
            sel_row = 0
            filter_df = df
        elif char == "w" and prev_cahr == "x":
            sel_rows = []
            adjust = True
            tdf = main_df[main_df['fid'] == sel_exp]
            spath = tdf.iloc[0]["path"]
            tdf.to_csv(spath, sep="\t", index=False)
        elif char == "/":
            old_search = search
            search = rowinput("/", search)
            if search and search == old_search:
                si += 1
            else:
                si = 0
            if search:
                mask = np.column_stack([df[col].astype(str).str.contains(search, na=False) for col in df])
                si = min(si, len(mask) - 1)
                sel_row = df.loc[mask.any(axis=1)].index[si]
        elif char == ":":
            cmd = rowinput() #default=prev_cmd)
        elif char == "q":
            save_df(df)
            # if prev_char != "q": mbeep()
            consts["exit"] = "hit q another time to exit"
            prev_char = "q" # temporary line for exit on one key  #comment me
        if cmd.startswith("cp="):
            _, folder, dest = cmd.split("=")
            spath = main_df.iloc[0]["path"]
            dest = os.path.join(home, "logs", folder, dest)
            Path(folder).mkdir(exist_ok=True, parents=True)
            shutil.copyfile(spath, dest)
        if cmd.startswith("w="):
            _,val = cmd.split("=")[1]
            col = sel_cols[cur_col]
            col_widths[col] = int(val)
            adjust = False
        if cmd.startswith("cc"):
            name = cmd.split("=")[-1]
            if not name in rep_cmp:
                rep_cmp[name] = {}
            exp=df.iloc[sel_row]["eid"]
            tdf = main_df[main_df['eid'] == exp]
            _agg = {}
            for c in sel_cols:
                if c in df.columns: 
                    if c.endswith("score"):
                        _agg[c] = "mean"
                    else:
                        _agg[c] = "first"
            gdf = tdf.groupby(["prefix"], as_index=False).agg(_agg).reset_index(drop=True)
            all_rels = gdf['prefix'].unique()
            for rel in all_rels: 
                cond = (gdf['prefix'] == rel)
                val = gdf.loc[cond, "m_score"].iloc[0]
                val = "{:.2f}".format(val)
                if not rel in rep_cmp[name]:
                    rep_cmp[name][rel] = []
                rep_cmp[name][rel].append(val)
            save_obj(rep_cmp, "rep_cmp", "gtasks")
            char = "r"
        if cmd.startswith("conv"):
            to = ""
            if "@" in cmd:
                to = cmd.split("@")[1]
            col = sel_cols[cur_col]
            if to == "num":
                df[col] = df[col].astype(float)
        if cmd.startswith("="):
            label = cmd[1:]
            s_rows = sel_rows
            if not s_rows: s_rows = [sel_row]
            col = sel_cols[cur_col]
            for s_row in s_rows:
                df.iloc[s_row, df.columns.get_loc(col)] = label
                exp=df.iloc[s_row]["eid"]
                cond = f"(main_df['eid'] == '{exp}')"
                tdf = main_df[main_df.eid == exp]
                path=tdf.iloc[0]["path"]
                if Path(path).is_file():
                    tdf = pd.read_table(path)
                    tdf[col] = label
                    tdf.to_csv(path, sep="\t", index=False)
        if char == "X":
           backit(df, sel_cols)
           exprs = []
           scores = []
           for prefix in pcols:
                exps, scs = get_sel_rows(df, col=prefix, from_main=False) 
                exprs.extend(exps)
                scores.extend(scs)
           dfs = []
           _cols = ["prefix","target_text", "m_score"]
           for eid, pfx in zip(exprs, pcols): 
                tdf = main_df.loc[(main_df.eid == eid) & (main_df.prefix == pfx), _cols]
                gdf = tdf.groupby("target_text", as_index=False).first()
                accuracy_values = []
                for target_text in gdf['target_text']:
                    pred_text1_matches = main_df.loc[(main_df.eid == eid) & (main_df.prefix == pfx) & (main_df.target_text == target_text) & (main_df.target_text == main_df.pred_text1)]
                    total_matches = len(main_df.loc[(main_df.eid == eid) & (main_df.prefix == pfx) & (main_df.target_text == target_text)])
                    accuracy = len(pred_text1_matches) # / total_matches if total_matches != 0 else 0
                    accuracy_values.append(accuracy)

                gdf['accuracy'] = accuracy_values
                dfs.append(gdf.round(1))
           df = pd.concat(dfs, ignore_index=True)
           sel_cols = df.columns
           group_col = "prefix"
        if cmd.startswith("cross"):
            backit(df, sel_cols)
            eid = df.iloc[sel_row]['eid'] 
            if "prefix" in df:
                _, scores = get_sel_rows(df, None, col="rouge_score", from_main=False) 
                _, prefixes = get_sel_rows(df, None, col="prefix", from_main=False) 
                exprs = [eid] * len(prefixes)
                if "mask_type" in df:
                    _, mask_types = get_sel_rows(df, None, col="mask_type", from_main=False) 
                if "label" in df:
                   _, labels = get_sel_rows(df, None, col="label", from_main=False) 
                else:
                    labels = ["x"] * len(exprs)
            else:
                prefix = sel_cols[cur_col]
                exprs, scores = get_sel_rows(df, col=prefix, from_main=False) 
                prefixes = [prefix]*len(exprs)
                if "mask_type" in df:
                    _, mask_types = get_sel_rows(df, col="mask_type", from_main=False) 
                else:
                    mask_types = exprs.copy()
                _, labels = get_sel_rows(df, col="label", from_main=False) 


            _cols = ["pred_text1", "target_text"]
            dfs = []
            for eid, acc, mt, label, prefix in zip(exprs, scores, mask_types, labels, prefixes):
                tdf = main_df.loc[(main_df.eid == eid) & (main_df.prefix == prefix), _cols]
                canceled, val = False, "pred_text1" # list_values(sel_cols)
                if not canceled:
                    treatment = 'target_text' #sel_cols[cur_col]
                    tdf = pd.crosstab(tdf[val], tdf[treatment], margins=True)
                tdf["preds"] = list(tdf.axes[0])
                count_columns = tdf.columns[tdf.columns != 'preds']
                tdf['first_word'] = tdf['preds'].str.split().str[0]
                tdf['group'] = tdf['first_word'].str[:5]
                gdf = tdf.groupby('group').agg({
                    'preds': lambda x: x.head(1),  # Preserve the first 'text' value in each group
                    **{col: 'sum' for col in count_columns}  # Sum up the count columns
                }).reset_index()

                pr = {}
                B = {}
                for col in count_columns:
                    gdf[col] = gdf[col].astype(int)
                    for idx, row in gdf.iterrows():
                        if row['preds'] == "All":
                            B[col] = row[col]
                        elif row['preds'].strip() == col:
                            per = round(row[col] / B[col],2)
                            if per is not None:
                                per *= 100
                                per = str(per) + "\%"
                            pr[col] = per

                pr["preds"] = "precision"
                gdf.loc[len(gdf)] = pr
                gdf = gdf.drop(index=0)

                gdf["label"] = mt
                gdf["acc"] = acc

                #precision_recall_df = gdf.apply(calculate_precision_recall, axis=1)
                #gdf = pd.concat([gdf, precision_recall_df], axis=1)

                gdf["prefix"] = prefix 
                gdf["eid"] = eid
                #tdf["exp"] = label 
                #tdf["uid"] = str(mt) + " " + str(label) + " " + str(eid)
                dfs.append(gdf)
            df = pd.concat(dfs, ignore_index=True)
            all_sel_cols = ["preds"] + list(df.columns)
            sel_cols = all_sel_cols[:20] 
            sel_cols.remove("prefix")
            sel_cols.remove("eid")
            sel_cols.remove("group")
            sel_cols.remove("All")
            for col in sel_cols:
               col_widths[col] = len(col) + 2
            #adjust = False
            left = 0
            sel_rows= []
            context = "cross"
            group_col = "label"
        if cmd == "mline" or (char == "h" and context == "pivot"):
             try:
                 cur_col_name = sel_cols[cur_col]
                 if len(selected_cols) == 0:
                     selected_cols = [sel_cols[cur_col]]
                 if len(selected_cols) == 1 and cur_col_name not in selected_cols:
                     selected_cols.append(cur_col_name)
                 if len(selected_cols) == 2:
                     xcol = selected_cols[0]
                     ycol = selected_cols[1]
                     df.plot.line(x=xcol, y=ycol, label="my")
                 if len(selected_cols) == 3:
                     gcol = selected_cols[0]
                     xcol = selected_cols[1]
                     ycol = selected_cols[2]
                     # df.set_index(xcol).groupby(gcol)[ycol].plot()
                     sns.lineplot(x=xcol,y=ycol, hue=gcol, data=df)
                 elif len(selected_cols) > 0:
                     tdf = df[selected_cols]
                     tdf.plot.line(subplots=True)
                 # Customize axis labels
                 plt.xlabel('X Axis Label')
                 plt.ylabel('Y Axis Label')
                 plt.show()
             except Exception as e:
                 show_msg("Error:" + repr(e))
                 mbeep()
        if cmd.startswith(">"):
            regkey = cmd[1:]
            df["key"] = regkey 
            register[regkey] = df
            consts["Register"] = str(register)
        if cmd == "clear":
            register = {}
        if cmd in ['nplot','eplot','plot', 'cplot','mplot']:
            #yyyyyyyy
           backit(df, sel_cols)
           fig, ax = plt.subplots()
           cols = selected_cols 
           scol = sel_cols[cur_col]
           if not cols:
               cols = ["label",scol, "All"]
           if len(cols) < 3:
               cols.append("All")
           if cmd in ['eplot']:
               cols = ["label","num_train_epochs","All"] 
           elif cmd == 'nplot':
               cols = ["label","max_train_samples","All"] 
           if cols:
               gcol = cols[0]
               xcol = cols[1]
               ycol = cols[2]
               gi = 0 
               name = ""
               labels = {}
               if cmd == "mplot":
                  labels = {'MSUM': 0, 'SSUM': 1, 'MCAT': 2}
               df[xcol] = df[xcol].astype(float)
               for key, grp in df.groupby([gcol]):
                     _label = key[0] if type(key) == tuple else key
                     label = _label
                     if label in labels:
                         gi = labels[label]
                     if cmd == 'cplot' and not label in labels:
                         label = rowinput("Order:Label [" + _label + "]:")
                         if not label:
                             label=_label
                         if ":" in label:
                             gi, label = label.split(":")
                         gi = int(gi)
                         labels[label] = gi 
                     ax = grp.sort_values(xcol).plot.line(x=xcol, y=ycol, ax=ax,
                             linestyle="--",marker="o", lw=3, 
                             label=label, color=colors[gi])
                     gi += 1
                     if gi > len(colors) - 1: gi = 0
                     name += "-".join([str(k) for k in key]) + "_"
               if labels:
                   desired_order = dict(sorted(labels.items(), key=lambda item: item[1])).keys() 
                   handles, labels = ax.get_legend_handles_labels()
                   ordered_handles=[handles[labels.index(label)] for label in desired_order]
                   ordered_labels = desired_order
                   ax.legend(ordered_handles, ordered_labels, fontsize=20)
               xmax = df[xcol].max() 
               ax.set_xlim([0, xmax]) 
               ax.set_xbound(lower=0.0, upper=xmax)
               ax.set_xticks(df[xcol].unique())
               if cmd == 'mplot': 
                  cmd = 'cplot'
               if cmd == 'cplot':
                  name = rowinput("Title:", name)
               if cmd == 'cplot':
                  xlabel = rowinput("X Label:")
               if cmd == 'cplot':
                  xlabel = rowinput("Y Label:")

               # Retrieve current labels
               xlabel = ax.get_xlabel()
               ylabel = ax.get_ylabel()
               if not ylabel: ylabel = "Accuracy"
               ax.set_ylabel(ylabel, fontsize=18)
               ax.set_xlabel(xlabel, fontsize=18)

               ax.set_title(ylabel + ' vs. ' + xlabel, fontsize=16)
               ax.legend(fontsize=20)  # You can set this to any desired font size

               plt.show()
               # char = "H"
        if char == "H":
            name = ax.get_title()
            pname = rowinput("Plot name:", name[:30])
            pics_dir = "/home/ahmad/Documents/Papers/Applied_Int_paper/pics" #os.getcwd() 
            if pname:
                folder = ""
                if "/" in pname:
                    folder, pname = pname.split("/")
                if folder:
                    folder = os.path.join(pics_dir, "plots", folder)
                else:
                    folder = os.path.join(pics_dir, "plots")
                Path(folder).mkdir(exist_ok=True, parents=True)
                pname = pname.replace(" ", "_")
                pname = os.path.join(folder, pname +  ".png")
                fig = ax.get_figure()
                fig.savefig(pname)
                ax = None
                subprocess.run(["eog", pname])
        if cmd.startswith("anova"):
            to = ""
            pics_dir = "/home/ahmad/Documents/Papers/Applied_Int_paper/pics" #os.getcwd() 
            canceled, val = False, "pred_text1" # list_values(sel_cols)
            if not canceled:
                treatment = 'target_text' #sel_cols[cur_col]
                df[val] = df[val].astype(float)
                ax = sns.boxplot(x=treatment, y=val, data=df, color='#99c2a2')
                ax = sns.swarmplot(x=treatment, y=val, data=df, color='#7d0013')
                ax.set_xlabel("Class")
                ax.set_ylabel("Prediction")
                title = df.iloc[0]["prefix"]
                # Ordinary Least Squares (OLS) model
                model = ols(f'{val} ~ C({treatment})', data=df).fit()
                backit(df, sel_cols)
                sel_cols = ["sum_sq","df","F","PR(>F)"]
                df = sm.stats.anova_lm(model, typ=2)
                pval = "{:.2f}".format(df.iloc[0]["PR(>F)"])
                df = df.round(2)
                fval = df.iloc[0]["F"]
                ax.set_title(title + "   F-Value:" + str(fval) + "  P-Value:" + str(pval))
                plt.show()
        if cmd.startswith("banova"):
            to = ""
            pics_dir = "/home/ahmad/Documents/Papers/Applied_Int_paper/pics" #os.getcwd() 
            canceled, val = False, "target_text" # list_values(sel_cols)
            if not canceled:
                treatment = 'pred_text1' #sel_cols[cur_col]
                df[val] = df[val].astype(float)
                title = df.iloc[0]["prefix"]
                ax = sns.boxplot(x=treatment, y=val, data=df, color='#99c2a2')
                ax = sns.swarmplot(x=treatment, y=val, data=df, color='#7d0013')
                ax.set_xlabel("Prediction")
                ax.set_ylabel("Class")
                # Ordinary Least Squares (OLS) model
                model = ols(f'{val} ~ C({treatment})', data=df).fit()
                backit(df, sel_cols)
                sel_cols = ["sum_sq","df","F","PR(>F)"]
                df = sm.stats.anova_lm(model, typ=2)
                pval = "{:.2f}".format(df.iloc[0]["PR(>F)"])
                df = df.round(2)
                fval = df.iloc[0]["F"]
                ax.set_title(title + "   F-Value:" + str(fval) + "  P-Value:" + str(pval))
                plt.show()
        if cmd.startswith("hanova"):
            to = ""
            pics_dir = "/home/ahmad/Documents/Papers/Applied_Int_paper/pics" #os.getcwd() 
            dest_folder = rowinput("Dest Folder:")
            pic_dir = os.path.join(pics_dir, "pics", dest_folder)
            for prefix in main_df["prefix"].unique():
                if "stsb" in prefix and prefix != "stsb_stsb":
                    df = main_df.loc[(main_df.prefix == prefix)] 
                    if "_stsb" in prefix:
                        val = 'pred_text1' #sel_cols[cur_col]
                        treatment = "target_text"
                    else:
                        treatment = 'pred_text1' #sel_cols[cur_col]
                        val = "target_text"

                    df[val] = df[val].astype(float)
                    samples = str(len(df))
                    fig, ax = plt.subplots()
                    ax = sns.boxplot(x=treatment, y=val, data=df, color='#99c2a2')
                    ax = sns.swarmplot(x=treatment, y=val, data=df, color='#7d0013')
                    if val == "target_text":
                        ax.set_xlabel("Prediction")
                        ax.set_ylabel("Class")
                    else:
                        ax.set_ylabel("Prediction")
                        ax.set_xlabel("Class")
                    # Ordinary Least Squares (OLS) model
                    model = ols(f'{val} ~ C({treatment})', data=df).fit()
                    backit(df, sel_cols)
                    sel_cols = ["sum_sq","df","F","PR(>F)"]
                    try:
                        df = sm.stats.anova_lm(model, typ=2)
                        pval = "{:.8f}".format(df.iloc[0]["PR(>F)"])
                        df = df.round(2)
                        fval = df.iloc[0]["F"]
                    except:
                        pval, fval = "na", "na"
                    title = prefix.replace("_"," by ") 
                    ax.set_title(title + "   F-Value:" + str(fval) + "  P-Value:" + str(pval))
                    Path(pic_dir).mkdir(parents=True, exist_ok=True)
                    plt.savefig(pic_dir + "/" + prefix + ".png")
        if cmd.startswith("describe"):
            df = df.groupby(selected_cols)['All'].describe()
            sel_cols = df.columns
        if cmd.startswith("agg"):
            if not selected_cols:
                selected_cols = ["model_temp", "template"]
            tdf = df.groupby(selected_cols)['All'].mean().unstack()
            plt.figure(figsize=(10, 6))
            sns.heatmap(tdf, annot=True, cmap='coolwarm')
            plt.title('Heatmap of Mean Scores by Category1 and Category2')
            plt.show()
        if cmd.startswith("violin"):
            if not selected_cols:
                selected_cols = ["model_temp", "template"]
            plt.figure(figsize=(12, 6))
            tdf = get_sel_dfs(df)
            sns.violinplot(x=selected_cols[0], y=selected_cols[-1], 
                    hue=selected_cols[1], data=tdf, split=True)
            plt.title('Boxplot of Scores by Category1 and Category2')
            plt.show()
        if cmd.startswith("box"):
            if not selected_cols:
                selected_cols = ["model_temp", "template"]
            plt.figure(figsize=(12, 6))
            tdf = get_sel_dfs(df)
            sns.boxplot(x=selected_cols[0], y=selected_cols[-1], 
                    hue=selected_cols[1], data=tdf)
            plt.title('Boxplot of Scores by Category1 and Category2')
            plt.show()
        if cmd.startswith("ebar"):
            sns.set(style="whitegrid")
            plt.figure(figsize=(10, 6))

            sns.boxplot(x='prefix', y='depth_score', hue='max_train_samples', data=df, ci=None, palette="Blues", edgecolor='w')

            for index, row in df.iterrows():
                plt.errorbar(x=index, y=row['depth_score'], yerr=row['depth_score_std'], fmt='none', c='black', capsize=5)

            # plt.title('Depth Score and Standard Deviation by Relation Type and Training Samples')
            plt.xlabel('Relation Type')
            plt.ylabel('Depth Score')
            plt.legend(title='Training Samples (tn)')
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.show()            


        if cmd.startswith("2bar"):
            category1_mapping = {
                    "t5-large-sup-free-8000":"t5-large-lm-omcs",
                    "t5-large-sup-opsent-6500":"t5-large-lm-wikidata",
                    }
            df['model_name_or_path'] = df['model_name_or_path'].map(category1_mapping)
            tdf = df.pivot(index=selected_cols[0], 
                    columns=selected_cols[1], values=selected_cols[-1])

            ax = tdf.plot(kind='bar', figsize=(10, 6))
            ax.legend(title='Model Type')

            # plt.title('Mean Comparison of tn across Models')
            plt.xlabel('Template')
            plt.ylabel('Accuracy Mean Value')
            plt.xticks(rotation=0)

            plt.tight_layout()
            plt.show()
        if cmd.startswith("bar") or cmd.startswith("line"):
            show_extra = True
            consts["columns"] = sel_cols 
            if len(selected_cols) < 3:
                ms_cmd = True
                i = len(selected_cols)
                sel_col = sel_cols[cur_col]
                inp = ["Filter col", "X col", "Y col"]
                col = rowinput(inp[i] + ":", default=sel_col)
                if not col:
                    ms_cmd = False
                else:
                    consts[inp[i]] = col 
                    selected_cols.append(col)
            else:
                ms_cmd = False
                show_extra = False
                filter_col = selected_cols[0]
                x_col = selected_cols[1]
                y_col = selected_cols[2]
                # unique_filter_values = df[filter_col].unique()
                # desired_order = ['xAttr', 'AtLocation', 'ObjectUse', 'xIntent', 'xWant', 'xNeed']

                # df[filter_col] = pd.Categorical(df[filter_col], categories=desired_order, ordered=True)
                category3_mapping = {
                        'AnswerPrompting': 'AP', 'ChoicePrompting': 'CP',
                        'MaskedAnswerPrompting': 'MAP', 'MaskedChoicePrompting': 'MCP' 
                        }

                df[x_col] = df[x_col].map(category3_mapping)

                palette = ['#2c903c', '#4173c4']
                g = sns.FacetGrid(df, col=filter_col, col_wrap=3, height=4, aspect=1)  #
                if cmd == "bar":
                    hue_col = selected_cols[3]  if len(selected_cols) > 3 else x_col
                    g.map_dataframe(sns.barplot, x=x_col, y=y_col, 
                            hue=hue_col, palette="muted")
                else:
                    hue_col = selected_cols[3]  if len(selected_cols) > 3 else filter_col
                    g.map_dataframe(sns.lineplot, x=x_col, y=y_col, 
                            hue=hue_col, palette="muted", marker='o')
                g.set_axis_labels("Templates", 'Mean Accuracy Score')
                g.set_titles('{col_name}')
                # plt.xticks(rotation=90)
                g.add_legend(loc='upper right', title_fontsize=14) 
                plt.tight_layout()
                plt.show()
        if cmd.startswith("sbar"):
            #tdf = df.groupby(selected_cols + ['model_base'])['All'].mean().unstack().reset_index()
            if not selected_cols:
                selected_cols = ["model_temp", "template"]
            grouped = df.groupby(selected_cols + ['model_base'])['All'].mean().reset_index()
            pivoted = grouped.pivot_table(index=selected_cols, columns='model_base', values='All').reset_index()
            model_bases = df['model_base'].unique()
            palette = sns.color_palette("husl", len(df[selected_cols[1]].unique()))
            ncols = 3
            nrows = (len(model_bases) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), 
                    constrained_layout=True)
            axes = axes.flatten()
            for idx, model_base in enumerate(model_bases):
                ax = axes[idx]
                subset = pivoted[[selected_cols[0], selected_cols[1], model_base]].dropna()

                subset.plot(kind='bar', x=selected_cols[0], 
                        y=model_base, ax=ax, label=model_base, color=palette[idx % len(palette)])

                #subset = tdf[tdf['model_base'] == model_base]
                #subset.plot(kind='bar', x=selected_cols[0], y=subset.columns[1:], ax=ax)

                ax.set_title(f'Model Base: {model_base}')
                ax.set_xlabel('Category1')
                ax.set_ylabel('Mean Scores')
                ax.legend(title='Category2')
                ax.tick_params(axis='x', rotation=45)

            for i in range(idx + 1, len(axes)):
                fig.delaxes(axes[i])

            plt.show()
        elif cmd.startswith("qbar"):
            if False: #context == "main":
                score_col = "rouge_score"
                if sel_cols[cur_col].endswith("_score"):
                    score_col = sel_cols[cur_col]
                backit(df, sel_cols)
                tdf = summarize(df, score_col=score_col)
            else:
                tdf = df.copy()
            #rename_dict = {
            #    "model_temp": 'model',
            #}
            #tdf = tdf.rename(columns=rename_dict)
            if not selected_cols:
                selected_cols = ["model_base", "model_temp", "template"]
            for col in selected_cols:
                tdf[col] = tdf[col].fillna('none')

            cat_col = selected_cols[0]
            num_cats = tdf[cat_col].nunique()
            templates = tdf[selected_cols[-1]].unique()
            category1_order = tdf[selected_cols[1]].unique()
            category2_order = templates
            facet_order = tdf[cat_col].unique()
            convert_labels = False #settings.get("convert_labels", False)

            palette = ['orange','#B33', '#4cc', '#24f', 'pink','#909', '#b7d99c', '#293']
            if  "model_temp" in selected_cols:
                category1_mapping = {
                    'none': '---', 
                    'sup': 'LM', 
                    'unsup': 'Denoising',
                    "mixed": "Mixed"
                    }
                category1_order = [
                        '---', 
                        'LM', 
                        'Denoising', 
                        'Mixed'
                        ]
                tdf['model_temp'] = tdf['model_temp'].map(category1_mapping)
            if  "template" in selected_cols:
                if any("ptar" in t for t in templates):
                    category2_order = ['PrePT', 'PostPT', 
                            'MaskedPrePT', 'MaskedPostPT',
                            'PreAnswerPT','PostAnswerPT',
                            'PreMaskedAnswerPT','PostMaskedAnswerPT']
                    category2_mapping = {
                        '0-ptar-sup': 'PostPT', 
                        "ptar-sup":"PreSup", 
                        'ptar-unsup-nat':'MaskedPrePT',
                        'ptar-sup-nat': 'PrePT', 
                        '0-ptar-unsup': 'MaskedPostPT',
                        '0-ptar-vnat-v3': 'MaskedAnswerPT', 
                        '0-ptar-vnat_1-vs1': 'AnswerPT',
                        'ptar-vnat_0-v4':'PreMaskedAnswerPT',
                        '0-ptar-vnat_0-v4':'PostMaskedAnswerPT',
                        'ptar-vnat_0-vs2':'PreAnswerPT',
                        '0-ptar-vnat_0-vs2':'PostAnswerPT',
                        }
                else:
                    cat_color = {
                            'Mapping':'orange', 
                            'Prompting':'#B33', 
                            'MaskedMapping':'#4cc', 
                            'MaskedPrompting':'#24f', 
                            "AnswerPrompting":'pink', 
                            "ChoicePrompting":'#909',
                            "MaskedAnswerPrompting":'#b7d99c',
                            "MaskedChoicePrompting":'#283',
                            }
                
                    category2_order = [
                            'Mapping', 
                            'Prompting', 
                            'MaskedMapping', 
                            'MaskedPrompting', 
                            "AnswerPrompting", 
                            "ChoicePrompting",
                            "MaskedAnswerPrompting",
                            "MaskedChoicePrompting",
                            ]
                    palette = [v for k,v in cat_color.items() if k in category2_order]
                    category2_mapping = {
                         'sup': 'Mapping', 
                         'unsup': 'MaskedMapping',
                         'sup-nat': 'Prompting', 
                         'unsup-nat': 'MaskedPrompting',
                         'vnat-v3': 'MaskedChoicePrompting',
                         'vnat_1-vs2': "ChoicePrompting",
                         'vnat_0-v4': "MaskedAnswerPrompting",
                         'vnat_0-vs2': "AnswerPrompting",
                         # 'vnat_1-vs2': "Predict Choice Number",
                        }
                cat2_mapping ={k:v for k,v in category2_mapping.items() if k in templates}
                # category2_order = [v for v in category2_order if v in cat2_mapping.values()]
                if convert_labels:
                    tdf['template'] = tdf['template'].map(cat2_mapping)
            if "qpos" in selected_cols:
                category3_mapping = {
                        'start': 'question position: start',
                        'end': 'question position: end',
                        }

                #facet_order = category3_mapping.values() 
                #tdf['qpos'] = tdf['qpos'].map(category3_mapping)
                category2_order = ["start","end"]
                palette = ['#4cc', 'teal', 'orange', '#24f', '#b7d99c', '#293']
                # palette = ['pink','#909', '#b7d99c', '#293']
            if convert_labels and "model_base" in selected_cols:
                category3_mapping = {
                        'v1': 'T5-v1', 'base': 'T5-base',
                        'lm': 'T5-lm', 'large': 'T5-large' 
                        }

                facet_order = ['T5-base','T5-large'] #, 'T5-large']  # specify the desired order
                tdf['model_base'] = tdf['model_base'].map(category3_mapping)
                palette = ['pink','#909', '#b7d99c', '#293']

                # facet_order = ['T5-v1', 'T5-lm', 'T5-base']  # specify the desired order

            if cur_col_name.endswith("_score"):
                score_col = cur_col_name
            else:
                score_col = 'All' # if context == "pivot" else 'rouge_score'
            tdf = tdf.groupby(selected_cols)[score_col].mean().reset_index()
            hue_palette = sns.color_palette("husl", len(tdf[selected_cols[-1]].unique()))
            g = sns.FacetGrid(tdf, col=cat_col, col_order=facet_order,
                    col_wrap=num_cats, height=4, aspect=1.5)
            g.map_dataframe(sns.barplot, x=selected_cols[1], 
                    y=score_col, hue=selected_cols[-1], 
                    palette=palette, ci=None, 
                    order=category1_order, hue_order=category2_order)

            g.set_axis_labels("Unsupervised Training Objective", 'Mean Scores')
            g.set_titles('{col_name}')
            #g.fig.text(0.5, 0.04, 'Unsupervised Training Objective', 
            #        ha='center', va='center', fontsize=12)
            if "@" in cmd:
                _, ll = cmd.split("@")
                g.fig.text(0.05, 0.95, ll, ha='left', va='top', fontsize=12, transform=g.fig.transFigure)
            # Rotate x labels for better readability
            #for ax in g.axes.flat:
            #    for label in ax.get_xticklabels():
            #        label.set_rotation(45)
            g.add_legend(loc='upper right', title_fontsize=14) 
            plt.show()

        elif cmd.startswith("dprop"):

            # Group by prefix and training samples (tn*) to calculate proportion
            df_grouped = df.groupby(['prefix', 'max_train_samples'])

            results = []

            for (prefix, tn_star), group in df_grouped:
                prompting = group[group['template'] == 'Prompting']['depth_score'].values
                masked_prompting = group[group['template'] == 'MaskedPrompting']['depth_score'].values
                if prompting.size > 0 and masked_prompting.size > 0:
                    proportion = ((prompting[0] - masked_prompting[0]) / prompting[0]) * 100
                    results.append({'prefix': prefix, 'max_train_samples': tn_star, 'proportion (%)': round(proportion, 2)})

            df = pd.DataFrame(results)
            sel_cols = df.columns
        elif cmd.startswith("mscat"):
            category_column = selected_cols[2]
            categories = df[category_column].unique()

            for category in categories:
                category_df = df[df[category_column] == category]
                
                x = category_df[selected_cols[0]]
                y = category_df[selected_cols[1]]
                
                plt.scatter(x, y, label=f'Category: {category}')
                
                # Fit a polynomial (linear fit)
                coefficients = np.polyfit(x, y, 1)
                polynomial = np.poly1d(coefficients)
                y_fit = polynomial(x)
                
                # Plot the fit line for each category
                plt.plot(x, y_fit, label=f'Fit Line ({category})')

            xlabel = selected_cols[0].replace('_', ' ').title()
            ylabel = selected_cols[1].replace('_', ' ').title()

            plt.xlabel(xlabel, fontsize=16)
            plt.ylabel(ylabel, fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend(fontsize=10, loc='best')
            plt.show()
        elif cmd.startswith("corr"):
            correlation_matrix = df[selected_cols].corr()
            df = pd.DataFrame(correlation_matrix)
        elif cmd.startswith("scat"):
            x = df[selected_cols[0]]
            y = df[selected_cols[1]]
            # sns.lmplot(x=selected_cols[0], y=selected_cols[1], data=df, ci=None)

            plt.scatter(x, y, label='Data Points')

            coefficients = np.polyfit(x, y, 1)
            polynomial = np.poly1d(coefficients)
            y_fit = polynomial(x)

            plt.plot(x, y_fit, color='green', label='Fit Line')
            xlabel = selected_cols[0]
            ylabel = selected_cols[1]
            xlabel = xlabel.replace('_', ' ').title()
            ylabel = ylabel.replace('_', ' ').title()

            plt.xlabel(xlabel, fontsize=16)
            plt.ylabel(ylabel, fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend(fontsize=16)

            plt.title(xlabel + ' vs. ' + ylabel, fontsize=20)
            plt.show()
        elif cmd.startswith("mbar"):
            tdf = df.groupby(selected_cols)['All'].mean().unstack()
            plt.figure(figsize=(10, 6))
            tdf.plot(kind='bar')
            plt.title('Bar Plot of Mean Scores by Category1 and Category2')
            plt.xlabel('Category1')
            plt.ylabel('Mean Scores')
            plt.show()
        elif cmd.startswith("apply"):
            _, function_name = cmd.split("@")
            if not function_name in function_map:
                show_msg("There is no funciton named " + function_name)
            elif "fid" in df: 
                exprs, scores= get_sel_rows(df, row_id="fid", col="rouge_score", from_main=False) 
                s_rows = sel_rows
                if not s_rows:
                    s_rows = [sel_row]
                for s_row, exp, score in zip(s_rows, exprs, scores):
                    tdf = main_df[main_df['fid'] == exp]
                    spath = tdf.iloc[0]["path"]
                    function_map[function_name](tdf,spath)
            mbeep()
        if cmd.startswith("rem"):
            if "@" in cmd:
                exp_names = cmd.split("@")[2:]
                _from = cmd.split("@")[1]
            rep = load_obj(_from, "gtasks", {})
            if rep:
                for k, cat in rep.items():
                    for exp in exp_names:
                        if exp in cat:
                            del rep[k][exp]
            save_obj(rep, _from, "gtasks")
        if char in ["1","2"]:
            if not pcols or not all(col in df for col in pcols):
                mbeep()
            else:
                if prev_char not in ["1","2"]:
                    backit(df, sel_cols)
                if score_cols:
                    ss = int(char) - 1
                    if ss < len(score_cols):
                        score_col = score_cols[ss]
                #scols = selected_cols
                #if not scols:
                #    scols = [sel_cols[cur_col]]
                #sel_cols = index_cols.copy() + ["All"] + scols 
                sort_col = sort if sort else "All"
                if score_col == score_cols[0]:
                    for col in df.columns:
                        if col in pcols:
                            sel_cols.append(col)
                else:
                    for col in df.columns:
                        if col.startswith(score_col[0:2] + "-"):
                            sel_cols.append(col)
                    sort_col = score_col[0:2] + "-All"
                    if sort_col in sel_cols:
                        sel_cols.remove(sort_col)
                        sel_cols.insert(len(index_cols), sort_col)
                sort = sort_col
                # df = df.sort_values(by=sort_col, ascending=False)
        if char == "!":
            doc_dir = "/home/ahmad/findings" #os.getcwd() 
            note_file = os.path.join(doc_dir, "notes.csv")
            context = "notes"
            if not "comment" in df:
                backit(df, sel_cols)
                if Path(note_file).is_file():
                    df = pd.read_csv(note_file)
                else:
                    df = pd.DataFrame(columns=["date","cat","comment"])
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                sel_cols = df.columns 
                info_cols = []
                df = df.sort_values("date", ascending=False)
                cond_colors["cat"] = cat_colors
                # search_df = df
        if ch == cur.KEY_IC or char == "e" and context == "notes":
            doc_dir = "/home/ahmad/findings" #os.getcwd() 
            note_file = os.path.join(doc_dir, "notes.csv")
            if not "comment" in df or context != "notes":
                if Path(note_file).is_file():
                    tdf = pd.read_csv(note_file)
                else:
                    tdf = pd.DataFrame(columns=["date","cat","comment"])
                tdf = tdf.loc[:, ~tdf.columns.str.contains('^Unnamed')]
            else:
                tdf = df
            bg_color = HL_COLOR
            win_height = 8
            note_title = ""
            _default = ""
            cat = ""
            if char == "e" and len(df) > 0:
                cat = df.iloc[sel_row]["cat"] 
                _default = cat + "\n" + df.iloc[sel_row]["comment"]
            _comment, ret_ch = biginput("", default=_default)
            if _comment:
                lines = _comment.split("\n")
                comment = _comment
                if len(lines) > 1:
                    cat = lines[0]
                    comment = "\n".join(lines[1:]) 
                new_note = {}
                new_note["date"] = now
                new_note["comment"] = comment
                new_note["cat"] = cat
                if char != "e":
                    tdf = pd.concat([tdf, pd.DataFrame([new_note])], ignore_index=True)
                else:
                    tdf.iloc[sel_row] = new_note 
                if Path(note_file).is_file():
                    shutil.copyfile(note_file, note_file.replace("notes", now + "_notes"))
                tdf.to_csv(note_file)
            if "comment" in df:
                df = tdf
                df = df.sort_values(by=["date"], ascending=False) 
                cond_colors["cat"] = cat_colors
        # rrrrrrrrr
        if cmd.startswith("rep") or char == "Z" or char == "r": 
            score_col = None
            if sel_cols[cur_col].endswith("_score"):
                score_col = sel_cols[cur_col]
            if not global_summary:
                backit(df, sel_cols)
                pdf = summarize(df, rep_cols=selected_cols, score_col=score_col)
            else:
                pdf = df
            avg_col = "All"
            context = "pivot"
            shortkeys["pivot"] = {"o":"open image", "h": "plot line"}
            score_col = score_cols[0]
            cond_colors["eid"] = index_colors
            cond_colors["All"] = score_colors
            cond_colors["time"] = time_colors
            cond_colors["expid"] = time_colors
            pcols = []
            if not global_summary:
                for col in pivot_cols:
                    pcols.extend(df[col].unique())
            else:
                for col in df.columns:
                    if "nu-" in col:
                        col = col.replace("nu-","")
                        pcols.extend(col)
            for col in pcols:
                cond_colors[col] = pivot_colors

            _sel_cols = [] 
            if score_col == score_cols[0]:
                avg_col = "All"
                for col in pdf.columns:
                    if col in df or col in pcols:
                        _sel_cols.append(col)
            else:
                avg_col = score_col[0:2] + "-All"
                for col in pdf.columns:
                    if col in df or col.startswith(score_col[0:2] + "-"):
                        _sel_cols.append(col)
            if "time" in pdf:
                df = pdf.sort_values(by="time", ascending=False)
            else:
                df = pdf.sort_values(by="All", ascending=False)
            #sort = "time"
            sel_cols = list(dict.fromkeys(sel_cols + _sel_cols))
            if len(df) > 1:
                sel_cols, info_cols_back, tag_cols = remove_uniques(df, sel_cols, 
                        keep_cols=pivot_cols + info_cols + pcols)
            for col in ["folder", "output_dir", "eid", "expname"]:
                if col in sel_cols:
                    sel_cols.remove(col)
                if not col in info_cols_back:
                    info_cols_back.append(col)

            for i, col in enumerate(info_cols + [avg_col]):
                if col in sel_cols:
                    sel_cols.remove(col)
                sel_cols.insert(i, col)

            for col in exclude_cols:
                if col in sel_cols:
                    sel_cols.remove(col)

            if "time" in sel_cols:
                sel_cols.remove("time")
                sel_cols.append("time")

            pivot_df = df
            info_cols = []
            #df.columns = [map_cols[col].replace("_","-") if col in map_cols else col 
            #              for col in pdf.columns]
        if char == "L" or cmd.startswith("Latex"):
            tdf = df
            cols = selected_cols if selected_cols else sel_cols
            latex_table=tabulate(tdf[cols],  #[rep_cols+score_cols], 
                    headers='keys', tablefmt='latex_raw', showindex=False, floatfmt=".2f")
            def rotate_columns(latex_table, cols_to_rotate):
                for col in cols_to_rotate:
                    latex_table = latex_table.replace(col, f"\\rot{{{col}}}")
                return latex_table
            #latex_table = latex_table.replace("tabular", "longtable")
            latex_table = latex_table.replace("_", "-")
            if "rot" in cmd:
                latex_table = rotate_columns(latex_table, rot_cols)
            latex_lines = latex_table.split('\n')
            modified_latex_lines = []

            for i, line in enumerate(latex_lines):
                if i > 0 and '\\multirow{' in line:  
                    modified_latex_lines.append(r'\hline')  
                    modified_latex_lines.append(line)
                else:
                    modified_latex_lines.append(line)

            latex_table = '\n'.join(modified_latex_lines)
            pyperclip.copy(latex_table)
            # clipboard_content = pyperclip.paste()
            tname = rowinput("Table name:")
            table_dir = "/home/ahmad/tehran-thesis/tables"
            Path(table_dir).mkdir(exist_ok=True, parents=True)
            with open(os.path.join(table_dir, tname + ".tex"), "w") as f:
                f.write(latex_table)
            #report = report.replace("mytable", latex_table + "\n\n \\newpage mytable")
            #report = report.replace("mytable", "\n \\newpage mytable")
            # tdf = pdf
            # iiiiiiiiiiiii
            #report = report.replace("mytable","")
            #with open(m_report, "w") as f:
            #    f.write(main_report)
            mbeep()
            #subprocess.run(["pdflatex", tex])
            #subprocess.run(["okular", pdf])
        if char == "l" or char == "Z" or cmd.startswith("latex"):
            _dir = Path(__file__).parent
            doc_dir = "/home/ahmad/logs" #os.getcwd() 
            table_dir = "/home/ahmad/Documents/Papers/Applied_Int_paper/tables" #os.getcwd() 
            if len(score_cols) > 1:
                # m_report = f"{_dir}/report_templates/report_colored_template.tex"
                m_report = f"{_dir}/report_templates/report_template.tex"
            else:
                m_report = f"{_dir}/report_templates/report_template.tex"
            with open(m_report, "r") as f:
                report = f.read()
            #with open(os.path.join(doc_dir, "report.tex"), "w") as f:
            #    f.write("")
            backit(df, sel_cols)

            exprs, _ = get_sel_rows(df, from_main=False) 
            cond = False
            for exp in exprs:
                cond = cond | (group_df["eid"] == exp) 
            bdf = group_df[cond]
            pdf = summarize(bdf, score_col=score_col, rename=False)
            pdf.reset_index(drop=True)
            dfs = []
            if not selected_cols:
                selected_cols = ["model_base","model_temp","template"]
            for score_col in ["rouge_score","bert_score","bleu_score"]:
                sid = score_col[0:2] + "-"
                cols = [col for col in pdf.columns if col.startswith(sid)]
                tdf = pdf[selected_cols + cols]
                tdf.columns = [col.replace(sid, "") for col in tdf.columns]
                tdf["metric"] = score_col
                dfs.append(tdf)
            df = tdf = pd.concat(dfs, axis=0) #, ignore_index=True)
            sel_cols = list(tdf.columns)
            numeric_cols = [col for col in tdf.columns if pd.api.types.is_numeric_dtype(tdf[col])]
            max_values = tdf[numeric_cols].max()

            def format_with_bold_max(val, max_val, col):
                if val == max_val:
                    if col == 'All':
                        return r'\textbf{' + "{:.2f}".format(val) + '}'
                    else:
                        return r'\textbf{' + "{:.2f}".format(val) + '}'
                else:
                    if col == 'All':
                        return "{:.2f}".format(val)
                    else:
                        return "{:.2f}".format(val)            

            grouped = tdf.groupby('metric')

            formatted_dfs = []
            for name, group in grouped:
                # Calculate max values for the group
                max_values = group.max()
                
                # Apply bold formatting to each column
                formatted_group = group.copy()
                for col in numeric_cols: 
                    max_value = max_values[col]
                    formatted_group[col] = formatted_group[col].apply(format_with_bold_max, args=(max_value,col))
                
                formatted_dfs.append(formatted_group)

            tdf = pd.concat(formatted_dfs)
            tdf = tdf.reset_index()
            rot_cols = pcols
            if "template" in tdf:
                templates = tdf["template"].unique()
                category1_mapping = {
                        'none': '---', 'sup': 'LM', 
                        'unsup': 'Denoising',"per_sample": "Mixed"
                        }
                if any("ptar" in temp for temp in templates):
                    category2_mapping = {
                        '0-ptar-sup': 'PostPT', 
                        'ptar-unsup-nat':'MaskedPrePT',
                        'ptar-sup-nat': 'PrePT', '0-ptar-unsup': 'MaskedPostPT'
                        }
                else:
                    category2_mapping = {
                         'sup': 'Mapping', 'unsup': 'MaskedMapping',
                         'sup-nat': 'Prompting', 'unsup-nat': 'MP'
                        }
                category3_mapping = {
                        'v1': 'T5-v1', 'base': 'T5-base',
                        'lm': 'T5-lm' 
                        }
                metric_mapping = {
                        'bert_score': 'BERT', 'bleu_score': 'BLEURT',
                        'rouge_score': 'Rouge' 
                        }

                tdf['model_temp'] = tdf['model_temp'].map(category1_mapping)
                tdf['template'] = tdf['template'].map(category2_mapping)
                tdf['model_base'] = tdf['model_base'].map(category3_mapping)
                tdf['metric'] = tdf['metric'].map(metric_mapping)
                rename_dict = {
                    "model_base": 'model',
                    "model_temp": 'objective',
                }
                tdf = tdf.rename(columns=rename_dict)
                rot_cols.extend(["template"])
            #for col in numeric_cols:
            #    tdf[col] = tdf[col].apply(format_with_bold_max, args=(max_values[col],col))
            
            #sorter = ['PT', 'MCAT', 'MSUM', 'SSUM'] 
            #tdf.sort_values(by="label", 
            #        key=lambda column: column.map(lambda e: sorter.index(e)), inplace=True)

            cols = list(tdf.columns)
            cols.remove("index")
            method = 'model_name_or_path'
            #score_col = score_cols[0]
            #if selected_cols: 
            #    cols = selected_cols
            #cols += numeric_cols
            group_col = "metric"
            ng = str(len(sel_rows))
            if group_col in tdf:
                tdf['score'] = ''
                tdf['group'] = ''
                old_val = ''
                for idx, row in tdf.iterrows():
                    if row[group_col] != old_val: 
                        old_val = str(row[group_col])
                        tdf.at[idx, 'group'] ='\\multirow{' + ng + '}{*}{' + old_val + '}'
                        if "acc" in cols:
                            tdf.at[idx, 'score'] = '\\multirow{' + ng + '}{*}{' + row['acc'] + '}'
                        
                    if "preds" in cols and row['preds'] == 'precision':
                        tdf.at[idx, method] = '\\rowcolor{gray!20}' + tdf.at[idx, group_col]
 
                if "max_train_samples" in tdf:
                    cols.insert(1, "max_train_samples")
                    cols.remove("max_train_samples")
                if "acc" in cols:
                    cols.remove("acc")
                    cols.append("score")
                tdf.drop(columns=[group_col], inplace=True)
                tdf.rename(columns={'group':group_col}, inplace=True)

            latex_table=tabulate(tdf[cols],  #[rep_cols+score_cols], 
                    headers='keys', tablefmt='latex_raw', showindex=False, floatfmt=".2f")
            def rotate_columns(latex_table, cols_to_rotate):
                for col in cols_to_rotate:
                    latex_table = latex_table.replace(col, f"\\rot{{{col}}}")
                return latex_table
            #latex_table = latex_table.replace("tabular", "longtable")
            latex_table = latex_table.replace("_", "-")
            if "rot" in cmd:
                latex_table = rotate_columns(latex_table, rot_cols)
            latex_lines = latex_table.split('\n')
            modified_latex_lines = []

            for i, line in enumerate(latex_lines):
                if i > 0 and '\\multirow{' in line:  
                    modified_latex_lines.append(r'\hline')  
                    modified_latex_lines.append(line)
                else:
                    modified_latex_lines.append(line)

            latex_table = '\n'.join(modified_latex_lines)
            tname = rowinput("Table name:")
            with open(os.path.join(table_dir, tname + ".tex"), "w") as f:
                f.write(latex_table)
            report = report.replace("mytable", latex_table + "\n\n \\newpage mytable")
            report = report.replace("mytable", "\n \\newpage mytable")
            # tdf = pdf
            # iiiiiiiiiiiii
            report = report.replace("mytable","")
            tex = f"{doc_dir}/report.tex"
            pdf = f"{doc_dir}/report.pdf"
            with open(tex, "w") as f:
                f.write(report)
            #with open(m_report, "w") as f:
            #    f.write(main_report)
            mbeep()
            #subprocess.run(["pdflatex", tex])
            #subprocess.run(["okular", pdf])
        if "getimage" in cmd or char == "Z":
            show_msg("Generating images ...", bottom=True, delay=2000)
            _dir = Path(__file__).parent
            doc_dir = "/home/ahmad/logs" #os.getcwd() 
            m_report = os.path.join(doc_dir, "report.tex")
            with open(m_report, "r") as f:
                report = f.read()

            image = """
                \\begin{{figure}}
                    \centering
                    \includegraphics[width=\\textwidth]{{{}}}
                    \caption[image]{{{}}}
                    \label{{{}}}
                \end{{figure}}
            """
            multi_image = """
                \\begin{figure}
                    \centering
                    \caption[image]{mycaption}
                    mypicture 
                    \label{fig:all}
                \end{figure}
            """
            graphic = "\includegraphics[width=\\textwidth]{{{}}}"
            pics_dir = doc_dir + "/pics"
            #ii = image.format(havg, "havg", "fig:havg")
            #report = report.replace("myimage", ii +"\n\n" + "myimage")
            Path(pics_dir).mkdir(parents=True, exist_ok=True)
            #pname = plot_bar(pics_dir, train_num)
            #ii = image.format(pname, "bar", "fig:bar")
            #report = report.replace("myimage", ii +"\n\n" + "myimage")
            all_exps = df["eid"].unique()
            experiment_images, fnames = get_images(df, all_exps, 'eid')
            all_images = {}
            kk = 0
            id = "other"
            images_str = ""
            cols = ["eid"] + rep_cols + score_cols
            img_string = ""
            for key, img_list in experiment_images.items():
                mkey = key
                caption_dict = {}
                if not df.loc[df['eid'] == key].empty:
                    caption_dict = df.loc[df['eid'] == key, sel_cols].iloc[0].to_dict()
                caption = ""
                name = key
                key = str(key)
                for new_im in img_list:
                    name = key + str(name)
                    _exp = key.replace("_","-")
                    _exp = _exp.split("-")[0]
                    fname = fnames[kk]
                    for k,v in caption_dict.items():
                        if k in map_cols:
                            k = map_cols[k]
                        if type(v) == float:
                            v = f"{v:.2f}"
                        if k == "cat":
                            v = v.split("-")[0]
                        caption += " \\textcolor{gray}{" + str(k).replace("_","-") \
                            + "}: \\textcolor{blue}{" + str(v).replace("_","-")+ "}" 
                    ss = "_scores" if "score" in fname else "_sim"
                    if "@" in fname:
                        ss = "_" + fname.split("@")[1]
                    pname = doc_dir + "/pics/" + id + name.strip("-") + ss + ".png" 
                    dest = os.path.join(doc_dir, pname) 
                    new_im.save(dest)
                    label = "fig:" + key
                    ii = image.format(pname, caption, label)
                    if kk % 10 == 0:
                        tex = f"{doc_dir}/hm_img_{kk}.tex"
                        with open(tex, "w") as f:
                            f.write(img_string)
                        img_string = ""
                        report = report.replace("myimage", 
                                f"\clearpage \n \input{{hm_img_{kk}.tex}} \n myimage") 
                    img_string +=  ii +"\n\n" 
                    if not _exp in all_images:
                        all_images[_exp] = {}
                    all_images[_exp][ss] = pname
                    kk += 1

            if img_string: 
                tex = f"{doc_dir}/hm_img_{kk}.tex"
                with open(tex, "w") as f:
                    f.write(img_string)
                img_string = ""
                report = report.replace("myimage", 
                        f"\clearpage \n \input{{hm_img_{kk}.tex}} \n myimage") 

            g1 = ["SIL","SILP","SIP"] 
            g2 = ["SILPI","SLPI","SLP", "SL"] 
            ii = 0
            multi_image3 = multi_image
            for k,v in all_images.items():
                if ii % 2 == 0:
                    multi_image3 += f" \\newpage \n \\subsection{{{k}}}"
                ii += 1
                for p,q in v.items():
                    multi_image3 += multi_image.replace("mypicture", 
                            graphic.format(q) + "\n").replace("mycaption",
                                    str(p) + ":" + str(q))

            multi_image3 = multi_image3.replace("mypicture","")
            tex = f"{doc_dir}/scores_img.tex"
            with open(tex, "w") as f:
                f.write(multi_image3)
            #report = report.replace("myimage", 
            #        "\n\n \input{scores_img.tex} \n\n myimage") 
            #report = report.replace("myimage", "\n\n \input{sim_img.tex} \n\n myimage") 
            #report = report.replace("myimage", "\n\n \input{other_img.tex} \n\n") 
            ####################
            report = report.replace("mytable","")
            report = report.replace("myimage","")
            tex = f"{doc_dir}/report.tex"
            pdf = f"{doc_dir}/report.pdf"
            with open(tex, "w") as f:
                f.write(report)
            #with open(m_report, "w") as f:
            #    f.write(main_report)
            show_msg(pdf)
            mbeep()
            #subprocess.run(["pdflatex", tex])
            subprocess.run(["okular", pdf])

        if cmd == "fix_types":
            for col in ["target_text", "pred_text1"]: 
                main_df[col] = main_df[col].astype(str)
                for col in ["steps", "epochs", "val_steps"]: 
                    main_df[col] = main_df[col].astype(int)
                char = "SS"
        if cmd == "clean":
            main_df = main_df.replace(r'\n',' ', regex=True)
            char = "SS"
        if cmd == "fix_template":
            main_df.loc[(df["template"] == "unsup-tokens") & 
                    (main_df["wrap"] == "wrapped-lstm"), "template"] = "unsup-tokens-wrap"
            main_df.loc[(main_df["template"] == "sup-tokens") & 
                    (main_df["wrap"] == "wrapped-lstm"), "template"] = "sup-tokens-wrap"
        
        if cmd == "ren":
            sel_col = sel_cols[cur_col]
            new_name = rowinput("Rename " + sel_col + " to:", default="")
            map_cols[sel_col] = new_name
            save_obj(map_cols, "map_cols", "atomic")
            cur_col += 1
        if cmd == "copy" or char == "\\":
            exp=df.iloc[sel_row]["eid"]
            exp = str(exp)
            spath = tdf.iloc[0]["path"]
            oldpath = Path(spath).parent.parent
            pred_file = os.path.join(oldpath, "images", "pred_router_" + exp + ".png") 
            oldpath = os.path.join(oldpath, exp)
            newpath = rowinput(f"copy {oldpath} to:", default=oldpath)
            new_pred_file = os.path.join(newpath, "images", "pred_router_" + exp + ".png") 
            shutil.copyfile(pred_file, new_pred_file)
            copy_tree(oldpath, newpath)
        if cmd == "repall":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                _a = rowinput("from")
                _b = rowinput("to")
                main_df[col] = main_df[col].str.replace(_a,_b)
                char = "SS"
        if cmd == "replace" or cmd == "replace@":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                vals = df[col].unique()
                d = {}
                for v in vals:
                    rep = rowinput(str(v) + "=" ,v)
                    if not rep:
                        break
                    if type(v) == int:
                        d[v] = int(rep)
                    else:
                        d[v] = rep
                if rowinput("Apply?") == "y":
                    if "@" in cmd:
                        df = df.replace(d)
                    else:
                        df = df.replace(d)
                        main_df = main_df.replace(d)
                        char = "SS"
        if cmd in ["set", "set@", "add", "add@", "setcond"]:
            if "add" in cmd:
                col = rowinput("New col name:")
            col = sel_cols[cur_col]
            cond = ""
            if "cond" in cmd:
                cond = get_cond(df, for_col=col, num=5, op="&")
            if cond:
                val = rowinput(f"Set {col} under {cond} to:")
            else:
                val = rowinput("Set " + col + " to:")
            if val:
                if cond:
                    if "@" in cmd:
                        main_df.loc[eval(cond), col] = val
                        char = "SS"
                    else:
                        df.loc[eval(cond), col] =val
                else:
                    if "@" in cmd:
                        main_df[col] =val
                        char = "SS"
                    else:
                        df[col] = val
        if ":=" in cmd:
            var, val = cmd.split(":=")
            if type(val) == str and val.lower() == "true": val = True
            if type(val) == str and val.lower() == "false": val = False 
            settings[var] = val
            save_obj(settings, "settings", "gtasks")
        elif "==" in cmd:
            col, val = cmd.split("==")
            df = df[df[col] == val]
        elif "top@" in cmd:
            backit(df, sel_cols)
            tresh = float(cmd.split("@")[1])
            df = df[df["bert_score"] > tresh]
            df = df[["prefix","input_text","target_text", "pred_text1"]] 
        if cmd == "cp" or cmd == "cp@":
            canceled, col,val = list_df_values(main_df, get_val=False)
            if not canceled:
                copy = rowinput("Copy " + col + " to:", col)
                if copy:
                    if "@" in cmd:
                        df[copy] = df[col]
                    else:
                        main_df[copy] = main_df[col]
                        char = "SS"
        if cmd.isnumeric():
            sel_row = int(cmd)
        elif cmd == "q" or cmd == "wq":
            save_df(df)
            prev_char = "q" 
        elif not char in ["q", "S","R"]:
            pass
            #mbeep()
        if char in ["S", "}"]:
            _name = "main_df" if char == "S" else "df"
            _dfname = dfname
            if dfname == "merged":
                _dfname = "test"
            cmd, _ = minput(cmd_win, 0, 1, f"File Name for {_name} (without extension)=", default=_dfname, all_chars=True)
            cmd = cmd.split(".")[0]
            if cmd != "<ESC>":
                if char == "}":
                    dfname = os.path.join(base_dir, cmd+".tsv")
                    df.to_csv(dfname, sep="\t", index=False)
                    show_msg("written in " + dfname)
                else:
                    dfname = cmd
                    char = "SS"
        if char == "SS":
                df = main_df[["prefix","input_text","target_text"]]
                df = df.groupby(['input_text','prefix','target_text'],as_index=False).first()

                save_path = os.path.join(base_dir, dfname+".tsv")
                sel_cols = ["prefix", "input_text", "target_text"]
                Path(base_dir).mkdir(parents = True, exist_ok=True)
                df.to_csv(save_path, sep="\t", index=False)

                save_obj(dfname, "dfname", dfname)
        if char == "R" and prev_char != "x":
            filter_df = orig_df
            df = filter_df
            FID = "fid" 
            reset = True
            #sel_cols = group_sel_cols 
            #save_obj([], "sel_cols", context)
            #save_obj([], "info_cols", context)
            hotkey = hk
        if char == "R" and prev_char == "x":
            df = main_df
            sel_cols = list(df.columns)
            save_obj(sel_cols,"sel_cols",dfname)
            extra["filter"] = []
            info_cols = []
        if (ch == cur.KEY_BACKSPACE or char == "b") and back:
            if back:
                cur_df = back.pop()
                df = cur_df.df
                sel_cols = cur_df.sel_cols 
                sel_row = cur_df.sel_row
                cur_row = cur_df.cur_row
                sel_rows = cur_df.sel_rows
                info_cols = cur_df.info_cols
                context = cur_df.context
                cur_col = cur_df.cur_col
                left = cur_df.left
                is_filtered = cur_df.is_filtered
                group_col = cur_df.group_col
                if back:
                    back_df = back[-1].df
                else:
                    if "b" in general_keys:
                        del general_keys["b"]
            else:
                if "b" in general_keys:
                    del general_keys["b"]
                mbeep()
            if extra["filter"]:
                extra["filter"].pop()

def render_mpl_table(data, wrate, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        # mlog.info("Size %s", size)
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    mpl_table.auto_set_column_width(col=list(range(len(data.columns)))) # Provide integer list of columns to adjust

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax
def get_cond(df, for_col = "", num = 1, op="|"):
    canceled = False
    sels = []
    cond = ""
    while not canceled and len(sels) < num:
        canceled, col, val = list_df_values(df, col=for_col, get_val=True,sels=sels)
        if not canceled:
            cond += f"{op} (df['{col}'] == '{val}') "
            sels.append(val)
    cond = cond.strip(op)
    return cond

def get_cols(df, num = 1):
    canceled = False
    sels = []
    while not canceled and len(sels) < num:
        canceled, col,_ = list_df_values(df, get_val=False, sels = sels)
        if not canceled:
            sels.append(col)
    return sels

def biginput(prompt=":", default=""):
    rows, cols = std.getmaxyx()
    win = cur.newwin(12, cols, 5, 0)
    win.bkgd(' ', cur.color_pair(CUR_ITEM_COLOR))  # | cur.A_REVERSE)
    _comment, ret_ch = minput(win, 0, 0, "Enter text", 
            default=default, mode =MULTI_LINE)
    if _comment == "<ESC>":
        _comment = ""
    return _comment, ret_ch

def rowinput(prompt=":", default=""):
    prompt = str(prompt)
    default = str(default)
    ch = UP
    history = load_obj("history","", ["test1", "test2", "test3"])
    ii = len(history) - 1
    hh = history.copy()
    while ch == UP or ch == DOWN:
        cmd, ch = minput(cmd_win, 0, 1, prompt, default=default, all_chars=True)
        if ch == UP:
            if cmd != "" and cmd != default:
                jj = ii -1
                while jj > 0: 
                    if hh[jj].startswith(cmd):
                      ii = jj
                      break
                    jj -= 1
            elif ii > 0: 
                ii -= 1 
            else: 
                ii = 0
                mbeep()
        elif ch == DOWN:
            if cmd != "" and cmd != default:
                jj = ii + 1
                while jj < len(hh) - 1: 
                    if hh[jj].startswith(cmd):
                      ii = jj
                      break
                    jj += 1
            elif ii < len(hh) - 1: 
               ii += 1 
            else:
               ii = len(hh) - 1
               mbeep()
        if hh:
            ii = max(ii, 0)
            ii = min(ii, len(hh) -1)
            default = hh[ii]
    if cmd == "<ESC>":
        cmd = ""
    if cmd:
        history.append(cmd)
    save_obj(history, "history", "")
    return cmd

def order(sel_cols, cols, pos=0):
    z = [item for item in sel_cols if item not in cols] 
    z[pos:pos] = cols
    save_obj(z, "sel_cols",dfname)
    return z

def subwin(infos):
    ii = 0
    infos.append("[OK]")
    inf = infos[ii:ii+30]
    change_info(inf)
    cc = std.getch()
    while not is_enter(cc): 
        if cc == DOWN:
            ii += 1
        if cc == UP:
            ii -= 1
        if cc == cur.KEY_NPAGE:
            ii += 10
        if cc == cur.KEY_PPAGE:
            ii -= 10
        if cc == cur.KEY_HOME:
            ii = 0
        if cc == cur.KEY_END:
            ii = len(infos) - 20 
        ii = max(ii, 0)
        ii = min(ii, len(infos)-10)
        inf = infos[ii:ii+30]
        change_info(inf)
        cc = std.getch()
                
def change_info(infos):
    info_bar.erase()
    h,w = info_bar.getmaxyx()
    w = 80
    lnum = 0
    infos.insert(0, "-"*30)
    for msg in infos:
        lines = textwrap.wrap(msg, width=w, placeholder=".")
        for line in lines: 
            mprint(str(line).replace("@","   "), info_bar, color=HL_COLOR)
            lnum += 1
    rows,cols = std.getmaxyx()
    info_bar.noutrefresh(0,0, rows -lnum - 1,0, rows-1, cols - 1)
    return lnum

si_hash = {}

def list_values(vals,si=0, sels=[], is_combo=False):
    tag_win = cur.newwin(15, 70, 3, 5)
    tag_win.bkgd(' ', cur.color_pair(TEXT_COLOR))  # | cur.A_REVERSE)
    tag_win.border()
    vals = sorted(vals)
    key = "_".join([str(x) for x in vals[:4]])
    if si == 0:
        if key in si_hash:
            si = si_hash[key]
    opts = {"items":{"sels":sels, "range":["Done!"] + vals}}
    if is_combo: opts["items"]["type"] = "combo-box"
    is_cancled = True
    si,canceled, st = open_submenu(tag_win, opts, "items", si, "Select a value", std)
    val = st
    if not canceled and si > 0: 
        val = vals[si - 1]
        si_hash[key] = si
        is_cancled = False
    return is_cancled, val

def list_df_values(df, col ="", get_val=True,si=0,vi=0, sels=[], extra=[]):
    is_cancled = False
    if not col:
        cols = extra + list(df.columns) 
        is_cancled, col = list_values(cols,si, sels)
    val = ""
    if col in df and col and get_val and not is_cancled:
        df[col] = df[col].astype(str)
        vals = sorted(list(df[col].unique()))
        is_cancled, val = list_values(vals,vi, sels)
    return is_cancled, col, val 


text_win = None
info_bar = None
cmd_win = None
main_win = None
text_width = 60
std = None
check_time = False
hotkey = ""
dfname = ""
global_cmd = ""
global_search = ""
global_summary = False
root_path = ""
base_dir = os.path.join(home, "datasets", "comet")
data_frame = None

def get_files(dfpath, dfname, dftype, summary, limit, file_id):
    if not dfname:
        mlog.info("No file name provided")
    else:
        path = os.path.join(dfpath, *dfname)
        if Path(path).is_file():
            files = [path]
            dfname = Path(dfname).stem
        else:
            files = []
            ii = 0
            for root, dirs, _files in os.walk(dfpath):
                for _file in _files:
                    root_file = os.path.join(root,_file)
                    cond = all(s.strip() in root_file for s in dfname)
                    if check_time:
                        ts = os.path.getctime(root_file)
                        ctime = datetime.fromtimestamp(ts)
                        last_hour = datetime.now() - timedelta(hours = 5)
                        cond = cond and ctime > last_hour
                    if dftype in _file and cond: 
                        files.append(root_file)
                        ii += 1
                if limit > 0 and ii > limit:
                    break
        # mlog.info("files: %s",files)
        if not files:
            print("No file was selected")
            return
        dfs = []
        s_dfs = []
        ii = 0
        ff = 0
        prev_exp = -1
        folders = {}
        for f in tqdm(files):
            if f.endswith(".tsv"):
                df = pd.read_table(f, low_memory=False)
            elif f.endswith(".json"):
                df = load_results(f)
            force_fid = False
            sfid = file_id.split("@")
            fid = sfid[0]
            if global_search: 
                col = "pred_text1"
                val = global_search
                if "@" in global_search:
                    val, col = global_search.split("@")
                values = df[col].unique()
                if val in values:
                    print("path:", f)
                    print("values:", values)
                    assert False, "found!" + f
                continue
            if len(sfid) > 1:
                force_fid = sfid[1] == "force"
            if True: #force_fid:
                df["path"] = f
                df["fid"] = ii
                _dir = str(Path(f).parent)
                folder = str(Path(f).parent) 
                if not folder in folders:
                    folders[folder] = ff
                    ff += 1
                df["folder"] = folder 
                eid = folders[folder]
                df["eid"] = eid 
                _pp = _dir + "/*.png"
                png_files = glob(_pp)
                if not png_files:
                    _pp = str(Path(_dir).parent) + "/hf*/*.png"
                    png_files = glob(_pp)
                for i,png in enumerate(png_files):
                    key = Path(png).stem
                    if not key in df:
                       df[key] = png
                if fid == "parent":
                    _ff = "@".join(f.split("/")[5:]) 
                    df["exp_name"] = Path(f).parent.stem #.replace("=","+").replace("_","+")
                else:
                    df["exp_name"] =  "_" + Path(f).stem
            if not summary:
                dfs.append(df)
            else:
                if prev_exp != eid:
                    if prev_exp >= 0:
                        sdf = pd.concat(s_dfs, ignore_index = True)
                        sdf = add_cols(sdf)
                        sdf = add_scores(sdf)
                        sdf = grouping(sdf)
                        sdf = summarize(sdf)
                        dfs.append(sdf)
                        s_dfs = []
                    prev_exp = eid
            s_dfs.append(df)
            ii += 1
        if len(dfs) > 0:
            df = pd.concat(dfs, ignore_index=True)
            return df
        return None

def start(stdscr):
    global info_bar, text_win, cmd_win, std, main_win, colors, dfname, STD_ROWS, STD_COLS
    std = stdscr
    now = datetime.now()
    std.bkgd(' ', cur.color_pair(TEXT_COLOR)) # | cur.A_REVERSE)
    rows, cols = std.getmaxyx()
    set_max_rows_cols(rows, cols) 
    height = rows - 1
    width = cols
    # mouse = cur.mousemask(cur.ALL_MOUSE_EVENTS)
    text_win = cur.newpad(rows * 10, cols * 10)
    text_win.bkgd(' ', cur.color_pair(TEXT_COLOR)) # | cur.A_REVERSE)
    cmd_win = cur.newwin(1, cols, rows - 1, 0)

    info_bar = cur.newpad(rows*10, cols*10)
    info_bar.bkgd(' ', cur.color_pair(HL_COLOR)) # | cur.A_REVERSE)

    cur.start_color()
    cur.curs_set(0)
    # std.keypad(1)
    cur.use_default_colors()

    colors = [str(y) for y in range(-1, cur.COLORS)]
    if cur.COLORS > 100:
        colors = [str(y) for y in range(-1, 100)] + [str(y) for y in range(107, cur.COLORS)]


    theme = {'preset': 'default', "sep1": "colors", 'text-color': '247', 'back-color': '233', 'item-color': '71','cur-item-color': '251', 'sel-item-color': '33', 'title-color': '28', "sep2": "reading mode",           "dim-color": '241', 'bright-color':"251", "highlight-color": '236', "hl-text-color": "250", "inverse-highlight": "True", "bold-highlight": "True", "bold-text": "False", "input-color":"234", "sep5": "Feedback Colors"}

    reset_colors(theme)
    show_df(data_frame, summary=global_summary)
    cur.flash()

@click.command(context_settings=dict(
            ignore_unknown_options=True,
            allow_extra_args=True,))
@click.argument("fname", nargs=-1, type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
@click.option(
    "--fid",
    "-fid",
    default="parent",
    type=str,
    help=""
)
@click.option(
    "--ftype",
    "-ft",
    default="tsv",
    type=str,
    help=""
)
@click.option(
    "--dpy",
    "-d",
    is_flag=True,
    help=""
)
@click.option(
    "--chk_time",
    "-t",
    is_flag=True,
    help=""
)
@click.option(
    "--summary",
    "-s",
    is_flag=True,
    help=""
)
@click.option(
    "--hkey",
    "-h",
    default="CG",
    type=str,
    help=""
)
@click.option(
    "--cmd",
    "-c",
    default="",
    type=str,
    help=""
)
@click.option(
    "--search",
    "-f",
    default="",
    type=str,
    help=""
)
@click.option(
    "--limit",
    "-l",
    default=-1,
    type=int,
    help="Limit of datasets to load"
)
@click.pass_context
def main(ctx, fname, path, fid, ftype, dpy, summary, hkey, cmd, search, limit, chk_time):
    if dpy:
        port = 1234
        debugpy.listen(('0.0.0.0', int(port)))
        print("Waiting for client at run...port:", port)
        debugpy.wait_for_client()  # blocks execution until client is attached
    global dfname, hotkey, global_cmd, global_search,check_time, data_frame, root_path, global_summary
    if summary:
        hkey = hkey.replace("C","").replace("G","")
    global_summary = summary
    check_time = chk_time
    global_search = search
    root_path = path
    hotkey = hkey 
    global_cmd = cmd
    dfname = fname
    if not fname:
        fname = [ftype]
    set_app("showdf")
    data_frame = get_files(path, fname, ftype, summary=summary, limit=limit, file_id= fid)
    if data_frame is not None:
        dfname = "merged"
        wrapper(start)
    else:
        mlog.info("No tsv or json file was found")

if __name__ == "__main__":
    main()
