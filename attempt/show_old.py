import curses as cur
from distutils.dir_util import copy_tree
import subprocess
from functools import reduce
import matplotlib.pyplot as plt
from curses import wrapper
from tabulate import tabulate
import click
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
import json
from tqdm import tqdm
# from comet.utils.myutils import *
from attempt.utils.utils import combine_x,combine_y,add_margin
file_id = "name"
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import sklearn
import sklearn.metrics
import attempt.metrics.metrics as mets

table_mid_template = """
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
table_fp_template = """
    \\begin{{table*}}[h!]
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

def remove_uniques(df, sel_cols, tag_cols, keep_cols = []):
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
        if c in ["exp_id", "expid", "rouge_score", "pred_max_num"] + keep_cols:
            _sel_cols.append(c)
        elif _count > 1: 
           _sel_cols.append(c)
        else:
           _info_cols.append(c) 
    if _sel_cols:
        for _col in tag_cols:
            if not _col in _sel_cols:
                _sel_cols.append(_col)

    return _sel_cols, _info_cols, tag_cols

def list_dfs(df, main_df, s_rows, FID):
    dfs_items = [] 
    dfs = []
    ii = 0
    dfs_val = {}
    for s_row in s_rows:
        exp=df.iloc[s_row]["exp_id"]
        prefix=df.iloc[s_row]["prefix"]
        dfs_val["exp" + str(ii)] = exp
        mlog.info("%s == %s", FID, exp)
        cond = f"(main_df['{FID}'] == '{exp}') & (main_df['prefix'] == '{prefix}')"
        tdf = main_df[(main_df[FID] == exp) & (main_df['prefix'] == prefix)]
        tdf = tdf[["pred_text1", "exp_name", "id","hscore", "bert_score","query", "resp", "template", "rouge_score", "fid","prefix", "input_text","target_text", "sel"]]
        tdf = tdf.sort_values(by="rouge_score", ascending=False)
        sort = "rouge_score"
        dfs.append(tdf)
    return dfs

def find_common(df, main_df, on_col_list, s_rows, FID, char, tag_cols):
    dfs_items = [] 
    dfs = []
    ii = 0
    dfs_val = {}
    for s_row in s_rows:
        exp=df.iloc[s_row]["exp_id"]
        prefix=df.iloc[s_row]["prefix"]
        dfs_val["exp" + str(ii)] = exp
        mlog.info("%s == %s", FID, exp)
        cond = f"(main_df['{FID}'] == '{exp}') & (main_df['prefix'] == '{prefix}')"
        tdf = main_df[(main_df[FID] == exp) & (main_df['prefix'] == prefix)]
        tdf = tdf[tag_cols + ["pred_text1", "top_pred", "top", "exp_name", "id","hscore", "bert_score","query", "resp", "template", "rouge_score", "fid","prefix", "input_text","target_text", "sel"]]
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
        if char == "n":
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
    all_exps = main_df['expid'].unique()
    for exp in all_exps:
        for task in main_df["prefix"].unique():
            cond = ((main_df['expid'] == exp) & (main_df["prefix"] == task))
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
            for met in metrics_list:
                for k,v in met.items():
                    infos.append(exp + ":" + task + ":" + str(k) + ":" + str(v))
                    infos.append("---------------------------------------------")
    return infos

def show_df(df):
    global dfname, hotkey, global_cmd

    hk = hotkey
    cmd = global_cmd 
    sel_row = 0
    cur_col = 0
    ROWS, COLS = std.getmaxyx()
    ch = 1
    left = 0
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
    stats = []
    col_widths = load_obj("widths", "")
    def refresh():
        text_win.refresh(0, left, 0, 0, ROWS-1, COLS-2)
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
            exp=df.iloc[s_row]["exp_id"]
            tdf = main_df[main_df["fid"] == exp]
            spath = tdf.iloc[0]["path"]
            tdf.to_csv(spath, sep="\t", index=False)


    if not col_widths:
        col_widths = {"query":50, "model":30, "pred_text1":30, "epochs":30, "date":30, "rouge_score":7, "bert_score":7, "input_text":50}

    df['id']=df.index
    df = df.reset_index(drop=True)
    if not "tag" in df:
        df["tag"] = np.NaN 

    #if not "word_score" in df:
    #    df['word_score'] = df['pred_text1'].str.split().str.len()

    if not "hscore" in df:
        df["hscore"] = np.NaN 

    if not "pid" in df:
        df["pid"] = 0
    if not "l1_decoder" in df:
        df["l1_decoder"] ="" 
        df["l1_encoder"] ="" 
        df["cossim_decoder"] ="" 
        df["cossim_encoder"] ="" 

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

    if "exp_id" in df:
        df = df.rename(columns={"exp_id":"expid"})

    if "input_text" in df:
        df['input_text'] = df['input_text'].str.replace('##','')
        df['input_text'] = df['input_text'].str.split('>>').str[0]
        df['input_text'] = df['input_text'].str.strip()

    if not "m_score" in df:
        calc_metrics(df)

    main_df = df
    edit_col = ""
    count_col = ""
    extra = {"filter":[], "inp":""}
    save_obj(dfname, "dfname", "")
    sel_cols = list(df.columns)
    for col in df.columns:
        if col.endswith("score"):
            df[col] = pd.to_numeric(df[col])
    fav_path = os.path.join(base_dir, dfname + "_fav.tsv")
    if Path(fav_path).exists():
        fav_df = pd.read_table(fav_path)
    else:
        fav_df = pd.DataFrame(columns = df.columns)
    sel_path = os.path.join(home, "atomic2020", "sel-test", "test.tsv")
    if Path(sel_path).exists():
        sel_df = pd.read_table(sel_path)
        if not "sel" in sel_df:
            sel_df["sel"] = False
    else:
        sel_df = pd.DataFrame(columns = ["prefix","input_text","target_text", "sel"])
        sel_df.to_csv(sel_path, sep="\t", index=False)

    back = []
    sels = []
    filter_df = main_df
    tag_cols = []
    if "taginfo" in df:
        tags = df.loc[0, "ftag"]
        tags = tags.replace("'", "\"")
        tags = json.loads(tags)
        tag_cols = list(tags.keys())
    if "expid" in tag_cols:
        tag_cols.remove("expid")
    if "expid" in df:
        df["expid"] = df["expid"].astype(str)

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

    #wwwwwwwwww
    colors = ['blue','teal','orange', 'red', 'purple', 'brown', 'pink','gray','olive','cyan']
    contexts = {"g":"main", "G":"main", "X":"view", "r":"main"}
    ax = None
    if "Z" in hotkey:
        df["m_score"] = df["rouge_score"]
    context = dfname
    font = ImageFont.truetype("/usr/share/vlc/skins2/fonts/FreeSans.ttf", 48)
    seq = ""
    reset = False
    back_sel_cols = []
    search = ""
    sort = "rouge_score"
    on_col_list = []
    keep_cols = []
    rep_cols = load_obj("rep_cols", "gtasks", [])
    score_cols = [] #load_obj("score_cols", "gtasks", ["m_score"])
    unique_cols = []
    group_sel_cols = []
    sel_fid = "" 
    df_cond = True
    open_dfnames = [dfname]
    dot_cols = {}
    selected_cols = []
    rep_cmp = load_obj("rep_cmp", "gtasks", {})
    settings = load_obj("settings", "gtasks", {})
    rname = settings.setdefault("rname", "rpp")
    task = ""
    if "prefix" in df:
        task = df["prefix"][0]
    #if not "learning_rate" in df:
    #    df[['fid_no_lr', 'learning_rate']] = df['fid'].str.split('_lr_', 1, expand=True)
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
    group_rows = []
    def row_print(df, col_widths ={}, _print=False):
        nonlocal group_rows
        infos = []
        margin = min(len(df), 5)
        sel_dict = {}
        g_row = ""
        g = 0
        g_start = -1
        row_color = TEXT_COLOR
        sel_col_color = TITLE_COLOR
        cross_color = HL_COLOR   
        sel_row_color = SEL_COLOR
        g_color = row_color
        group_mode = group_col and group_col in sel_cols 
        _sel_row = -1 if group_mode else sel_row 
        ii = 0 
        for idx, row in df.iterrows():
           text = "{:<5}".format(ii)
           _sels = []
           _infs = []
           if (group_mode and group_col in row and row[group_col] != g_row):
               g_row = row[group_col]
               if _print and _sel_row >= 0 and ii >= _sel_row - 1:
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
               if g == sel_group:
                  _sel_row = ii
                  #row_color = SEL_COLOR
                  #g_color = WARNING_COLOR
                  g_start = ii
               g+=1
           if _sel_row < 0 or ii < _sel_row - margin:
               ii += 1
               continue

           if group_mode: cross_color = sel_col_color
           _color = row_color
           if cur_col < 0:
              _color = sel_col_color
           if ii in sel_rows:
               _color = MSG_COLOR
           if ii == _sel_row and not group_mode:
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
                   if ii == _sel_row and not sel_col in _infs:
                      infos.append(_info)
                      _infs.append(sel_col)
               if ii == _sel_row:
                   sel_dict[sel_col] = row[sel_col]
               if not sel_col in col_widths:
                   col_widths[sel_col] = len(content) + 4
               if len(content) > col_widths[sel_col]:
                   col_widths[sel_col] = len(content) + 4
               min_width = 30
               if content.strip().isdigit():
                   min_width = 5 
               col_widths[sel_col] = min(col_widths[sel_col], min_width)
               _w = col_widths[sel_col] 
               if sel_col in sel_cols:
                   if (cur_col >=0 and cur_col < len(sel_cols) 
                          and sel_col == sel_cols[cur_col]):
                       if ii == _sel_row: 
                          cell_color = cross_color 
                       else:
                          cell_color = sel_col_color
                   else:
                       if sel_col in selected_cols:
                          cell_color = sel_col_color
                       elif sel_col == group_col:
                          cell_color = g_color
                       elif ii in sel_rows:
                          cell_color = MSG_COLOR
                       elif ii == _sel_row:
                          cell_color = sel_row_color
                       else:
                          cell_color = row_color
                   text = textwrap.shorten(text, width=36, placeholder="...")
                   text = "{:<{x}}".format(content, x= _w)
                   if _print:
                       mprint(text, text_win, color = cell_color, end="") 
                   _sels.append(sel_col)

           _end = "\n"
           if _print:
               mprint("", text_win, color = _color, end="\n") 
           ii += 1
           if ii > _sel_row + ROWS:
               break
        return infos, col_widths

    def backit(df, sel_cols):
        back.append(df)
        sels.append(sel_cols.copy())
        back_rows.append(sel_row)
        back_infos.append(info_cols.copy())
    for _col in ["input_text","pred_text1","target_text"]:
        if _col in df:
            df[_col] = df[_col].astype(str)

    map_cols =  load_obj("map_cols", "atomic", {})
    def get_images(df, exps, fid='expid'):
        imgs = {}
        dest = ""
        start = "pred"
        fnames = []
        for exp in exps:
            cond = f"(main_df['{fid}'] == '{exp}')"
            tdf = main_df[main_df[fid] == exp]
            if tdf.empty:
                return dest, imgs, fnames
            path=tdf.iloc[0]["path"]
            path = Path(path)
            #_selpath = os.path.join(path.parent, "pred_sel" + path.name) 
            #shutil.copy(path, _selpath)
            # grm = tdf.iloc[0]["gen_route_methods"]
            runid = tdf.iloc[0]["runid"]
            run = "wandb/offline*" + str(runid) + f"/files/media/images/{start}*.png"
            paths = glob(str(path.parent.parent) +"/" +run)
            # paths = glob(run)
            spath = "images/" + str(runid)
            if Path(spath).exists():
                shutil.rmtree(spath)
            Path(spath).mkdir(parents=True, exist_ok=True)
            images = []
            kk = 1
            key = exp # "single"
            ii = 0
            for img in paths: 
                fname = Path(img).stem
                if fname in fnames:
                    continue
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
                sorted_keys = reversed(sorted(img_dict.keys()))
                img_list = [img_dict[k] for k in sorted_keys] 
                if len(img_list) > 1:
                    new_im = combine_x(img_list)
                    name = str(key) 
                    dest = os.path.join("temp", name.strip("-") + ".png")
                    new_im.save(dest)
                    c_imgs[key] = [new_im] 
                    fnames.append(dest)
            imgs = c_imgs
        return dest, imgs, fnames




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
            "bert_score":"bt",
            }
    wraps = {
            "tag":20,
            }
    adjust = True
    show_consts = True
    show_extra = False
    consts = {}
    extra = {"filter":[]}
    orig_df = main_df.copy()
    prev_char = ""
    while prev_char != "q":
        text_win.clear()
        group_rows = []
        left = min(left, max_col  - width)
        left = max(left, 0)
        top = min(top, max_row  - height)
        top = max(top, 0)
        sel_row = min(sel_row, len(df) - 1)
        sel_row = max(sel_row, 0)
        sel_rows = list(set(sel_rows))
        sel_group = max(sel_group, 0)
        #sel_group = min(sel_row, sel_group)
        cur_col = min(cur_col, len(sel_cols) - 1)
        cur_col = max(cur_col, -1)
        if not hotkey:
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
               text += "{:<{x}}".format(head, x=_w) 
            mprint(text, text_win) 
            #fffff
            infos,_ = row_print(df, col_widths, True)
            refresh()
        if cur_col < len(sel_cols) and len(sel_cols) > 0:
            _sel_col = sel_cols[cur_col]
            infos.append(_sel_col)
        for c in info_cols:
            if not c in df:
                continue
            if "score" in c:
                mean = df[c].mean()
                _info = f"Mean {c}:" + "{:.2f}".format(mean)
                infos.append(_info)
        infos.append("-------------------------")
        if show_consts:
            consts["len"] = str(len(df))
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
        change_info(infos)

        prev_char = chr(ch)
        prev_cmd = cmd
        if global_cmd and not hotkey or hotkey == "q":
            cmd = global_cmd
            global_cmd = ""
        else:
            cmd = ""
        if hotkey == "":
            ch = std.getch()
        else:
            ch, hotkey = ord(hotkey[0]), hotkey[1:]
        char = chr(ch)
        if char != "q" and prev_char == "q": 
            consts["exit"] = ""
            prev_char = ""
        extra["inp"] = char

        seq += char
        vals = []
        get_cmd = False
        adjust = True
        # context = contexts[char] if char in contexts else char
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
        if ch == SUP:
            info_cols = info_cols_back.copy()
        if ch == LEFT:
            cur_col -= 1
            cur_col = max(-1, cur_col)
            width = col_widths[sel_cols[cur_col]]
            _sw = sum([col_widths[x] for x in sel_cols[:cur_col]])
            if _sw < left:
                left = _sw - width - 10 
            adjust = False
        if ch == RIGHT:
            cur_col += 1
            cur_col = min(len(sel_cols)-1, cur_col)
            width = col_widths[sel_cols[cur_col]]
            _sw = sum([col_widths[x] for x in sel_cols[:cur_col]])
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
            elif group_col and group_col in sel_cols:
                sel_group +=1
            else:
                sel_row += 1
            adjust = False
        elif ch == UP: 
            if context == "inp":
                back_rows[-1] -= 1
                hotkey = "bp"
            elif group_col and group_col in sel_cols:
                sel_group -=1
            else:
                sel_row -= 1
            adjust = False
        elif ch == cur.KEY_SRIGHT:
            sel_row += ROWS - 4
        elif ch == cur.KEY_HOME:
            sel_row = 0 
            sel_group = 0
        elif ch == cur.KEY_SHOME:
            left = 0 
        elif ch == cur.KEY_END:
            sel_row = len(df) -1
        elif ch == cur.KEY_SLEFT:
            sel_row -= ROWS - 4
        elif char == "l" and prev_char == "l":
            seq = ""
        elif char == "s":
            col = sel_cols[cur_col]
            if col in selected_cols:
                if len(selected_cols) == 1:
                    df = df.sort_values(by=col, ascending=asc)
                    asc = not asc
                selected_cols.remove(col)
            else:
                selected_cols.append(col)
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
            if col == "exp_id": col = FID
            if "filter" in consts:
                consts["filter"] += " " + col + "='" + str(val) + "'"
            else:
                consts["filter"] = col + "='" + str(val) + "'"
            df_cond = df_cond & (df[col] == val)
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
        elif char in ["e","E"]:
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
            s_rows = sel_rows
            from comet.train.eval import do_score
            if not s_rows:
                s_rows = [sel_row]
            if prev_char == "x":
                s_rows = range(len(df))
            for s_row in s_rows:
                exp=df.iloc[s_row]["exp_id"]
                _score=df.iloc[s_row]["bert_score"]
                #if _score > 0:
                #    continue
                cond = f"(main_df['{FID}'] == '{exp}')"
                tdf = main_df[main_df[FID] == exp]
                #df = tdf[["pred_text1", "id", "bert_score","query", "template", "rouge_score", "fid","prefix", "input_text","target_text"]]
                spath = tdf.iloc[0]["path"]
                spath = str(Path(spath).parent)
                tdf = do_score(tdf, "rouge-bert", spath, reval=True) 
                tdf = tdf.reset_index()
                #main_df.loc[eval(cond), "bert_score"] = tdf["bert_score"]
            df = main_df
            hotkey = hk
        elif char in ["l"]:
            if char == "m": char = "i"
            backit(df, sel_cols)
            s_rows = sel_rows
            if not sel_rows:
                s_rows = group_rows
                if not group_rows:
                    s_rows = [sel_row]
            #_, start = list_values(["start","pred"])
            s_rows = set(s_rows)
            exprs = []
            for s_row in s_rows:
                exp=df.iloc[s_row]["expid"]
                exprs.append(exp)
            dest, imgs, fnames = get_images(df, exprs)
            subprocess.run(["eog", dest])
        if char in ["o","O"]:
            _agg = {}
            for c in df.columns:
                if c.endswith("score"):
                    _agg[c] = "mean"
                else:
                    _agg[c] = "first"
            if char == "O":
                pdf = df
            else:
                pdf = df.groupby(["expid","prefix"]).agg(_agg).reset_index(drop=True)
                pdf = pdf.sort_values(by=["expid","rouge_score"], ascending=False)
            images = []
            i = 0
            eid = -1
            start = 0
            for idx, row in pdf.iterrows(): 
                if i < sel_row:
                    i += 1
                    start += 1
                    continue
                if row["expid"] != eid:
                    eid = row["expid"]
                    paths = glob("**/images/pred_router_"+ str(row["expid"]) + ".png")
                    if not paths:
                        paths = glob("images/pred_router_"+ str(row["expid"]) + ".png")
                    if paths:
                        im = Image.open(paths[0])
                        xx = 100
                        _image = add_margin(im, 0, 0, 0, 700, (255, 255, 255))
                        draw = ImageDraw.Draw(_image)
                yy = 10
                if False:
                    for cc in ["prefix", "rouge_score", "bert_score", "num_preds"] + tag_cols:
                        if cc.endswith("score"):
                            mm = map_cols[cc] if cc in map_cols else cc
                            if xx == 100:
                                draw.text((10, yy),"{}".format(mm),
                                    (20,25,255),font=font)
                            draw.text((xx, yy),"{:.2f}".format(row[cc]),
                                    (230,5,5),font=font)
                        else:
                            mm = map_cols[cc] if cc in map_cols else cc
                            if xx == 100:
                                draw.text((10, yy),"{}".format(mm),(20,25,255),font=font)
                            draw.text((xx, yy),"{}".format(row[cc]),(0,5,5),font=font)
                        yy += 60
                if xx == 100:
                    images.append(_image)
                xx += 180
                i+=1 
                if i >= start + 10:
                    break
            pic = combine_y(images)
            dest = os.path.join("routers.png")
            pic.save(dest)
            #pname=df.iloc[sel_row]["image"]
            subprocess.run(["eog", dest])
        elif char == "L":
            s_rows = sel_rows
            if not sel_rows:
                s_rows = group_rows
                if not group_rows:
                    s_rows = [sel_row]
            all_rows = range(len(df))
            Path("temp").mkdir(parents=True, exist_ok=True)
            imgs = []
            for s_row in all_rows:
                exp=df.iloc[s_row]["exp_id"]
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
            exp=df.iloc[sel_row]["exp_id"]
            cond = f"(main_df['{FID}'] == '{exp}')"
            tdf = main_df[main_df[FID] == exp]
            path=tdf.iloc[0]["path"]
            conf = os.path.join(str(Path(path).parent), "exp.json")
            with open(conf,"r") as f:
                infos = f.readlines()
            subwin(infos)
        elif char == "l" and prev_char == "x":
            exp=df.iloc[sel_row]["expid"]
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
        elif (char == "i" and not prev_char == "x" and hk=="G"):
            backit(df,sel_cols)
            exp=df.iloc[sel_row]["exp_id"]
            cond = f"(main_df['{FID}'] == '{exp}')"
            df = main_df[main_df[FID] == exp]
            sel_cols=tag_cols + ["bert_score","pred_text1","target_text","input_text","rouge_score","prefix"]
            sel_cols, info_cols, tag_cols = remove_uniques(df, sel_cols, orig_tag_cols)
            unique_cols = info_cols.copy()
            df = df[sel_cols]
            df = df.sort_values(by="input_text", ascending=False)
        elif char == "m":
            canceled, col = list_values(rep_cols)
            if not canceled:
                if col in rep_cols: 
                    rep_cols.remove(col)
            save_obj(rep_cols, "rep_cols", "gtasks")
            cmd = "report"
        elif char == "N":
            canceled, col, val = list_df_values(df, get_val=False)
            if not canceled:
                if col in rep_cols: 
                    rep_cols.remove(col)
                rep_cols.insert(0, col)
            save_obj(rep_cols, "rep_cols", "gtasks")
            cmd = "report"
        elif char == "Q":
            canceled, col = list_values(["m_score","word_score","preds_num","rouge_score","bert_score"])
            if not canceled:
                if col in score_cols: 
                    score_cols.remove(col)
                score_cols = [col]
            save_obj(score_cols, "score_cols", "gtasks")
            char = "Z"
        elif char == "I" or ch == cur.KEY_IC:
            canceled, col, val = list_df_values(main_df, get_val=False)
            if not canceled:
                if col in sel_cols: 
                    sel_cols.remove(col)
                if col in info_cols:
                    info_cols.remove(col)
                if ch == cur.KEY_IC:
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
        elif char == "h":
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
                mlog.info("g_list: %s", g_list)
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
        elif char == "x" and prev_char == "x":
            backit(df, sel_cols)
            df = sel_df
        # png files
        elif char == "l" and prev_char == "x":
            df = main_df.groupby(["l1_decoder", "template", "model", "prefix"], as_index=False).first()
            sel_cols = ["l1_decoder", "template", "model", "prefix"]
        elif char == "z" and prev_char == "x":
            fav_df = fav_df.append(df.iloc[sel_row])
            mbeep()
            fav_df.to_csv(fav_path, sep="\t", index=False)
        elif char == "Z" and prev_char == "x":
            main_df["m_score"] = main_df["rouge_score"]
            df = main_df
            hotkey = "CGR"
            #backit(df, sel_cols)
            #df = fav_df
        elif char == "j":
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
            if col == "exp_id": col = FID
            if col == "fid":
                sel_fid = exp
            mlog.info("%s == %s", col, exp)
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
        elif char == "g":
            if cur_col < len(sel_cols):
                col = sel_cols[cur_col]
                if col == group_col:
                    group_col = ""
                    sel_row = 0
                    sel_group = 0
                else:
                    group_col = col
                    sel_row = 0
                    sel_group = 0
                    df = df.sort_values(by=[group_col, sort], ascending=[True, False])
        elif char == "c" and not prev_char in ["c", "p"]:
            backit(df, sel_cols)
            if not "expid" in sel_cols:
                sel_cols.insert(1, "expid")
            _agg = {}
            for c in sel_cols:
                if c.endswith("score"):
                    _agg[c] = "mean"
                else:
                    _agg[c] = "first"
            consts["options"] = "c: for expid,   p: for expid and prefix"
        elif char == "p" and prev_char == "c":
            df = back[-1]
            consts["options"] = "b: back"
            df = df.groupby(["expid","prefix"]).agg(_agg).reset_index(drop=True)
            df = df.sort_values(by=["expid","prefix"], ascending=False)
        elif char == "A": 
            consts["options"] = "b: back"
            backit(df, sel_cols)
            if not "expid" in sel_cols:
                sel_cols.insert(1, "expid")
            _agg = {}
            for c in sel_cols:
                if c.endswith("score"):
                    _agg[c] = "mean"
                else:
                    _agg[c] = "first"
            df = df.groupby(["expid"]).agg(_agg).reset_index(drop=True)
            df = df.sort_values(by=["m_score"], ascending=False)
            sort = "rouge_score"
        elif char == "a": 
            consts["options"] = "b: back"
            backit(df, sel_cols)
            col = sel_cols[cur_col]
            _agg = {}
            for c in sel_cols:
                if c.endswith("score"):
                    _agg[c] = "mean"
                else:
                    _agg[c] = "first"
            df = df.groupby([col]).agg(_agg).reset_index(drop=True)
            df = df.sort_values(by=["m_score"], ascending=False)
            sort = "m_score"
        elif char == "u":
            infos = calc_metrics(main_df)
            subwin(infos)
        elif char == "U" and prev_char == "x": 
            if sel_col:
                df = df[sel_col].value_counts(ascending=False).reset_index()
                sel_cols = list(df.columns)
                col_widths["index"]=50
                info_cols = []
        elif char == "C": 
            score_col = "rouge_score"
            backit(df, sel_cols)
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

            extra["filter"].append("group predictions")
        elif char == " ":
            if sel_row in sel_rows:
                sel_rows.remove(sel_row)
            else:
                sel_rows.append(sel_row)
            adjust = False
        elif char == "?": 
            if not sel_rows:
                tinfo=df.iloc[sel_row]["ftag"]
                infos = tinfo.split(",")
                infos.append(main_df.loc[0, "path"])
                subwin(infos)
            else:
                s1 = sel_rows[0]
                s2 = sel_rows[1]
                infos = []
                for col in df.columns:
                    if (col == "ftag" or 
                        col == "extra_fields" or col == "path" or col == "full_tag" 
                        or col.startswith("test_") or "output_dir" in col):
                        continue
                    i1 = df.iloc[s1][col]
                    i2 = df.iloc[s2][col]
                    if str(i1) != str(i2):
                       infos.append(col + ":" + str(i1))
                       infos.append(col + ":" + str(i2))
                subwin(infos)
        elif char == "z" and prev_char == "z":
            consts["context"] = context
            sel_cols =  load_obj("sel_cols", context, [])
            info_cols = load_obj("info_cols", context, [])
        elif char == "G":
            backit(df, sel_cols)
            context = "main"
            if FID == "input_text":
                context = "inp2"
            col = FID
            left = 0
            col = [col, "prefix"]
            sel_cols =  load_obj("sel_cols", context, [])
            sel_cols = list(set(sel_cols))
            info_cols = load_obj("info_cols", context, [])
            if reset:
                info_cols = ["bert_score", "num_preds"]
            if reset: #col == "fid":
                sel_cols = ["expid", "rouge_score"] + tag_cols + ["method", "trial", "prefix","num_preds", "bert_score", "pred_max_num","pred_max", "steps","max_acc","best_step", "st_score", "learning_rate",  "num_targets", "num_inps", "train_records", "train_records_nunique", "group_records", "wrap", "frozen", "prefixed"] 
            reset = False

            _agg = {}
            group_sel_cols = sel_cols.copy()
            for c in df.columns:
                if c.endswith("score"):
                    _agg[c] = "mean"
                else:
                    _agg[c] = ["first", "nunique"]
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
                if c == FID + "_first":
                    ren[c] = "exp_id"
                elif c.endswith("_mean"):
                    ren[c] = c.replace("_mean","")
                elif c.endswith("_first"):
                    ren[c] = c.replace("_first","")
            df = df.rename(columns=ren)
            if not "num_preds" in sel_cols:
                sel_cols.append("num_preds")
            df["avg_len"] = avg_len
            df = df.sort_values(by = ["rouge_score"], ascending=False)
            sort = "rouge_score"
        elif char == "z": 
            if len(df) > 1:
                sel_cols, info_cols, tag_cols = remove_uniques(df, sel_cols, 
                        orig_tag_cols, keep_cols)
                unique_cols = info_cols.copy()
            info_cols_back = info_cols.copy()
            info_cols = []
            save_obj(sel_cols, "sel_cols", context)
            save_obj(info_cols, "info_cols", context)
        elif char == "T":
            s_rows = sel_rows
            if not sel_rows:
                s_rows = [sel_row]
            pfix = ""
            for s_row in s_rows:
                exp=df.iloc[s_row]["exp_id"]
                score=df.iloc[s_row]["rouge_score"]
                cond = f"(main_df['{FID}'] == '{exp}')"
                tdf = main_df[main_df[FID] == exp]
                prefix=tdf.iloc[0]["prefix"]
                expid=tdf.iloc[0]["expid"]
                path=tdf.iloc[0]["path"]
                js = os.path.join(str(Path(path).parent), "exp.json")
                fname ="exp_" + str(expid) + "_" + prefix + "_" + str(round(score,2)) + ".json"
                if not pfix:
                    pfix = rowinput("prefix:")
                fname = pfix + "_" + fname
                dest = os.path.join(home, "results", fname)
                shutil.copyfile(js, dest)
        elif char == "U":
            left = 0
            backit(df, sel_cols)

            s_rows = sel_rows
            if not sel_rows:
                s_rows = [sel_row]
            cond = ""
            for s_row in s_rows:
                exp=df.iloc[s_row]["exp_id"]
                cond += f"| (main_df['{FID}'] == '{exp}') "
            cond = cond.strip("|")
            filter_df = main_df[eval(cond)]
            df = filter_df.copy()
            sel_rows = []
            FID = "input_text"
            hotkey = hk
        elif char in ["n", "p", "t", "i"]: # and prev_cahr != "x" and hk == "gG":
            left = 0
            context= "comp"
            cur_col = -1
            sel_group = 0
            s_rows = sel_rows
            if not sel_rows:
                s_rows = group_rows
                if not group_rows:
                    s_rows = [sel_row]
            sel_rows = sorted(sel_rows)
            if sel_rows:
                sel_row = sel_rows[-1]
            backit(df, sel_cols)
            sel_rows = []
            on_col_list = ["pred_text1"]
            other_col = "target_text"
            if char =="i": 
                group_col = "input_text"
            #    on_col_list = ["input_text"] 
            #    other_col = "pred_text1"
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
                            _df, sel_exp = find_common(df, filter_df, on_col_list, _rows, FID, char)
                            dfs.append(_df)
                df = pd.concat(dfs,ignore_index=True)
                #df = df.sort_values(by="int", ascending=False)
            elif len(s_rows) > 1:
                sel_cols=orig_tag_cols + ["bert_score","pred_text1","target_text","input_text","rouge_score","prefix"]
                sel_cols, info_cols, tag_cols = remove_uniques(df, sel_cols, orig_tag_cols)
                unique_cols = info_cols.copy()
                _cols = tag_cols
                df, sel_exp, dfs = find_common(df, main_df, on_col_list, _rows, 
                                               FID, char, _cols)
                df = pd.concat(dfs).sort_index(kind='mergesort')
                _all = len(df)
                df =df.sort_values(by='input_text').drop_duplicates(subset=['input_text', 'pred_text1',"prefix"], keep=False)
                _common = _all - len(df)
                consts["Common"] = str(_common) + "| {:.2f}".format(_common / _all)
                #df = df.sort_values(by="input_text", ascending=False)
            else:
                path = df.iloc[sel_row]["path"]
                path = Path(path)
                #_selpath = os.path.join(path.parent, "sel_" + path.name) 
                #shutil.copyfile(path, _selpath)
                exp=df.iloc[sel_row]["exp_id"]
                sel_exp = exp
                #FID="expid"
                cond = f"(main_df['{FID}'] == '{exp}')"
                df = main_df[main_df[FID] == exp]
                if "prefix" in df:
                    task = df.iloc[0]["prefix"]
                sel_cols=orig_tag_cols + ["bert_score","pred_text1","top_pred", "top", "target_text","input_text","rouge_score","prefix"]
                sel_cols, info_cols, tag_cols = remove_uniques(df, sel_cols, orig_tag_cols)
                unique_cols = info_cols.copy()
                df = df[sel_cols]
                df = df.sort_values(by="input_text", ascending=False)
                sort = "input_text"
                info_cols = []
                df = df.reset_index()
            if len(df) > 1:
                sel_cols=orig_tag_cols + ["bert_score","pred_text1", "target_text", "top_pred", "input_text", "rouge_score","prefix"]
                sel_cols, info_cols, tag_cols = remove_uniques(df, sel_cols, orig_tag_cols)
                unique_cols = info_cols.copy()
                info_cols_back = info_cols.copy()
                info_cols = []

        elif char == "M" and prev_char != "x":
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
        elif char == "D": 
            s_rows = sel_rows
            if FID == "fid":
                mdf = main_df.groupby("fid", as_index=False).first()
                mdf = mdf.copy()
                _sels = df["exp_id"]
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

        elif char == "P" and prev_char == "x":
            cols = get_cols(df,2)
            if cols:
                df = df.sort_values(cols[1])
                x = cols[0]
                y = cols[1]
                #ax = df.plot.scatter(ax=ax, x=x, y=y)
                ax = sns.regplot(df[x],df[y])
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
           df = df[df_cond]
           df_cond = True
        elif is_enter(ch) or char in ["f", "F"]:
            backit(df, sel_cols)
            if is_enter(ch): char = "f"
            col = sel_cols[cur_col]
            if col == "exp_id": col = FID
            if char == "f":
                canceled, col, val = list_df_values(main_df, col, get_val=True)
            else:
                canceled, col, val = list_df_values(df, col, get_val=True)
            cond = ""
            if not canceled:
               if type(val) == str:
                  cond = f"df['{col}'] == '{val}'"
               else:
                  cond = f"df['{col}'] == {val}"
            mlog.info("cond %s, ", cond)
            if char == "f":
               df = main_df
            if cond:
               df = df[eval(cond)]
               #df = df.reset_index()
               filter_df = df
               if not "filter" in extra:
                  extra["filter"] = []
               extra["filter"].append(cond)
               sel_row = 0
            if char == "f":
               hotkey = hk
               keep_cols.append(col)
        if char == "V":
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
        if char in ["y","Y"]:
            #yyyyyyyy
           cols = get_cols(df, 2)
           backit(df, sel_cols)
           if cols:
               gcol = cols[0]
               y_col = cols[1]
               if char == "Y":
                   cond = get_cond(df, gcol, 10)
                   df = df[eval(cond)]
               gi = 0 
               name = ""
               for key, grp in df.groupby([gcol]):
                     ax = grp.sort_values('steps').plot.line(ax=ax,linestyle="--",marker="o",  x='steps', y=y_col, label=key, color=colors[gi])
                     gi += 1
                     if gi > len(colors) - 1: gi = 0
                     name += key + "_"
               ax.set_xticks(df["steps"].unique())
               ax.set_title(name)
               if not "filter" in extra:
                   extra["filter"] = []
               extra["filter"].append("group by " + name)
               char = "H"
        if char == "H":
            name = ax.get_title()
            pname = rowinput("Plot name:", name[:30])
            if pname:
                folder = ""
                if "/" in pname:
                    folder, pname = pname.split("/")
                ax.set_title(pname)
                if folder:
                    folder = os.path.join(base_dir, "plots", folder)
                else:
                    folder = os.path.join(base_dir, "plots")
                Path(folder).mkdir(exist_ok=True, parents=True)
                pname = pname.replace(" ", "_")
                pname = os.path.join(folder, now + "_" + pname +  ".png")
                fig = ax.get_figure()
                fig.savefig(pname)
                ax = None
                if "ahmad" in home:
                    subprocess.run(["eog", pname])

        elif char == "R" and prev_char == "x":
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
        elif ch == cur.KEY_DC or char == "d":
            col = sel_cols[cur_col]
            if col in orig_tag_cols:
                orig_tag_cols.remove(col)
            if col in tag_cols:
                tag_cols.remove(col)
            sel_cols.remove(col)
            save_obj(sel_cols, "sel_cols", context)
        elif ch == cur.KEY_SDC:
            #col = sel_cols[cur_col]
            #sel_cols.remove(col)
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
        elif char == "v":
            #do_wrap = not do_wrap
            sel_rows = []
            selected_cols = []
            dot_cols = {}
            keep_cols = []
            consts = {}
            if prev_char == "x":
                info_cols = ["bert_score", "num_preds"]
            if prev_char == "x": 
                sel_cols = ["expid", "rouge_score"] + tag_cols + ["method", "trial", "prefix","num_preds", "bert_score", "pred_max_num","pred_max", "steps","max_acc","best_step", "st_score", "learning_rate",  "num_targets", "num_inps", "train_records", "train_records_nunique", "group_records", "wrap", "frozen", "prefixed"] 
                save_obj(sel_cols, "sel_cols", context)
        elif char == "M" and prev_char == "x":
            info_cols = []
            for col in df.columns:
                info_cols.append(col)
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
            if search == old_search:
                si += 1
            else:
                si = 0
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
            exp=df.iloc[sel_row]["expid"]
            tdf = main_df[main_df['expid'] == exp]
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
            char = "R"
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
        if cmd.startswith("comp"):
            if "=" in cmd:
                command = cmd.split("=")[0]
                rname = cmd.split("=")[1]
                col = cmd.split("=")[2]
                if not "@" in col:
                    sel_col = col + "@123@m_score"
                else:
                    sel_col = col + "@m_score"
            pp = f"{_dir}/report_templates/report.tex.temp"
            with open(pp, "r") as f:
                report = f.read()
            mdf = main_df
            rep = load_obj(rname, "gtasks", {})
            cats = set(rep["cats"])
            sel_cats = []
            size, seed, _ = sel_col.split("@")
            for cat in cats:
                cat_parts = cat.split("-")
                if (cat_parts[0].startswith(rname) and cat_parts[-1] == size 
                        and cat_parts[-2] == seed):
                    sel_cats.append(cat)
            all_exps = gdf['expid'].unique()
            table = latex_table(rep, rname, mdf, all_exps, sel_col, sel_cats[0])
            report = report.replace("mytable", table +"\n\n" + "mytable")
            for cat in sel_cats:
                report = report.replace("myimage", 
                        "\input{" + cat +  "_images.tex}\n\nmyimage")
            with open(f"{_dir}/report_templates/report.tex.temp2", "w") as f:
                f.write(report)
            cmd = "comp_rep"
        if (cmd.startswith("put") or 
                cmd.startswith("shv") or 
                cmd.startswith("shav") or 
                cmd.startswith("show")):
            _dir = Path(__file__).parent
            doc_dir = "/home/ahmad/logs/" #os.getcwd()
            # rname = cmd.split("@")[-1]
            # settings["rname"] = rname
            # save_obj(settings, "settings", "gtasks")
            command = "show"
            sizes, seeds = [],[]
            if "@" in cmd:
                command = cmd.split("@")[0]
                rep_names = cmd.split("@")[1:]
            elif "=" in cmd:
                command = cmd.split("=")[0]
                _sizes = cmd.split("=")[1]
                if not _sizes == "def":
                    sizes = [int(s) for s in _sizes.split("-")]
                _seeds = cmd.split("=")[2]
                if not _seeds == "def" and not _seeds == "avg":
                    seeds = [int(s) for s in _seeds.split("-")]
                rep_names = cmd.split("=")[3:]
            com3 = ""
            if rep_names[0] in ["avg", "img"]:
                com3 = rep_names[0]
                rep_names = rep_names[1:]
            pp = f"{_dir}/report_templates/report.tex.temp"
            #if Path(f"{_dir}/report_templates/report.tex." + rname).is_file():
            #    pp = f"{_dir}/report_templates/report.tex." + rname
            with open(pp, "r") as f:
                report = f.read()
            _agg = {}
            if not "m_score" in sel_cols:
                sel_cols.append("m_score")
            for c in sel_cols:
                if c in df.columns: 
                    if c.endswith("score"):
                        _agg[c] = "mean"
                    else:
                        _agg[c] = "first"
            rdf = df.groupby(["expid", "prefix"], as_index=False).agg(
                    _agg).reset_index(drop=True)
            gdf = df.groupby(["expid"], as_index=False).agg(_agg).reset_index(drop=True)
            gdf = gdf.sort_values(by=["m_score"], ascending=False)
            all_exps = gdf['expid'].unique()
            exp_names = []
            for exp in all_exps:
                exp = exp.replace("_","-")
                exp = exp.split("-")[0]
                exp_names.append(exp)

            tables, heads = [], []
            reps = []
            all_tables = ""
            table_env = table_env_template
            table_env_sdev = table_env_template
            rep_avg = {}
            for rname in rep_names:
                rep = load_obj(rname, "gtasks", {})
                reps.append(rep)
            for rep, rname in zip(reps, rep_names):
                table_cont1=""
                table_cont1 += "method & "
                head1 = "|r|"
                cols = []
                if not sizes:
                    sizes = [10, 20, 50, 100, 200]
                if not seeds:
                    seeds = [123,45,76]
                for n in sizes: 
                    for seed in seeds: 
                        col = str(n) + "@" + str(seed) + "@m_score"
                        cols.append(col)
                        if command != "shav": 
                            head1 += "r|"
                            header = str(n) + "--" + str(seed) 
                            scol = col.replace("_","-")
                            table_cont1 += " \hyperref[table:"+rname+scol+"]{"+header+"} &"
                    if command == "shv":
                        head1 += "r|"
                        table_cont1 += " \hyperref[table:" +rname+ str(n) + f"]{n} &"
                mdf = main_df

                category = mdf["experiment"].unique()[0]
                category = category.split("/")[0]
                if not "cats" in rep:
                    rep["cats"] = []
                if not category in rep["cats"]:
                    rep["cats"].append(category)

                table_cont1 = table_cont1.strip("&")
                table_cont1 += "\\\\\n"
                table_cont1 += "\\hline\n"
                train_num = str(mdf["max_train_samples"].unique()[0])
                seed = mdf["d_seed"].unique()[0]
                cur_sel_col = cc = train_num + "@" + str(seed) + "@m_score"
                if command == "put":
                    rep_exps = all_exps
                else:
                    rep_exps = list(rep[list(rep.keys())[1]].keys())
                for ii, _exp in enumerate(rep_exps):
                    exp = _exp.replace("_","-")
                    exp = exp.split("-")[0]
                    table_cont1 += "\hyperref[fig:"+ _exp + "]{"+ exp +"} & " 
                    cond = (gdf['expid'] == _exp)
                    for jj, sel_col in enumerate(cols):
                        if command != "put" and not sel_col in rep:
                            continue
                        if command != "put" and not exp in rep[sel_col]:
                            continue
                        val = ""
                        val_col = sel_col
                        cat_col = rname + sel_col
                        if "@" in sel_col:
                            val_col = sel_col.split("@")[-1]
                            cat_col = rname + sel_col.split("@")[0]
                        if command == "put":
                            if not sel_col in rep:
                                rep[sel_col] = {}
                            if not exp in rep[sel_col]:
                                rep[sel_col][exp] = {"avg":1000}
                            if exp in rep[sel_col] and rep[sel_col][exp]["avg"] != 1000:
                                val = rep[sel_col][exp]["avg"] 
                            if sel_col == cc and val_col in gdf.columns:
                                val = gdf.loc[cond, val_col].iloc[0]
                            if val:
                                val = float(val)
                                table_cont1 += f" $ {val:.1f} $ &"
                            else:
                                table_cont1 += f" $ na $ &"
                            rep[sel_col][exp]["avg"] = val
                            #########
                            if sel_col == cc and val_col in mdf.columns:
                                for rel in rdf['prefix'].unique(): 
                                    cond2=((mdf['prefix'] == rel) & (mdf["expid"] == _exp))
                                    sc = val_col 
                                    s_val = mdf.loc[cond2, sc].mean()
                                    #preds_num = mdf.loc[cond, "pred_text1"].unique()
                                    #preds_num = len(preds_num)
                                    one_pred = False #int(preds_num) == 1
                                    maxval = rdf.loc[(rdf["prefix"] == rel), sc].max()
                                    if not rel in rep[sel_col][exp]:
                                        rep[sel_col][exp][rel] = []
                                    if not cat_col in rep:
                                        rep[cat_col] = {}
                                    if not exp in rep[cat_col]:
                                        rep[cat_col][exp] = {}
                                    if not rel in rep[cat_col][exp]:
                                        rep[cat_col][exp][rel] = []
                                    if s_val or s_val == 0:
                                        if rel == "stsb" and s_val == 0:
                                            s_val = 50
                                        s_val = round(s_val,2)
                                        rep[cat_col][exp][rel].append(s_val)
                                        rep[sel_col][exp][rel].append(s_val)
                                        maxval = round(maxval,2)
                                        bold = s_val == maxval
                                        s_val = "{:.2f}".format(s_val)
                                        if bold: 
                                            mval = "\\textcolor{teal}{\\textbf{" + s_val + "}}"
                                        else:
                                            mval = "\\textcolor{black}{" + s_val + "}"
                            ##########
                            # all_tables+=latex_table(rep, rname, mdf, rep_exps,cur_sel_col)
                        elif command == "show":
                            table_cont1 += " $ "
                            for cc, rr in enumerate(reps):
                                val = "na"
                                if sel_col in rr:
                                   if exp in rr[sel_col]: 
                                      val = rr[sel_col][exp]["avg"] 
                                      if val:
                                          val = "{:.1f}".format(float(val))
                                table_cont1+="\\textcolor{"+colors[cc] + "}{" + val + "}-"
                                # table_cont1 += f" {val} "
                            table_cont1 =  table_cont1.strip("-")
                            table_cont1 += " $ &"
                        else:
                            val = rep[sel_col][exp]["avg"]
                            if val:
                               val = "{:.1f}".format(float(val))
                            if command != "shav":
                                table_cont1 += f" $ {val} $ &"
                            if not cat_col in rep_avg:
                                rep_avg[cat_col] = {}
                            if not exp in rep_avg[cat_col]:
                                rep_avg[cat_col][exp] = []
                            if not cat_col in rep:
                                rep[cat_col] = {}
                            if not exp in rep[cat_col]:
                                rep[cat_col][exp] = {}
                            if not "avg" in rep[cat_col][exp]:
                                rep[cat_col][exp]["avg"] = []
                            if val:
                                rep_avg[cat_col][exp].append(float(val))
                            rep[cat_col][exp]["avg"] = rep_avg_list = rep_avg[cat_col][exp]
                            if jj % 3 == 2: 
                                avg = 0
                                if len(rep_avg_list) > 0:
                                    avg = stat.mean(rep_avg_list)
                                sdev = 0
                                if len(rep_avg_list) > 1:
                                    sdev = stat.stdev(rep_avg_list)
                                avg = round(avg, 1)
                                table_cont1 += "" + str(avg) + " &" \
                                #table_cont1 += "\\textcolor{blue}{" + \
                                #        f" $ {avg:.1f}_{{{sdev:.1f}}} $ " + "}  &"
                    table_cont1 = table_cont1.strip("&")
                    table_cont1 += "\\\\\n"
                    table_cont1 += "\\hline \n"

                # table_cont1 += "\\hline \n"
                tables.append(table_cont1)
                heads.append(head1)
                if command == "show":
                    break
                save_obj(rep, rname, "gtasks")

            image = """
                \\newpage
                \\begin{{figure}}[h!]
                    \centering
                    \includegraphics[width=\\textwidth]{{{}}}
                    \caption[image]{{{}}}
                    \label{{{}}}
                \end{{figure}}
            """
            havg = ""
            if (com3 == "avg" or com3 == "img") and rep_avg:
                havg = doc_dir + "/pics/" + rname + "_havg.png"
                tab = []
                exp_names = []
                train_nums = []
                table_avg = " N & "
                table_sdev = " N & "
                head_avg = "|r|"
                #exp_names = list(rep_avg[list(rep_avg.keys())[0]].keys())
                #exp_names = ["SILPI","SILP","SIL","SLPI","P","PI", "SIP","SL", "SLP"] 
                exp_names = ["SILPI","SILP","SIL","SLPI","P","SIP"] 
                #exp_names = ["SIL","P","SIP"]
                train_nums = list(rep_avg.keys())
                for tn in train_nums:
                    head_avg += "r|"
                    table_avg += f" {tn} &"
                    table_sdev += f" {tn} &"
                table_avg = table_avg.strip("&")
                table_avg += "\\\\\n"
                table_sdev = table_sdev.strip("&")
                table_sdev += "\\\\\n"
                _min, _max = 100, 0
                _mins, _maxs = 100, 0
                for exp in exp_names: 
                    if exp == "P2":
                        continue
                    tt = []
                    table_avg += f" \\textbf{{{exp}}} &"
                    table_sdev += f" \\textbf{{{exp}}} &"
                    for tn in train_nums: 
                        if not exp in rep_avg[tn]:
                            continue
                        avg_list = rep_avg[tn][exp]
                        avg=0
                        if avg_list:
                            avg = stat.mean(avg_list)
                            sdev = 0
                            if len(avg_list) > 1:
                                sdev = stat.stdev(avg_list)
                            avg = round(avg,1)
                            if avg > _max: _max = avg
                            if avg < _min: _min = avg
                            if sdev > _maxs: _maxs = sdev 
                            if sdev < _mins: _mins = sdev
                            table_avg += f"{avg:.1f} &"
                            table_sdev += f"{sdev:.1f} &"
                           # table_avg += "\\textcolor{black}{" + \
                           #         f" $ {avg:.1f}_{{{sdev:.1f}}} $ " + "}  &"
                        tt.append(avg)
                    tab.append(tt)
                    table_avg = table_avg.strip("&")
                    table_avg += "\\\\\n"
                    table_sdev = table_sdev.strip("&")
                    table_sdev += "\\\\\n"

                table_avg = table_avg.strip("\n")
                table_sdev = table_sdev.strip("\n")
                table = table_hm_template.format("label", _min, _max, table_avg)
                with open(f"{doc_dir}/table_hm_{rname}.tex", "w") as f:
                    f.write(table)
                table = table_hm_template.format("label", _mins,_maxs,table_sdev)
                with open(f"{doc_dir}/table_hm_sdev_{rname}.tex", "w") as f:
                    f.write(table)

                hm_table = "\input{{table_hm_" + rname + ".tex}}"
                table_env = table_env.replace("minipage", 
                        hm_table + "\nminipage")
                hm_sdev_table = "\input{{table_hm_sdev_" + rname + ".tex}}"
                table_env_sdev = table_env_sdev.replace("minipage", 
                        hm_sdev_table + "\nminipage")
                report = report.replace("mytable", hm_sdev_table + "\n mytable") 
                #heads.append(head_avg)
                #tables.append(table_avg)
                #rep_names.append("avg")
                if "P2" in exp_names:
                    exp_names.remove("P2")
                # tab = np.array(tab)
                if False: #tab.any():
                    # row_norms = np.linalg.norm(tab, axis=1, keepdims=True)
                    # tab = tab / row_norms
                    fig, ax = plt.subplots()
                    fig.set_size_inches(len(exp_names)*1, len(train_nums) + 1)

                    sns.heatmap(tab, ax=ax, cmap="crest", annot=True, 
                            fmt=".1f",
                            vmin=60.,
                            yticklabels=exp_names,
                            xticklabels=train_nums,
                            linewidth=0.5)
                    ax.set_title('Results of different methods')
                    ax.set_xlabel('methods')
                    ax.set_ylabel('train nums')

                    fig.savefig(havg)
            ############### AAAAAAAAAAVVVVVVVVVVVVVV
            all_tables = "\\newpage"
            for size in sizes: 
                cat_col = rname + str(size)
                for seed in seeds: 
                    sel_col = str(cat_col) + "@" + str(seed) + "@m_score"
                    #all_tables += latex_table(rep, rname, mdf, rep_exps, sel_col,
                    #        category) + "\\newpage"
                if str(cat_col) in rep: 
                    cat_table = latex_table(rep, rname, mdf, rep_exps, 
                                    str(cat_col),
                                    category,
                                    caption="The performance of GLUE tasks for training size " + str(cat_col)) 
                    fname_avg = f"{doc_dir}/{rname}_{cat_col}_avg.tex"
                    with open(fname_avg, "w") as f:
                        f.write(cat_table)
                    all_tables += f"\input{{{fname_avg}}} \n\n \\newpage"
                    #all_tables += cat_table + "\n \\newpage"

            #################
            ii = image.format(havg, "havg", "fig:havg")
            all_tables = ii + "\n\n" + all_tables 
            with open(f"{doc_dir}/heatmap_avg.tex", "w") as f:
                f.write(ii)

            hm_table = table_env.replace("minipage","").format(rname, rname)
            report = report.replace("mytable", hm_table + "\n mytable") 
            hm_table = table_env_sdev.replace("minipage","").format(rname, rname)
            report = report.replace("mytable", hm_table + "\n mytable") 
            table_dir = os.path.join(doc_dir, "table")
            Path(table_dir).mkdir(parents=True, exist_ok=True)
            cat=mdf.iloc[0]["cat"]
            table_name = f"{cat}.txt"
            table_file = f"{table_dir}/{table_name}"
            out = open(table_file, "w")
            _input = f"table/{table_name}" 
            ii = 1
            for head, cont, capt in zip(heads, tables, rep_names):
                lable = "table:show"
                caption = f"{capt}:{exp}"
                caption = caption.replace("_","-")
                if command == "shv":
                    table = table_mid_template.format(head, cont, caption, lable)
                else:
                    table = table_fp_template.format(head, cont, caption, lable)
                # assert report.index("mytable"), "mytable not found"
                report = report.replace("mytable", table +"\n\n" + "mytable")
                with open(f"{doc_dir}/table_{capt}.tex", "w") as f:
                    f.write(table)
                ii += 1
            report = report.replace("mytable", all_tables +"\n\n" + "mytable")
            ############ images
            if com3 == "img":
                train_num = str(mdf["max_train_samples"].unique()[0])
                seed = mdf["d_seed"].unique()[0]
                pics_dir = doc_dir + "/pics"
                Path(pics_dir).mkdir(parents=True, exist_ok=True)
                dest, imgs, fnames = get_images(df, all_exps)
                images_rep = "myimage"
                for key, img_list in imgs.items():
                    name = key
                    for new_im in img_list:
                        _exp = key.replace("_","-")
                        _exp = _exp.split("-")[0]
                        caption = "\hyperref[table:show]{ \\textcolor{red}{"+category+"}}:"+_exp 
                        caption = caption.replace("_","-")
                        name = category + "-" + key + \
                                "-" + str(train_num) + "-" + str(seed)
                        label = "fig:" + category + _exp 
                        pname = doc_dir + "/pics/" + name.strip("-") + ".png"
                        dest = os.path.join(doc_dir, pname) 
                        new_im.save(dest)
                        ii = image.format(pname, caption, label)
                        images_rep = images_rep.replace("myimage", ii +"\n\n" + "myimage")

                ####################
                images_rep = images_rep.replace("myimage","")
                report = report.replace("myimage", "\input{" + rname + "_images.tex}")
                with open(f"{doc_dir}/{category}_images.tex", "a") as f:
                    f.write(images_rep)
                with open(f"{doc_dir}/{rname}_images.tex", "a") as f:
                    f.write(images_rep)
            #with open(f"{_dir}/report_templates/report.tex." + rname, "w") as f:
            #    f.write(report)
            tex = f"{doc_dir}/report.tex"
            with open(tex, "w") as f:
                f.write(report)
            mbeep()

        if cmd.startswith("avg"):
            _dir = Path(__file__).parent
            doc_dir = os.path.join(home, "Documents/Paper2/IJCA/FormattingGuidelines-IJCAI-23")
            rname = cmd.split("@")[-1]
            if Path(f"{_dir}/report_templates/report.tex." + rname).is_file():
                pp = f"{_dir}/report_templates/report.tex." + rname
            else:
                pp = f"{_dir}/report_templates/report.tex.temp"
            with open(pp, "r") as f:
                report = f.read()
            rep = load_obj(rname, "gtasks", {})
            _agg = {}
            for c in sel_cols:
                if c in df.columns: 
                    if c.endswith("score"):
                        _agg[c] = "mean"
                    else:
                        _agg[c] = "first"
            rdf = df.groupby(["expid", "prefix"], as_index=False).agg(
                    _agg).reset_index(drop=True)
            gdf = df.groupby(["expid"], as_index=False).agg(_agg).reset_index(drop=True)
            gdf = gdf.sort_values(by=["m_score"], ascending=False)
            all_exps = gdf['expid'].unique()

            exp_names = []
            for exp in all_exps:
                exp = exp.replace("_","-")
                exp = exp.split("-")[0]
                exp_names.append(exp)
            #####################
            train_num = str(mdf["max_train_samples"].unique()[0])
            seed = mdf["d_seed"].unique()[0]
            cols = [str(train_num) + "@" + str(seed) + "@m_score"]
            ii = 1
            for exp in all_exps: #gdf["expid"].unique():
                cond = (gdf['expid'] == exp)
                _exp = exp.replace("_","-")
                _exp = _exp.split("-")[0]
                for sel_col in cols:
                    val_col = sel_col
                    if "@" in sel_col:
                        val_col = sel_col.split("@")[-1]
                    if val_col in gdf.columns:
                        val = gdf.loc[cond, val_col].iloc[0]
                    if val_col.endswith("score"):
                        mval = val = "{:.2f}".format(val)
                        # val = "$ \\textcolor{teal}{" + str(ii) + \
                        #      "}-\\textcolor{blue}{ " + val + " }$"
                        val = "$ \\textcolor{blue}{ " + val + " }$"
                    else:
                        mval = val
                        val = "\\textbf{" + str(val) + "}"
                    report = report.replace("@" + _exp + "@" + sel_col, str(val))
                    if not sel_col in rep:
                        rep[sel_col] = {}
                    rep[sel_col][_exp] = str(mval) 
                ii += 1
            ii = 1
            save_obj(rep, rname, "gtasks")
            ##################
            rep2 = {}
            for k,v in rep.items():
                kk = k.split("@")
                tn = kk[0]
                if len(kk) <= 2:
                    continue
                if not tn in rep2:
                    rep2[tn] = {}
                for exp, val in v.items():
                    if not exp in rep2[tn]:
                        rep2[tn][exp] = []
                    rep2[tn][exp].append(val)

            mdf = main_df
            head1 = "|r|"
            table_avg = " tn  &"
            for exp in exp_names:
                head1 += "r|"
                table_avg += " \\textbf{" + exp + "} &"
            table_avg = table_avg.strip("&")
            table_avg += "\\\\\n"
            table_avg += "\\hline\n"
            tab = []
            train_nums = []
            for k,v in rep2.items():
                tn = str(k) + "@m_score" 
                train_nums.append(str(k))
                table_avg += " \\textbf{" + str(k) + "} &"
                tt = []
                #for exp, val in v.items():
                for exp in exp_names:
                    if not exp in rep2[k]:
                        continue
                    val = rep2[k][exp]
                    val = [float(v) for v in val if v]
                    try:
                        avg = stat.mean(val)
                        sdev = stat.stdev(val)
                    except:
                        continue
                    avg_sd = "{:.2f}_{{{:.2f}}}".format(avg, sdev)
                    if not tn in rep:
                        rep[tn] = {}
                    rep[tn][exp] = avg 
                    tt.append(avg)
                    table_avg += f" ${avg_sd}$ &"
                if tt:
                    tab.append(tt)
                table_avg = table_avg.strip("&")
                table_avg += "\\\\\n"
                table_avg += "\\hline\n"
            table_avg = table_avg.strip("&")
            table_avg += "\\\\\n"
            table_avg += "\\hline\n"

            havg = doc_dir + "/pics/" + "havg.png"
            try:
                tab = np.array(tab)
                if tab.any():
                    row_norms = np.linalg.norm(tab, axis=1, keepdims=True)
                    normalized_rows = tab / row_norms
                    fig, ax = plt.subplots()

                    sns.heatmap(tab, ax=ax, cmap="crest", annot=True, 
                            annot_kws={'rotation': 90}, 
                            fmt=".1f",
                            xticklabels=exp_names,
                            yticklabels=train_nums,
                            linewidth=0.5)
                    ax.set_title('Heatmap of a matrix')
                    ax.set_xlabel('X-Axis')
                    ax.set_ylabel('Y-Axis')

                    fig.savefig(havg)
            except:
                pass

            ##################
            head_cmp = "|r|"
            table_cmp = " tn  &"
            key_list = list(rep_cmp.keys())
            methods = []
            if key_list:
                key = key_list[0]
                methods = list(rep_cmp[key].keys())
            for exp in methods:
                head_cmp += "r|"
                table_cmp += " \\textbf{" + exp.replace("_","-") + "} &"
            table_cmp = table_cmp.strip("&")
            table_cmp += "\\\\\n"
            table_cmp += "\\hline\n"
            tab = []
            exp_cats = []
            for k,v in rep_cmp.items():
                tn = str(k) + "@m_score" 
                exp_cats.append(str(k))
                table_cmp += " \\textbf{" + str(k) + "} &"
                tt = []
                for exp in methods:
                    val = rep_cmp[k][exp]
                    val = [float(v) for v in val if v]
                    try:
                        avg = stat.mean(val)
                        sdev = stat.stdev(val)
                    except:
                        continue
                    avg_sd = "{:.2f}_{{{:.2f}}}".format(avg, sdev)
                    tt.append(avg)
                    table_cmp += f" ${avg_sd}$ &"
                tab.append(tt)
                table_cmp = table_cmp.strip("&")
                table_cmp += "\\\\\n"
                table_cmp += "\\hline\n"
            table_cmp = table_cmp.strip("&")
            table_cmp += "\\\\\n"
            table_cmp += "\\hline\n"
            ii = 1
            for head, cont in zip([head1, head_cmp],
                    [table_avg, table_cmp]):
                lable = "table:" + str(ii) 
                caption = f"{rname} {exp} {train_num} {seed}"
                caption = caption.replace("_","-")
                table = """
                    \\begin{{table*}}[h!]
                        \label{{{}}}
                        \caption{{{}}}
                        \\begin{{tabular}}{{{}}}
                        \hline
                        {}
                        \end{{tabular}}
                    \end{{table*}}
                    """
                table = table.format(lable, caption, head, cont)
                # assert report.index("mytable"), "mytable not found"
                report = report.replace("mytable", table +"\n\n" + "mytable")
                ii += 1

            image = """
                \\begin{{figure}}[h!]
                    \centering
                    \includegraphics[width=\\textwidth]{{{}}}
                    \caption[image]{{{}}}
                    \label{{{}}}
                \end{{figure}}
            """
            pics_dir = doc_dir + "/pics"
            ii = image.format(havg, "havg", "fig:havg")
            report = report.replace("myimage", ii +"\n\n" + "myimage")
            _dir = Path(__file__).parent
            doc_dir = os.path.join(home, 
                    "Documents/Paper2/IJCA/FormattingGuidelines-IJCAI-23")
            m_report = f"{_dir}/report_templates/report.tex.temp2"
            with open(m_report,"w") as f:
                f.write(report)
            char = "R"
        if False: # cmd == "report" or char == "Z" or char == "R" or cmd == "comp_rep":
            _dir = Path(__file__).parent
            doc_dir = os.path.join(home, 
                    "Documents/Paper2/IJCA/FormattingGuidelines-IJCAI-23")
            if cmd == "comp_rep":
                m_report = f"{_dir}/report_templates/report.tex.temp2"
            else:
                m_report = f"{_dir}/report_templates/report.tex.temp"
            _agg = {}
            if char == "R":
                score_cols = ["m_score"]
                rep_cols = ["expid"]
            elif char == "Z":
                # score_cols = ["preds_num"]
                rep_cols = ["model_name_or_path","template"]
            pivot_cols = ["model_name_or_path","prefix"]
            index_cols = ["template"]

            if not "m_score" in sel_cols:
                sel_cols.append("m_score")
            if not rep_cols:
                rep_cols = ["expid", "template","prefix"]
            main_score = score_cols[0] 
            # df = main_df
            gcol = rep_cols
            main_df["preds_num"] = main_df.groupby(gcol + ["prefix"], sort=False)["pred_text1"].transform("nunique")
            cols = set(rep_cols + score_cols)
            for c in cols:
                if c in main_df.columns: 
                    if c == "preds_num":
                        _agg[c] = "mean"
                    elif c.endswith("score"):
                        _agg[c] = "mean"
                    else:
                        _agg[c] = "first"
            with open(m_report, "r") as f:
                main_report = f.read()
            report = main_report

            #Define the category mapping
            category_mapping = {'AtLocation': 1, 'CapableOf': 1, 'HasProperty': 1, 
                    'ObjectUse': 1, 'isFilledBy':1, 'xAttr':1,
                    'xIntent':2, 'xNeed':2,
                    'qnli':3,'mnli':3, 'sst2':3}
            if not "category" in main_df:
                main_df['category'] = main_df['prefix'].map(category_mapping)


            tvalues=["sup-nat","unsup-nat","sup","unsup","pt-sup","pt-unsup","0-pt-unsup","0-pt-sup"] #,"0-pt-unsup-nat"]
            mdf = main_df[(main_df['template'].isin(tvalues))]
            for cat in [1, 2,3]:
                for model in ["t5-base","t5-lmb","t5-v1"]:
                    if True:# cat == 1:
                       tdf = mdf[(mdf.model_name_or_path == model) & (mdf.category == cat)]
                    else:
                        tdf = mdf[(mdf.category == cat)]
                        if model != "t5-base":
                            continue
                    gdf = tdf.groupby(gcol + ["prefix"], as_index=False).agg(_agg)#.reset_index(drop=True)
                    #gdf.set_index(gcol, inplace=True)
                    #gdf[score_cols] = gdf[score_cols].round(2)
                    #gdf = gdf.sort_values(by=gcol[0], ascending=False)

                    pivot_df = gdf.pivot_table(
                                index=index_cols, 
                                columns=pivot_cols,
                                #aggfunc='mean',
                                #margins=True
                            )
                    #pivot_df = pivot_df.rename(columns={'sup': 's','unsup-nat':'un',
                    #    'sup-nat':'sn','unsup':'u', 
                    #    'pt-sup':'ps','0-pt-unsup':'0pu', 
                    #    'pt-unsup':'pu','0-pt-sup':'0ps', 
                    #    'model_name_or_path':'model'})

                    # Flatten the resulting MultiIndex columns
                    #pivot_df.columns = [f'{col[0]}_{col[1]}' for col in pivot_df.columns]
                    # Format the labels of index level 1 using LaTeX formatting
                    # pivot_df.reset_index(inplace=True)

                    # Now 'pivot_df' has 'prefix' values as columns and non-repeated index

                    # Generate LaTeX table representation
                    #table = pivot_df.to_latex(index=True, escape=True)
                    ######
                    styler = pivot_df.style
                    formatted_styler = styler.format(escape=True, 
                            precision=2)  # Format to 2 decimal places

                    float_format = "%.2f"  # Format for 2 decimal places
                    #table = formatted_styler.to_latex()

                    # Calculate the average for each column
                    average_row = pivot_df.mean()
                    average_row.name = 'Average'
                    # Find the maximum value in each row
                    # max_values = pivot_df.max(axis=1)
                    # max_values = pivot_df.max(axis=0, level=1)
                    max_in_rows = pivot_df.groupby(level=0, axis=1).max()
                    # Create a function to format the cell value
                    def format_cell(value, max_value, is_avg=False):
                        if isinstance(value, (int, float)):
                            if value == max_value:
                                if is_avg:
                                    return '\\textcolor{brown}{\\textbf{' + "{:.2f}".format(value) + '}}'
                                else:
                                    return '\\textcolor{teal}{\\textbf{' + "{:.2f}".format(value) + '}}'
                            if is_avg:
                                return '\\textcolor{blue}{' + "{:.2f}".format(value) + '}'
                            else:
                                return "{:.2f}".format(value) 
                        return str(value)

                    # Apply the formatting function to each row
                    #pivot_df.loc['Average'] = [value for col, value in average_row.items()]
                    grouped = pivot_df.groupby(level=1, axis=1).mean()
                    formatted_rows = []
                    for index, row in grouped.iterrows():
                        for model in row.index:
                            pivot_df.loc[index,(main_score,model,'avg.')] = row[model]

                        # Rearrange the columns to move the average columns 
                    pivot_df = pivot_df.reorder_levels([0, 1, 2], axis=1)
                    pivot_df = pivot_df.sort_index(axis=1, level=[0, 1], 
                            sort_remaining=False)


                    max_in_cols = pivot_df.max(axis=0)
                    for index, row in pivot_df.iterrows():
                        max_value = row.max()
                        avg = row.mean()
                        is_avg = index == 'Average'
                        # Compare row values against the maximum values in each column
                        formatted_row = []
                        for col_idx, value in enumerate(row):
                            is_avg = pivot_df.columns[col_idx][-1] == 'avg.'
                            formatted_row.append(format_cell(value, 
                                max_in_cols[col_idx], is_avg)) 

                        pivot_df.loc[index] = formatted_row 


                    prod = 1
                    for l in pivot_df.columns.levels:
                        prod = prod*len(l)
                    num_subcategories = prod
                    last = len(pivot_df.columns.levels[-1])
                    f = '|c'
                    for i in range(1, prod):
                        if i % last == 0:
                            f += '|'
                        f += 'c'

                    # Generate LaTeX table representation with centered columns for subcategories
                    column_format = 'l' + f 


                    # Define the \rot command
                    def rot(text):
                        return f"\\rot{{{text}}}"

                    # Apply the \rot command to the column names in level 3
                    pivot_df.columns = pivot_df.columns.set_levels(pivot_df.columns.levels[2].map(rot), level=2)
                    
                    # Create a new DataFrame with the formatted rows
                    #formatted_df = pd.DataFrame(formatted_rows, pivot_df.columns)
                    table = pivot_df.to_latex(escape=False, 
                            multirow=True,
                            header=True,
                            multicolumn_format='c',
                            column_format=column_format,
                            multicolumn=True,
                            float_format=float_format)
                    table = table.replace(
                            "model_name_or_path","model"
                    )
                    table = table.replace("_","-")

                    report = report.replace("mytable", table +"\n\n" + "mytable")
            report = report.replace("mytable","")
            report = report.replace("myimage","")
            tex = f"{doc_dir}/report.tex"
            pdf = f"{doc_dir}/report.pdf"
            with open(tex, "w") as f:
                f.write(report)
            with open(m_report, "w") as f:
                f.write(main_report)
            mbeep()
        # if cmd == "report2" or char == "Z2" or char == "R2" or cmd == "comp_rep2":
        # rrrrrrrrr
        if cmd.startswith("report") or char == "Z" or char == "R" or cmd == "comp_rep":
            rep_cols = []
            if "=" in cmd:
                parts = cmd.split("=")
                cmd = parts[0]
                rep_cols = parts[1:]
            if "@" in cmd:
                parts = cmd.split("@")
                cmd = parts[0]
                score_cols = parts[1:]

            _dir = Path(__file__).parent
            doc_dir = "/home/ahmad/logs" #os.getcwd() 
            # doc_dir = os.path.join(home, 
            #        "Documents/Paper2/IJCA/FormattingGuidelines-IJCAI-23")
            if cmd == "comp_rep":
                m_report = f"{_dir}/report_templates/report.tex.temp2"
            else:
                m_report = f"{_dir}/report_templates/report.tex.temp"
            _agg = {}
            if char == "R" or cmd == "report":
                if not score_cols:
                    score_cols = ["m_score"]
                if not rep_cols:
                    rep_cols = ["cat"]
            elif char == "Z":
                score_cols = ["rouge_score"]
                if not rep_cols:
                    rep_cols = ["cat"]
                #rep_cols = ["model_name_or_path","template"]
            exp_cols = ["cat","num_source_prompts","compose_method"] 

            if not "m_score" in sel_cols:
                sel_cols.append("m_score")
            if not rep_cols:
                rep_cols = ["expid", "template"]
            main_score = "rouge_score"
            with open(m_report, "r") as f:
                report = f.read()
            with open(os.path.join(doc_dir, "report.tex"), "w") as f:
                f.write("")
            # df = main_df
            rep_cols = exp_cols
            gcol = rep_cols
            cols = ["prefix"]
            df["preds_num"] = df.groupby(gcol + ["prefix"], sort=False)["pred_text1"].transform("count")
            cols = set(cols + exp_cols + rep_cols + score_cols) # + ["preds_num"])
            for c in cols:
                if c in df.columns: 
                    if c == "preds_num":
                        _agg[c] = "mean"
                    elif c.endswith("score"):
                        _agg[c] = "mean"
                    else:
                        _agg[c] = "first"
            mdf = main_df
            rdf = df.groupby(gcol + ["prefix"], as_index=False).agg(
                    _agg).reset_index(drop=True)
            gdf = main_df.groupby(gcol, as_index=False).agg(_agg).reset_index(drop=True)
            gdf = gdf.sort_values(by=score_cols[0], ascending=False)
            gdf.columns = gdf.columns.to_flat_index()

            pdf = mdf.pivot_table(index=gcol, columns='prefix', 
                    values=score_cols, aggfunc='mean')
            pdf['avg'] = pdf.mean(axis=1, skipna=True)
            pdf.reset_index(inplace=True)
            pdf.columns = [col[1] if col[0] == "m_score" else col[0] for col in pdf.columns]
            pdf.columns = pdf.columns.to_flat_index()
            pdf = pdf.sort_values(by='avg', ascending=False)

            latex_table=tabulate(pdf, #[exp_cols+score_cols], 
                    headers='keys', tablefmt='latex_raw', showindex=False)
            report = report.replace("mytable", latex_table + "\n mytable")
            train_num = str(mdf["max_train_samples"].unique()[0])
            seed = mdf["d_seed"].unique()[0]
            all_exps = gdf[gcol[0]].unique()
            exp_names = []
            for exp in all_exps:
                exp = str(exp)
                exp = exp.replace("_","-")
                exp = exp.split("-")[0]
                exp_names.append(exp)

            rep_style="horiz"
            rep = load_obj(rname, "gtasks", {})
            table_cont2=""
            ### aaaaaaaaaaaaaaa
            head2 = "|"
            top_cat = "prefix"
            if rep_style == "vert":
                top_cat = rep_cols[1]
                all_rels = mdf[top_cat].unique()
            else:
                all_rels = mdf[top_cat].unique()
                for col in rep_cols: 
                    if col in map_cols:
                        col = map_cols[col]
                    col = col.replace("_","-")
                    table_cont2 += " \multirow{2}{*}{\\textbf{" + col + "}} &"
                    head2 += "l|"
            old_rel = ""
            for rel in all_rels: 
                for idx, sc in enumerate(score_cols):
                    head2 += "c"
                head2 +="|"
            if rep_style == "horiz":
                for sc in score_cols:
                    head2 += "l|"
            slen = str(len(score_cols))
            show_score_headings = False
            rlen = "1" if show_score_headings else "2"
            for rel in all_rels: 
                table_cont2 += "\multicolumn{" + slen + "}{l|}{ \multirow{" + rlen + "}{*}{\\textit{"+rel+"}}} &"
            if rep_style == "horiz":
                for sc in score_cols:
                    sc = sc.replace("_score","")
                    table_cont2 += " \multirow{" + rlen + "}{*}{avg.} &" 
            table_cont2 = table_cont2.strip("&")
            table_cont2 += "\\\\\n"
            start = str(len(rep_cols) + 1)
            end = str(len(rep_cols) + len(score_cols)*len(all_rels))
            if rep_style == "horiz":
                if len(score_cols) > 1:
                   table_cont2 += "\cline{" + start + "-" + end+ "}\n"

                for c in rep_cols:
                    table_cont2 += "& "
            for rel in all_rels: 
                for sc in score_cols:
                    sc = sc.replace("_score","") if show_score_headings else ""
                    table_cont2 += sc + " &"
            if rep_style == "horiz":
                for sc in score_cols:
                    sc = sc.replace("_score","") if show_score_headings else ""
                    table_cont2 += sc + " &" # for average
            table_cont2 = table_cont2.strip("&")
            table_cont2 += "\\\\\n"
            table_cont2 += "\\hline\n"

            values = []
            for col in gcol:
                ll = [l for l in gdf[col].unique()]
                values.append(ll)
            combs = [c for c in itertools.product(*values)]
            _first = ""
            for ii, comb in enumerate(combs): 
                cc = "@".join([str(c) for c in comb])
                first = comb[0]
                if rep_style == "vert":
                    table_cont2 += " &"
                else:
                    if first != _first:
                        if _first != "": table_cont2 += "\\hline \n"
                        _first = first
                        col = gcol[0]
                        table_cont2 += f" \multirow{{1}}{{*}}{{$ @{cc}@{col} $}} &"
                    else:
                        table_cont2 += " &"
                for col in gcol[1:]: 
                    table_cont2 += f" $ @{cc}@{col} $ &"
                if rep_style == "horiz":
                    for rel in all_rels:
                        for score_col in score_cols:
                            table_cont2 += f" $ @{cc}@{rel}@{score_col} $ &"
                    for score_col in score_cols:
                        table_cont2 += f" $ @{cc}@{score_col} $ &"
                table_cont2 = table_cont2.strip("&")
                table_cont2 += "\\\\\n"
            table_cont2 += "\\hline \n"
            if rep_style == "vert":
                for rel in all_rels:
                    for score_col in score_cols:
                        table_cont2 += f" $ @{cc}@{score_col} $ &"
                table_cont2 = table_cont2.strip("&")
                table_cont2 += "\\\\\n"
                table_cont2 += "\\hline \n"
            ii = 1
            category = mdf["experiment"].unique()[0]
            category = category.split("/")[0]
            for head, cont in zip([head2],
                    [table_cont2]):
                lable = "table:" + str(ii) 
                caption = f" {category} {exp} {train_num} {seed}"
                caption = caption.replace("_","-")
                table = """
                    \\begin{{table*}}[h!]
                        \label{{{}}}
                        \caption{{{}}}
                        \\begin{{tabular}}{{{}}}
                        \hline
                        {}
                        \end{{tabular}}
                    \end{{table*}}
                    """
                table = table.format(lable, caption, head, cont)
                # assert report.index("mytable"), "mytable not found"
                report = report.replace("mytable", table +"\n\n" + "mytable")
                ii += 1

            #table = gdf.to_latex(index=True, escape=True)
            #report = report.replace("mytable", table +"\n\n" + "mytable")
            caps = {}
            cols = []
            rep_cells = rep_cols + score_cols 
            for cell in rep_cells:
               cols.append(cell)

            ii = 1
            for comb in combs: #gdf["expid"].unique():
                img_cap = " Table: \hyperref[table:1]{Table}"
                cc = "@".join([str(c) for c in comb])
                cond = True
                for e, elem in enumerate(comb):
                    cond = cond & (gdf[gcol[e]] == elem)
                cond2 = (gdf[gcol[0]] == comb[0])
                for sel_col in cols:
                    val_col = sel_col
                    val = ""
                    if "@" in sel_col:
                        val_col = sel_col.split("@")[-1]
                    if val_col in gdf.columns:
                        rows = gdf.loc[cond, val_col]
                        if not rows.empty:
                            val = rows.iloc[0]
                    if val:
                        if val_col.endswith("score") or val_col.endswith("num"):
                            maxval = gdf.loc[cond2, val_col].max()
                            val = round(val,2)
                            maxval = round(maxval,2)
                            bold = val == maxval
                            if val_col.endswith("num"):
                                mval = val = "{:.1f}".format(val)
                            else:
                                mval = val = "{:.2f}".format(val)
                            if bold:
                                val = "$ \\textcolor{blue}{ " + val + " }$"
                            else:
                                val = "$ \\textcolor{black}{ " + val + " }$"
                        else:
                            mval = val
                            val = "\\textbf{" + str(val) + "}"
                    img_cap += sel_col.replace("_","-") + ": " + str(val) + " | "
                    report = report.replace("@" + cc + "@" + sel_col, str(val))
                caps[exp] = img_cap
                ii += 1
            for comb in combs: #gdf["expid"].unique():
                img_cap = " Table: \hyperref[table:1]{Table}"
                cc = "@".join([str(c) for c in comb])
                main_cond = True
                for e, elem in enumerate(comb):
                    main_cond = main_cond & (mdf[gcol[e]] == elem)
                scores = ""
                for rel in all_rels: 
                    cond = (mdf[top_cat] == rel) & main_cond 
                    for sc in score_cols: 
                        val = mdf.loc[cond, sc].mean()
                        preds_num = mdf.loc[cond, "pred_text1"].unique()
                        preds_num = len(preds_num)
                        one_pred = int(preds_num) == 1
                        maxval = rdf.loc[(rdf[top_cat] == rel), sc].max()
                        val = round(val,2)
                        maxval = round(maxval,2)
                        bold = val == maxval
                        if sc != "preds_num":
                            val = "{:.2f}".format(val)
                        else:
                            try:
                               val = str(int(val))
                            except:
                               val = "NA"
                        if one_pred:
                            val = "\\textcolor{orange}{" + val + "}"
                        elif bold: 
                            val = "\\textcolor{teal}{\\textbf{" + val + "}}"
                        else:
                            val = "\\textcolor{black}{" + val + "}"
                        scores += rel + ":" + "$" + val + "$ "
                        report = report.replace(
                                "@" + cc + "@" + rel + "@" + sc, val)
                caps[exp] += "\\\\" + scores            
                ii += 1
                
            if True: #char == "R": 
                image = """
                    \\begin{{figure}}[h!]
                        \centering
                        \includegraphics[width=\\textwidth]{{{}}}
                        \caption[image]{{{}}}
                        \label{{{}}}
                    \end{{figure}}
                """
                cat = mdf["experiment"].unique()[0]
                multi_image = """
                    \\begin{figure}[h!]
                        \centering
                        mypicture 
                        \caption[image]{mycaption}
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
                tdf = mdf.sort_values(by=["cat"])
                all_exps = tdf["fid"].unique()
                dest, imgs, fnames = get_images(df, all_exps, 'fid')
                sims = {}
                scores = {}
                all_images = {}
                kk = 0
                id = "other"
                images_str = ""
                if "preds_num" in _agg:
                    del _agg["preds_num"]
                if "num_preds" in _agg:
                    del _agg["num_preds"]
                _agg["fid"] = "first"
                edf = mdf.groupby(exp_cols, as_index=False).agg(_agg).reset_index(drop=True)
                edf = edf.sort_values(by=score_cols[0], ascending=False)
                cols = exp_cols + score_cols
                for key, img_list in imgs.items():
                    mkey = key
                    key = str(key)
                    name = key
                    for new_im in img_list:
                        name = key + str(name)
                        _exp = key.replace("_","-")
                        _exp = _exp.split("-")[0]
                        label = "fig:" + key 
                        fname = fnames[kk]
                        caption = ""
                        if not edf.loc[edf['fid'] == mkey].empty:
                            sels = edf.loc[edf['fid'] == mkey, cols].iloc[0].to_dict()
                            cat = json.dumps(sels).replace("_","-")
                            for k,v in sels.items():
                                caption += " \\textcolor{gray}{" + str(k).replace("_","-") \
                                    + "}: \\textcolor{blue}{" + str(v).replace("_","-")+ "}" 
                        else:
                            cat = "none" 
                        ss = "_scores" if "score" in fname else "_sim"
                        if "@" in fname:
                            ss = "_" + fname.split("@")[1]
                        pname = doc_dir + "/pics/" + id + name.strip("-") + ss + ".png" 
                        dest = os.path.join(doc_dir, pname) 
                        new_im.save(dest)
                        ii = image.format(pname, caption, label)
                        report = report.replace("myimage", ii +"\n\n" + "myimage")
                        if not _exp in all_images:
                            all_images[_exp] = {}
                        all_images[_exp][ss] = pname
                        if ss.endswith("scores"):
                            scores[_exp] = pname
                        else:
                            sims[_exp] = pname
                        kk += 1

                multi_image1 = multi_image
                for exp in ["SIL","SILP","SIP"]: 
                    if not exp in scores or not exp in sims:
                        continue
                    pname = scores[exp]
                    multi_image1 = multi_image1.replace("mypicture", 
                        graphic.format(pname) + "\n mypicture")
                    pname = sims[exp]
                    multi_image1 = multi_image1.replace("mypicture", 
                        graphic.format(pname) + "\n mypicture")

                multi_image2 = multi_image
                for exp in ["SILPI","SLPI","SLP", "SL"]: 
                    if not exp in scores or not exp in sims:
                        continue
                    pname = scores[exp]
                    multi_image2 = multi_image2.replace("mypicture", 
                        graphic.format(pname) + "\n mypicture")
                    pname = sims[exp]
                    multi_image2 = multi_image2.replace("mypicture", 
                        graphic.format(pname) + "\n mypicture")

                multi_image3 = ""
                ii = 0
                for k,v in all_images.items():
                    if ii % 2 == 0:
                        multi_image3 += f" \\newpage \n \\subsection{{{k}}}"
                    ii += 1
                    for p,q in v.items():
                        multi_image3 += multi_image.replace("mypicture", 
                                graphic.format(q) + "\n").replace("mycaption",
                                        str(p) + ":" + str(q))

                multi_image1 = multi_image1.replace("mypicture","")
                multi_image2 = multi_image2.replace("mypicture","")
                multi_image3 = multi_image3.replace("mypicture","")
                if False:
                    tex = f"{doc_dir}/group_si.tex"
                    with open(tex, "w") as f:
                        f.write(multi_image)
                    tex = f"{doc_dir}/group_slp.tex"
                    with open(tex, "w") as f:
                        f.write(multi_image2)
                tex = f"{doc_dir}/scores_img.tex"
                with open(tex, "w") as f:
                    f.write(multi_image1)
                tex = f"{doc_dir}/sim_img.tex"
                with open(tex, "w") as f:
                    f.write(multi_image2)
                tex = f"{doc_dir}/other_img.tex"
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
            mbeep()
            #subprocess.run(["pdflatex", tex])
            #subprocess.run(["okular", pdf])

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
            exp=df.iloc[sel_row]["expid"]
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
        if cmd == "rep" or cmd == "rep@":
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
        if "==" in cmd:
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
        elif not char in ["q", "S","r"]:
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
                    df.to_csv(os.path.join(base_dir, cmd+".tsv"), sep="\t", index=False)
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
        if char == "r" and prev_char != "x":
            filter_df = orig_df
            df = filter_df
            FID = "fid" 
            reset = True
            #sel_cols = group_sel_cols 
            #save_obj([], "sel_cols", context)
            #save_obj([], "info_cols", context)
            hotkey = hk
        if char == "r" and prev_char == "x":
            df = main_df
            sel_cols = list(df.columns)
            save_obj(sel_cols,"sel_cols",dfname)
            extra["filter"] = []
            info_cols = []
        if char == "b" and back:
            if back:
                df = back.pop()
                sel_cols = sels.pop() 
                sel_row = back_rows.pop()
                info_cols = back_infos.pop()
                left = 0
            else:
                mbeep()
            if extra["filter"]:
                extra["filter"].pop()

def render_mpl_table(data, wrate, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        mlog.info("Size %s", size)
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
    win = cur.newwin(12, cols - 10, 5, 5)
    _default = ""
    win.bkgd(' ', cur.color_pair(CUR_ITEM_COLOR))  # | cur.A_REVERSE)
    _comment, ret_ch = minput(win, 0, 0, "Enter text", 
            default=_default, mode =MULTI_LINE)
    if _comment == "<ESC>":
        _comment = ""
    return _comment

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
    for msg in infos:
        lines = textwrap.wrap(msg, width=w, placeholder=".")
        for line in lines: 
            mprint(str(line).replace("@","   "), info_bar, color=HL_COLOR)
            lnum += 1
    rows,cols = std.getmaxyx()
    info_bar.refresh(0,0, rows -lnum - 1,0, rows-1, cols - 2)
si_hash = {}
def list_values(vals,si=0, sels=[]):
    tag_win = cur.newwin(15, 70, 3, 5)
    tag_win.bkgd(' ', cur.color_pair(TEXT_COLOR))  # | cur.A_REVERSE)
    tag_win.border()
    vals = sorted(vals)
    key = "_".join([str(x) for x in vals[:4]])
    if si == 0:
        if key in si_hash:
            si = si_hash[key]
    opts = {"items":{"sels":sels, "range":["Done!"] + vals}}
    is_cancled = True
    si,canceled, _ = open_submenu(tag_win, opts, "items", si, "Select a value", std)
    val = ""
    if not canceled and si > 0: 
        val = vals[si - 1]
        si_hash[key] = si
        is_cancled = False
    return is_cancled, val

def list_df_values(df, col ="", get_val=True,si=0,vi=0, sels=[]):
    is_cancled = False
    if not col:
        cols = list(df.columns)
        is_cancled, col = list_values(cols,si, sels)
    val = ""
    if col and get_val and not is_cancled:
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
dfname = ""
dfpath = ""
dftype = "tsv"
hotkey = ""
global_cmd = ""
global_search = ""
base_dir = os.path.join(home, "mt5-comet", "comet", "data", "atomic2020")
def start(stdscr):
    global info_bar, text_win, cmd_win, std, main_win, colors, dfname
    stdscr.refresh()
    std = stdscr
    now = datetime.datetime.now()
    rows, cols = std.getmaxyx()
    height = rows - 1
    width = cols
    # mouse = cur.mousemask(cur.ALL_MOUSE_EVENTS)
    text_win = cur.newpad(rows * 50, cols * 20)
    text_win.bkgd(' ', cur.color_pair(TEXT_COLOR)) # | cur.A_REVERSE)
    cmd_win = cur.newwin(1, cols, rows - 1, 0)
    info_bar = cur.newpad(rows*10, cols*20)
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
    #if not dfname:
    #    fname = load_obj("dfname","","")
    #    dfname = fname + ".tsv"
    if not dfname:
        mlog.info("No file name provided")
    else:
        path = os.path.join(dfpath, *dfname)
        if Path(path).is_file():
            files = [path]
            dfname = Path(dfname).stem
        else:
            files = []
            for root, dirs, _files in os.walk(dfpath):
                for _file in _files:
                    root_file = os.path.join(root,_file)
                    cond = all(s.strip() in root_file for s in dfname)
                    if dftype in _file and cond: 
                        files.append(root_file)
        # mlog.info("files: %s",files)
        if not files:
            print("No file was selected")
            return
        dfs = []
        ii = 0
        for f in tqdm(files):
            # mlog.info(f)
            #print(f)
            #print("==================")
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
                    df["exp_name"] = _ff #.replace("=","+").replace("_","+")
                elif fid == "name":
                    df["exp_name"] =  "_" + Path(f).stem
                else:
                    df["exp_name"] =  "_" + df[fid]
            dfs.append(df)
            ii += 1
        if len(dfs) > 1:
            df = pd.concat(dfs, ignore_index=True)
        if files:
            dfname = "merged"
            show_df(df)
        else:
            mlog.info("No tsv or json file was found")

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
    "--hkey",
    "-h",
    default="CGR",
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
    "-s",
    default="",
    type=str,
    help=""
)
@click.pass_context
def main(ctx, fname, path, fid, ftype, dpy, hkey, cmd, search):
    if dpy:
        port = 1234
        debugpy.listen(('0.0.0.0', int(port)))
        print("Waiting for client at run...port:", port)
        debugpy.wait_for_client()  # blocks execution until client is attached
    global dfname,dfpath,file_id,dftype,hotkey, global_cmd, global_search
    file_id = fid
    global_search = search
    hotkey = hkey 
    global_cmd = cmd
    if not fname:
        fname = [dftype]
    if fname != "last":
        dfname = fname 
        dfpath = path
    dftype= ftype
    set_app("showdf")
    wrapper(start)

if __name__ == "__main__":
    main()
