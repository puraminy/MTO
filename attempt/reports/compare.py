import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import shutil
from pathlib import Path
from datetime import datetime
from pytz import timezone
import os
import platform
import subprocess

def open_pdf(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", path])
    else:  # Assume Linux or Unix
        subprocess.run(["xdg-open", path])



def line_2_plot(df, x_col, y_col, cat_col, x_label, y_label='Accuracy', get_input=False):
    df = df.sort_values(x_col)
    markers = ['o', 's']
    colors = ['orange', 'blue','green', 'brown', 'cyan']
    cats = df[cat_col].unique()
    mapping = {cat: (rowinput(cat + ":", cat) if get_input else cat) for cat in cats}

    if get_input:
        df[cat_col] = df[cat_col].map(mapping)

    for i, cat in enumerate(cats):
        filtered_df = df[df[cat_col] == mapping[cat]]
        plt.plot(filtered_df[x_col], filtered_df[y_col], marker=markers[i % len(markers)],
                 label=mapping[cat])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    tehran = timezone('Asia/Tehran')
    now = datetime.now(tehran)
    now = now.strftime("%m-%d-%H-%M-%S")  # Adds seconds
    fname = col + "-" + measure + ".pdf"
    if Path(fname).is_file():
        shutil.move(fname, fname + now + ".pdf")
    plt.savefig(fname, bbox_inches='tight')
    open_pdf(fname)

def line_plot(df, selected_cols, measure_cols, x_label, y_label='Accuracy'):
    col = selected_cols[0]
    measure = measure_cols[0]

    summary2 = df.groupby(col)[measure].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(8, 5))

    # Plot mean and error bars
    plt.plot(summary2[col], summary2['mean'], '-o', label='Mean', color='green')
    plt.errorbar(summary2[col], summary2['mean'], yerr=summary2['std'], fmt='o', capsize=5, 
                 ecolor='black', label='Std Dev')

    # Use larger font sizes for publication-quality output
    plt.xlabel(x_label) #, fontsize=18, fontweight='bold')
    plt.ylabel(y_label) # + ' ', fontsize=18, fontweight='bold')
    plt.xticks(summary2[col]) #, fontsize=16)
    #plt.yticks(fontsize=16)
    plt.grid(True)
    #plt.legend(fontsize=16)
    plt.tight_layout()
    tehran = timezone('Asia/Tehran')
    now = datetime.now(tehran)
    now = now.strftime("%m-%d-%H-%M-%S")  # Adds seconds
    fname = col + "-" + measure + ".pdf"
    if Path(fname).is_file():
        shutil.move(fname, fname + now + ".pdf")
    plt.savefig(fname, bbox_inches='tight')
    open_pdf(fname)


def compare(df, dim_cols, measure_cols, cat_cols):
    matplotlib.rcParams.update({
        'font.size': 12,
        'figure.dpi': 300,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'lines.linewidth': 2,
        'legend.fontsize': 10,
        'grid.alpha': 0.4,
    })

    # df = pd.read_table("compare.tsv")
    y_col = measure_cols[0]

    legend_map = {
        ('SLP', 'wsp1'): 'WSP',
        ('SLP', 'wcp1'): 'WCP',
        ('P', 'wavg'): 'P (private only)',
        ('SL', 'wavg'): 'S (single source only)',
    }
    color_map = {
        ('P', 'wavg'): 'orange',
        ('SL', 'wavg'): 'black',
        ('SLP', 'wsp1'): 'blue',
        ('SLP', 'wcp1'): 'green'
    }

# Plot setup
    plt.figure(figsize=(7.5, 4.5))  # more space for legend

    grouped = df.groupby(['prompts_conf', 'compose_method'])
    for (conf, cm), group in grouped:
        label = legend_map.get((conf, cm), f'{conf} ({cm})')
        color = color_map.get((conf, cm), 'gray')
        
        if len(group['num_target_prompts'].unique()) == 1:
            # Constant line for baseline
            y_val = group[y_col].values[0]
            plt.hlines(
                y=y_val,
                xmin=df['num_target_prompts'].min(),
                xmax=df['num_target_prompts'].max(),
                label=label,
                linestyles='dashed',
                colors=color,
                linewidth=2.5
            )
        else:
            group_sorted = group.sort_values('num_target_prompts')
            plt.plot(
                group_sorted['num_target_prompts'],
                group_sorted[y_col],
                marker='o',
                label=label,
                color=color
            )

    # Finalize plot
    plt.xlabel("Number of Source Prompts ")
    if not "All" in y_col:
        plt.ylabel(f"Mean Accuracy ({y_col})")
    else:
        plt.ylabel(f"Mean Accuracy ")
    plt.grid(True)
    plt.legend(
        loc='lower right',
        fontsize=8,
        bbox_to_anchor=(1, 0.1),  # default is (1, 0) for lower right, 0.39 inches = 1 cm above
        frameon=True
    )

    plt.tight_layout()
    tehran = timezone('Asia/Tehran')
    now = datetime.now(tehran)
    now = now.strftime("%m-%d-%H-%M-%S")  # Adds seconds
    if Path("compare.pdf").is_file():
        shutil.move("compare.pdf", "comapre_back" + now + ".pdf")
    plt.savefig("compare.pdf", bbox_inches='tight')
    open_pdf("compare.pdf")

