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
from PyPDF2 import PdfMerger

def add_open_pdf(fname):
    base_fname = os.path.join("/home","ahmad", "pics", fname + ".pdf")
    temp_plot = "temp_plot.pdf"
    final_pdf = base_fname
    now = "now"
    if Path(base_fname).is_file():
        shutil.move(base_fname, base_fname + "." + now + ".bak")

    plt.savefig(temp_plot, bbox_inches='tight')
    plt.close()

    merger = PdfMerger()
    if Path(base_fname + "." + now + ".bak").is_file():
        merger.append(base_fname + "." + now + ".bak")

    merger.append(temp_plot)
    merger.write(final_pdf)
    merger.close()
    Path(temp_plot).unlink()
    open_pdf(final_pdf)

def open_pdf(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", path])
    else:  # Assume Linux or Unix
        subprocess.run(["xdg-open", path])

def line_plot(df, x_col, y_cols, cat_cols, x_label, 
        y_labels=['Value'], use_std=False, new_file=False, normalize=False):
    if isinstance(cat_cols, str):
        cat_cols = [cat_cols]
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    if not cat_cols or cat_cols == [None]:
        cat_cols = []

    df = df.copy()

    if cat_cols:
        df['group'] = df[cat_cols].astype(str).agg('_'.join, axis=1)
    else:
        df['group'] = ''  # empty placeholder for group

    df = df.sort_values(x_col)

    if normalize:
        for y in y_cols:
            y_min, y_max = df[y].min(), df[y].max()
            df[f"{y}_norm"] = (df[y] - y_min) / (y_max - y_min + 1e-8)

    markers = ['o', 's', '^', 'D', 'v', 'x']
    linestyles = ['-', '--', ':', '-.']
    colors = ['brown', 'blue', 'green', 'black', 'red', 'magenta', 'teal', 'orange']

    plt.figure(figsize=(12, 6))

    for j, y_col in enumerate(y_cols):
        plot_col = f"{y_col}_norm" if normalize else y_col

        groups = df['group'].unique()
        for i, group in enumerate(groups):
            filtered = df[df['group'] == group]

            # Legend label: y_col if no group; otherwise group
            label = y_labels[j] if not cat_cols else group

            if use_std:
                summary = filtered.groupby(x_col)[plot_col].agg(['mean', 'std']).reset_index()
                plt.errorbar(
                    summary[x_col], summary['mean'], yerr=summary['std'],
                    fmt=markers[i % len(markers)], capsize=5,
                    linestyle=linestyles[j % len(linestyles)],
                    color=colors[(j + i) % len(colors)],
                    ecolor=colors[(j + i + 3) % len(colors)],
                    label=label
                )
            else:
                plt.plot(
                    filtered[x_col], filtered[plot_col],
                    marker=markers[i % len(markers)],
                    linestyle=linestyles[j % len(linestyles)],
                    color=colors[(j + i) % len(colors)],
                    label=label
                )

    plt.xlabel(x_label, fontweight='bold')
    y_label = y_labels[0] if len(y_labels) == 1 else ""
    plt.ylabel(y_label, fontweight='bold')
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fname = new_file
    if not new_file:
        base = x_col + "-" + "-".join(y_cols)
        base += "-" + "-".join(cat_cols) if cat_cols else "-ycols"
        base += ".std" if use_std else ""
        fname = base + ".pdf"

    if Path(fname).is_file():
        shutil.move(fname, fname + ".bak")
    # plt.savefig(fname, bbox_inches='tight')
    add_open_pdf(fname)

def bar_plot(df, x_col, y_cols, cat_cols, x_label,
             y_labels=['Value'], use_std=False, new_file=False, normalize=False):

    if isinstance(cat_cols, str):
        cat_cols = [cat_cols]
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    if not cat_cols or cat_cols == [None]:
        cat_cols = []

    df = df.copy()

    if cat_cols:
        df['group'] = df[cat_cols].astype(str).agg('_'.join, axis=1)
    else:
        df['group'] = ''

    df = df.sort_values(x_col)

    if normalize:
        for y in y_cols:
            y_min, y_max = df[y].min(), df[y].max()
            df[f"{y}_norm"] = (df[y] - y_min) / (y_max - y_min + 1e-8)

    colors = ['blue', 'green', 'orange', 'black', 'red', 'magenta', 'teal', 'brown']
    width = 0.8 / len(df['group'].unique())  # dynamic bar width
    x_ticks = df[x_col].unique()

    plt.figure(figsize=(12, 6))

    for j, y_col in enumerate(y_cols):
        plot_col = f"{y_col}_norm" if normalize else y_col
        groups = df['group'].unique()

        for i, group in enumerate(groups):
            filtered = df[df['group'] == group]

            # Ensure alignment on x-axis
            x_vals = filtered[x_col].values
            y_vals = filtered[plot_col].values
            x_positions = [x + i * width for x in range(len(x_vals))]

            label = y_labels[j] if not cat_cols else group

            if use_std:
                std_vals = filtered.groupby(x_col)[plot_col].std().reindex(x_vals).values
                plt.bar(x_positions, y_vals, width=width,
                        yerr=std_vals, capsize=5,
                        color=colors[i % len(colors)],
                        label=label)
            else:
                plt.bar(x_positions, y_vals, width=width,
                        color=colors[i % len(colors)],
                        label=label)

        # Reset x-ticks for each y_col loop if needed
        plt.xticks([r + width * (len(groups)-1)/2 for r in range(len(x_vals))], x_ticks)

    plt.ylim(70, 80)
    plt.xlabel(x_label, fontweight='bold')
    y_label = y_labels[0] if len(y_labels) == 1 else ""
    plt.ylabel(y_label, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    fname = new_file
    if not new_file:
        base = "bar-" + x_col + "-" + "-".join(y_cols)
        base += "-" + "-".join(cat_cols) if cat_cols else "-ycols"
        base += ".std" if use_std else ""
        fname = base + ".pdf"

    if Path(fname).is_file():
        shutil.move(fname, fname + ".bak")

    # plt.savefig(fname, bbox_inches='tight')
    add_open_pdf(fname)

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

