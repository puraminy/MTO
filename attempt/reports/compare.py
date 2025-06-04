import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import shutil
from pathlib import Path
from datetime import datetime
from pytz import timezone

def compare(df):
    matplotlib.rcParams.update({
        'font.size': 12,
        'figure.dpi': 300,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'lines.linewidth': 2,
        'legend.fontsize': 10,
        'grid.alpha': 0.4,
    })

# Sample Data
    df = pd.read_table("compare.tsv")

# Mapping for legend and color
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
            y_val = group['All_mean'].values[0]
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
                group_sorted['All_mean'],
                marker='o',
                label=label,
                color=color
            )

    # Finalize plot
    plt.xlabel("Number of Source Prompts ")
    plt.ylabel("Mean Accuracy")
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

