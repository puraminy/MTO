#import wandb
import seaborn as sns
#import PIL
import matplotlib.pyplot as plt
from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerCallback 
from math import floor
import attempt.mylogs as mylogs
from attempt.myutil import tag_to_image
import matplotlib.pyplot as plt
import json, os
import io
from PIL import Image
import torch
import logging
logger = logging.getLogger(__name__)
import numpy as np

def reduce_consecutive_zeros(matrix):
    # Find which columns are zero columns
    is_zero_column = np.all(matrix == 0, axis=0)

    # Initialize variables for reduced matrix
    reduced_matrix = []
    current_zero_sequence = False  # Flag to track if we are in a sequence of zero columns

    for col_idx in range(matrix.shape[1]):
        if is_zero_column[col_idx]:
            # Current column is a zero column
            if not current_zero_sequence:
                # Start a new zero sequence
                reduced_matrix.append(matrix[:, col_idx:col_idx+1])  # Add the zero column
                current_zero_sequence = True
        else:
            # Current column is not a zero column
            reduced_matrix.append(matrix[:, col_idx:col_idx+1])  # Add the non-zero column
            current_zero_sequence = False  # Reset the zero sequence flag

    # Stack the reduced matrix parts horizontally to form the final matrix
    if reduced_matrix:
        final_matrix = np.hstack(reduced_matrix)
    else:
        final_matrix = np.empty((matrix.shape[0], 0))  
    return final_matrix


class PTLearningRateCallback(TrainerCallback):
    def on_log(self, args, state, control, logs = None, **kwargs):
        model = kwargs.pop("model", None)
        mylogs.bp("ptlr")
        lr = kwargs.pop("lr_scheduler", None)
        optimizer = kwargs.pop("optimizer", None)
        #if optimizer:
        #    for i, param_group in enumerate(optimizer.param_groups):
        #       logger.info(f"Learning rate for parameter group {i}: {param_group['lr']}")

        if lr:
            #logs["slr"] = lr._last_lr[0]
            #logs["tlr"] = lr._last_lr[1]
            #logs["step"] = state.global_step 
            last_lrs = lr.get_last_lr()
            #for i, llr in enumerate(last_lrs):
            #    logs["lr" + str(i)] = '{:3}'.format('{}'.format(llr)) 
        logger.info(logs)

class AnnealCallback(TrainerCallback):
    def on_log(self, args, state, control, logs = None, **kwargs):
        model = kwargs.pop("model", None)
        e = model.encoder
        logs["temperature:"] = '{:3}'.format('{}'.format(e.temperature)) 
        logs["threshold:"] = '{:3}'.format('{}'.format(e.sel_thresh)) 

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs.pop("model", None)
        e = model.encoder
        e.anneal(state.global_step)
        # wandb.log({"temperature": e.temperature})
        #mylogs.winfo("router","%s: %s  (%s %s > %s)", state.global_step, 
        #        e.router_temperature, e.anneal_dir, e.anneal_rate, e.anneal_min)

class WBCallback(WandbCallback):
    cur_epoch = -1
    def __init__(self, save_path, save_router_image=False, **kwargs):
        self.save_path = save_path
        self.save_router_image = save_router_image
        super().__init__()

    @staticmethod
    def save_images(scores, x_labels, y_labels, state=None, fname="", 
            annot=True,title="", add_tags=True, vmin=None, vmax=None):
        if not title: title = fname
        if add_tags:
            fig, axes = plt.subplot_mosaic("ABB;ACC;ADD")
            ax1, ax2, ax3,ax4 = axes["A"], axes["B"], axes["C"], axes["D"]
            axes = [ax2, ax3, ax4]
            ax_t = ax2
        else:
            fig, axes = plt.subplot_mosaic("A;B;C")
            ax1, ax2, ax3 = axes["A"], axes["B"], axes["C"]
            axes = [ax1, ax2, ax3]
            ax_t = ax1
        if state is not None:
            ax_t.set_title(f"Epoch:{state.epoch}  Step:{state.global_step} Best:{state.best_metric}")
        else:
            ax_t.set_title(title)
        fig.set_size_inches(12.5, 6.5)
        if add_tags:
            ax1.axis("off")
            tags = mylogs.get_full_tag()
            img = tag_to_image(tags)
            fig.figimage(img, 5, 100)
        for score, ax in zip(scores, axes):
            np_score = score.detach().cpu().numpy()
            if np_score.size != 0:
                sns.heatmap(np_score, ax=ax, cmap="crest", annot=annot, 
                        # annot_kws={'rotation': 90}, 
                        vmin = vmin, vmax=vmax,
                        xticklabels=x_labels,
                        yticklabels=y_labels,
                        linewidth=0.5)
        #plt.tight_layout()
        mylogs.bp("wand")
        #if fname:
        #    wandb.log({fname:wandb.Image(fig)})
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close("all")
        return img_buf

    @staticmethod
    def save_image(scores, x_labels, y_labels, fpath="", mask_zeros=False,
            annot=True,title="", df=None, img_h=6.5, cbar=True, vmin=None, vmax=None):
        # if not title: title = fpath
        mylogs.bp("save_image")
        if len(scores) == 2:
            fig, axes = plt.subplot_mosaic("AB")
            ax1, ax2 = axes["A"], axes["B"]
            axes = [ax1, ax2]
            ax_t = ax2
        else:
            fig, axes = plt.subplot_mosaic("A")
            ax1 = axes["A"]
            axes = [ax1]
            ax_t = ax1
        ax1.set_title(title)
        if not type(scores) == list:
            scores = [scores]
        mask = None
        sns.set(font_scale=1.2)
        for ax, sc in zip(axes, scores):
            np_score = sc.detach().cpu().numpy()
            if mask_zeros:
                np_score = reduce_consecutive_zeros(np_score)
                zero_columns = np.where(np.all(np_score == 0, axis=0))[0]
                mask = np.zeros_like(np_score, dtype=bool)
                mask[:, zero_columns] = True  # Set mask to False for zero columns
                mm = (np_score == -10)
                np_score[mm] = 0

            fig.set_size_inches(np_score.shape[1], np_score.shape[0])
            sns.heatmap(np_score, ax=ax, cmap="crest", annot=annot, 
                    cbar=cbar, mask=mask, 
                    vmin = vmin, vmax=vmax,
                    # annot_kws={'rotation': 90}, 
                    xticklabels=x_labels,
                    yticklabels=y_labels,
                    linewidth=0.5)
        #plt.tight_layout()
        mylogs.bp("wand")
        if fpath:
            plt.savefig(fpath, format='png')
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close("all")
        return img_buf

    def on_epoch_begin(self, args, state, control, **kwargs):
        if not self.save_router_image:
            return
        mylogs.bp("save_router")
        model = kwargs.pop("model", None)
        self.save_router(model, state)

    def save_router(self, model, state):
        targets = model.encoder.target_encoders_idx
        y_labels = [model.encoder.prompt_names[i] for i in targets]
        y_labels = [y.replace("tar-","") for y in y_labels]
        p_labels = []
        for pl in model.encoder.prompt_names:
            if not "tar" in pl and not "input" in pl:
                pl = pl.replace("source_for_","") 
                pl = pl.replace("source_","") 
                pl = pl.replace("superglue-","") 
                pl = pl.replace("com","src") 
                p_labels.append(pl)
        router_scores = model.encoder.router.index_select(0, targets)
        square = False
        x_labels = y_labels
        if not square:
            if p_labels: x_labels = p_labels 
        tlen = router_scores.size(0)
       # rsim = torch.eye(tlen)
       # cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
       # for i in range(tlen):
       #     for j in range(tlen):
       #         if i != j:
       #             rsim[i][j] = cos(router_scores[i][:], 
       #                     router_scores[j][:])

        vmin = 0 if tlen <=3 else None
        fname = "pred@router@router_" + str(state.epoch)  + ".png"
        fpath = os.path.join(self.save_path, fname)
        self.save_image(router_scores, x_labels, y_labels, fpath,
                    annot=True,  vmin=vmin, vmax=1)

    def setup(self, args, state, model, **kwargs):
        epoch = floor(state.epoch)
        mylogs.bp("wand")
        epoch = int(epoch)
        if state.global_step % 50 == 1 or state.global_step == 2:
            self.save_router(model, state)

