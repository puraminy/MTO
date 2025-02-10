# several of the evaluation metrics are from https://github.com/google-research/text-to-text-transfer-transformer/blob/a1352e625db7ec114062f99d99b0565b9e45c155/t5/evaluation/metrics.py
"""Defines different metrics used for evaluation of tasks."""
import numpy as np
import scipy
import math
import sklearn
import collections
from logging import getLogger
from .qa_utils import normalize_squad, qa_metrics
import sklearn.metrics
import functools

## My imports
from pandas.plotting import table
import matplotlib.pyplot as plt
import matplotlib.colors as clr 
import torch
import wandb
from rouge import Rouge
from attempt.mylogs import *
import attempt.mylogs as mylogs
from attempt.myutil import df_to_image
if not colab:
    from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import re
import pandas as pd
import json



from bleurt import score
import tensorflow as tf


TASK_TO_METRICS = {
                   "default":["rouge"],
                   "atomic": ["rouge"],
                   "xIntent": ["rouge"],
                   "xAttr": ["rouge"],
                   "xWant": ["rouge"],
                   "xEffect": ["rouge"],
                   "xNeed": ["rouge"],
                   "atomic-rels": ["rouge"],
                   "xReact": ["rouge"],
                   "oReact": ["rouge"],
                   "oEffect": ["rouge"],
                   "oWant": ["rouge"],
                   "isBefore": ["rouge"],
                   "isAfter": ["rouge"],
                   "mrpc": ["accuracy"], #, "f1_score_with_invalid"],
                   "cola": ["accuracy"], # ['matthews_corrcoef'],
                   "stsb": ['pearson_corrcoef', 'spearman_corrcoef'],
                   'sst2': ['accuracy'],
                   "mnli": ["accuracy"],
                   "mnli_mismatched": ["accuracy"],
                   "mnli_matched": ["accuracy"],
                   "qnli": ["accuracy"],
                   "rte": ["accuracy"],
                   "wnli": ["accuracy"],
                   "tweet-eval": ["accuracy"],
                   "qqp": ["accuracy"],
                   "superglue-boolq": ["accuracy"],
                   "superglue-rte": ["accuracy"],
                   "superglue-cb": ["accuracy"],
                   "superglue-copa": ["accuracy"],
                   "superglue-multirc": ["f1_score_with_invalid"], # "exact_match"],
                   "superglue-wic": ["accuracy"],
                   "superglue-wsc.fixed": ["accuracy"],
                   "superglue-record": ["f1_score_with_invalid", "exact_match"],
                   "multi_nli": ["accuracy"],
                   "squad": ["exact_match", "f1_score_with_invalid"],
                   "snli": ["accuracy"],
                   "nq": ["exact_match", "f1_score_with_invalid"],
                   "hotpotqa": ["exact_match", "f1_score_with_invalid"],
                   "searchqa": ["exact_match", "f1_score_with_invalid"],
                   "newsqa": ["exact_match", "f1_score_with_invalid"],
                   "triviaqa": ["exact_match", "f1_score_with_invalid"],
                   "imdb": ["accuracy"],
                   "winogrande": ["accuracy"],
                   "scitail": ["accuracy"],
                   "amazon_polarity": ["accuracy"],
                   "yelp_polarity": ["accuracy"],
                   "paws": ["accuracy"], }

logger = getLogger(__name__)

def rouge(predictions, targets) -> dict:
    """Computes rouge score."""
    #breakpoint()
    rouge_scorer = Rouge()
    rouge_score = -1
    try:
        rouge_score = rouge_scorer.get_scores(predictions, targets,
                                            avg=True, ignore_empty=True)
        rouge_score = rouge_score["rouge-l"]["f"]
    except:
        pass
    return {"rouge": rouge_score}

def accuracy(predictions, targets) -> dict:
    """Computes the average accuracy."""
    mylogs.bp("compute")
    targets = [str(t).strip() for t in targets]
    predictions = [str(p).strip() for p in predictions]
    preds = []
    for t,p in zip(targets, predictions):
        if len(p.split(" ")) > len(t.split(" ")):
            preds.append(p.split(" ")[0])
        else:
            preds.append(p)
    acc = 100 * ((np.array(preds) == np.array(targets)).mean())
    return {"accuracy": "{:.2f}".format(acc)}


def pearson_corrcoef(predictions, targets) -> dict:
    """Computes Pearson correlation coefficient."""
    from data.postprocessors import string_to_float
    targets = [string_to_float(target) for target in targets]
    predictions = [string_to_float(prediction) for prediction in predictions]
    if np.isnan(predictions).any():
        predictions = np.nan_to_num(predictions)
    pearson_corrcoef = 100 * scipy.stats.pearsonr(targets, predictions)[0]

    # Note that if all the predictions will be the same, spearman
    # correlation is nan, to gaurad against this, we check the output
    # and return 0 in this case.
    if math.isnan(pearson_corrcoef):
        pearson_corrcoef = 0
    return {"pearson": "{:.2f}".format(pearson_corrcoef)}


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def spearman_corrcoef(predictions, targets) -> dict:
    """Computes Spearman correlation coefficient."""
    # TODO: we need to do postprocessors in a clean way for each dataset.
    from data.postprocessors import string_to_float
    targets = [string_to_float(target) for target in targets]
    predictions = [string_to_float(prediction) for prediction in predictions]
    spearman_corrcoef = 100 * scipy.stats.spearmanr(targets, predictions)[0]

    # Note that if all the predictions will be the same, spearman
    # correlation is nan, to gaurad against this, we check the output
    # and return 0 in this case.
    if math.isnan(spearman_corrcoef):
        spearman_corrcoef = 0
    return {"spearmanr": "{:.2f}".format(spearman_corrcoef)}


def f1_score_with_invalid(predictions, targets) -> dict:
    """Computes F1 score,  with any prediction != 0 or 1 is counted as incorrect.
    Args:
      targets: list of targets, either 0 or 1
      predictions: list of predictions, any integer value
    Returns:
      F1 score, where any prediction != 0 or 1 is counted as wrong.
    """
    def binary_reverse(labels):
        return ['0' if label == '1' or label.lower() == 'true' else '1' for label in labels]

    targets, predictions = np.asarray(targets), np.asarray(predictions)
    # Get indices of invalid predictions.
    invalid_idx_mask = np.logical_and(predictions != 'False', predictions != 'True')
    # For any prediction != 0 or 1, we set the prediction to the opposite of its corresponding target.
    predictions[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask])
    try:
        return {"f1": 100 * sklearn.metrics.f1_score(targets, predictions, pos_label='False')}
        #targets = targets.astype(np.int32)
        #predictions = predictions.astype(np.int32)
        #return {"f1": 100 * sklearn.metrics.f1_score(targets, predictions)}
    except:
        return {"f1": -1}

# TODO: maybe gaurd against invalid values https://stackoverflow.com/questions/56865344/how-do-i-calculate-the-matthews-correlation-coefficient-in-tensorflow


def matthews_corrcoef(predictions, targets) -> dict:
    """Computes the Matthews correlation coefficient."""
    targets = [t.strip() for t in targets]
    predictions = [p.strip() for p in predictions]
    return {"matthews_correlation": 100 * sklearn.metrics.matthews_corrcoef(targets, predictions)}


def squad(predictions, targets):
    """Computes SQuAD metrics, maximizing over answers per question.
    Args:
      targets: list of lists of strings
      predictions: list of strings
    Returns:
      dict with score_key: squad score across all targets and predictions
    """

    if type(targets[0]) is list:
        targets = [[normalize_squad(t) for t in u] for u in targets]
    else:
        targets = [[normalize_squad(u)] for u in targets]

    predictions = [normalize_squad(p) for p in predictions]
    return qa_metrics(targets, predictions)


def exact_match(predictions, targets):
    """Computes whether the targets match predictions exactly."""
    return {"em": 100 * float(np.array_equal(targets, predictions))}


def sklearn_metrics_wrapper(metric_str,
                            metric_dict_str=None,
                            metric_post_process_fn=None,
                            **metric_fn_kwargs):
    """Wraps any sklearn.metric function and returns a t5 metric function.
    Args:
      metric_str: string, the function from `sklearn.metrics` to use.
      metric_dict_str: optional string, if not specified `metric_str` is used as
        the key in the returned dictionary.
      metric_post_process_fn: callable, if specified the final computed metric
        will be passed through this.
      **metric_fn_kwargs: kwargs, passed to the metric function we are calling.
    Returns:
      the function that calculates the metric in a dict.
    """
    if not hasattr(sklearn.metrics, metric_str):
        raise ValueError("sklearn.metrics does not have: %s" % metric_str)

    def fn(predictions, targets):
        metric_fn = getattr(sklearn.metrics, metric_str)
        metric_val = metric_fn(targets, predictions, **metric_fn_kwargs)
        if metric_post_process_fn is not None:
            metric_val = metric_post_process_fn(metric_val)
        return {metric_dict_str or metric_str: metric_val}
    return fn


def mean_multiclass_f1(num_classes, **metric_fn_kwargs):
    """Computes the unweighted average of the F1 per class."""
    return sklearn_metrics_wrapper(
        "fbeta_score",
        metric_dict_str="f1_multiclass",
        metric_post_process_fn=lambda x: 100 * x,
        beta=1,
        labels=range(num_classes),
        average="macro",
        **metric_fn_kwargs)


def multirc_f1_over_all_answers(targets, predictions):
    """Special metric for MultiRC which computes F1 score over all examples.
    This is necessary because the targets/predictions for MultiRC are dicts and
    the f1_score_with_invalid expects a list of True/False labels, not dicts. As
    a result we just need to key in the "value" for each of the example dicts
    before feeding into f1_score_with_invalid.
    Args:
      targets: list of dicts, where each dict has a "value" key.
      predictions: list of dicts, where each dict has a "value" key.
    Returns:
      F1 score over values, where any prediction != 0 or 1 is counted as wrong.
    """
    return f1_score_with_invalid(
         targets, predictions
        # [t["value"] for t in targets], [p["value"] for p in predictions]
    )


def mean_group_metric(metric_fn, group_key="group", value_key="value"):
    """Returns a metric that averages `metric_fn` on sub-groups of results.
    The sub-groups are defined by aggregating results (targets and predictions)
    by accessing the feature specified by `group_key` in the target dicts.
    **WARNING**: Using this function can produce unreliable results if you do not
    pass in full groups. For example, if you evaluate over a random subsample of a
    validation set and do not retain all of the examples in each group, you may
    get results which aren't directly comparable to using the full validation set.
    Args:
      metric_fn: function, the metric to compute on the subgroups.
      group_key: string, the key for the grouping value in the target dictionary.
      value_key: string, the key for the value in the dictionaries.
    """
    def my_metric(targets, predictions):
        """Computes mean of `metric_fn` over subgroups of results."""
        grouped_values = collections.defaultdict(lambda: ([], []))
        for targ, pred in zip(targets, predictions):
            g = targ[group_key]
            grouped_values[g][0].append(targ[value_key])
            grouped_values[g][1].append(pred[value_key])
        group_scores = collections.defaultdict(list)
        for (targets, predictions) in grouped_values.values():
            for metric, score in metric_fn(targets, predictions).items():
                group_scores[metric].append(score)
        return {metric: np.mean(scores) for metric, scores in group_scores.items()}
    return my_metric


def build_compute_metrics_fn(task_names, tokenizer, ignore_pad_token_for_loss):
    """Builds a dictionary from each task to the task metric."""

    print(task_names)

    def compute_metrics(eval_preds, eval_metrics, post_processor=None):
        preds, labels, data_info = eval_preds
        decoded_preds, decoded_labels = post_processor.process(
            preds, labels, data_info)
        result = {}
        for metric in eval_metrics:
            result.update(metric(decoded_preds, decoded_labels))
        return result

    def tasks_metrics(task):
        from data.tasks import TASK_MAPPING
        from data.postprocessors import AutoPostProcessor
        task_metric = TASK_TO_METRICS[task] if task in TASK_TO_METRICS else ["rouge"] 
        post_processor = AutoPostProcessor.get(
            task, tokenizer, ignore_pad_token_for_loss)
        return functools.partial(compute_metrics, metrics=task_metric, post_processor=post_processor)

    return {task: tasks_metrics(task) for task in task_names}

def bleu_score(scorer, hyps, refs, device):
    if scorer == None:
        return 0, 0, 0.0

    # Clean the input sentences
    hyps = [p.strip() for p in hyps]
    refs = [g.strip() for g in refs]
    
    # Compute BLEURT scores for each hypothesis-reference pair
    scores = scorer.score(references=refs, candidates=hyps)
    
    # Find the pair with the highest BLEURT score
    best_score = max(scores)
    best_index = scores.index(best_score)
    
    best_hyp_index = best_index
    best_ref_index = best_index
    
    return best_hyp_index, best_ref_index, best_score

######## My functions
def bert_score(scorer, hyps, refs, device):
    if scorer == None:
        return 0, 0, 0.0

    hyps = [p.strip() for p in hyps]
    refs = [g.strip() for g in refs]

    embeddings1 = scorer.encode(hyps, device=device, convert_to_tensor=True)
    embeddings2 = scorer.encode(refs, device=device, convert_to_tensor=True)

    #Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    #Find the pairs with the highest cosine similarity scores
    pairs = []
    rows = cosine_scores.shape[0]
    cols = cosine_scores.shape[1]
    for i in range(rows):
        for j in range(cols):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
        #logging.info({'index': [i, j], 'score': cosine_scores[i][j]})

    #Sort scores in decreasing order
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

    top = pairs[0]
    best_hyp_index = top["index"][0]
    best_ref_index = top["index"][1]

    return best_hyp_index, best_ref_index, top["score"] 

rel_target_omits = {
   # "xIntent":"to",
}
def do_score(df, scorers, save_path, reval=False, scores_to_image=False, use_wandb=False):
    #try:
    #    nltk_path = str(nltk.data.find("tokenizers/punkt"))
    #    mlog.info(f"using nltk from: {nltk_path}")
    #except LookupError:
    #    nltk.download('punkt')
    base_path = "/content/drive/MyDrive/pret"
    if not colab:
        base_path = os.path.join(home, "pret")
    checkpoint = f"{base_path}/paraphrase-MiniLM-L6-v2"
    if not Path(checkpoint).exists():
        checkpoint = 'sentence-transformers/paraphrase-MiniLM-L6-v2'

    bert_scorer = None
    if colab:
        scorers = scorers.replace("bert","")
    if "bert" in scorers:
        bert_scorer = SentenceTransformer(checkpoint)

    checkpoint = f"{base_path}/bleurt-20"
    if not Path(checkpoint).exists():
        # checkpoint = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
        raise FileNotFoundError(checkpoint + " doesn't exists!")

    bleu_scorer = None
    if "bleu" in scorers:
        bleu_scorer = score.BleurtScorer(checkpoint)

    rouge_scorer = None
    if "rouge" in scorers:
        rouge_scorer = Rouge()

    checkpoint = f"{base_path}/nli-roberta-base-v2"
    if not Path(checkpoint).exists():
        checkpoint = 'sentence-transformers/nli-roberta-base-v2'
    nli_model = None
    if "nli" in scorers:
        nli_model = CrossEncoder(checkpoint)
    nli_counter = {}
    nli_map = ['contradiction', 'entailment', 'neutral']
    for l in nli_map:
        nli_counter[l] = 0
    counter = {"all":0}
    sum_match = {"all":0} 
    mean_match = {}
    sum_bert = {"all":0} 
    mean_bert = {}
    sum_rouge = {"all":0}
    mean_rouge = {}
    sum_bleu = {"all":0}
    mean_bleu = {}
    sum_out = {"all":0}
    mean_out = {}
    new_results = {}
    #smoothie = SmoothingFunction().method4 # a function for smooth
    hyp_counter = [0]*5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_predictions = []
    all_golds = []
    if not reval:
        mlog.info("Preparing iterator ...")
        mlog.info("Scoring....")
    if scorers:
        rows = []
        pbar = tqdm(total=len(df), position=0, leave=True) #,dynamic_ncols=True)
        df = df.sort_values(by=["prefix","input_text"])
        preds = []
        tails = []
        mylogs.bp("score")
        oldinp = df.iloc[0]["input_text"]
        rel =  df.iloc[0]["prefix"]
        gold_set = [w.strip() for w in df["target_text"].unique()]
        for step, row in df.iterrows():
            oldrel = rel
            rel = row["prefix"]
            lang = row["langs"] 
            scope = rel + "_" + lang
            if not scope in sum_bert: 
                sum_bert[scope] = 0
                sum_rouge[scope] = 0
                sum_bleu[scope] = 0
                sum_match[scope] = 0
                counter[scope] = 0
                sum_out[scope] = 0
            #Compute embeddings
            cur_hyp = str(row["pred_text1"])

            inp = row["input_text"]
            tail = re.sub(r'<extra_.*?>','',str(row["target_text"]))
            tail = tail.strip()
            all_predictions.append(cur_hyp)
            all_golds.append(tail)
            if oldinp == inp:
                preds.append(cur_hyp)
                tails.append(tail)
                pbar.update()
                continue

            data = {"prefix":oldrel, "input_text":oldinp}
            counter[scope] += 1
            counter["all"] += 1
            oldinp = inp
            hi, ri = 0, 0
            if bert_scorer is not None or bleu_scorer is not None:
                if bert_scorer is not None:
                    hi, ri, cur_score = bert_score(bert_scorer, preds, tails, device)
                    best_hyp = preds[hi]
                    best_ref = tails[ri]
                    data["bert_score"] = float("{:.2f}".format(cur_score))
                    sum_bert[scope] += cur_score
                    sum_bert["all"] += cur_score
                    mean_bert[scope] = "{:.4f}".format(sum_bert[scope] / counter[scope])
                    mean_bert["all"] = "{:.4f}".format(sum_bert["all"] / counter["all"])
                if bleu_scorer is not None:
                    hi, ri, cur_score = bleu_score(bleu_scorer, preds, tails, device)
                    best_hyp = preds[hi]
                    best_ref = tails[ri]
                    data["bleu_score"] = float("{:.2f}".format(cur_score))
                    sum_bleu[scope] += cur_score
                    sum_bleu["all"] += cur_score
                    mean_bleu[scope] = "{:.4f}".format(sum_bleu[scope] / counter[scope])
                    mean_bleu["all"] = "{:.4f}".format(sum_bleu["all"] / counter["all"])
                #hyp_counter[hi] += 1
                data["match_score"] = 0
                data["top"] = best_ref
                data["all_preds"] = "<br />".join(preds) 
                data["top_pred"] = best_hyp
                if best_hyp.strip() == best_ref:
                    data["match_score"] = 1
                    sum_match[scope] += 1
                    sum_match["all"] += 1
                data["out_score"] = 0
                if best_hyp.strip() not in gold_set:
                    data["out_score"] = 1
                    sum_out[scope] += 1
                    sum_out["all"] += 1
                mean_out[scope] = "{:.4f}".format(sum_out[scope] / counter[scope])
                mean_out["all"] = "{:.4f}".format(sum_out["all"] / counter["all"])
            if nli_model:
                pair = (best_hyp, best_ref)
                nli_scores = nli_model.predict(pair)  
                _max  = nli_scores.argmax()
                label = nli_map[_max]
                nli_counter[label] += 1
                data["nli_group"] = label
            #### BLUE score
            #tokenized_rs = []
            #for r in tails:
            #    tokenized_rs.append(word_tokenize(r))
            #hypo = word_tokenize(top_hyp)
            #bleu_score = 0.0
            #try:
            #    bleu_score = sentence_bleu(tokenized_rs, hypo, smoothing_function=smoothie)
            #except ValueError: # TODO ZeroDivisionError
            #    vlog.warning("math domain error in bleu, set to 0.0. generated sentence: {}".format(hypo))
            #data["bleu_score"] = bleu_score 
            #sum_bleu[scope] += bleu_score 
            #mean_bleu[scope] = "{:.4f}".format(sum_bleu[scope] / counter[scope])
            #### Rouge score
            rouge_score = 0
            if rel in rel_target_omits:
                omit = rel_target_omits[rel]
                preds = [p.replace(omit, "") for p in preds]
                tails = [t.replace(omit,"") for t in tails]
            if rouge_scorer and preds and tails:
                try:
                    _scores = []
                    for _pred in preds:
                        for _tail in tails:
                            _score = rouge_scorer.get_scores(_pred, _tail, 
                                                    avg=True, ignore_empty=True)
                            _score = _score["rouge-l"]["f"]
                            _scores.append(_score)
                    rouge_score = max(_scores)
                except:
                    rouge_score = 0
            inp_key = inp + rel
            mean_match[scope] = "{:.4f}".format(sum_match[scope] / counter[scope])
            data["rouge_score"] = rouge_score
            #if reval or pd.isnull(row['rouge_score']):
            #    for key, value in data.items():
            #        df.loc[step, key] = value
            sum_rouge[scope] += rouge_score
            sum_rouge["all"] += rouge_score
            mean_rouge[scope] = "{:.4f}".format(sum_rouge[scope] / counter[scope])
            mean_rouge_all = sum_rouge["all"] / counter["all"]
            mean_rouge["all"] = "{:.4f}".format(mean_rouge_all)
            if bert_scorer is not None:
                pbar.set_description(f"{scope:<20} :Bert:{mean_bert[scope]:<7} | {mean_bert['all']:<7} Rouge {mean_rouge[scope]:<7}|{mean_rouge['all']:<7} ")
            else:
                pbar.set_description(f"{scope:<20} Rouge {mean_rouge[scope]:<7}|{mean_rouge['all']:<7} ")
            preds = [cur_hyp]
            tails = [tail]
            pbar.update()
            rows.append(data)

    scored_df = pd.DataFrame(rows)
    if reval:
        df = pd.merge(df, scored_df, on=["input_text", "prefix"], how="left", suffixes=('', '_new'))
        if "bert" in scorers:
            # Update existing rows with new scores
            #df["bert_score"] = df["bert_score_new"]
            df['bert_score'] = df.apply(lambda row: row['bert_score_new'] if pd.notnull(row['bert_score_new']) else row['bert_score'], axis=1)
            df.drop(columns=['bert_score_new'], inplace=True)
        if "bleu" in scorers:
            df.drop(columns=['bleu_score_new'], inplace=True)
            df['bleu_score'] = df.apply(lambda row: row['bleu_score_new'] if pd.notnull(row['bleu_score_new']) else row['bleu_score'], axis=1)
            df.drop(columns=['bleu_score_new'], inplace=True)
    else:
        #merged_df = pd.concat([df, scored_df], axis=1)
        merged_df = pd.merge(df, scored_df, on=["input_text", "prefix"])
        df = merged_df
    df.reset_index(drop=True, inplace=True)
    mlog.info("Saving results %s", save_path)
    save_fname = now + "_full_results.tsv"
    if not save_path.endswith("tsv"):
        save_path = os.path.join(save_path, save_fname) 
    print("Saving results %s", save_path)
    df.to_csv(save_path, index=False, sep="\t")
#################
    ret_scores = {}
    for sname, metric in zip(
            ['mean_rouge',  'mean_bert', 'mean_match', 'mean_bleu'],
            [mean_rouge, mean_bert, mean_match, mean_bleu]):
        s =0 
        ii = 0
        jj = 0
        for key,val in metric.items():
            metric[key] = str(val) + "--" + str(counter[key])
            s += float(val)
            ii += 1
            jj += counter[key]

        mm = s/(1 if ii == 0 else ii)
        ret_scores[sname] = mm 
        metric["AVG"] = "{:.2f}--{}".format(mm, jj)

    mean_bert_str = json.dumps(mean_bert, indent=2)
    mean_out_str = json.dumps(mean_out, indent=2)
    mean_rouge_str = json.dumps(mean_rouge, indent=2)
    mean_bleu_str = json.dumps(mean_bleu, indent=2)
    mean_match_str = json.dumps(mean_match, indent=2)

    mlog.info("-----------------------------------------------------")
    pbar.close()
    pred_counts = df['pred_text1'].unique()
    ret_scores["num_preds"] = len(pred_counts)
    mlog.info("Pred counts")
    vlog.info("Pred counts")
    if len(pred_counts) < 5:
        for  r in pred_counts:
            mlog.info(r)
            vlog.info(r)

    #for logger in [mlog, vlog, clog]:
    #    logger.info("Len data frame: {}".format(len(df)))
    #    logger.info("Rouge:{} ".format(mean_rouge_str)) 
        #if "bert" in scorers:
        #    logger.info("BERT:{} ".format(mean_bert_str)) 
        #logger.info("nli_counter: {}".format(nli_counter))
        #logger.info("hyp_counter: {}".format(hyp_counter))
    #    logger.info("Distinct preds:{}".format(len(pred_counts)))

#################
    #gdf = scored_df
    #gdf["prefix"] = df["prefix"]
    #gdf["input_text"] = df["input_text"]
    #gdf["target_text"] = df["target_text"]
    #gdf["pred_text1"] = df["pred_text1"]
    #group_col = "pred_text1"
    #gdf["rouge_score"] = gdf.groupby(['prefix','input_text'])["rouge_score"].transform("max")
    #gdf["bert_score"] = gdf.groupby(['prefix','input_text'])["bert_score"].transform("max")
    #gdf["pred_freq"] = gdf.groupby(['prefix','pred_text1'],
    #                 sort=False)["pred_text1"].transform("count")
    #cols = ['prefix']
    #gdf = gdf.merge(gdf[cols+['pred_text1']]
    #     .value_counts().groupby(cols).head(1)
    #     .reset_index(name='pred_max_num').rename(columns={'pred_text1': 'pred_max'})
    #   )

    #res_df = gdf[["prefix","input_text","pred_text1",
    #              "target_text","rouge_score","bert_score"]]
    #res_df = res_df.groupby(["prefix","input_text"]).first()
    #scores = res_df[["rouge_score","bert_score"]]
    #scores = scores.sort_values(by = ["bert_score"], ascending=False)
    #scores = scores.to_numpy()
    #fig = df_to_image(scores, annot=False)
    #wandb.run.log({"results": wandb.Image(fig)})
    #res_table = wandb.Table(dataframe=res_df)
    #wandb.run.log({"attn_scores": res_table})

########################
    #col = ["prefix"]
    #_agg = {}
    #for c in gdf.columns:
    #    if c.endswith("score"):
    #        _agg[c] = "mean"
    #    else:
    #        _agg[c] = ["first", "nunique"]
    #gb = gdf.groupby(col)
    ##counts = gb.size().to_frame(name='group_records')
    ##gdf = (counts.join(gb.agg(_agg)))
    #gdf = gb.agg(_agg)
    #gdf.columns = [ '_'.join(str(i) for i in col) for col in gdf.columns]
    #avg_len = 1
    #ren = {
    #        "target_text_nunique":"num_targets",
    #        "pred_text1_nunique":"num_preds",
    #        "input_text_nunique":"num_inps",
    #        }
    #for c in gdf.columns:
    #    if c.endswith("_mean"):
    #        ren[c] = c.replace("_mean","")
    #    elif c.endswith("_first"):
    #        ren[c] = c.replace("_first","")
    #gdf = gdf.rename(columns=ren)
    #gdf = gdf.reset_index(drop=True)
    #prefix = gdf.iloc[0]["prefix"]
    #num_preds = gdf.iloc[0]["num_preds"]
    #pred_max = gdf.iloc[0]["pred_max"]
    #pred_max_num = gdf.iloc[0]["pred_max_num"]
    #test_rouge = gdf.iloc[0]["rouge_score"]
    #test_bert = gdf.iloc[0]["bert_score"]
    #test_rouge = "{:.2f}".format(test_rouge)
    #test_bert = "{:.2f}".format(test_bert)
    #gdf = gdf[["rouge_score","bert_score"]]
    # [["num_preds","pred_max_num","num_inps","num_targets"]]
    #gtable = wandb.Table(dataframe=gdf)
    #wandb.run.log({"Summary": gtable})
    #title = f"{prefix}: rg {test_rouge} | {test_bert} | {num_preds} up, mp: {pred_max_num} | {pred_max}"
    #tdf = {"rouge_score":0, "bert_score":1}
    #gdf = pd.concat([gdf, pd.DataFrame([tdf])], ignore_index=True)
    #gdf = pd.concat([gdf, scores])
    #if scores_to_image:
    #    fig = df_to_image(gdf.to_numpy(), title = title, annot = False)
    #    if use_wandb:
    #        wandb.run.log({prefix: wandb.Image(fig)})
    #if use_wandb:
    #   wandb.run.summary["test_rouge"] = test_rouge 
    #   wandb.run.summary["test_bert"] = test_bert
    #   wandb.run.summary["num_preds"] = num_preds 

    return ret_scores
