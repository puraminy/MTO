import logging
import os
from os.path import expanduser
from pytz import timezone
import datetime
from pathlib import Path
from deepdiff import DeepDiff
import pickle
from pathlib import Path
from appdirs import *
from appdirs import AppDirs
import json

appname = "attempt"
appauthor = "ahmad"
profile = "ahmad"

main_args = {}
prev_args = {}
prev_main_vars = {}

class colors:
    HEADER = '\033[95m'
    INFO = '\033[94m'
    INFO2 = '\033[96m'
    SUCCESS = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def args(key, default="no_default"):
    if key in main_args:
        return main_args[key]
    else:
        return default

def is_debug():
    return main_args["is_debug"]

def get_full_tag(as_str=False):
    return get_tag(main_args["full_tag"], main_args, as_str)

def get_tag(tags=None, args=None, as_str=False):
    if args is None: args = main_args
    if tags is None: tags = args["tag"]
    tag_dict = {}
    tag_str = ""
    for _t in tags:
        if _t in args:
            val = args[_t]
            if type(val) == list: val = "@".join([str(v) for v in val])
            val = str(val).split("/")[-1]
            tag_dict[_t] = val
            tag_str += "|" + _t + "=" + val
        else:
            tag_dict[_t] = ""
    if as_str:
        return tag_str
    return tag_dict

tehran = timezone('Asia/Tehran')
now = datetime.datetime.now(tehran)
today = now.strftime('%Y-%m-%d')
now = now.strftime('%m-%d-%H:%M')
home = expanduser("~")
colab = not "ahmad" in home and not "pouramini" in home
if not colab: 
    logPath = os.path.join(home, "logs")
    resPath = os.path.join(home, "results") 
    pretPath = os.path.join(home, "pret") 
    confPath = os.path.join(home, "confs") 
else:
    home = "/content/drive/MyDrive/"
    pretPath = "/content/drive/MyDrive/pret"
    logPath = "/content/drive/MyDrive/logs"
    resPath = "/content/drive/MyDrive/logs/results"
    confPath = "/content/drive/MyDrive/logs/confs"

pp = Path(__file__).parent.parent.resolve()
dataPath = os.path.join(pp, "data", "atomic2020")

Path(resPath).mkdir(exist_ok=True, parents=True)
Path(logPath).mkdir(exist_ok=True, parents=True)

#logFilename = os.path.join(logPath, "all.log") #app_path + '/log_file.log'
FORMAT = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)10s() ] %(message)s")
FORMAT2 = logging.Formatter("%(message)s")
#logging.basicConfig(filename=logFilename)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(FORMAT2)
mlog = logging.getLogger("att.main")
mlog.setLevel(logging.INFO)
mlog.addHandler(consoleHandler)
clog = logging.getLogger("att.cfg")
dlog = logging.getLogger("att.data")
vlog = logging.getLogger("att.eval")
tlog = logging.getLogger("att.train")
timelog = logging.getLogger("att.time")
plog = logging.getLogger("att.preview")

def getFname(name, path=""):
    if not path:
        if "ahmad" in home or "pouramini" in home:
            path = os.path.join(home, "logs")
        else:
            path = "/content"
    logFilename = os.path.join(path, f"{name}.log")
    return logFilename

def minfo(text, *args, **kwargs):
    log = kwargs.pop("log", True)
    if log:
        mlog.info(colors.INFO2 + text + colors.ENDC + "\n", *args)
    else:
        print(colors.INFO2 + text + colors.ENDC + "\n", *args)

def success(text, *args, **kwargs):
    log = kwargs.pop("log", True)
    if log:
        mlog.info(colors.SUCCESS + text + colors.ENDC + "\n", *args)
    else:
        print(colors.SUCCESS + text + colors.ENDC + "\n", *args)

def warning(text, *args, **kwargs):
    mlog.info(colors.WARNING + text + colors.ENDC + "\n", *args)

def tinfo(text, *args, **kwargs):
    tlog.info(text, *args)

def winfo(lname, text, *args, **kwargs):
    logger = logging.getLogger(lname)
    logger.info(text, *args)

import inspect
import sys
BREAK_POINT = 0

def setbp(bpoint):
    global BREAK_POINT
    BREAK_POINT=bpoint

def bp(break_point):
    if colab: return
    cond = False 
    equal = False
    if str(break_point).startswith(">"):  
        break_point = str(break_point).strip("=") 
        cond = break_point in str(BREAK_POINT) 
        equal = True
    if str(BREAK_POINT).startswith("="):
        cond = str(BREAK_POINT) in str(break_point) 
        equal = True
    if not equal:
        # cond = break_point in str(BREAK_POINT) 
        cond = str(BREAK_POINT) in break_point # or cond 
    if cond:
        fname = sys._getframe().f_back.f_code.co_name
        line = sys._getframe().f_back.f_lineno
        mlog.info(">>>>>> break point %s at %s line %s",break_point, fname, line)
        breakpoint()

def trace(frame, event, arg):
    if event == "call":
        filename = frame.f_code.co_filename
        if filename.endswith("train/train.py"):
            lineno = frame.f_lineno
            # Here I'm printing the file and line number,
            # but you can examine the frame, locals, etc too.
            print("%s @ %s" % (filename, lineno))
    return trace

mlog.info(now)
#sys.settrace(trace)
def add_handler(logger, fname, set_format=False, base_folder=""):
    log_folder = os.path.join(base_folder, "logs")
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logFilename = os.path.join(log_folder, fname + ".log")
    handler = logging.FileHandler(logFilename, mode="w")
    if set_format:
        handler.setFormatter(FORMAT)
    logger.addHandler(handler)
    return logFilename

def init_logs(base_folder):
    for logger, fname in zip([mlog,dlog,clog,vlog,tlog,timelog], ["main","data","cfg","eval","train", "time"]):
        add_handler(logger, fname, base_folder)

def diff_args():
    if not prev_args:
        return None
    diff = DeepDiff(prev_args, main_args) 
    return diff

def set_args(args):
    global main_args, prev_args 
    prev_args = main_args.copy()
    main_args =args
    tlog.handlers.clear()
    tags = "_".join(list(get_tag(args["tag"]).values()))
    exp = str(args["expid"]) + "_" + tags 
    exp = exp.strip("-")
    tHandler = logging.FileHandler(getFname(exp + "_time", 
        path=args["save_path"]), mode='w')
    tHandler.setFormatter(FORMAT)
    tlog.addHandler(tHandler)
    tlog.setLevel(logging.INFO)



def save_obj(obj, name, directory, data_dir=True, common=False, is_json=True):
    if obj is None or name.strip() == "":
        logging.info(f"Empty object to save: {name}")
        return
    if not data_dir or name.startswith("chk_def_"):
        folder = directory
    elif common:
        folder = user_data_dir(appname, appauthor) + "/" + directory  
    else:
        folder = user_data_dir(appname, appauthor) + "/profiles/" + profile + "/" + directory
    Path(folder).mkdir(parents=True, exist_ok=True)
    if not is_json:
        fname = os.path.join(folder, name + '.pkl')
        with open(fname, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        fname = os.path.join(folder, name + '.json')
        with open(fname, 'w') as f:
            json.dump(obj, f)


def load_obj(name, directory, default=None, data_dir=True, common =False, is_json=True):
    if not data_dir:
        folder = directory
    elif common:
        folder = user_data_dir(appname, appauthor) + "/" + directory  
    else:
        folder = user_data_dir(appname, appauthor) + "/profiles/" + profile + "/" + directory

    if not is_json:
        fname = os.path.join(folder, name + ".pkl")
        obj_file = Path(fname)
        if not obj_file.is_file():
            return default
        with open(fname, 'rb') as f:
            return pickle.load(f)
    else:
        fname = os.path.join(folder, name + ".json")
        obj_file = Path(fname)
        if not obj_file.is_file():
            return default
        with open(fname, 'rb') as f:
            return json.load(f)


def is_obj(name, directory, common = False, ext=".pkl"):
    if common:
        folder = user_data_dir(appname, appauthor) + "/" + directory  
    else:
        folder = user_data_dir(appname, appauthor) + "/profiles/" + profile + "/" + directory
    if not name.endswith(ext):
        name = name + ext
    fname = os.path.join(folder, name)


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
