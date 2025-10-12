import logging
import pickle
from pathlib import Path
from appdirs import *
import os
appname = "mycur"
appauthor = "App"
user = 'na'
profile = "---"

def set_app(name,author="App",_user="na",_profile="---"):
    global appname, appauthor, user, profile
    appname = name
    appauthor = author
    user = _user
    profile = _profile

def save_obj(obj, name, directory, data_dir=True, common=False):
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
    fname = os.path.join(folder, name + '.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, directory, default=None, data_dir=True, common =False):
    if not data_dir:
        folder = directory
    elif common:
        folder = user_data_dir(appname, appauthor) + "/" + directory  
    else:
        folder = user_data_dir(appname, appauthor) + "/profiles/" + profile + "/" + directory

    fname = os.path.join(folder, name + ".pkl")
    obj_file = Path(fname)
    if not obj_file.is_file():
        return default
    with open(fname, 'rb') as f:
        return pickle.load(f)


def is_obj(name, directory, common = False):
    if common:
        folder = user_data_dir(appname, appauthor) + "/" + directory  
    else:
        folder = user_data_dir(appname, appauthor) + "/profiles/" + profile + "/" + directory
    if not name.endswith('.pkl'):
        name = name + '.pkl'
    fname = os.path.join(folder, name)
    obj_file = Path(fname)
    if not obj_file.is_file():
        return False
    else:
        return True
