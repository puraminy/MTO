import glob
import json
import click
import mylogs
import os
from pathlib import Path

@click.command(context_settings=dict(
            ignore_unknown_options=True,
            allow_extra_args=True,))
@click.argument("fname", nargs=-1, type=str)
@click.pass_context
def main(ctx, fname):
    cur_path = os.getcwd()
    for f in fname:
        files = glob.glob(os.path.join(cur_path, "*" + f + "*"))
    inp_kv = ""
    new_kv = ""
    for f in files:
        with open(f) as j:
            d = json.load(j)
        print("-----------------------")
        mylogs.minfo(Path(f).stem)
        print("old kv is:", inp_kv)
        if new_kv == "q":
            break
        if new_kv != "all":
            new_kv = input("key=value:")
        if new_kv and new_kv != "s" and new_kv != "all":
            inp_kv = new_kv
        while True:
            k,v = inp_kv.split("=")
            k = k.strip()
            v = v.strip()
            mylogs.success(k + ":" + (str(d[k]) if k in d else "()") + " ===> " + v)
            if v.lower() == "true": v = True
            elif v.lower() == "false": v = False
            elif v.isnumeric() and "." in v: v = float(v)
            elif v.isnumeric(): v = int(v)
            elif "@" in v: v = [val for val in v.split("@") if val]
            elif v == "none": v = None
            d[k] = v
            print("old kv is:", inp_kv)
            if new_kv != "all":
                new_kv = input("key=value:",)
            if new_kv and new_kv != "s" and new_kv != "all":
                inp_kv = new_kv
            if not new_kv or new_kv == "all" or new_kv == "q":
                break
        with open(f, 'w') as j:
            json.dump(d, j, indent=3)

if __name__ == "__main__":
    main()
