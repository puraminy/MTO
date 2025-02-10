from io import BytesIO
from matplotlib.figure import Figure
from matplotlib.transforms import IdentityTransform
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as clr
import attempt.mylogs as mylogs
import json
from PIL import Image, ImageChops


def trim_image(im):
    im = im.convert("RGB")
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def text_to_image(s, *, dpi, xpos=10, ypos=0, **kwargs):
    # To convert a text string to an image, we can:
    # - draw it on an empty and transparent figure;
    # - save the figure to a temporary buffer using ``bbox_inches="tight",
    #   pad_inches=0`` which will pick the correct area to save;
    # - load the buffer using ``plt.imread``.
    #
    # (If desired, one can also directly save the image to the filesystem.)
    fig = Figure(facecolor="none")
    fig.text(xpos, ypos, s, **kwargs)
    with BytesIO() as buf:
        fig.savefig(buf, dpi=dpi, format="png", bbox_inches="tight",
                    pad_inches=0)
        buf.seek(0)
        rgba = plt.imread(buf)
    return rgba

import matplotlib.pyplot as plt
from matplotlib import transforms

def tag_to_image(tags, get_image=False):
    tags = dict(sorted(tags.items()))
    tag_labels = list(tags.keys())
    tag_values = list(tags.values())
    mylogs.bp("image")

    if get_image:
        fig = plt.figure(facecolor="white")
    else:
        fig = plt.figure(facecolor="none")
    fig.set_size_inches(12.5, 3.5)
    xpos = 0
    ypos = 0
    t = plt.gca().transData
    r = fig.canvas.get_renderer()
    plt.axis('off')
    color_list = ["blue","green","red","orange","black", "brown"]
    for i, (label, value) in enumerate(tags.items()):
        text = plt.text(xpos, ypos, label + ": ", # transform=t, 
                color=color_list[i % len(color_list)], 
                fontsize=16)
        ex = text.get_window_extent(renderer=r)
        t = transforms.offset_copy(text._transform, x=ex.width, units='dots')
        text = plt.text(xpos, ypos, value, transform=t, 
                color=color_list[i % len(color_list)], 
                weight="bold",fontsize=14)
        ypos += 0.08
    buf = BytesIO()
    plt.savefig(buf,  format="png")
    if get_image:
        img = Image.open(buf)
        plt.close("all")
        return img 
    else:
        buf.seek(0)
        rgba = plt.imread(buf)
        plt.close("all")
        return rgba

def df_to_image(df, annot=True, title="results"):
    # Set background to white
    tags = mylogs.get_full_tag()
    tag_img = tag_to_image(tags)
    fig, axes = plt.subplot_mosaic("ABB")
    ax1, ax2 = axes["A"], axes["B"]
    ax2.set_title(title)
    fig.set_size_inches(12.5, 6.5)
    ax1.axis("off")
    sns.heatmap(df, ax=ax2, annot=annot, cbar=False)
    fig.figimage(tag_img, 5, 120)
    return fig 
