
import sys
import os
import curses as cur
import locale
import sys
import string
import textwrap
import subprocess
import datetime
from .colors import *
from .obj_util import *

pyperclip_imported =False
try:
    import pyperclip 
    pyperclip_imported =True
except:
    pass
#locale.setlocale(locale.LC_ALL, '')
code = "utf-8" #locale.getpreferredencoding()
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


colors = []

if os.name == 'nt':
    import ctypes
    from ctypes import POINTER, WinDLL, Structure, sizeof, byref
    from ctypes.wintypes import BOOL, SHORT, WCHAR, UINT, ULONG, DWORD, HANDLE

    import subprocess
    import msvcrt
    import winsound

    from ctypes import wintypes

def fix_borders():
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    hWnd = kernel32.GetConsoleWindow()
    win32gui.SetWindowLong(hWnd, win32con.GWL_STYLE, 
            win32gui.GetWindowLong(hWnd, win32com.GWL_STYLE) & win32con.WS_MAXIMIZEBOX & win32con.WS_SIZEBOX)


def maximize_console(lines=None):
    if os.name == "nt":
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        hWnd = kernel32.GetConsoleWindow()
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        user32.ShowWindow(hWnd, 1)
        subprocess.check_call('mode.com con cols=124 lines=29')

    #kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    #user32 = ctypes.WinDLL('user32', use_last_error=True)

    #SW_MAXIMIZE = 3

    #kernel32.GetConsoleWindow.restype = wintypes.HWND
    #kernel32.GetLargestConsoleWindowSize.restype = wintypes._COORD
    #kernel32.GetLargestConsoleWindowSize.argtypes = (wintypes.HANDLE,)
    #user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
    #fd = os.open('CONOUT$', os.O_RDWR)
    #try:
    #    hCon = msvcrt.get_osfhandle(fd)
    #    max_size = kernel32.GetLargestConsoleWindowSize(hCon)
    #    if max_size.X == 0 and max_size.Y == 0:
    #        raise ctypes.WinError(ctypes.get_last_error())
    #finally:
    #    os.close(fd)
    #cols = max_size.X
    #hWnd = kernel32.GetConsoleWindow()
    #if cols and hWnd:
    #    if lines is None:
    #        lines = max_size.Y
    #    else:
    #        lines = max(min(lines, 9999), max_size.Y)

def resize_font_on_windows(height, get_size = False):
    LF_FACESIZE = 32
    STD_OUTPUT_HANDLE = -11

    class COORD(Structure):
        _fields_ = [
            ("X", SHORT),
            ("Y", SHORT),
        ]


    class CONSOLE_FONT_INFOEX(Structure):
        _fields_ = [
            ("cbSize", ULONG),
            ("nFont", DWORD),
            ("dwFontSize", COORD),
            ("FontFamily", UINT),
            ("FontWeight", UINT),
            ("FaceName", WCHAR * LF_FACESIZE)
        ]


    kernel32_dll = WinDLL("kernel32.dll")

    get_last_error_func = kernel32_dll.GetLastError
    get_last_error_func.argtypes = []
    get_last_error_func.restype = DWORD


    get_std_handle_func = kernel32_dll.GetStdHandle
    get_std_handle_func.argtypes = [DWORD]
    get_std_handle_func.restype = HANDLE

    get_current_console_font_ex_func = kernel32_dll.GetCurrentConsoleFontEx
    get_current_console_font_ex_func.argtypes = [HANDLE, BOOL, POINTER(CONSOLE_FONT_INFOEX)]
    get_current_console_font_ex_func.restype = BOOL

    set_current_console_font_ex_func = kernel32_dll.SetCurrentConsoleFontEx
    set_current_console_font_ex_func.argtypes = [HANDLE, BOOL, POINTER(CONSOLE_FONT_INFOEX)]
    set_current_console_font_ex_func.restype = BOOL

    # Get stdout handle
    stdout = get_std_handle_func(STD_OUTPUT_HANDLE)
    if not stdout:
        return ("{:s} error: {:d}".format(get_std_handle_func.__name__, get_last_error_func()))
    # Get current font characteristics
    font = CONSOLE_FONT_INFOEX()
    font.cbSize = sizeof(CONSOLE_FONT_INFOEX)
    res = get_current_console_font_ex_func(stdout, False, byref(font))
    if not res:
        return ("{:s} error: {:d}".format(get_current_console_font_ex_func.__name__, get_last_error_func()))
    # Display font information
    for field_name, _ in font._fields_:
        field_data = getattr(font, field_name)
        if field_name == "dwFontSize" and get_size:
            return field_data.Y
    # Alter font height
    font.dwFontSize.X = 10  # Changing X has no effect (at least on my machine)
    font.dwFontSize.Y = height
    # Apply changes
    res = set_current_console_font_ex_func(stdout, False, byref(font))
    if not res:
        return("{:s} error: {:d}".format(set_current_console_font_ex_func.__name__, get_last_error_func()))
    # Get current font characteristics again and display font size
    res = get_current_console_font_ex_func(stdout, False, byref(font))
    if not res:
        return("{:s} error: {:d}".format(get_current_console_font_ex_func.__name__, get_last_error_func()))
    return ""

if os.name == 'nt':
    import msvcrt

    class _CursorInfo(ctypes.Structure):
        _fields_ = [("size", ctypes.c_int),
                    ("visible", ctypes.c_byte)]

DOWN = cur.KEY_DOWN
UP = cur.KEY_UP
LEFT = cur.KEY_LEFT
RIGHT = cur.KEY_RIGHT
SLEFT = cur.KEY_SLEFT
SRIGHT = cur.KEY_SRIGHT
SUP = 337
SDOWN = 336
ARROWS = [UP, DOWN, LEFT, RIGHT, SLEFT, SRIGHT, SUP, SDOWN]

hotkey = ""
old_hotkey = "non-blank"
def get_key(win = None):
    global hotkey, old_hotkey
    #if hotkey == old_hotkey:
    #   hotkey = ""
    #   old_hotkey = "non-blank"
    cur.flushinp()
    cur.flash()
    if hotkey == "":
        ch = win.getch()
    else:
    #    old_hotkey = hotkey
        ch, hotkey = ord(hotkey[0]), hotkey[1:]
    return ch

# Global heatmap constant (color pair numbers)
HEATMAP = [201, 202, 203, 204, 205]

def init_heatmap_colors():
    # Choose foreground color for contrast (white on red/orange, black on yellow/white)
    WHITE = 15 if cur.COLORS >= 16 else cur.COLOR_WHITE
    BLACK = 0 if cur.COLORS >= 16 else cur.COLOR_BLACK

    # Define background color indexes (standard or 256-color mode)
    BG_WHITE = 15 if cur.COLORS >= 16 else cur.COLOR_WHITE
    BG_YELLOW = 11 if cur.COLORS >= 16 else cur.COLOR_YELLOW
    BG_ORANGE = 208  # xterm-256 color for orange
    BG_RED = 9 if cur.COLORS >= 16 else cur.COLOR_RED

    # Initialize color pairs with appropriate foreground/background
    cur.init_pair(HEATMAP[0], WHITE, BG_WHITE)   # white bg, black text
    cur.init_pair(HEATMAP[1], WHITE, BG_YELLOW)  # yellow bg, black text
    cur.init_pair(HEATMAP[2], WHITE, cur.COLOR_GREEN)  # orange bg, white text
    cur.init_pair(HEATMAP[3], WHITE, cur.COLOR_BLUE)     # red bg, white text
    cur.init_pair(HEATMAP[4], WHITE, cur.COLOR_BLACK)     # red bg, white text

def reset_colors(theme, bg=None):
    global back_color, TEXT_COLOR, ITEM_COLOR, SEL_ITEM_COLOR, TITLE_COLOR, DIM_COLOR, color_map
    if bg is None:
        bg = int(theme["back-color"])
    back_color = bg
    for each in range(1, min(256, cur.COLORS)):
        cur.init_pair(each, each, bg)
    TEXT_COLOR = int(theme["text-color"])
    ITEM_COLOR = int(theme["item-color"]) 
    TITLE_COLOR = int(theme["title-color"])
    DIM_COLOR =  int(theme["dim-color"]) 
    init_heatmap_colors()
    reset_hl(theme)
    cur.init_pair(CUR_ITEM_COLOR, bg, int(theme["cur-item-color"]) % cur.COLORS)
    cur.init_pair(SEL_ITEM_COLOR, bg, int(theme["sel-item-color"]) % cur.COLORS)
    cur.init_pair(INPUT_COLOR, TEXT_COLOR, int(theme["input-color"]) % cur.COLORS)
    cur.init_pair(ERR_COLOR, cW, cR % cur.COLORS)
    cur.init_pair(MSG_COLOR, 232, 30 % cur.COLORS)
    #cur.init_pair(INFO_COLOR, 235, cG % cur.COLORS)
    cur.init_pair(SEL_COLOR, 253, 24 % cur.COLORS)
    cur.init_pair(INFO_COLOR, 246, 235 % cur.COLORS)
    cur.init_pair(WARNING_COLOR, cW, cO % cur.COLORS)

def reset_hl(theme):
    global HL_COLOR
    if theme["inverse-highlight"] == "True":
        cur.init_pair(HL_COLOR, int(theme["hl-text-color"]) % cur.COLORS,
                      int(theme["highlight-color"]) % cur.COLORS)
    else:
        cur.init_pair(HL_COLOR, int(theme["highlight-color"]) % cur.COLORS,
                      int(theme["hl-text-color"]) % cur.COLORS)
def hide_cursor(useCur = True):
    if useCur:
        cur.curs_set(0)
    elif os.name == 'nt':
        ci = _CursorInfo()
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        ctypes.windll.kernel32.GetConsoleCursorInfo(handle, ctypes.byref(ci))
        ci.visible = False
        ctypes.windll.kernel32.SetConsoleCursorInfo(handle, ctypes.byref(ci))
    elif os.name == 'posix':
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

def show_cursor(useCur = True):
    if useCur:
        cur.curs_set(1)
    elif os.name == 'nt':
        ci = _CursorInfo()
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        ctypes.windll.kernel32.GetConsoleCursorInfo(handle, ctypes.byref(ci))
        ci.visible = True
        ctypes.windll.kernel32.SetConsoleCursorInfo(handle, ctypes.byref(ci))
    elif os.name == 'posix':
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()

def mprint(text, win, color=None, attr = None, end="\n", refresh = False, color_start=0):
    if color is None:
        color = TEXT_COLOR
    m_print(text, win, color, attr, end, refresh, color_start)

def m_print(text, win, color, attr = None, end="\n", refresh = False, color_start=0):
    if win is None:
        print(text, end=end)
    else:
        try:
            color = int(color)
        except:
            color = ERR_COLOR
        c = cur.color_pair(color)
        if attr is not None:
            c = cur.color_pair(color) | attr
        height, width = win.getmaxyx()
        #win.addnstr(text + end, height*width-1, c)
        #text = textwrap.shorten(text, width=height*width-5)
        text = text + end
        if color_start == 0:
            win.addstr((text).encode(code), c)
        else:
            p1 = text[:color_start]
            p2 = text[color_start:]
            win.addstr((p1).encode(code), c)
            win.addstr((p2).encode(code), HL_COLOR)
        if not refresh:
            pass #win.refresh(0,0, 0,0, height -5, width)
        else:
            #win.refresh()
            pass

def print_there(x, y, text, win = None, color=0, attr = None, pad = False):
    if win is not None:
        c = cur.color_pair(color)
        if attr is not None:
            c = cur.color_pair(color) | attr
        height, width = win.getmaxyx()
        #win.addnstr(x, y, text, height*width-1, c)
        _len = (height*width)-x
        win.addstr(x, y, text[:_len].encode(code), c)
        if pad:
            pass #win.refresh(0,0, x,y, height -5, width)
        else:
            pass # win.refresh()
    else:
        sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (x, y, text))
        sys.stdout.flush()

def clear_screen(win = None):
    if win is not None:
        win.erase()
        win.refresh()
    else:
        os.system('clear')
def rinput(win, r, c, prompt_string, default=""):
    show_cursor()
    cur.echo() 
    win.addstr(r, c, prompt_string.encode(code))
    win.refresh()
    input = win.getstr(r, len(prompt_string), 30)
    clear_screen(win)
    hide_cursor()
    try:
        inp = input.decode(code)  
        cur.noecho()
        return inp
    except:
        hide_cursor()
        cur.noecho()
        return default

def valid_date(datestring):
    try:
        datetime.datetime.strptime(datestring, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def m_input(prompt=":", default=""):
    cmd, _ = minput(win_info, 0, 1, prompt, default=default, all_chars=True)
    return cmd

PROMPT_LINE = 0
SINGLE_LINE = 1
MULTI_LINE = 2
def minput(mwin, row, col, prompt_string, exit_on = [], default="", mode = PROMPT_LINE, footer="", color=HL_COLOR, return_on_char = False, all_chars=False):
    multi_line = mode == MULTI_LINE
    #subprocess.call('setxkbmap -layout ir', shell=True)
    if mode > 0:
        mrows, mcols = mwin.getmaxyx()
        mwin.border()
        print_there(row, col, prompt_string, mwin)
        if footer == "":
            footer =  "<Insert> or \: Insert | <ESC>: Close | Shift + Del: Clear | Shift + Left: Delete line"
            footer = textwrap.shorten(footer, mcols)
        if mode == MULTI_LINE:
            print_there(mrows-1, col, footer, mwin)
            win = mwin.derwin(mrows - 2, mcols-2, 1, 1)
        else:
            print_there(mrows-1, col, footer, mwin)
            win = mwin.derwin(mrows - 2, mcols-2, 1, 1)
        win.bkgd(' ', cur.color_pair(color))  # | cur.A_REVERSE)
        mwin.refresh()
    else:
        win = mwin
        attr = cur.A_BOLD
        c = cur.color_pair(color) | attr
        win.addstr(row, col, prompt_string.encode(code), c)
        win.clrtobot()
    rows, cols = win.getmaxyx()
    if not multi_line:
        exit_on = ['\n']
        next_line = "+"
    else:
        next_line = '\n'
        if not exit_on:
            exit_on = ['\t']
    show_cursor()
    cur.noecho() 
    win.keypad(True)
    out = default.split('\n')
    out = list(filter(None, out))
    #out = out[:rows]
    #inp = "\n".join(out) 
    inp = default
    pos = len(inp)
    ch = 0
    rtl = False
    start_line = 0
    max_lines = 40
    row, col = win.getyx()
    start = col
    ret_now = False
    while ch not in exit_on or ret_now:
        if rtl:
            cur.curs_set(0)
        pos = max(pos, 0)
        pos = min(pos, len(inp))
        if multi_line:
            text = inp
            out = []
            temp = text.split("\n")
            for line in temp:
                if  len(line) < cols - 2:
                    out += [line]
                else:
                    out += textwrap.wrap(line, width = cols -2, break_long_words=False, replace_whitespace=False, drop_whitespace=False)
            #out = filter(None, out)
            if not out:
                out = [""]
            r = row
            if len(out) > 1:
                pass
            c = 0
            yloc = r
            xloc = pos - c
            for i, l in enumerate(out):
                enters = inp.count("\n", c, c + len(l) + 1)
                if pos >= c and pos <= c + len(l):
                    yloc = r
                    xloc = pos - c 
                r += 1
                c += len(l) + enters 
            start_line = max(0, yloc - rows + 1) 
            for ii,l in enumerate(out[start_line:]):
                if ii < rows:
                   if rtl and False:
                       start = cols - len(l)-2
                   win.addstr(ii, start, l.encode(code))
                win.clrtoeol()
            win.clrtobot()
        else:
            win.addstr(row, start, inp.encode(code))
            win.clrtoeol()
            xloc = start + pos
            yloc = row + (xloc // cols)
            xloc = xloc % cols
        if yloc < rows:
            win.move(yloc, xloc)
        else:
            win.move(rows -1 , xloc)
        ch = win.get_wch()
        if type(ch) == str and ch == '٫':
            ch = '#'
        if (type(ch) == str and ord(ch) == 127) or ch == 8 or ch == 127 or ch == cur.KEY_BACKSPACE:
            if pos > 0:
                inp = inp[:pos-1] + inp[pos:]
                pos -= 1
            else:
                mbeep()
        elif ch == cur.KEY_DC:
            if pos < len(inp):
                inp = inp[:pos] + inp[pos+1:]
            else:
                mbeep()
        elif ch == cur.KEY_SDC:
            inp = ""
            pos = 0
        elif not all_chars and (ch == '=' or ch == "|"):
            mbeep()
            #if pyperclip_imported:
            #    pyperclip.copy(inp)
            break
        elif ch == cur.KEY_SLEFT:
            if len(inp) > 0:
                temp = inp.split("\n")
                c = 0
                inp = ""
                for line in temp:
                    if pos >= c and pos < c + len(line):
                        pos -= min(xloc, len(line) - 1)
                    else:
                        inp += line + "\n"
                    c += len(line)
                inp = inp.strip("\n")
        elif  ch == next_line:
            if multi_line and type(ch) == str:
                enters = inp.count("\n")
                if enters < max_lines - 1:
                    inp = inp[:pos] + "\n" + inp[pos:]
                    pos += 1
                else:
                    mbeep()
                    print_there(mrows-1, col, f"You reached the maximum number of {max_lines} lines", mwin, color=WARNING_COLOR)
                    mwin.clrtoeol()
                    mwin.get_wch()
                    print_there(mrows-1, col, footer, mwin)
                    mwin.refresh()
        elif ch == cur.KEY_HOME:
            pos = 0
        elif ch == cur.KEY_END:
            pos = len(inp)
        elif ch == cur.KEY_LEFT:
            if pos > 0:
                pos -= 1 
            else:
                mbeep()
        elif ch == cur.KEY_RIGHT:
            pos += 1
        elif ch == cur.KEY_UP: 
            if not multi_line:
                break
            if yloc < 1:
                mbeep()
            else:
                pos = 0
                for i in range(0, yloc - 1):
                    pos += len(out[i])
                enters = inp.count("\n", 0, pos + yloc - 1)
                pos += enters
                pos += min(xloc, len(out[yloc - 1]))
        elif ch == cur.KEY_DOWN:
            if not multi_line:
                break
            if yloc >= max_lines - 1 or yloc >= len(out) - 1:
                mbeep()
            else:
                pos = 0
                for i in range(0, yloc + 1):
                    pos += len(out[i])
                enters = inp.count("\n", 0, pos + yloc + 1)
                pos += enters
                pos += min(xloc, len(out[yloc + 1]))
        elif type(ch) == str and len(ch) == 1 and  ord(ch) == 27:
            hide_cursor()
            cur.noecho()
            return "<ESC>",ch
        elif ch == cur.KEY_IC or ch == "\\":
            #if pyperclip_imported:
            #    pyperclip.copy(inp)
            break
        else:
            letter = ch
            if ch in exit_on:
                break
            else:
                inp = inp[:pos] + str(letter) + inp[pos:]
                pos += 1
                if return_on_char:
                    break
    cur.noecho()
    hide_cursor()
    #subprocess.call('setxkbmap -layout us', shell=True)
    return inp, ch

def mbeep(repeat=1):
    if os.name == "nt":
        winsound.Beep(500, 100)
    else:
        cur.beep()

# -*- coding: utf-8 -*-

def confirm_all(msg):
    return confirm(msg, acc=['y', 'n', 'a'])


def confirm(msg, acc=['y', 'n'], color=WARNING_COLOR, list_opts=True, bottom = True):
    mbeep()
    if list_opts:
        msg = msg + " (" + "/".join(acc) + ")"
    if not bottom:
        ch = show_info(msg, color, bottom, title="Confirm", acc=acc)
    else:
        win = show_info(msg, color)
        ch = 0
        while chr(ch).lower() not in acc:
            ch = win.getch()
            if not chr(ch).lower() in acc:
                mbeep()
            else:
                break
        ch = chr(ch).lower()
    show_info(old_msg)
    return ch

old_msg = ''
STD_ROWS = 21
STD_COLS = 90

def set_max_rows_cols(rows, cols):
    global STD_ROWS, STD_COLS
    STD_ROWS = rows
    STD_COLS = cols

win_info = None
def show_info(msg, color=INFO_COLOR, bottom=True, title = "Info", acc =[], refresh=True):
    global win_info, old_msg 
    rows, cols = STD_ROWS, STD_COLS
    start_col = 1
    start_row = rows -1
    mrows = 1
    mcols = cols
    if not bottom:
        mcols = 2*cols//3 - 2
        start_col = (cols - mcols) // 2 
        nlines = 0
        for line in msg.splitlines():
            wrap = textwrap.wrap(line,mcols, 
                    break_long_words=False,replace_whitespace=False)
            nlines += len(wrap)
        nlines += 1
        mrows = nlines + 2
        start_row = (rows - mrows) // 2 
    old_msg = msg
    win_info = cur.newwin(mrows, mcols, 3, 7)
    win_info.bkgd(' ', cur.color_pair(color))  # | cur.A_REVERSE)
    win_info.border()
    win_info.erase()
    msg = msg.replace("\n","")
    if len(msg) > cols - 15:
        msg = msg[:(cols - 16)] + "..."
    print_there(0, 1, " {} ".format(msg), win_info, color)
    win_info.clrtoeol()
    if refresh:
        win_info.refresh()
    else:
        win_info.noutrefresh()
    return win_info

def show_msg(msg, color=MSG_COLOR, bottom=False, delay=-1):
    temp = old_msg
    mbeep()
    win = show_info(msg, color, bottom)
    if delay > 0:
        win.timeout(delay)
        win.getch()
        win.timeout(-1)
        show_info(temp)
    else:
        win.getch()


def show_warn(msg, color=WARNING_COLOR, bottom=True, stop=True, delay=3000):
    if bottom:
        msg += "; press any key..."
        temp = old_msg
    win = show_info(msg, color, bottom)
    ch = ''
    if bottom and stop:
        if delay > 0:
            win.timeout(delay)
            ch = win.getch()
            win.timeout(-1)
            show_info(temp)
        else:
            ch = win.getch()
    return ch

def show_err(msg, color=ERR_COLOR, bottom=True):
    if bottom:
        msg += "; press any key..."
        temp = old_msg
    win = show_info(msg, color, bottom)
    if bottom:
        win.getch()
        show_info(temp)

def select_box(in_opts, mwin, list_index = 0, ni=0, in_row=False, title="", border = False, in_colors=[], color = INPUT_COLOR, ret_index = False):
    ch = 0
    list_names = in_opts.keys()
    mrows, mcols = mwin.getmaxyx() 
    if in_row:
        footer =  "Enter/number: Insert | q/ESC: Close "
    else:
        footer = "Right: Select, Left:Cancel"
    okay = RIGHT
    cancel = LEFT
    footer = textwrap.shorten(footer, mcols)
    print_there(mrows-1, 1, footer, mwin)
    mwin.refresh()
    if not border:
        win = mwin.derwin(mrows - 2, mcols, 1, 0)
    else:
        win = mwin.derwin(mrows - 2, mcols - 2, 1, 1)
    win.bkgd(' ', cur.color_pair(color))  # | cur.A_REVERSE)
    if not in_opts:
        show_err("No item to list")
        return ni, list_index
    horiz = False
    row_cap = 3 if in_row else 1
    col_cap = 5 
    opt_colors = {}
    if not horiz:
       _cap = col_cap
    while ch != 27 and ch != ord('q'):
        opts = []
        list_index = min(max(0, list_index), len(in_opts) -1)
        for i, k in enumerate(list(in_opts.values())[list_index]):
            new_k = str(i) + ") " + k
            opts.append(new_k)
            if in_colors:
                opt_colors[new_k] = in_colors[k] if k in in_colors else TEXT_COLOR 

        _size = max([len(s) for s in opts]) + 2
        ni = max(ni, 0)
        if ni > len(opts) - 1:
            ni = 0
        win.erase()
        cc = len(in_opts)*8 
        if len(in_opts) > 1:
            for i, st in enumerate(list_names):
                st = " " + st.ljust(8)
                if i == list_index:
                    print_there(0, cc, st, mwin, color)
                else:
                    print_there(0, cc, st, mwin, INFO_COLOR)
                cc -= len(st) + 2



def show_submenu(sub_menu_win, opts, si, in_colors={}, color=None, active_sel=True, search=False, colors=None):
    if color is None:
        color = ITEM_COLOR
    win_rows, win_cols = sub_menu_win.getmaxyx()
    blank = 3 if search else 2
    if len(opts) > win_rows - 1:
        win_rows = min(win_rows - blank, 10)
    start = si - win_rows // 2
    start = max(start, 0)
    if len(opts) > win_rows:
        start = min(start, len(opts) - win_rows)
    if search and start > 0:
        mprint("...", sub_menu_win, color)
    footer = ""
    is_color = in_colors or opts == colors
    c_attr = None
    for vi, v in enumerate(opts[start:start + win_rows]):
        v = str(v)
        v = v.strip().replace("\n","")
        if is_color:
            if opts == colors:
                cc = v
                c_attr = cur.A_REVERSE
                item_w = 8
            else:
                cc = in_colors[v] if in_colors and v in in_colors else TEXT_COLOR
                c_attr = cur.A_REVERSE
                item_w = win_cols - 6
        if start + vi == si:
            sel_v = v
            if len(v) > win_cols:
                footer = v
                v = v[:win_cols - 5] + "..."
            _v = str(v)
            if colors and v in colors: 
                _v += " *"
            if is_color:
                mprint(" {:<{}}".format(str(v),item_w + 4), sub_menu_win, int(cc), attr=c_attr)
            elif active_sel:
                mprint(" {:<8}".format(_v), sub_menu_win, CUR_ITEM_COLOR)
            else:
                mprint(" {:<8}".format(_v), sub_menu_win, SEL_ITEM_COLOR)
        else:
            if len(v) > win_cols:
                v = v[:win_cols - 5] + "..."
            if is_color:
                mprint(" {:<{}}".format(v, item_w), sub_menu_win, int(cc), attr=c_attr)
            else:
                _color = color
                if colors and v in colors: 
                    _color = colors[v]
                mprint(" {:<8}".format(str(v)), sub_menu_win, _color)
    if start + win_rows < len(opts):
        mprint("...", sub_menu_win, color)
    # if footer != "":
    #   mprint(footer, sub_menu_win, cW)

def is_enter(ch):
    return ch == cur.KEY_ENTER or ch == 10 or ch == 13


def open_submenu(sub_menu_win, options, sel, si, title, std):
    ch = 0
    st = ""
    back_st = ""
    prev_si = si
    cancel = False
    temp_msg = old_msg
    is_combo = "type" in options[sel] and options[sel]["type"] == "combo-box"
    sel_range = options[sel]["range"]
    info = "Enter/Right: select | qq/ESC/Left: cancel "    
    if sel == "preset" or is_combo:
        info += " | Del: delete an item"

    show_info(info)
    while not is_enter(ch) and not ch == RIGHT:
        if ch == UP:
            if si == 0:
                si = prev_si
                cancel = True
                break
            else:
                si -= 1
        elif ch == DOWN:
            si += 1
        elif ch == cur.KEY_NPAGE:
            si += 10
        elif ch == cur.KEY_PPAGE:
            si -= 10
        elif ch == cur.KEY_HOME:
            si = 0
        elif ch == cur.KEY_END:
            si = len(sel_range) - 1
        elif ch == LEFT or ch == 27 or (back_st.lower() == "q" and chr(ch) == "q"):
            si = prev_si
            cancel = True
            break
        elif ch == cur.KEY_DC:
            can_delete = False
            if sel == "preset" and len(sel_range) == 1 or sel_range[si] == "default":
                show_warn("You can't delete the default " + title)
                can_delete = False
            elif is_combo:
                if "---"  in sel_range:
                    sep_index = sel_range.index("---")
                    if si > sep_index:
                        show_warn("You can't remove predefined profiles which appear below separate line (---)")
                    can_delete = False
            if can_delete:
                item = sel_range[si]
                _confirm = confirm("Are you sure you want to delete '" + item)
                if _confirm == "y" or _confirm == "a":
                    del_obj(item, title, common = True)
                if item in sel_range:
                    sel_range.remove(item)
                    if si > len(sel_range):
                        si = len(sel_range) 
                    if is_combo and "list-file" in sel_range:
                        save_obj(sel_range["range"], sel_range["list-file"], "", common = True)
        elif ch != 0:
            if ch == 127 or ch == cur.KEY_BACKSPACE:
                if st == "":
                    si = prev_si
                    cancel = True
                    break
                else:
                    st = st[:-1]
                    back_st = back_st[:-1]
                    si, st = find(sel_range, st, "", si)
            if chr(ch).lower() in string.printable:
                back_st += chr(ch)
                si, new_st = find(sel_range, st, chr(ch), si)
                if not is_combo:
                    if st == new_st:
                        mbeep()
                    else:
                        st = new_st
                else:
                    st += chr(ch)
        si = min(si, len(sel_range) - 1)
        si = max(si, 0)
        sub_menu_win.erase()
        searchable = is_combo or len(sel_range) > 8
        colors = None
        if "colors" in options[sel]:
            colors = options[sel]["colors"]
        elif "sels" in options[sel]:
            sels = options[sel]["sels"]
            colors = {}
            for s in sels:
                colors[s] = SEL_ITEM_COLOR
        show_submenu(sub_menu_win, sel_range, si, search=searchable, colors=colors)
        if is_combo: 
            show_cursor()
            mprint("Search or Add:" + st, sub_menu_win, end ="")
        elif len(sel_range) > 8:
            show_cursor()
            mprint("Search:" + st, sub_menu_win, end ="")
        sub_menu_win.refresh()
        cur.flushinp()
        cur.flash()
        ch = get_key(std)

    si = min(si, len(sel_range) - 1)
    si = max(si, 0)
    hide_cursor()
    sub_menu_win.erase()
    show_info(temp_msg)
    return si, cancel, st

def find(list, st, ch, default):
    _find = st + ch
    _find = _find.lower().strip()
    for i, item in enumerate(list):
        item = str(item)
        if item.lower().startswith(_find): #or _find in item.lower() or item.lower() in _find: 
            return i, _find
    for i, item in enumerate(list):
        item = str(item)
        if _find in item.lower(): 
            return i, _find
    return default, st


