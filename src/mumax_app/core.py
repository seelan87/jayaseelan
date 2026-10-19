# MuMax3 OVF → Image | Image Sequence → MP4
# Standalone four-tab application, extracted from the current GUI notebook.
# Python packages: numpy, Pillow, opencv-python, matplotlib, pyvista, vtk. Tkinter must be available locally.

import os, re, sys, math, shutil, struct, tempfile, subprocess, threading, queue, json, time
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, ScalarFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = IMAGE_EXTS
COLOR_PRESETS = (
    "blue,white,red", "red,white,blue", "black,gray,white",
    "white,gray,black", "blue,white,green", "green,white,red",
    "blue,cyan,yellow", "black,red,yellow", "black,blue,cyan",
    "magenta,white,green", "cyan,white,magenta", "blue,green,red",
)
VALID_COLORS = {"white", "black", "gray", "transparent", "red", "green", "blue",
                "lightred", "lightgreen", "lightblue", "yellow", "darkyellow",
                "cyan", "darkcyan", "magenta", "darkmagenta"}

MUMAX_RGB = {'white': (255,255,255), 'black': (0,0,0), 'gray': (128,128,128),
                   'transparent': (255,255,255), 'red': (255,0,0), 'green': (0,255,0),
                   'blue': (0,0,255), 'lightred': (255,127,127), 'lightgreen': (127,255,127),
                   'lightblue': (127,127,255), 'yellow': (255,255,0), 'darkyellow': (127,127,0),
                   'cyan': (0,255,255), 'darkcyan': (0,127,127), 'magenta': (255,0,255), 'darkmagenta': (127,0,127)}

class OVFReadError(ValueError):
    """A file-specific converter/read failure that may be skipped."""
RESOLUTIONS = {"360p": (640, 360), "480p": (854, 480), "720p": (1280, 720), "1080p": (1920, 1080)}

class ConversionCancelled(Exception):
    def __init__(self, output_dir=None):
        super().__init__("Conversion stopped by user.")
        self.output_dir = output_dir

def check_cancel(event, output_dir=None):
    if event is not None and event.is_set():
        raise ConversionCancelled(output_dir)

def run_converter(command, cwd, cancel_event=None):
    check_cancel(cancel_event)
    if cancel_event is None:
        return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        while True:
            check_cancel(cancel_event)
            try:
                stdout, stderr = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pass
        check_cancel(cancel_event)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()

def select_frame_range(files, frame_range=None):
    if frame_range is None:
        return files
    start, end = frame_range
    if not isinstance(start, int) or not isinstance(end, int) or not 1 <= start <= end <= len(files):
        raise ValueError(f"Frame range must satisfy 1 <= start <= end <= {len(files)} (inclusive).")
    return files[start-1:end]

def natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", Path(path).name)]

def open_folder(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        messagebox.showwarning("Folder", f"Conversion finished, but the folder could not be opened automatically:\n{exc}")

def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or "output"

def auto_export_folder(parent, tab, source, kind):
    """Create a unique export bundle without replacing an existing directory."""
    from datetime import timezone, timedelta
    parent = Path(parent).resolve()
    if not parent.is_dir():
        raise ValueError('Choose an existing output folder.')
    stamp = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%Y%m%d_%H%M%S_%f')
    source_name = safe_name(Path(source).parent.name)[:32]
    name = f'MuMax_Tab{tab}_{kind}_{source_name}_{stamp}'
    for number in range(1000):
        folder = parent/(name if number == 0 else name+f'_{number}')
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            continue
    raise OSError('Could not create a unique export folder.')


def auto_export_filename(source, view, component, factor, extension, first=None, last=None, fps=None):
    stem = safe_name(Path(source).stem)[:48]
    parts = [stem, safe_name(view), safe_name(component), f'{int(factor)}x']
    if first is not None:
        parts.append(f'frame{first+1:06d}' if last is None else f'frames{first+1:06d}-{last+1:06d}')
    if fps is not None:
        parts.append(f'{fps:g}fps')
    return '_'.join(parts)+'.'+extension


def discard_empty_export_folder(folder):
    """Only remove the newly created, empty bundle after a failed/cancelled export."""
    if folder is not None:
        try:
            Path(folder).rmdir()
        except OSError:
            pass


_MUMAX_HEADER_CACHE = {}


def figure_header_tiles(lines, size, scale=1):
    key = (tuple(lines), size, scale)
    if key in _MUMAX_HEADER_CACHE:
        return _MUMAX_HEADER_CACHE[key]
    tiles = []
    for index, line in enumerate(lines):
        tiles.append(component_title_image(line, annotation_font(round((size+(3 if index == 0 else 0))*scale))))
    if len(_MUMAX_HEADER_CACHE) >= 64:
        _MUMAX_HEADER_CACHE.clear()
    _MUMAX_HEADER_CACHE[key] = tiles
    return tiles


def figure_header_height(lines, size, scale=1):
    tiles = figure_header_tiles(lines, size, scale)
    return sum(tile.height for tile in tiles)+round((16+6*max(0, len(tiles)-1))*scale)


def draw_figure_header(image, lines, size, scale=1):
    """Center complete glyph images so subscripts cannot be clipped by a baseline."""
    height = figure_header_height(lines, size, scale)
    ImageDraw.Draw(image).rectangle((0, 0, image.width, height-1), fill='white')
    y = round(8*scale)
    for tile in figure_header_tiles(lines, size, scale):
        image.paste(tile, (round((image.width-tile.width)/2), y), tile)
        y += tile.height+round(6*scale)
    return image


def parse_ovf_vectors(path):
    """Read common OVF 2 text, Binary 4, or Binary 8 data blocks into (nz, ny, nx, 3)."""
    raw = Path(path).read_bytes()
    header_text = raw[:min(len(raw), 200000)].decode("latin1", errors="ignore")
    def get_int(key, default=None):
        m = re.search(r"^\s*#\s*" + re.escape(key) + r"\s*:\s*(\d+)", header_text, re.I | re.M)
        return int(m.group(1)) if m else default
    nx, ny, nz = get_int("xnodes"), get_int("ynodes"), get_int("znodes")
    if not all(v and v > 0 for v in (nx, ny, nz)):
        raise ValueError("Could not read xnodes/ynodes/znodes from the OVF header.")
    if get_int("valuedim", 3) != 3:
        raise ValueError("Arrow overlays require a three-component vector OVF.")
    mesh = re.search(r"^\s*#\s*meshtype\s*:\s*(\w+)", header_text, re.I | re.M)
    if mesh and mesh.group(1).lower() != "rectangular":
        raise ValueError("Only rectangular OVF meshes are supported for arrows.")
    count = nx * ny * nz * 3
    m4 = re.search(rb"# Begin:\s*Data Binary 4\s*\r?\n", raw, re.I)
    m8 = re.search(rb"# Begin:\s*Data Binary 8\s*\r?\n", raw, re.I)
    mt = re.search(rb"# Begin:\s*Data Text\s*\r?\n", raw, re.I)
    if m4 or m8:
        match = m4 or m8
        width = 4 if m4 else 8
        check_value = 1234567.0 if m4 else 123456789012345.0
        start = match.end()
        if len(raw) < start + width * (count + 1):
            raise ValueError("OVF binary data block is incomplete.")
        endian = next((e for e in ("<", ">")
                       if struct.unpack(e + ("f" if m4 else "d"), raw[start:start+width])[0] == check_value), None)
        if endian is None:
            raise ValueError("Invalid OVF binary check value.")
        start += width
        arr = np.frombuffer(raw, dtype=endian + f"f{width}", count=count, offset=start)
        trailer = raw[start + count * width:]
        if not re.match(rb"\s*#\s*End:\s*Data Binary " + str(width).encode(), trailer, re.I):
            raise ValueError("OVF binary data size does not match its header.")
        arr = arr.astype(np.float32, copy=False)
    elif mt:
        start = mt.end()
        end_match = re.search(rb"# End:\s*Data Text", raw[start:], re.I)
        if end_match is None:
            raise ValueError("OVF text data block has no end marker.")
        block = raw[start:start + end_match.start()].decode("ascii")
        values = " ".join(line.split("#", 1)[0] for line in block.splitlines())
        vals = np.array(values.split(), dtype=np.float32)
        if vals.size != count:
            raise ValueError("OVF text data size does not match its header.")
        arr = vals
    else:
        raise ValueError("Unsupported OVF data block. Expected OVF Text, Binary 4, or Binary 8.")
    if not np.isfinite(arr).all():
        raise ValueError("OVF vector data contains non-finite values.")
    # OVF stores x fastest, then y, then z, with x/y/z components interleaved.
    return arr.reshape((nz, ny, nx, 3))

def read_ovf_time_ns(path):
    """Read simulation time from a MuMax/OVF header; never infer it from video FPS."""
    with Path(path).open('rb') as stream:
        header = stream.read(65536).split(b'# Begin: Data', 1)[0].decode('latin1')
    number = r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)'
    match = re.search(r'(?:total\s+simulation\s+)?time\s*(?:\((ns|ps|us|ms|s)\))?\s*:\s*'
                      + number + r'\s*(ns|ps|us|ms|s)?\b', header, re.I)
    if not match:
        return None
    unit = (match.group(3) or match.group(1) or 's').lower()
    value = float(match.group(2)) * {'s': 1e9, 'ms': 1e6, 'us': 1e3, 'ns': 1, 'ps': 1e-3}[unit]
    return value if math.isfinite(value) else None


def annotation_font(size, font_path=''):
    candidates = [font_path] if font_path else []
    candidates += [str(Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / 'segoeui.ttf'),
                   'DejaVuSans.ttf']
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def scientific_runs(text, font):
    """Render component suffixes as real lowered, smaller glyphs, including z."""
    small = annotation_font(max(1, round(font.size*0.7)), getattr(font, 'path', ''))
    runs, previous = [], 0
    for match in re.finditer(r'\bm_?([xyz])\b', text):
        if match.start() > previous:
            runs.append((text[previous:match.start()], font, 0))
        runs.extend([('m', font, 0), (match.group(1), small, max(1, round(font.size*0.22)))])
        previous = match.end()
    if previous < len(text):
        runs.append((text[previous:], font, 0))
    return runs


def scientific_width(text, font):
    return sum(float(f.getlength(value)) for value,f,shift in scientific_runs(text,font))


def draw_scientific_text(draw, xy, text, font, fill='black'):
    x, baseline = xy
    for value, run_font, shift in scientific_runs(text,font):
        draw.text((x, baseline+shift), value, font=run_font, fill=fill, anchor='ls')
        x += run_font.getlength(value)


def component_title_image(component, font):
    ascent, descent = font.getmetrics()
    result = Image.new('RGBA', (max(1, math.ceil(scientific_width(component,font))+2),
                              ascent+descent+max(1, round(font.size*.22))+2), (0,0,0,0))
    draw_scientific_text(ImageDraw.Draw(result), (0,ascent), component, font)
    bbox = result.getbbox()
    return result.crop(bbox) if bbox else result


def layout_annotation_lines(text, font, available):
    """Keep explicit paragraph breaks; wrap at words and split overlong words."""
    result = []
    for paragraph in text.split('\n'):
        current = ''
        for token in re.findall(r'\S+|[^\S\n]+', paragraph):
            if current and not token.isspace() and scientific_width(current+token,font) > available:
                result.append((current.rstrip(), False))
                current = ''
            for char in token:
                if current and scientific_width(current+char,font) > available:
                    result.append((current.rstrip(), False))
                    current = ''
                if current or not char.isspace():
                    current += char
        result.append((current.rstrip(), True))
    return result


def annotation_text_panel(image, text, location, font, pad, line_spacing=1.0, alignment='Align left'):
    """Reserve margins and lay out text with component subscripts and paragraph alignment."""
    spacing = float(line_spacing)
    if spacing not in (1.0, 1.15, 1.5, 2.0):
        raise ValueError('Choose line spacing 1, 1.15, 1.5, or 2.')
    if alignment not in ('Align left', 'Center', 'Align right', 'Justified'):
        raise ValueError('Choose a supported text alignment.')
    available = max(image.width, font.size*8)-2*pad
    lines = layout_annotation_lines(text, font, available)
    widths = [scientific_width(line,font) for line,last in lines]
    content_width = math.ceil(max(widths,default=0))
    if alignment == 'Justified' and any(not last for line,last in lines):
        content_width = max(content_width,available)
    ascent, descent = font.getmetrics()
    natural_height = ascent+max(descent, round(font.size*.22)+1)
    advance = math.ceil(natural_height*spacing)
    width = max(1,content_width+2*pad)
    height = max(1,(len(lines)-1)*advance+natural_height+2*pad)
    if location in {'Left','Right'}:
        canvas = Image.new('RGB',(image.width+width,max(image.height,height)),'white')
        canvas.paste(image,(width if location == 'Left' else 0,(canvas.height-image.height)//2))
        x = pad if location == 'Left' else image.width+pad
        y = (canvas.height-height)//2+pad
    else:
        canvas = Image.new('RGB',(max(image.width,width),image.height+height),'white')
        bottom = location.startswith('Bottom')
        canvas.paste(image,((canvas.width-image.width)//2,0 if bottom else height))
        x = pad if location.endswith('left') else (canvas.width-width)//2+pad
        if location.endswith('right'):
            x = canvas.width-width+pad
        y = image.height+pad if bottom else pad
    draw = ImageDraw.Draw(canvas)
    for index, ((line,last),line_width) in enumerate(zip(lines,widths)):
        baseline = y+ascent+index*advance
        offset = (content_width-line_width)/2 if alignment == 'Center' else content_width-line_width if alignment == 'Align right' else 0
        words = line.split()
        if alignment == 'Justified' and not last and len(words) > 1:
            gap = (content_width-sum(scientific_width(w,font) for w in words))/(len(words)-1)
            cursor = x
            for word in words:
                draw_scientific_text(draw,(cursor,baseline),word,font)
                cursor += scientific_width(word,font)+gap
        else:
            draw_scientific_text(draw,(x+offset,baseline),line,font)
    return canvas


def format_colorbar_value(value, decimals=False):
    return f"{value:.3f}" if decimals else str(round(value))

def decorate_image(image, options=None, *, colors='blue,white,red', component='m', time_ns=None, scale=1):
    opts = options or {}
    if not any(opts.get(k) for k in ('colorbar', 'legend', 'display_time')):
        return image
    image = image.convert('RGB')
    size = max(6, int(opts.get('font_size', 8) * scale))
    font = annotation_font(size, opts.get('font_path', ''))
    pad = max(3, size//3)
    if opts.get('colorbar'):
        low, high = float(opts.get('vmin', -1)), float(opts.get('vmax', 1))
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError('Color-bar minimum must be less than maximum; both must be finite.')
        cmap = stream_color_map(colors)
        style = opts.get("colorbar_size", "Small")
        font_factor, height_factor = {"Small": (0.65, 0.45), "Medium": (0.82, 0.7), "Bigger": (1.0, 1.0)}.get(style, (0.65, 0.45))
        # Fit title/ticks to the original OVF raster height, excluding annotations.
        size = max(1, min(round(size * font_factor), max(1, image.height//7)))
        font = annotation_font(size, opts.get("font_path", ""))
        pad = max(1, min(max(1, size//3), max(1, image.height//20)))
        height = image.height
        bar_width = max(6, round(size*0.8))
        labels = [format_colorbar_value(value, opts.get("colorbar_decimals", False))
                  for value in (high, (low+high)/2, low)]
        draw = ImageDraw.Draw(image)
        title = component_title_image(component, font)
        band_width = math.ceil(max([draw.textlength(s,font=font) for s in labels]+[title.width])) + bar_width + 5*pad
        canvas = Image.new('RGB', (image.width+band_width, height), 'white')
        canvas.paste(image, (0, 0))
        x = image.width+pad
        title_height = title.height
        top_limit = min(height-1, 2*pad+title_height)
        available = max(1, height-top_limit-pad)
        bar_height = min(available, max(size*3, round(available*height_factor)))
        position = opts.get("colorbar_position", "Middle")
        if position == "Top":
            y = top_limit
        elif position == "Bottom":
            y = height-pad-bar_height
        else:
            y = max(top_limit, (height-bar_height)//2)
        y = max(0, min(y, height-bar_height))
        rgb = cmap(np.linspace(1, 0, bar_height))[:, :3]
        gradient = np.repeat(np.uint8(np.rint(rgb*255))[:,None,:], bar_width, axis=1)
        canvas.paste(Image.fromarray(gradient), (x,y))
        draw = ImageDraw.Draw(canvas)
        canvas.paste(title,(x,max(0,y-title_height-pad)),title)
        draw.rectangle((x,y,x+bar_width-1,y+bar_height-1), outline='black', width=max(1, min(scale, bar_width//3)))
        for index, label in enumerate(labels):
            box = font.getbbox(label)
            label_height = max(1, box[3]-box[1])
            yy = (y, y+(bar_height-label_height)//2, y+bar_height-label_height)[index]
            yy = max(0, min(yy, height-label_height))
            draw.text((x+bar_width+pad-box[0], yy-box[1]), label, font=font, fill='black')
        image = canvas
    # Legend/time keep their own text size, independent of color-bar size.
    size = max(6, int(opts.get('font_size', 8) * scale))
    font = annotation_font(size, opts.get('font_path', ''))
    pad = max(3, size//3)
    if opts.get('legend') and opts.get('legend_text', ''):
        image = annotation_text_panel(image, opts.get('legend_text', ''),
                                      opts.get('legend_location', 'Top center'), font, pad,
                                      line_spacing=opts.get('line_spacing',1.0), alignment=opts.get('text_alignment','Align left'))
    if opts.get('display_time'):
        text = f't = {time_ns:.3f} ns' if time_ns is not None else 't = unavailable (OVF time missing)'
        image = annotation_text_panel(image, text, opts.get('time_location', 'Top left'), font, pad)
    return image


def load_annotation_manifest(folder):
    path = Path(folder) / 'ovf_annotations.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {}


def solid_arrow_polygon_2d(tail,tip,shaft_half,head_length,head_half):
    """A single outline: the shaft stops at the head base; only the tip reaches the endpoint."""
    x1,y1=tail;x2,y2=tip
    length=math.hypot(x2-x1,y2-y1)
    if length<1e-12:return []
    ux,uy=(x2-x1)/length,(y2-y1)/length
    head_length=min(max(0.,head_length),length*.7)
    head_half=max(head_half,shaft_half*1.6)
    half=min(shaft_half,head_half*.6)
    neck_x,neck_y=x2-ux*head_length,y2-uy*head_length
    px,py=-uy,ux
    return [(x1+px*half,y1+py*half),(neck_x+px*half,neck_y+py*half),
            (neck_x+px*head_half,neck_y+py*head_half),(x2,y2),
            (neck_x-px*head_half,neck_y-py*head_half),(neck_x-px*half,neck_y-py*half),
            (x1-px*half,y1-py*half)]


def overlay_arrows(image, vectors, density=20, length_scale=1.0, head_size=8, mode="Normalized", cancel_event=None, sample_indices=None, opacity=1.):
    """Overlay solid arrows from in-plane mx/my vectors; density is approximate arrows across x."""
    if mode not in {"Magnitude", "Normalized"}:
        raise ValueError("Choose Magnitude or Normalized arrow mode.")
    img = image.convert("RGB")
    # Render only the arrow layer at higher resolution, then downsample it.
    # Bound temporary memory for large exports.
    aa = max(1, min(4, int(math.sqrt(16_000_000 / (img.width * img.height)))))
    layer = Image.new("RGBA", (img.width * aa, img.height * aa), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    def polygon(points):
        draw.polygon([(x * aa, y * aa) for x, y in points], fill=(15, 15, 15, 255))
    nz, ny, nx, _ = vectors.shape
    # Match the first z layer explicitly selected in mumax3-convert.
    v = vectors[0]
    stride = max(1, int(round(nx / max(1, density))))
    xs = list(range(stride // 2, nx, stride))
    # Choose approximately the same physical sampling interval in x/y.
    stride_y = max(1, int(round(stride * (image.width / nx) / (image.height / ny))))
    ys = list(range(stride_y // 2, ny, stride_y))
    if sample_indices is not None:
        xs,ys = sample_indices
        if len(xs)>1: stride = int(xs[1]-xs[0])
        if len(ys)>1: stride_y = int(ys[1]-ys[0])
    W, H = img.size
    sx, sy = W / nx, H / ny
    maxlen = max(2.0, min(sx * stride, sy * stride_y) * 0.42 * float(length_scale))
    base_head = min(max(1.0, float(head_size)), maxlen * 0.45)
    # Use the actual in-plane magnitude, with no per-image rescaling.
    # Unit in-plane magnetization uses the configured reference size.
    for gy in ys:
        check_cancel(cancel_event)
        for gx in xs:
            mx, my = float(v[gy, gx, 0]), float(v[gy, gx, 1])
            mag = math.hypot(mx, my)
            if not math.isfinite(mag) or mag < 1e-8:
                continue
            factor = mag if mode == "Magnitude" else 1.0
            arrow_length = maxlen * factor
            head = base_head * factor
            ux, uy = mx / mag, -my / mag  # image y-axis points down
            cx, cy = (gx + 0.5) * sx, (ny - gy - 0.5) * sy
            x1, y1 = cx - ux * arrow_length / 2, cy - uy * arrow_length / 2
            x2, y2 = cx + ux * arrow_length / 2, cy + uy * arrow_length / 2
            # A polygon shaft retains fractional widths for smoother small arrows.
            half_width = max(0.3, head / 8)
            back = head * 1.7
            polygon(solid_arrow_polygon_2d((x1,y1),(x2,y2),half_width,
                                          back*math.cos(.55),back*math.sin(.55)))
    if aa > 1:
        layer = layer.resize(img.size, Image.Resampling.LANCZOS)
    if opacity<1:layer.putalpha(layer.getchannel("A").point(lambda v:round(v*max(0.,opacity))))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")

def convert_ovf_folder(input_dir, output_parent, component, fmt, colors, ppi, upscale,
                       arrows, arrow_density, arrow_length, arrow_head, progress=None, *, arrow_mode="Normalized", annotations=None, cancel_event=None, frame_range=None):
    if not str(input_dir).strip() or not str(output_parent).strip():
        raise ValueError("Select both input and output folders.")
    inp, outparent = Path(input_dir).expanduser().resolve(), Path(output_parent).expanduser().resolve()
    if not inp.is_dir() or not outparent.is_dir():
        raise ValueError("Select valid existing input and output folders.")
    if component not in {"mx", "my", "mz", "ALL"} or fmt not in {"JPG", "PNG", "TIFF"}:
        raise ValueError("Invalid component or output format.")
    if ppi <= 0 or upscale not in (1, 2, 4, 8, 10):
        raise ValueError("PPI must be positive and upscaling must be 1, 2, 4, 8, or 10.")
    colors = colors.strip()
    stream_color_map(colors)
    if arrows and (arrow_density < 1 or not all(math.isfinite(v) and v > 0 for v in (arrow_length, arrow_head))):
        raise ValueError("Arrow density, length, and head size must be positive and finite.")
    check_cancel(cancel_event)
    annotations = dict(annotations or {})
    if annotations.get("colorbar"):
        low, high = float(annotations.get("vmin", -1)), float(annotations.get("vmax", 1))
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("Color-bar minimum must be less than maximum.")
    files = field_files(inp)
    files = select_frame_range(files, frame_range)
    if not files:
        raise ValueError("No OVF, OOMMF, or VTK vector-field files found directly inside the selected input folder.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    outdir = outparent / f"{safe_name(inp.name)}_images_{stamp}"
    outdir.mkdir()
    if arrow_mode not in {"Magnitude", "Normalized"}:
        raise ValueError("Choose Magnitude or Normalized arrow mode.")
    comps = {"mx": "x", "my": "y", "mz": "z"}
    comp_list = ["x", "y", "z"] if component == "ALL" else [comps[component]]
    ext = {"JPG": ".jpg", "PNG": ".png", "TIFF": ".tiff"}[fmt]
    produced = 0
    manifest = {}
    skipped = []
    stem_counts = {}
    for path in files:stem_counts[path.stem] = stem_counts.get(path.stem, 0)+1
    repeated_stems = {stem for stem, count in stem_counts.items() if count > 1}
    def record_skip(ovf, error):
        skipped.append(f"{ovf.name}: {error}")
        (outdir / "skipped_ovf_files.txt").write_text("\n\n".join(skipped) + "\n", encoding="utf-8")
    try:
        for idx, ovf in enumerate(files, 1):
            check_cancel(cancel_event, outdir)
            try:
                vectors = parse_ovf_vectors(ovf)
                time_ns = read_ovf_time_ns(ovf)
            except (ValueError, OSError, UnicodeError) as e:
                record_skip(ovf, e)
                if progress:
                    progress(idx, len(files), f"Skipped: {ovf.name}")
                continue
            saved_for_file = []
            try:
                for comp in comp_list:
                    check_cancel(cancel_event, outdir)
                    with tempfile.TemporaryDirectory(prefix="mumax_convert_") as tmp:
                        low, high = float(annotations.get('vmin', -1)), float(annotations.get('vmax', 1))
                        image = scalar_slice_image(vectors[0, ..., 'xyz'.index(comp)], colors, low, high)
                        if upscale != 1:
                            image = image.resize((image.width*upscale, image.height*upscale), Image.Resampling.LANCZOS)
                        if arrows:
                            image = overlay_arrows(image, vectors, arrow_density, arrow_length, arrow_head * upscale, mode=arrow_mode, cancel_event=cancel_event)
                        check_cancel(cancel_event, outdir)
                        image = decorate_image(image, annotations, colors=colors, component=f"m{comp}",
                                               time_ns=time_ns, scale=upscale)
                        tag = f"_m{comp}"
                        source_stem = ovf.stem + ("_"+ovf.suffix[1:].lower() if ovf.stem in repeated_stems else "")
                        outname = f"{safe_name(source_stem)}{tag}{ext}"
                        save_path = outdir / outname
                        check_cancel(cancel_event, outdir)
                        if fmt == "JPG":
                            image.save(save_path, format="JPEG", quality=95, dpi=(ppi, ppi), subsampling=0)
                        elif fmt == "PNG":
                            image.save(save_path, format="PNG", dpi=(ppi, ppi))
                        else:
                            image.save(save_path, format="TIFF", dpi=(ppi, ppi), compression="tiff_lzw")
                        produced += 1
                        saved_for_file.append(save_path)
                        manifest[save_path.name] = {"time_ns": time_ns, "component": f"m{comp}", "colors": colors,
                                                   "rendered": {k: bool(annotations.get(k)) for k in ("colorbar", "legend", "display_time")}}
            except OVFReadError as exc:
                for saved in saved_for_file:
                    saved.unlink()
                    manifest.pop(saved.name, None)
                produced -= len(saved_for_file)
                record_skip(ovf, exc)
                if progress:
                    progress(idx, len(files), f"Skipped: {ovf.name}")
                continue
            if progress:
                progress(idx, len(files), ovf.name)
        check_cancel(cancel_event, outdir)
    except ConversionCancelled as exc:
        exc.output_dir = outdir
        raise
    finally:
        (outdir / "ovf_annotations.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return outdir, produced

def fit_frame(image, target_w, target_h):
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(round(w*scale))), max(1, int(round(h*scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x, y = (target_w-nw)//2, (target_h-nh)//2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas

def prepare_video_frame(p, metadata, annotations=None, colors="blue,white,red", field_settings=None):
    p = Path(p)
    if p.suffix.lower() in FIELD_EXTS:
        image = render_field_image(p, colors=colors, annotations=annotations, **dict(field_settings or {}))
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    frame = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read image: {p.name}")
    info = metadata.get(p.name, {})
    options = dict(annotations or {})
    # Exported decorations already appear in the pixels. Add only missing elements.
    for key, rendered in info.get("rendered", {}).items():
        if rendered:
            options[key] = False
    if any(options.get(k) for k in ("colorbar", "legend", "display_time")):
        time_ns = info.get("time_ns")
        if options.get("display_time") and time_ns is None:
            ovf = p.with_name(re.sub(r"_m[xyz]$", "", p.stem) + ".ovf")
            if ovf.exists():
                try:
                    time_ns = read_ovf_time_ns(ovf)
                except OSError:
                    pass
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil = decorate_image(pil, options, colors=info.get("colors", colors),
                             component=info.get("component", "m"), time_ns=time_ns)
        frame = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return frame


def convert_images_to_video(input_dir, output_parent, fps, resolution, progress=None, *, image_paths=None, output_label=None, annotations=None, colors="blue,white,red", cancel_event=None, frame_range=None, field_settings=None):
    if not str(input_dir).strip() or not str(output_parent).strip():
        raise ValueError("Select both input and output folders.")
    inp, outparent = Path(input_dir).expanduser().resolve(), Path(output_parent).expanduser().resolve()
    if not inp.is_dir() or not outparent.is_dir():
        raise ValueError("Select valid existing input and output folders.")
    if not math.isfinite(fps) or not 0.1 <= fps <= 240:
        raise ValueError("FPS must be from 0.1 to 240.")
    if resolution not in RESOLUTIONS:
        raise ValueError("Choose a supported output resolution.")
    images = sorted(image_paths if image_paths is not None else
                    video_source_files(inp), key=natural_key)
    images = select_frame_range(images, frame_range)
    if not images:
        raise ValueError("No supported images or OVF/OOMMF/VTK vector fields found in the selected input folder.")
    check_cancel(cancel_event)
    metadata = load_annotation_manifest(inp)
    w, h = RESOLUTIONS[resolution]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = safe_name(output_label or inp.name)
    outdir = outparent / f"{stem}_mp4_{stamp}"
    outdir.mkdir()
    outfile = outdir / f"{stem}_{resolution}_{fps:g}fps.mp4"
    writer = cv2.VideoWriter(str(outfile), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w,h))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("Could not initialize MP4 encoder. Reinstall opencv-python or check codec support.")
    try:
        for i, p in enumerate(images, 1):
            check_cancel(cancel_event, outdir)
            frame = prepare_video_frame(p, metadata, annotations, colors, field_settings)
            check_cancel(cancel_event, outdir)
            writer.write(fit_frame(frame, w, h))
            if progress:
                progress(i, len(images), p.name)
        check_cancel(cancel_event, outdir)
    except BaseException:
        writer.release()
        outfile.unlink(missing_ok=True)
        raise
    finally:
        writer.release()
    if not outfile.exists() or outfile.stat().st_size == 0:
        raise RuntimeError("MP4 output was not created.")
    return outdir, outfile, len(images)

class OutputPreview:
    """Independent image/video viewer; preview never changes the output file."""
    def __init__(self, parent, paths=()):
        self.window = tk.Toplevel(parent)
        self.window.title("Output preview")
        self.window.geometry("1000x740")
        self.window.minsize(480, 360)
        self.paths = [Path(p) for p in paths]
        self.capture = None
        self.frame_image = None
        self.photo = None
        self.timer = None
        self.resize_timer = None
        self.playing = False
        self.closed = False
        self.updating_slider = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0
        self.index = tk.IntVar(value=0)
        self.choice = tk.StringVar()
        self.status = tk.StringVar(value="Select an image or MP4 output to preview.")
        top = ttk.Frame(self.window, padding=8)
        top.pack(fill="x")
        ttk.Button(top, text="Previous", command=lambda: self.select_offset(-1)).pack(side="left")
        ttk.Button(top, text="Next", command=lambda: self.select_offset(1)).pack(side="left", padx=4)
        self.file_box = ttk.Combobox(top, textvariable=self.choice, state="readonly")
        self.file_box.pack(side="left", fill="x", expand=True, padx=6)
        self.file_box.bind("<<ComboboxSelected>>", lambda e: self.load(self.file_box.current()))
        ttk.Button(top, text="Open file…", command=self.browse).pack(side="right")
        self.canvas = tk.Canvas(self.window, background="#252525", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self.on_resize)
        controls = ttk.Frame(self.window, padding=8)
        controls.pack(fill="x")
        self.play_button = ttk.Button(controls, text="Play", command=self.toggle_play, state="disabled")
        self.play_button.pack(side="left")
        self.restart_button = ttk.Button(controls, text="Restart", command=self.restart, state="disabled")
        self.restart_button.pack(side="left", padx=6)
        self.position = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(controls, from_=0, to=1, variable=self.position, command=self.seek, state="disabled")
        self.slider.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Label(self.window, textvariable=self.status, padding=8, wraplength=950).pack(fill="x")
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh_choices()
        if self.paths:
            self.load(0)

    def refresh_choices(self):
        self.file_box.configure(values=[f"{i+1}: {p.name}" for i,p in enumerate(self.paths)])

    def browse(self):
        filename = filedialog.askopenfilename(parent=self.window, title="Preview output file",
                    filetypes=[("Images and MP4", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.mp4"), ("All files", "*.*")])
        if filename:
            path = Path(filename)
            if path not in self.paths:
                self.paths.append(path)
                self.refresh_choices()
            self.load(self.paths.index(path))

    def select_offset(self, step):
        if self.paths:
            self.load((self.index.get()+step) % len(self.paths))

    def pause(self):
        self.playing = False
        self.play_button.configure(text="Play")
        if self.timer is not None:
            self.window.after_cancel(self.timer)
            self.timer = None

    def load(self, index):
        self.pause()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.frame_image = None
        self.canvas.delete("all")
        self.play_button.configure(state="disabled")
        self.restart_button.configure(state="disabled")
        self.slider.configure(state="disabled")
        self.total_frames = 0
        self.index.set(index)
        self.file_box.current(index)
        path = self.paths[index]
        self.window.title(f"Output preview — {path.name}")
        try:
            if path.suffix.lower() == '.mp4':
                self.capture = cv2.VideoCapture(str(path))
                if not self.capture.isOpened():
                    raise ValueError("Cannot open MP4. Check that the file exists and is readable.")
                self.total_frames = max(1, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                fps = self.capture.get(cv2.CAP_PROP_FPS)
                self.fps = fps if math.isfinite(fps) and fps > 0 else 25.0
                self.slider.configure(to=max(1, self.total_frames-1), state="normal")
                self.play_button.configure(state="normal")
                self.restart_button.configure(state="normal")
                if not self.read_frame(0):
                    raise ValueError("Cannot decode the first video frame.")
            else:
                with Image.open(path) as image:
                    self.frame_image = image.convert('RGB')
                self.status.set(f"{path.name} | {self.frame_image.width} × {self.frame_image.height} px")
                self.render()
        except (OSError, ValueError, cv2.error) as exc:
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            self.play_button.configure(state="disabled")
            self.restart_button.configure(state="disabled")
            self.slider.configure(state="disabled")
            self.status.set(f"Preview unavailable: {exc}")

    def read_frame(self, index=None):
        if self.capture is None:
            return False
        if index is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok:
            return False
        self.current_frame = max(0, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))-1)
        self.frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.updating_slider = True
        self.position.set(self.current_frame)
        self.updating_slider = False
        self.status.set(f"{self.paths[self.index.get()].name} | Frame {self.current_frame+1}/{self.total_frames} | "
                        f"Playback {self.current_frame/self.fps:.3f} s | {self.fps:g} FPS")
        self.render()
        return True

    def seek(self, value):
        if not self.updating_slider and self.capture is not None:
            self.pause()
            self.read_frame(min(self.total_frames-1, max(0, int(float(value)))))

    def toggle_play(self):
        if self.playing:
            self.pause()
        elif self.capture is not None:
            if self.current_frame >= self.total_frames-1:
                self.read_frame(0)
            self.playing = True
            self.play_button.configure(text="Pause")
            self.timer = self.window.after(max(1, round(1000/self.fps)), self.tick)

    def tick(self):
        self.timer = None
        if self.closed or not self.playing:
            return
        start = time.monotonic()
        if not self.read_frame():
            self.pause()
            return
        if self.current_frame >= self.total_frames-1:
            self.pause()
            return
        delay = max(1, round(1000/self.fps - (time.monotonic()-start)*1000))
        self.timer = self.window.after(delay, self.tick)

    def restart(self):
        self.pause()
        if self.capture is not None:
            self.read_frame(0)

    def on_resize(self, event):
        if self.resize_timer is not None:
            self.window.after_cancel(self.resize_timer)
        self.resize_timer = self.window.after(60, self.render)

    def render(self):
        if self.resize_timer is not None:
            self.window.after_cancel(self.resize_timer)
        self.resize_timer = None
        if self.closed or self.frame_image is None:
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 2 or height < 2:
            return
        image = self.frame_image
        ratio = min(width/image.width, height/image.height)
        size = (max(1, round(image.width*ratio)), max(1, round(image.height*ratio)))
        self.photo = ImageTk.PhotoImage(image.resize(size, Image.Resampling.LANCZOS), master=self.window)
        self.canvas.delete("preview")
        self.canvas.create_image(width//2, height//2, image=self.photo, anchor="center", tags="preview")

    def close(self):
        if self.closed:
            return
        self.pause()
        self.closed = True
        if self.resize_timer is not None:
            self.window.after_cancel(self.resize_timer)
            self.resize_timer = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.window.destroy()


class InputPreview(OutputPreview):
    """Render one selected source frame with live GUI settings, into a temporary cache."""
    def __init__(self, gui, kind):
        self.gui, self.kind = gui, kind
        self.cache = tempfile.TemporaryDirectory(prefix="mumax_settings_preview_")
        self.busy = False
        self.refresh_pending = False
        self.cancel_event = threading.Event()
        super().__init__(gui.root)
        self.window.title("Preview before conversion")
        self.source_frame = tk.StringVar(value="1")
        self.preview_mode = tk.StringVar(value="Image" if kind == "ovf" else "MP4 frame")
        settings = ttk.Frame(self.window, padding=6)
        settings.pack(side="top", before=self.canvas, fill="x")
        ttk.Label(settings, text="Source frame (1-based):").pack(side="left",padx=4)
        self.frame_spin = ttk.Spinbox(settings, from_=1, to=1, textvariable=self.source_frame, width=8)
        self.frame_spin.pack(side="left",padx=4)
        ttk.Combobox(settings, textvariable=self.preview_mode, values=["Image", "MP4 frame"], state="readonly", width=12).pack(side="left",padx=4)
        self.refresh_button = ttk.Button(settings, text="Refresh from current settings", command=self.refresh)
        self.refresh_button.pack(side="left",padx=8)
        self.status.set("Choose a source frame and refresh. No output files are written to your output folder.")
        try:
            _, files = self.source_files()
            self.source_frame.set(str(files[0][0]))
            self.refresh()
        except (ValueError, OSError) as exc:
            self.status.set(str(exc))

    def source_files(self):
        folder_text = (self.gui.ovf_input if self.kind == "ovf" else self.gui.video_input).get().strip()
        folder = Path(folder_text).expanduser()
        if not folder_text or not folder.is_dir():
            raise ValueError("Choose an input folder in the main window first.")
        files = field_files(folder) if self.kind == "ovf" else video_source_files(folder)
        selected = select_frame_range(files, self.gui.snapshot_frame_range())
        if not selected:
            raise ValueError("No input frames found.")
        first = files.index(selected[0])+1
        numbered = list(enumerate(selected, first))
        self.frame_spin.configure(from_=first, to=numbered[-1][0])
        return folder, numbered

    def refresh(self):
        if self.closed:
            return
        if self.busy:
            self.refresh_pending = True
            self.status.set("Rendering preview… refresh with latest settings queued.")
            return
        self.refresh_pending = False
        try:
            folder, files = self.source_files()
            number = int(self.source_frame.get())
            sources = dict(files)
            if number not in sources:
                raise ValueError(f"Choose a frame from {files[0][0]} to {files[-1][0]}.")
            source = sources[number]
            annotations = self.gui.snapshot_annotations()
            colors = self.gui.colors.get()
            mode = self.preview_mode.get()
            resolution = self.gui.resolution.get()
            field_settings = field_render_settings(self.gui) if self.kind == 'video' and source.suffix.lower() in FIELD_EXTS else None
            if self.kind == "ovf":
                image_args = (self.gui.component.get(), self.gui.out_format.get(), colors,
                              int(self.gui.ppi.get()), int(self.gui.upscale.get().replace('x','')),
                              self.gui.arrows.get(), int(self.gui.arrow_density.get()),
                              float(self.gui.arrow_length.get()), float(self.gui.arrow_head.get()))
                arrow_mode = self.gui.arrow_mode.get()
        except (ValueError, OSError) as exc:
            self.status.set(str(exc))
            return
        self.busy = True
        self.refresh_button.configure(state="disabled")
        self.status.set(f"Rendering source frame {number}: {source.name}…")
        def worker():
            try:
                if self.kind == "ovf":
                    rendered, count = convert_ovf_folder(folder, self.cache.name, *image_args, arrow_mode=arrow_mode,
                            annotations=annotations, cancel_event=self.cancel_event, frame_range=(number,number))
                    if not count:
                        report = rendered / "skipped_ovf_files.txt"
                        raise ValueError(report.read_text(encoding="utf-8") if report.exists() else "Source frame is unreadable.")
                    paths = sorted([p for p in rendered.iterdir() if p.suffix.lower() in IMAGE_EXTS], key=natural_key)
                    if mode == "MP4 frame":
                        outputs = []
                        for p in paths:
                            frame = prepare_video_frame(p, {}, {}, colors)
                            preview = rendered / (p.stem + "_mp4_preview.png")
                            rgb = cv2.cvtColor(fit_frame(frame, *RESOLUTIONS[resolution]), cv2.COLOR_BGR2RGB)
                            Image.fromarray(rgb).save(preview)
                            outputs.append(preview)
                        paths = outputs
                else:
                    frame = prepare_video_frame(source, load_annotation_manifest(folder), annotations, colors, field_settings)
                    if mode == "MP4 frame":
                        frame = fit_frame(frame, *RESOLUTIONS[resolution])
                    preview = Path(self.cache.name) / (source.stem + "_preview.png")
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(preview)
                    paths = [preview]
                if not self.closed:
                    self.gui.events.put((self.complete, (paths, None, source.name, number)))
            except Exception as exc:
                if not self.closed:
                    self.gui.events.put((self.complete, ([], str(exc), source.name, number)))
            finally:
                if self.closed:
                    self.cache.cleanup()
        threading.Thread(target=worker, daemon=True).start()

    def complete(self, paths, error, source_name, number):
        self.busy = False
        if self.closed:
            self.cache.cleanup()
            return
        self.refresh_button.configure(state="normal")
        if error:
            self.status.set(f"Preview unavailable: {error}")
            if self.refresh_pending:
                self.refresh()
            return
        self.paths = paths
        self.refresh_choices()
        self.load(0)
        self.window.title(f"Preview before conversion — source frame {number}: {source_name}")
        self.status.set(f"Source frame {number}: {source_name} | {self.preview_mode.get()} preview | Refresh to apply changed settings")
        if self.refresh_pending:
            self.refresh()

    def close(self):
        if self.closed:
            return
        self.cancel_event.set()
        super().close()
        if not self.busy:
            self.cache.cleanup()


def ovf_mesh_coordinates(path, shape):
    with Path(path).open('rb') as stream:
        header = stream.read(65536).split(b'# Begin: Data',1)[0].decode('latin1')
    unit_match = re.search(r'^\s*#\s*meshunit\s*:\s*(\S+)',header,re.M|re.I)
    unit = unit_match.group(1) if unit_match else 'm'
    factor = {'m':1e9,'nm':1,'um':1e3,'µm':1e3}.get(unit,1)
    shown_unit = 'nm' if unit in ('m','nm','um','µm') else unit
    result = {}
    for axis,count in zip('xyz',(shape[2],shape[1],shape[0])):
        def number(key):
            match = re.search(r'^\s*#\s*'+key+r'\s*:\s*([-+0-9.eE]+)',header,re.M|re.I)
            return float(match.group(1)) if match else None
        step,base = number(axis+'stepsize'),number(axis+'base')
        if step is None or base is None or not math.isfinite(step) or step <= 0 or not math.isfinite(base):
            result[axis] = dict(centers=np.arange(count)+.5,edges=(0,count),unit='cell')
        else:
            result[axis] = dict(centers=(base+np.arange(count)*step)*factor,
                               edges=((base-step/2)*factor,(base+(count-.5)*step)*factor),unit=shown_unit)
    return result


def extract_cross_section(vectors, axis, index):
    if axis == 'X':
        if not 0 <= index < vectors.shape[2]:
            raise ValueError('X slice index is outside the mesh.')
        return vectors[:,:,index,:], 'y', (1,2), 0
    if axis == 'Y':
        if not 0 <= index < vectors.shape[1]:
            raise ValueError('Y slice index is outside the mesh.')
        return vectors[:,index,:,:], 'x', (0,2), 1
    if axis == 'Z':
        if not 0 <= index < vectors.shape[0]:
            raise ValueError('Z slice index is outside the mesh.')
        return vectors[index,:,:,:], 'x', (0,1), 2
    raise ValueError('Choose X, Y, or Z slice.')


def scalar_slice_image(values, colors, low=-1, high=1):
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError('Color minimum must be less than maximum.')
    return stream_scalar_rgb(values, colors, low, high)


def cross_section_arrows(image, plane, axis, selection='ALL', mode='Magnitude', density=80, length=2.5, head=1, sample_indices=None, opacity=1.):
    pair,normal = {'X':((1,2),0),'Y':((0,2),1),'Z':((0,1),2)}[axis]
    chosen = {'mx':0,'my':1,'mz':2}.get(selection)
    projected = np.zeros((1,*plane.shape[:2],3),np.float32)
    for target,component in enumerate(pair):
        if selection=='ALL' or chosen==component:
            projected[0,:,:,target] = plane[:,:,component]
    # Out-of-plane dot/cross markers are intentionally omitted.
    return overlay_arrows(image,projected,density,length,head,mode=mode,sample_indices=sample_indices,opacity=opacity)


LAYER_FILL_BASIC_COLORS = {
    'Black':'#000000', 'Dark gray':'#666666', 'Dark red':'#800000',
    'Red':'#ff0000', 'Orange':'#ff8000', 'Yellow':'#ffff00',
    'Green':'#008000', 'Cyan':'#00b7c3', 'Blue':'#0078d7', 'Purple':'#800080',
    'White':'#ffffff', 'Light gray':'#cccccc', 'Brown':'#8b4513',
    'Pink':'#ff80ab', 'Gold':'#ffc000', 'Lime':'#80ff00',
    'Mint':'#80e0c0', 'Sky blue':'#80cfff', 'Indigo':'#4040a0', 'Lilac':'#c080ff'}
LAYER_FILL_PASTEL_COLORS = {
    'Pastel blue':'#8ecae6', 'Pastel red':'#f6a6a6', 'Pastel green':'#a8ddb5',
    'Pastel yellow':'#ffe699', 'Lavender':'#cbb8ef', 'Peach':'#ffd0a8',
    'Soft blue':'#2585e6', 'Soft red':'#e63946', 'Soft green':'#2ca25f', 'Soft yellow':'#ffd43b',
    'Soft orange':'#f8961e', 'Soft purple':'#9b5de5', 'Soft cyan':'#22cbd6', 'Magenta':'#df55b7',
    'White':'#ffffff', 'Gray':'#9e9e9e', 'Charcoal':'#222222'}
LAYER_FILL_COLORS = {**LAYER_FILL_BASIC_COLORS,**LAYER_FILL_PASTEL_COLORS}
LAYER_FILL_PALETTES = {'Paint 3D style':LAYER_FILL_BASIC_COLORS,'Pastels':LAYER_FILL_PASTEL_COLORS}
# Phong finishes shared by the CPU and GPU layer renderers.
LAYER_FILL_FINISHES = {
    'Matte':dict(diffuse=1.,specular=0.,power=8.,metal=False),
    'Gloss':dict(diffuse=.9,specular=.65,power=60.,metal=False),
    'Dull metal':dict(diffuse=.65,specular=.5,power=18.,metal=True),
    'Polished metal':dict(diffuse=.4,specular=.95,power=110.,metal=True),
    'Glass':dict(diffuse=.25,specular=1.,power=120.,metal=False)}


def layer_finish_material(record,ambient):
    finish=LAYER_FILL_FINISHES.get(record.get('finish','Matte'),LAYER_FILL_FINISHES['Matte'])
    reflection=record.get('_reflection')
    if reflection is not None and not reflection.get('enabled'):reflection=None
    return dict(lighting=True,ambient=ambient,diffuse=(1-ambient)*finish['diffuse'],
        specular=reflection_specular(reflection,'layer',ambient) if reflection is not None else (1-ambient)*finish['specular'],
        specular_power=reflection['sharpness'] if reflection is not None else finish['power'])


def apply_layer_finish(actor,record,ambient):
    for name,value in layer_finish_material(record,ambient).items():setattr(actor.prop,name,value)
    from matplotlib.colors import to_rgb
    finish=LAYER_FILL_FINISHES.get(record.get('finish','Matte'),LAYER_FILL_FINISHES['Matte'])
    tint=to_rgb(record['color']) if finish['metal'] and record['mode']=='Solid color' else (1.,1.,1.)
    actor.prop.SetSpecularColor(*tint)


def shade_layer_fill(rgb,directions,record,ambient,normals=None,lighting=None):
    finish=LAYER_FILL_FINISHES.get(record.get('finish','Matte'),LAYER_FILL_FINISHES['Matte'])
    normal=np.zeros_like(directions) if normals is None else np.asarray(normals,float).copy()
    if normals is None:normal[:,2]=np.where(directions[:,2]<0,1.,-1.)
    if lighting is None:
        key=np.array([1.,-1.,2.]);key/=np.linalg.norm(key)
        fill=np.array([-1.,-.5,1.]);fill/=np.linalg.norm(fill);intensity=1.
    else:key,fill=lighting['directions'];intensity=lighting['intensity']
    # Two-sided sheet lighting, matching the glyph camera and light coordinates.
    light=np.broadcast_to(key,directions.shape).copy()
    secondary=np.broadcast_to(fill,directions.shape).copy()
    if normals is None and lighting is None:
        light[:,2]*=normal[:,2];secondary[:,2]*=normal[:,2]
    lambert=.1+.75*np.maximum(np.sum(normal*light,axis=1),0)+.15*np.maximum(np.sum(normal*secondary,axis=1),0)
    half=light-directions;half/=np.maximum(np.linalg.norm(half,axis=1)[:,None],1e-15)
    reflection=record.get('_reflection')
    if reflection is not None and not reflection.get('enabled'):reflection=None
    power=reflection['sharpness'] if reflection is not None else finish['power']
    specular=reflection_specular(reflection,'layer',ambient) if reflection is not None else (1-ambient)*finish['specular']
    shine=np.maximum(np.sum(normal*half,axis=1),0)**power
    tint=.12+.88*rgb if finish['metal'] else np.ones_like(rgb)
    return np.clip(rgb*(ambient+(1-ambient)*finish['diffuse']*lambert[:,None]*intensity)+
        tint*(specular*shine*intensity)[:,None],0,1)


class LayerFillControl(ttk.LabelFrame):
    """Independent settings for original OVF Z layers, expressed in physical units."""
    def __init__(self,parent,changed,status):
        super().__init__(parent,text='Layer Fill',padding=5)
        self.changed,self.status=changed,status
        self.records={};self.source_z=None;self.current=None;self.loading=False;self.busy=False
        self.enabled=tk.BooleanVar(master=self,value=False)
        self.merge=tk.BooleanVar(master=self,value=False);self.merge_style=None;self.merge_sets={}
        self.merge_set=tk.StringVar(master=self,value='Set 1')
        self.merge_layers=tk.StringVar(master=self,value='')
        self.mode=tk.StringVar(master=self,value='Solid color')
        self.preset=tk.StringVar(master=self,value='Pastel blue')
        self.color=tk.StringVar(master=self,value=LAYER_FILL_COLORS['Pastel blue'])
        self.colors=tk.StringVar(master=self,value='Use tab color map')
        self.finish=tk.StringVar(master=self,value='Matte')
        self.palette=tk.StringVar(master=self,value='Paint 3D style')
        self.transparency=tk.StringVar(master=self,value='70');self.alpha_slider=tk.DoubleVar(master=self,value=70)
        ttk.Checkbutton(self,text='Enable color fill',variable=self.enabled,command=self.master_changed).pack(anchor='w')
        self.body=ttk.Frame(self)
        ttk.Label(self.body,text='Check layers; select a row to edit it.').pack(anchor='w')
        ttk.Checkbutton(self.body,text='Merge Layer',variable=self.merge,command=self.merge_changed).pack(anchor='w',pady=(2,0))
        self.merge_row=ttk.Frame(self.body)
        self.merge_choice=ttk.Combobox(self.merge_row,textvariable=self.merge_set,values=('Set 1',),state='readonly')
        self.merge_choice.pack(fill='x');self.merge_choice.bind('<<ComboboxSelected>>',self.merge_set_changed)
        row=ttk.Frame(self.merge_row);row.pack(fill='x',pady=2)
        ttk.Button(row,text='New set',command=self.new_merge_set).pack(side='left',fill='x',expand=True)
        ttk.Button(row,text='Remove set',command=self.remove_merge_set).pack(side='left',fill='x',expand=True,padx=(3,0))
        row=ttk.Frame(self.merge_row);row.pack(fill='x',pady=2)
        ttk.Label(row,text='Layers').pack(side='left')
        entry=ttk.Entry(row,textvariable=self.merge_layers,width=12);entry.pack(side='left',fill='x',expand=True,padx=3)
        entry.bind('<Return>',self.set_merge_range)
        ttk.Button(row,text='Set layers',command=self.set_merge_range).pack(side='right')
        frame=self.tree_frame=ttk.Frame(self.body);frame.pack(fill='x',pady=3)
        self.tree=ttk.Treeview(frame,columns=('use','layer','set','z','alpha'),displaycolumns=('use','layer','z','alpha'),show='headings',height=5,selectmode='browse')
        for key,label,width in (('use','Fill',33),('layer','Layer',42),('set','Set',48),('z','Z (nm)',62),('alpha','Transp. %',67)):
            self.tree.heading(key,text=label);self.tree.column(key,width=width,minwidth=25,anchor='center',stretch=key in ('z','alpha'))
        scroll=ttk.Scrollbar(frame,orient='vertical',command=self.tree.yview);self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y');self.tree.pack(fill='both',expand=True)
        self.tree.bind('<Button-1>',self.click_layer);self.tree.bind('<<TreeviewSelect>>',self.load_selected)
        self.tree.bind('<space>',self.toggle_current)
        row=ttk.Frame(self.body);row.pack(fill='x')
        ttk.Button(row,text='Select all',command=lambda:self.select_all(True)).pack(side='left',fill='x',expand=True)
        ttk.Button(row,text='Clear',command=lambda:self.select_all(False)).pack(side='left',fill='x',expand=True,padx=(3,0))
        self.selection_label=tk.StringVar(master=self,value='Load an OVF to list Z layers.')
        ttk.Label(self.body,textvariable=self.selection_label,wraplength=250).pack(anchor='w',pady=3)
        ttk.Label(self.body,text='Fill style').pack(anchor='w')
        box=ttk.Combobox(self.body,textvariable=self.mode,values=('Solid color','Component color map'),state='readonly')
        box.pack(fill='x');box.bind('<<ComboboxSelected>>',self.mode_changed)
        self.solid=ttk.Frame(self.body);self.mapping=ttk.Frame(self.body)
        box=ttk.Combobox(self.solid,textvariable=self.palette,values=tuple(LAYER_FILL_PALETTES),state='readonly')
        box.pack(fill='x',pady=(3,0));box.bind('<<ComboboxSelected>>',lambda event:self.paint_palette())
        self.palette_frame=ttk.Frame(self.solid);self.palette_frame.pack(fill='x',pady=3)
        self.paint_palette()
        ttk.Label(self.solid,text='Color preset').pack(anchor='w')
        box=ttk.Combobox(self.solid,textvariable=self.preset,values=tuple(LAYER_FILL_COLORS)+('Custom',),state='readonly')
        box.pack(fill='x');box.bind('<<ComboboxSelected>>',self.preset_changed)
        row=ttk.Frame(self.solid);row.pack(fill='x',pady=3)
        self.swatch=tk.Label(row,width=3,background=self.color.get(),relief='sunken');self.swatch.pack(side='left',padx=(0,4))
        ttk.Button(row,text='Custom color…',command=self.choose_color).pack(side='left',fill='x',expand=True)
        entry=ttk.Entry(row,textvariable=self.color,width=9);entry.pack(side='right',padx=(3,0))
        entry.bind('<Return>',self.commit);entry.bind('<FocusOut>',self.commit)
        ttk.Label(self.mapping,text='Color map preset').pack(anchor='w')
        box=ttk.Combobox(self.mapping,textvariable=self.colors,values=('Use tab color map',)+tuple(STREAM_COLOR_MAPS),state='normal')
        box.pack(fill='x');box.bind('<<ComboboxSelected>>',self.commit);box.bind('<Return>',self.commit);box.bind('<FocusOut>',self.commit)
        self.opacity_frame=ttk.Frame(self.body);self.opacity_frame.pack(fill='x',pady=3)
        ttk.Label(self.opacity_frame,text='Finish (3D)').pack(anchor='w')
        box=self.finish_choice=ttk.Combobox(self.opacity_frame,textvariable=self.finish,values=tuple(LAYER_FILL_FINISHES),state='readonly')
        box.pack(fill='x',pady=(0,4));box.bind('<<ComboboxSelected>>',self.finish_changed)
        ttk.Label(self.opacity_frame,text='Transparency (%)').pack(anchor='w')
        row=ttk.Frame(self.opacity_frame);row.pack(fill='x')
        ttk.Scale(row,from_=0,to=100,variable=self.alpha_slider,command=self.slider_changed).pack(side='left',fill='x',expand=True)
        spin=ttk.Spinbox(row,from_=0,to=100,increment=1,textvariable=self.transparency,width=5,command=self.commit)
        spin.pack(side='right',padx=3);spin.bind('<Return>',self.commit);spin.bind('<FocusOut>',self.commit)
        ttk.Button(self.body,text='Apply style to checked layers',command=self.apply_checked).pack(fill='x',pady=3)
        self.show_mode()

    def paint_palette(self):
        for child in self.palette_frame.winfo_children():child.destroy()
        for i,(name,color) in enumerate(LAYER_FILL_PALETTES[self.palette.get()].items()):
            swatch=tk.Button(self.palette_frame,background=color,activebackground=color,width=2,height=1,
                relief='raised',borderwidth=1,takefocus=True,command=lambda n=name,c=color:self.pick_swatch(n,c))
            swatch.grid(row=i//10,column=i%10,padx=1,pady=1,sticky='ew')
        for i in range(10):self.palette_frame.columnconfigure(i,weight=1)

    def pick_swatch(self,name,color):
        if self.busy:return
        self.mode.set('Solid color');self.preset.set(name);self.color.set(color);self.show_mode();self.commit()

    def master_changed(self):
        if self.enabled.get():self.body.pack(fill='x')
        else:self.body.pack_forget()
        if not self.busy:self.changed()

    def merge_changed(self):
        if self.busy:return
        if self.merge.get() and not self.merge_sets:
            seed=self.records.get(self.current) or next(iter(self.records.values()),None)
            style={k:v for k,v in seed.items() if k not in ('index','checked')} if seed else dict(
                mode=self.mode.get(),preset=self.preset.get(),color=self.color.get(),colors=self.colors.get(),
                finish=self.finish.get(),transparency=float(self.transparency.get()))
            self.merge_sets[1]=dict(indices=[i for i,r in self.records.items() if r['checked']],style=style)
            self.merge_set.set('Set 1')
        if self.merge.get():self.merge_row.pack(fill='x',before=self.tree_frame,pady=2)
        else:self.merge_row.pack_forget()
        self.tree.configure(displaycolumns=('use','layer','set','z','alpha') if self.merge.get() else ('use','layer','z','alpha'))
        self.tree.heading('use',text='Merge' if self.merge.get() else 'Fill')
        self.update_merge_choice()
        self.refresh_rows();self.load_selected();self.changed()

    def current_merge_id(self):
        return next((i for i in self.merge_sets if self.merge_set.get()==f'Set {i}'),next(iter(self.merge_sets),None))

    def update_merge_choice(self):
        self.merge_choice.configure(values=tuple(f'Set {i}' for i in self.merge_sets))

    def merge_set_changed(self,event=None):
        if self.busy:return
        self.refresh_rows();self.load_selected();self.changed()

    def new_merge_set(self):
        if self.busy or not self.merge.get():return
        current=self.merge_sets.get(self.current_merge_id(),{})
        style=dict(current.get('style',self.merge_style or {}));index=max(self.merge_sets,default=0)+1
        self.merge_sets[index]=dict(indices=[],style=style);self.merge_set.set(f'Set {index}')
        self.update_merge_choice();self.merge_set_changed()

    def remove_merge_set(self):
        if self.busy or not self.merge.get():return
        removed=self.merge_sets.pop(self.current_merge_id(),None)
        if not self.merge_sets:self.merge_sets[1]=dict(indices=[],style=dict((removed or {}).get('style',self.merge_style or {})))
        self.merge_set.set('Set '+str(next(iter(self.merge_sets))))
        self.update_merge_choice();self.merge_set_changed()

    def assign_merge_members(self,indices):
        index=self.current_merge_id();members=set(indices)
        if index is None:return
        for key,group in self.merge_sets.items():
            group['indices']=sorted(members if key==index else set(group['indices'])-members)

    def set_merge_range(self,event=None):
        if self.busy or not self.merge.get():return
        import re
        try:
            members=set()
            for token in self.merge_layers.get().split(','):
                if not token.strip():continue
                match=re.fullmatch(r'\s*(\d+)(?:\s*[-–—]\s*(\d+))?\s*',token)
                if match is None:raise ValueError('Use layer numbers or ranges, for example 1-3, 5.')
                first=int(match[1]);last=int(match[2] or match[1])
                if not 1<=first<=last<=len(self.records):raise ValueError(f'Layer range must be within 1–{len(self.records)}.')
                members.update(range(first-1,last))
        except ValueError as exc:self.status.set(str(exc));return
        self.assign_merge_members(members);self.refresh_rows();self.load_selected();self.changed()

    def show_mode(self):
        self.solid.pack_forget();self.mapping.pack_forget()
        (self.solid if self.mode.get()=='Solid color' else self.mapping).pack(fill='x',before=self.opacity_frame,pady=3)

    def sync_layers(self,z):
        key=(tuple(np.asarray(z['centers'],float)),tuple(z['edges']),z['unit'])
        if self.source_z is not None and self.source_z[0]==key:return
        self.source_z=(key,z)
        self.tree.heading('z',text='Z ('+z['unit']+')')
        count=len(z['centers']);self.loading=True
        try:
            old=self.records;self.records={}
            self.tree.delete(*self.tree.get_children())
            for index,value in enumerate(z['centers']):
                self.records[index]=dict(old.get(index) or dict(index=index,checked=False,mode='Solid color',
                    preset='Pastel blue',color=LAYER_FILL_COLORS['Pastel blue'],colors='Use tab color map',transparency=70.,finish='Matte'))
                self.tree.insert('','end',iid=str(index),values=self.row_values(index))
            self.current=min(self.current or 0,max(0,count-1)) if count else None
            if self.current is not None:self.tree.selection_set(str(self.current))
        finally:self.loading=False
        self.load_selected()

    def row_values(self,index):
        record=self.records[index];z=self.source_z[1]
        owner=next((i for i,g in self.merge_sets.items() if index in g['indices']),None)
        checked=owner==self.current_merge_id() if self.merge.get() else record['checked']
        style=self.merge_sets[owner]['style'] if self.merge.get() and owner is not None else record
        return ('☑' if checked else '☐',index+1,f'Set {owner}' if owner is not None else '',f'{z["centers"][index]:.3f}',f'{style["transparency"]:g}')

    def refresh_rows(self):
        for index in self.records:self.tree.item(str(index),values=self.row_values(index))

    def load_selected(self,event=None):
        if self.loading:return
        rows=self.tree.selection()
        if not rows:return
        self.current=int(rows[0]);record=self.records[self.current];self.loading=True
        try:
            if self.merge.get():
                group=self.merge_sets.get(self.current_merge_id())
                if group is not None:self.merge_style=group['style']
                record=self.merge_style or record
            self.mode.set(record['mode']);self.preset.set(record['preset']);self.color.set(record['color']);self.colors.set(record['colors'])
            self.finish.set(record.get('finish','Matte'))
            self.transparency.set(f'{record["transparency"]:g}');self.alpha_slider.set(record['transparency'])
            self.swatch.configure(background=record['color'])
            z=self.source_z[1];step=(z['edges'][1]-z['edges'][0])/len(z['centers'])
            start=z['edges'][0]+self.current*step
            if self.merge.get():
                group=self.merge_sets.get(self.current_merge_id(),{})
                layers=', '.join(str(i+1) for i in group.get('indices',[]))
                self.merge_layers.set(format_layer_ranges(group.get('indices',[])))
                self.selection_label.set(self.merge_set.get()+' | Layers '+(layers or 'none selected'))
            else:self.selection_label.set(f'Layer {self.current+1} | Z {start:.3f}–{start+step:.3f} {z["unit"]}')
            self.show_mode()
        finally:self.loading=False

    def click_layer(self,event):
        if self.busy:return 'break'
        item=self.tree.identify_row(event.y)
        if item and self.tree.identify_column(event.x)=='#1':
            self.tree.selection_set(item);self.toggle(int(item));return 'break'

    def toggle(self,index):
        if self.busy:return
        if self.merge.get():
            members=set(self.merge_sets.get(self.current_merge_id(),{}).get('indices',[]))
            if index in members:members.remove(index)
            else:members.add(index)
            self.assign_merge_members(members)
        else:self.records[index]['checked']=not self.records[index]['checked']
        self.refresh_rows();self.load_selected();self.changed()

    def toggle_current(self,event=None):
        rows=self.tree.selection()
        if rows:self.toggle(int(rows[0]))
        return 'break'

    def select_all(self,checked):
        if self.busy:return
        if self.merge.get():self.assign_merge_members(self.records if checked else [])
        else:
            for record in self.records.values():record['checked']=checked
        self.refresh_rows();self.load_selected();self.changed()

    def mode_changed(self,event=None):self.show_mode();self.commit()

    def preset_changed(self,event=None):
        if self.preset.get() in LAYER_FILL_COLORS:self.color.set(LAYER_FILL_COLORS[self.preset.get()])
        self.commit()

    def choose_color(self):
        if self.busy:return
        from tkinter import colorchooser
        color=colorchooser.askcolor(color=self.color.get(),parent=self.winfo_toplevel(),title='Layer fill color')[1]
        if color:self.preset.set('Custom');self.color.set(color);self.commit()

    def slider_changed(self,value):
        if self.loading or self.busy:return
        self.transparency.set(str(round(float(value))));self.commit()

    def finish_changed(self,event=None):
        if self.loading or self.busy or self.current is None:return
        try:
            if self.finish.get()=='Glass' and float(self.transparency.get())==0:
                self.transparency.set('70')
        except (ValueError,tk.TclError):pass
        self.commit()

    def commit(self,event=None):
        if self.loading or self.busy or self.current is None:return False
        from matplotlib.colors import to_hex,to_rgba
        try:
            transparency=float(self.transparency.get())
            if not math.isfinite(transparency) or not 0<=transparency<=100:raise ValueError('Transparency must be 0–100%.')
            color=to_hex(to_rgba(self.color.get()))
            if self.mode.get()=='Component color map' and self.colors.get()!='Use tab color map':stream_color_map(self.colors.get())
        except (ValueError,TypeError) as exc:self.status.set(str(exc));return False
        style=dict(mode=self.mode.get(),preset=self.preset.get(),color=color,colors=self.colors.get(),transparency=transparency,finish=self.finish.get())
        if self.merge.get():
            self.merge_style=style
            group=self.merge_sets.get(self.current_merge_id())
            if group is not None:group['style']=dict(style)
        else:self.records[self.current].update(style)
        self.loading=True
        try:self.alpha_slider.set(transparency);self.swatch.configure(background=color)
        finally:self.loading=False
        self.refresh_rows();self.changed();return True

    def apply_checked(self):
        if not self.commit():return
        if self.merge.get():return
        style={k:v for k,v in self.records[self.current].items() if k not in ('index','checked')}
        for record in self.records.values():
            if record['checked']:record.update(style)
        self.refresh_rows();self.changed()

    def snapshot(self):
        members={i for g in self.merge_sets.values() for i in g['indices'] if i in self.records}
        groups=[dict(id=i,indices=[j for j in g['indices'] if j in self.records],style=dict(g['style'])) for i,g in self.merge_sets.items()]
        return dict(enabled=self.enabled.get(),merge=self.merge.get(),merge_style=dict(self.merge_style or {}),
            merge_groups=groups,layers=[dict(record) for i,record in self.records.items() if i in members] if self.merge.get()
            else [dict(record) for record in self.records.values() if record['checked']])

    def set_busy(self,busy):self.busy=busy


def layer_fill_source_z(view):
    current=getattr(view,'layer_fill_native_data',None)
    if current is not None:return current[1]['z']
    source=getattr(view,'native_result',None)
    if source is not None:return source[2][1]['z']
    cache=getattr(view,'native_cache',None)
    if cache is not None:return cache[1][1]['z']
    return view.mesh['z']


def layer_fill_native_data(view):
    current=getattr(view,'layer_fill_native_data',None)
    if current is not None:return current
    style=getattr(view,'style',{})
    if 'layer_native' in style:return style['layer_native']
    source=getattr(view,'native_result',None)
    if source is not None:return source[2][:2]
    cache=getattr(view,'native_cache',None)
    return cache[1][:2] if cache is not None else None


def layer_native_mask(native,index,shape,xy=None):
    if native is None:return None
    vectors,mesh=native
    if index>=len(vectors):return np.zeros(shape,bool)
    occupied=np.isfinite(vectors[index]).all(axis=-1)&(np.linalg.norm(vectors[index],axis=-1)>1e-8)
    if xy is None:
        return np.asarray(Image.fromarray(occupied).resize(shape[::-1],Image.Resampling.NEAREST),bool)
    x0,x1,y0,y1=xy;h,w=shape
    dx=(mesh['x']['edges'][1]-mesh['x']['edges'][0])/vectors.shape[2]
    dy=(mesh['y']['edges'][1]-mesh['y']['edges'][0])/vectors.shape[1]
    x=np.floor((x0+(np.arange(w)+.5)*(x1-x0)/w-mesh['x']['edges'][0])/dx).astype(int)
    y=np.floor((y0+(np.arange(h)+.5)*(y1-y0)/h-mesh['y']['edges'][0])/dy).astype(int)
    valid=(y[:,None]>=0)&(y[:,None]<occupied.shape[0])&(x[None,:]>=0)&(x[None,:]<occupied.shape[1])
    return occupied[y.clip(0,occupied.shape[0]-1)[:,None],x.clip(0,occupied.shape[1]-1)[None,:]]&valid


def layer_fill_effective_state(state):
    if not state.get('merge') or not state.get('layers'):return state
    if 'merge_groups' in state:
        owners={i:group for group in state['merge_groups'] for i in group['indices']}
        layers=[]
        for record in state['layers']:
            group=owners.get(record['index'])
            if group is not None:
                style={k:v for k,v in group['style'].items() if k not in ('index','checked','merge_group')}
                layers.append(dict(record,**style,merge_group=group['id']))
            else:layers.append(dict(record,merge_group=None))
        return dict(state,layers=layers)
    style=state.get('merge_style') or state['layers'][0]
    style={k:v for k,v in style.items() if k not in ('index','checked')}
    return dict(state,layers=[dict(record,**style) for record in state['layers']])


def format_layer_ranges(indices):
    runs=[]
    for i in sorted(set(indices)):
        value=i+1
        if runs and value==runs[-1][1]+1:runs[-1][1]=value
        else:runs.append([value,value])
    return ', '.join(str(a) if a==b else f'{a}-{b}' for a,b in runs)


def layer_fill_record_at(state,source_z,z):
    state=layer_fill_effective_state(state)
    if not state.get('enabled'):return None
    step=(source_z['edges'][1]-source_z['edges'][0])/len(source_z['centers'])
    index=int(math.floor((z-source_z['edges'][0])/step))
    return next((r for r in state['layers'] if r['index']==index and r['transparency']<100),None)


def layer_fill_rgba(plane,record,component,colors,clim,material_mask=None):
    from matplotlib.colors import to_rgba
    occupied=np.isfinite(plane).all(axis=-1)
    occupied &= material_mask if material_mask is not None else np.linalg.norm(plane,axis=-1)>1e-8
    if record['mode']=='Solid color':
        rgba=np.empty((*plane.shape[:2],4),np.uint8);rgba[:]=np.rint(np.array(to_rgba(record['color']))*255).astype(np.uint8)
    else:
        name=colors if record['colors']=='Use tab color map' else record['colors']
        values=np.linalg.norm(plane,axis=-1) if component=='mag' else plane[:,:,{'mx':0,'my':1,'mz':2}[component]]
        lo,hi=clim;hi=hi if hi>lo else lo+1
        rgba=np.uint8(np.rint(stream_color_map(name)(np.clip((values-lo)/(hi-lo),0,1))*255))
    rgba[:,:,3]=np.where(occupied,255,0)
    return rgba


def layer_fill_2d(image,plane,z,source_z,state,component,colors,clim,extent=None,native=None):
    record=layer_fill_record_at(state,source_z,z)
    if record is None:return image
    mask=layer_native_mask(native,record['index'],plane.shape[:2])
    rgba=layer_fill_rgba(plane,record,component,colors,clim,mask)[::-1].copy()
    rgba[:,:,3]=np.rint(rgba[:,:,3]*(1-record['transparency']/100)).astype(np.uint8)
    overlay=Image.fromarray(rgba)
    if extent is None:overlay=overlay.resize(image.size,Image.Resampling.NEAREST)
    else:overlay=overlay.transform(image.size,Image.Transform.EXTENT,extent,resample=Image.Resampling.NEAREST)
    return Image.alpha_composite(image.convert('RGBA'),overlay).convert('RGB')


def build_layer_fill_planes(vectors,mesh,source_z,state,component,colors,clim,xy_bounds=None,z_bounds=None,slice_z=None,native=None):
    state=layer_fill_effective_state(state)
    if not state.get('enabled'):return []
    dx=(mesh['x']['edges'][1]-mesh['x']['edges'][0])/vectors.shape[2]
    dy=(mesh['y']['edges'][1]-mesh['y']['edges'][0])/vectors.shape[1]
    bounds=xy_bounds or {'x':(0,vectors.shape[2]),'y':(0,vectors.shape[1])}
    x0,x1=bounds['x'];y0,y1=bounds['y']
    zcenters=np.asarray(mesh['z']['centers']);planes=[]
    for record in state['layers']:
        index=record['index']
        if index>=len(source_z['centers']) or record['transparency']>=100:continue
        step=(source_z['edges'][1]-source_z['edges'][0])/len(source_z['centers'])
        bottom=source_z['edges'][0]+index*step;top=bottom+step
        if z_bounds is not None:
            bottom=max(bottom,z_bounds[0]);top=min(top,z_bounds[1])
        if top<=bottom:continue
        z=(bottom+top)/2
        if slice_z is not None:
            selected=layer_fill_record_at(state,source_z,slice_z)
            if selected is None or selected['index']!=index:continue
            # Even an XY view retains the selected source layer's full thickness.
        hi=min(len(zcenters)-1,int(np.searchsorted(zcenters,z)));lo=max(0,hi-1)
        t=0. if hi==lo else float(np.clip((z-zcenters[lo])/(zcenters[hi]-zcenters[lo]),0,1))
        patch=vectors[lo,y0:y1,x0:x1,:]
        if t:patch=patch*(1-t)+vectors[hi,y0:y1,x0:x1,:]*t
        xy=(mesh['x']['edges'][0]+x0*dx,mesh['x']['edges'][0]+x1*dx,
            mesh['y']['edges'][0]+y0*dy,mesh['y']['edges'][0]+y1*dy)
        mask=layer_native_mask(native,index,patch.shape[:2],xy)
        rgba=layer_fill_rgba(patch,record,component,colors,clim,mask)
        if not np.any(rgba[:,:,3]):continue
        # Trim only empty margins. Holes and the actual occupied-cell outline remain intact.
        yy,xx=np.where(rgba[:,:,3]>0);r0,r1=int(yy.min()),int(yy.max())+1;c0,c1=int(xx.min()),int(xx.max())+1
        h,w=rgba.shape[:2];a,b,c,d=xy
        xy=(a+(b-a)*c0/w,a+(b-a)*c1/w,c+(d-c)*r0/h,c+(d-c)*r1/h)
        rgba=rgba[r0:r1,c0:c1].copy()
        planes.append(dict(index=index,z=z,z_bounds=(bottom,top),xy=xy,rgba=rgba,
            uniform=record['mode']=='Solid color' and bool(np.all(rgba[:,:,3])),
            opacity=1-record['transparency']/100,record=record))
    return merge_layer_fill_planes(planes) if state.get('merge') else planes


def merge_layer_fill_planes(planes):
    """Only adjoining selected material layers form a union; no gap is bridged."""
    groups=[]
    for plane in sorted(planes,key=lambda p:p['z_bounds'][0]):
        tolerance=64*np.finfo(float).eps*max(1.,abs(plane['z_bounds'][0]))
        key=plane['record'].get('merge_group',0)
        same_set=groups and key is not None and groups[-1][-1]['record'].get('merge_group',0)==key
        if same_set and abs(groups[-1][-1]['z_bounds'][1]-plane['z_bounds'][0])<=tolerance:groups[-1].append(plane)
        else:groups.append([plane])
    result=[]
    for group in groups:
        if len(group)==1:result.append(group[0]);continue
        xy=(min(p['xy'][0] for p in group),max(p['xy'][1] for p in group),
            min(p['xy'][2] for p in group),max(p['xy'][3] for p in group))
        bottom,top=group[0]['z_bounds'][0],group[-1]['z_bounds'][1]
        result.append(dict(group[0],parts=group,indices=tuple(p['index'] for p in group),
            xy=xy,z=(bottom+top)/2,z_bounds=(bottom,top),uniform=False))
    return result


def composite_layer_fill_rays(artist,image,depth,width,height,matrix,box):
    planes=getattr(artist,'fill_planes',None)
    if not planes:return image
    if any('z_bounds' in p for p in planes):
        return composite_layer_fill_volumes(artist,image,depth,width,height,matrix,box)
    lighting=artist_lighting(artist) if getattr(artist,'light_settings',None) is not None else None
    inverse=np.linalg.inv(matrix);inv_data=artist.ax.transData.inverted().get_affine().get_matrix()
    metric=artist.metric;result=image.reshape(-1,4).copy();count=width*height
    for first in range(0,count,60000):
        ids=np.arange(first,min(count,first+60000));yy,xx=np.divmod(ids,width)
        xd=(box.x0+(xx+.5)*box.width/width)*inv_data[0,0]+inv_data[0,2]
        yd=(box.y0+(yy+.5)*box.height/height)*inv_data[1,1]+inv_data[1,2]
        if np.isinf(artist.ax._focal_length):
            near=xd[:,None]*inverse[:,0]+yd[:,None]*inverse[:,1]-10*inverse[:,2]+inverse[:,3]
            origins=near[:,:3]/near[:,3,None]/metric
            direction=inverse[:3,2]/metric;direction/=max(np.linalg.norm(direction),1e-15)
            directions=np.broadcast_to(direction,origins.shape)
        else:
            eye=inverse[:3,2]/inverse[3,2]/metric
            point=xd[:,None]*inverse[:,0]+yd[:,None]*inverse[:,1]-inverse[:,2]+inverse[:,3]
            directions=point[:,:3]/point[:,3,None]/metric-eye
            directions/=np.maximum(np.linalg.norm(directions,axis=1)[:,None],1e-15)
            origins=np.broadcast_to(eye,directions.shape)
        alpha=result[ids,3].astype(float)/255;rgb=result[ids,:3].astype(float)/255*alpha[:,None]
        for positive in (True,False):
            side=directions[:,2]>1e-12 if positive else directions[:,2]<-1e-12
            if not side.any():continue
            for plane in sorted(planes,key=lambda p:p['z'],reverse=positive):
                t=np.divide(plane['z']/metric[2]-origins[:,2],directions[:,2],out=np.full(len(ids),np.inf),where=side)
                valid=side&(t>0)&(t<depth[ids])
                pixel=np.flatnonzero(valid)
                if not len(pixel):continue
                xy=(origins[pixel,:2]+directions[pixel,:2]*t[pixel,None])*metric[:2]
                x0,x1,y0,y1=plane['xy'];h,w=plane['rgba'].shape[:2]
                px=np.floor((xy[:,0]-x0)/(x1-x0)*w).astype(int);py=np.floor((xy[:,1]-y0)/(y1-y0)*h).astype(int)
                inside=(px>=0)&(px<w)&(py>=0)&(py<h)
                pixel=pixel[inside];color=plane['rgba'][py[inside],px[inside]].astype(float)/255
                color[:,:3]=shade_layer_fill(color[:,:3],directions[pixel],plane['record'],artist.ambient,lighting=lighting)
                a=color[:,3]*plane['opacity'];rgb[pixel]=color[:,:3]*a[:,None]+rgb[pixel]*(1-a[:,None])
                alpha[pixel]=a+alpha[pixel]*(1-a)
        rgb=np.divide(rgb,alpha[:,None],out=np.zeros_like(rgb),where=alpha[:,None]>1e-10)
        result[ids,:3]=np.uint8(np.clip(np.rint(rgb*255),0,255));result[ids,3]=np.uint8(np.clip(np.rint(alpha*255),0,255))
    return result.reshape(height,width,4)


def composite_layer_fill_volumes(artist,image,depth,width,height,matrix,box):
    """Trace native-thickness, footprint-masked slabs with glyph depth occlusion.

    Rectangular solid colors use one interval per ray. Irregular outlines / mapped
    colors traverse XY cells, avoiding any geometry that bridges vacuum or holes.
    """
    planes=[]
    for group in artist.fill_planes:
        if group.get('parts'):
            bounds=(*group['xy'],*group['z_bounds'])
            planes.extend(dict(part,merge_bounds=bounds) for part in group['parts'])
        else:planes.append(group)
    inverse=np.linalg.inv(matrix)
    lighting=artist_lighting(artist) if getattr(artist,'light_settings',None) is not None else None
    inv_data=artist.ax.transData.inverted().get_affine().get_matrix();metric=artist.metric
    result=image.reshape(-1,4).copy()
    for first in range(0,width*height,60000):
        ids=np.arange(first,min(width*height,first+60000));yy,xx=np.divmod(ids,width)
        xd=(box.x0+(xx+.5)*box.width/width)*inv_data[0,0]+inv_data[0,2]
        yd=(box.y0+(yy+.5)*box.height/height)*inv_data[1,1]+inv_data[1,2]
        if np.isinf(artist.ax._focal_length):
            near=xd[:,None]*inverse[:,0]+yd[:,None]*inverse[:,1]-10*inverse[:,2]+inverse[:,3]
            origins=near[:,:3]/near[:,3,None]/metric
            direction=inverse[:3,2]/metric;direction/=max(np.linalg.norm(direction),1e-15)
            directions=np.broadcast_to(direction,origins.shape)
        else:
            eye=inverse[:3,2]/inverse[3,2]/metric
            point=xd[:,None]*inverse[:,0]+yd[:,None]*inverse[:,1]-inverse[:,2]+inverse[:,3]
            directions=point[:,:3]/point[:,3,None]/metric-eye
            directions/=np.maximum(np.linalg.norm(directions,axis=1)[:,None],1e-15)
            origins=np.broadcast_to(eye,directions.shape)
        alpha=result[ids,3].astype(float)/255;rgb=result[ids,:3].astype(float)/255*alpha[:,None]
        for positive in (True,False):
            side=directions[:,2]>=0 if positive else directions[:,2]<0
            if not side.any():continue
            for plane in sorted(planes,key=lambda p:p['z'],reverse=positive):
                x0,x1,y0,y1=plane['xy'];z0,z1=plane['z_bounds']
                lower=np.array([x0,y0,z0])/metric;upper=np.array([x1,y1,z1])/metric
                parallel=np.abs(directions)<1e-13
                a=np.divide(lower-origins,directions,out=np.full_like(origins,-np.inf),where=~parallel)
                b=np.divide(upper-origins,directions,out=np.full_like(origins,np.inf),where=~parallel)
                enter=np.minimum(a,b);leave=np.maximum(a,b)
                outside=np.any(parallel&((origins<lower)|(origins>upper)),axis=1)
                near=np.maximum(0,enter.max(axis=1));far=leave.min(axis=1)
                end=np.minimum(far,depth[ids]);pixel=np.flatnonzero(side&~outside&(end>near+1e-12))
                if not len(pixel):continue
                ray=directions[pixel];start=near[pixel];stop=end[pixel]
                path=np.maximum(far[pixel]-start,1e-15)
                normal=np.zeros_like(ray);axis=enter[pixel].argmax(axis=1)
                normal[np.arange(len(pixel)),axis]=-np.sign(ray[np.arange(len(pixel)),axis])
                if 'merge_bounds' in plane:
                    gx0,gx1,gy0,gy1,gz0,gz1=plane['merge_bounds']
                    ga=np.divide(np.array([gx0,gy0,gz0])/metric-origins[pixel],ray,
                        out=np.full_like(ray,-np.inf),where=np.abs(ray)>1e-13)
                    gb=np.divide(np.array([gx1,gy1,gz1])/metric-origins[pixel],ray,
                        out=np.full_like(ray,np.inf),where=np.abs(ray)>1e-13)
                    group_enter=np.minimum(ga,gb);group_leave=np.maximum(ga,gb)
                    path=np.maximum(group_leave.min(axis=1)-np.maximum(0,group_enter.max(axis=1)),1e-15)
                    normal[:]=0;axis=group_enter.argmax(axis=1)
                    normal[np.arange(len(pixel)),axis]=-np.sign(ray[np.arange(len(pixel)),axis])
                transparency=1-plane['opacity']
                if plane.get('uniform',False):
                    color=np.broadcast_to(plane['rgba'][0,0,:3]/255,(len(pixel),3))
                    color=shade_layer_fill(color,ray,plane['record'],artist.ambient,normal,lighting)
                    amount=1-transparency**((stop-start)/path)
                    rgb[pixel]=color*amount[:,None]+rgb[pixel]*(1-amount[:,None])
                    alpha[pixel]=amount+alpha[pixel]*(1-amount)
                    continue
                h,w=plane['rgba'].shape[:2];cell=np.array([(x1-x0)/w,(y1-y0)/h])/metric[:2]
                # Walk from the far visible end back towards the camera.
                current=stop.copy();position=origins[pixel,:2]+ray[:,:2]*(stop-1e-10*path)[:,None]
                ij=np.floor((position-lower[:2])/cell).astype(int);ij[:,0]=ij[:,0].clip(0,w-1);ij[:,1]=ij[:,1].clip(0,h-1)
                active=np.arange(len(pixel))
                for _ in range(w+h+2):
                    if not len(active):break
                    rr=ray[active,:2];boundary=lower[:2]+(ij[active]+(rr<0))*cell
                    crossing=np.divide(boundary-origins[pixel[active],:2],rr,out=np.full_like(rr,-np.inf),where=np.abs(rr)>1e-13)
                    next_t=np.maximum(start[active],crossing.max(axis=1));next_t=np.minimum(next_t,current[active])
                    rgba=plane['rgba'][ij[active,1],ij[active,0]].astype(float)/255
                    amount=(1-transparency**((current[active]-next_t)/path[active]))*rgba[:,3]
                    color=shade_layer_fill(rgba[:,:3],ray[active],plane['record'],artist.ambient,normal[active],lighting)
                    target=pixel[active];rgb[target]=color*amount[:,None]+rgb[target]*(1-amount[:,None])
                    alpha[target]=amount+alpha[target]*(1-amount)
                    hit=np.abs(crossing-next_t[:,None])<=1e-9*np.maximum(1,path[active,None])
                    ij[active]-=(np.sign(rr)*hit).astype(int);current[active]=next_t
                    active=active[(current[active]>start[active]+1e-12)&(ij[active,0]>=0)&(ij[active,0]<w)&(ij[active,1]>=0)&(ij[active,1]<h)]
        rgb=np.divide(rgb,alpha[:,None],out=np.zeros_like(rgb),where=alpha[:,None]>1e-10)
        result[ids,:3]=np.uint8(np.clip(np.rint(rgb*255),0,255));result[ids,3]=np.uint8(np.clip(np.rint(alpha*255),0,255))
    return result.reshape(height,width,4)


def layer_fill_rectilinear_data(plane):
    h,w=plane['rgba'].shape[:2];x0,x1,y0,y1=plane['xy']
    bottom,top=plane['z_bounds']
    rgba=plane['rgba']
    # Remove repeated rows/columns exactly, retaining physical cell edges. A solid
    # rectangular layer becomes one cuboid even at 8x; holes keep their boundaries.
    cols=np.r_[0,1+np.flatnonzero(np.any(rgba[:,1:]!=rgba[:,:-1],axis=(0,2)))]
    reduced=rgba[:,cols]
    rows=np.r_[0,1+np.flatnonzero(np.any(reduced[1:]!=reduced[:-1],axis=(1,2)))]
    reduced=reduced[rows]
    return (x0+(x1-x0)*np.r_[cols,w]/w,y0+(y1-y0)*np.r_[rows,h]/h,np.array([bottom,top]),reduced)


def layer_fill_coalesce_edges(edges):
    values=np.unique(np.concatenate(edges));tolerance=64*np.finfo(float).eps*max(1.,np.abs(values).max())
    return values[np.r_[True,np.diff(values)>tolerance]]


def vtk_layer_fill_mesh(plane):
    pv=require_tab4_backend()
    if not plane.get('parts'):
        x,y,z,reduced=layer_fill_rectilinear_data(plane)
        rgba=reduced[None]
    else:
        parts=[layer_fill_rectilinear_data(p) for p in plane['parts']]
        x=layer_fill_coalesce_edges([p[0] for p in parts]);y=layer_fill_coalesce_edges([p[1] for p in parts])
        z=layer_fill_coalesce_edges([p[2] for p in parts])
        xc=(x[:-1]+x[1:])/2;yc=(y[:-1]+y[1:])/2;zc=(z[:-1]+z[1:])/2
        rgba=np.zeros((len(z)-1,len(y)-1,len(x)-1,4),np.uint8)
        for px,py,pz,colors in parts:
            ix=np.flatnonzero((xc>px[0])&(xc<px[-1]));iy=np.flatnonzero((yc>py[0])&(yc<py[-1]))
            iz=np.flatnonzero((zc>pz[0])&(zc<pz[-1]))
            sx=np.searchsorted(px,xc[ix],side='right')-1;sy=np.searchsorted(py,yc[iy],side='right')-1
            patch=colors[sy[:,None],sx[None,:]]
            for k in iz:rgba[k][iy[:,None],ix[None,:]]=patch
        # Matching neighboring Z cells become one thick cell, removing even the
        # coplanar subdivisions on the outer side walls when shapes match.
        starts=np.r_[0,1+np.flatnonzero(np.any(rgba[1:]!=rgba[:-1],axis=(1,2,3)))]
        rgba=rgba[starts];z=z[np.r_[starts,len(z)-1]]
    grid=pv.RectilinearGrid(x,y,z)
    grid.cell_data['layer_rgba']=rgba.reshape(-1,4)
    valid=np.flatnonzero(rgba[:,:,:,3].ravel())
    return grid.extract_cells(valid) if len(valid)<grid.n_cells else grid


def stream_layer_fill_planes(view,component,clim):
    state=view.style.get('layer_fill',{})
    if not state.get('enabled'):return []
    source_z=view.style.get('layer_source_z',layer_fill_source_z(view))
    section=view.result.get('section');xy_bounds=None;z_bounds=None;slice_z=None
    if section:
        bounds=np.asarray(view.result['display_bounds'],float).reshape(3,2).copy()
        if section['mode']=='Plane':
            axis={'XY':2,'YZ':0,'XZ':1}[section['view']]
            if axis==2:slice_z=section['position']
            else:
                name='xyz'[axis];step=(view.mesh[name]['edges'][1]-view.mesh[name]['edges'][0])/len(view.mesh[name]['centers'])
                bounds[axis]=[section['position']-step/2,section['position']+step/2]
        xy_bounds={}
        for axis,name in enumerate('xy'):
            values=view.mesh[name]['centers'];lo,hi=np.searchsorted(values,bounds[axis],side='left')
            hi=int(np.searchsorted(values,bounds[axis,1],side='right'))
            lo=max(0,min(len(values)-1,int(lo)));hi=max(lo+1,min(len(values),hi))
            xy_bounds[name]=(lo,hi)
        if section['mode']=='Slab':z_bounds=tuple(bounds[2])
    return build_layer_fill_planes(view.vectors,view.mesh,source_z,state,component,view.style['colors'],clim,xy_bounds,z_bounds,slice_z,layer_fill_native_data(view))


class CrossSectionBase(ttk.Frame):
    def __init__(self,gui,parent):
        super().__init__(parent,padding=10)
        self.gui = gui
        self.files = []
        self.vectors = None
        self.mesh = None
        self.time_ns = None
        self.loaded_index = None
        self.loading = False
        self.requested_index = 0
        self.frame_timer = self.render_timer = None
        self.updating = False
        self.photo = self.rendered_image = None
        self.folder = tk.StringVar()
        self.axis = tk.StringVar(value='X')
        self.frame = tk.DoubleVar(value=1)
        self.slice = tk.DoubleVar(value=1)
        self.component = tk.StringVar(value='mz')
        self.arrows = tk.BooleanVar(value=False)
        self.arrow_component = tk.StringVar(value='ALL')
        self.arrow_mode = tk.StringVar(value='Magnitude')
        self.density = tk.StringVar(value='80')
        self.length = tk.StringVar(value='2.5')
        self.head = tk.StringVar(value='1')
        self.colors = tk.StringVar(value=gui.colors.get())
        self.colorbar = tk.BooleanVar(value=True)
        self.show_time = tk.BooleanVar(value=True)
        self.stretch = tk.BooleanVar(value=True)
        self.low,self.high = tk.StringVar(value='-1'),tk.StringVar(value='1')
        self.status = tk.StringVar(value='Choose an OVF folder, then Load frames. X → YZ plane; Y → XZ plane.')
        self.frame_label,self.slice_label = tk.StringVar(),tk.StringVar()
        self.columnconfigure(1,weight=1)
        self.rowconfigure(6,weight=1)
        ttk.Label(self,text='OVF folder:').grid(row=0,column=0,sticky='w')
        ttk.Entry(self,textvariable=self.folder).grid(row=0,column=1,sticky='ew',padx=6)
        buttons = ttk.Frame(self)
        buttons.grid(row=0,column=2)
        ttk.Button(buttons,text='Browse…',command=self.browse).pack(side='left',padx=4)
        ttk.Button(buttons,text='Load frames',command=self.load_folder).pack(side='left',padx=4)
        frames = ttk.Frame(self)
        frames.grid(row=1,column=0,columnspan=3,sticky='ew',pady=5)
        frames.columnconfigure(1,weight=1)
        ttk.Label(frames,text='Frame').grid(row=0,column=0,padx=4)
        self.frame_slider = ttk.Scale(frames,from_=1,to=1,variable=self.frame,command=self.frame_changed)
        self.frame_slider.grid(row=0,column=1,sticky='ew',padx=6)
        ttk.Label(frames,textvariable=self.frame_label,width=28).grid(row=0,column=2)
        slices = ttk.Frame(self)
        slices.grid(row=2,column=0,columnspan=3,sticky='ew',pady=5)
        slices.columnconfigure(2,weight=1)
        ttk.Label(slices,text='Slice axis').grid(row=0,column=0,padx=4)
        box = ttk.Combobox(slices,textvariable=self.axis,values=['X','Y'],state='readonly',width=5)
        box.grid(row=0,column=1,padx=4)
        box.bind('<<ComboboxSelected>>',lambda e:self.update_slice())
        self.slice_slider = ttk.Scale(slices,from_=1,to=1,variable=self.slice,command=self.slice_changed)
        self.slice_slider.grid(row=0,column=2,sticky='ew',padx=6)
        ttk.Label(slices,textvariable=self.slice_label,width=36).grid(row=0,column=3)
        settings = ttk.Frame(self)
        settings.grid(row=3,column=0,columnspan=3,sticky='ew',pady=4)
        ttk.Label(settings,text='Color component').grid(row=0,column=0,sticky='w',padx=4)
        self.component_picker(settings,self.component,0,1)
        ttk.Checkbutton(settings,text='Arrows',variable=self.arrows,command=self.schedule_render).grid(row=0,column=2,padx=8)
        ttk.Label(settings,text='Arrow components').grid(row=0,column=3,padx=4)
        self.component_picker(settings,self.arrow_component,0,4)
        ttk.Label(settings,text='Color map').grid(row=1,column=0,sticky='w',padx=4,pady=5)
        box = ttk.Combobox(settings,textvariable=self.colors,values=COLOR_PRESETS,width=25)
        box.grid(row=1,column=1,sticky='w')
        box.bind('<<ComboboxSelected>>',lambda e:self.schedule_render())
        box.bind('<Return>',lambda e:self.schedule_render())
        ttk.Checkbutton(settings,text='Display time (ns)',variable=self.show_time,command=self.schedule_render).grid(row=1,column=2,padx=8)
        ttk.Checkbutton(settings,text='Color bar scale',variable=self.colorbar,command=self.schedule_render).grid(row=1,column=3,padx=4)
        bar = ttk.Frame(settings)
        bar.grid(row=1,column=4,sticky='w')
        for var,values,width in ((gui.colorbar_size,['Small','Medium','Bigger'],8),(gui.colorbar_position,['Top','Bottom','Middle'],7)):
            box = ttk.Combobox(bar,textvariable=var,values=values,state='readonly',width=width)
            box.pack(side='left',padx=3)
            box.bind('<<ComboboxSelected>>',lambda e:self.schedule_render())
        ttk.Checkbutton(bar,text='Decimals',variable=gui.colorbar_decimals,command=self.schedule_render).pack(side='left',padx=3)
        options = ttk.Frame(self)
        options.grid(row=4,column=0,columnspan=3,sticky='ew',pady=4)
        for label,var,width in (('Density',self.density,5),('Length',self.length,5),('Head',self.head,5),('Min',self.low,6),('Max',self.high,6)):
            ttk.Label(options,text=label).pack(side='left',padx=4)
            entry = ttk.Entry(options,textvariable=var,width=width)
            entry.pack(side='left')
            entry.bind('<Return>',lambda e:self.schedule_render())
        box = ttk.Combobox(options,textvariable=self.arrow_mode,values=['Magnitude','Normalized'],state='readonly',width=11)
        box.pack(side='left',padx=6)
        box.bind('<<ComboboxSelected>>',lambda e:self.schedule_render())
        ttk.Checkbutton(options,text='Stretch slice to window',variable=self.stretch,command=self.schedule_render).pack(side='left',padx=6)
        ttk.Button(options,text='Refresh',command=self.schedule_render).pack(side='left',padx=6)
        ttk.Button(options,text='Save slice…',command=self.save).pack(side='left',padx=6)
        ttk.Label(self,text='X slice = YZ plane; Y slice = XZ plane. In-plane arrows show direction; normal components use ⊙ toward / ⊗ away. Stretch changes display aspect only.',wraplength=900).grid(row=5,column=0,columnspan=3,sticky='w',pady=4)
        self.canvas = tk.Canvas(self,background='#252525',highlightthickness=0)
        self.canvas.grid(row=6,column=0,columnspan=3,sticky='nsew')
        self.canvas.bind('<Configure>',lambda e:self.schedule_render())
        ttk.Label(self,textvariable=self.status,wraplength=900).grid(row=7,column=0,columnspan=3,sticky='w',pady=4)

    def component_picker(self,parent,variable,row,column):
        picker = ttk.Frame(parent)
        picker.grid(row=row,column=column,sticky='w')
        for key,icon in zip(('mx','my','mz'),self.gui.component_icons):
            ttk.Radiobutton(picker,image=icon,variable=variable,value=key,command=self.schedule_render).pack(side='left',padx=2)
        ttk.Radiobutton(picker,text='ALL',variable=variable,value='ALL',command=self.schedule_render).pack(side='left',padx=2)

    def browse(self):
        folder = filedialog.askdirectory(parent=self.gui.root,title='Choose vector-field folder')
        if folder:
            self.folder.set(folder)
            self.load_folder()

    def load_folder(self):
        text = self.folder.get().strip() or self.gui.ovf_input.get().strip()
        folder = Path(text).expanduser()
        if not text or not folder.is_dir():
            self.status.set('Choose a valid field folder.')
            return
        files = field_files(folder)
        if not files:
            self.status.set('No OVF, OOMMF, or VTK vector fields found in this folder.')
            return
        self.folder.set(str(folder))
        self.files = files
        self.vectors = None
        self.rendered_image = None
        self.loaded_index = None
        self.canvas.delete('all')
        self.updating = True
        self.frame_slider.configure(from_=1,to=max(1,len(files)))
        self.frame.set(1)
        self.updating = False
        self.request_frame(0)

    def frame_changed(self,value):
        if self.updating or not self.files:
            return
        index = min(len(self.files)-1,max(0,round(float(value))-1))
        self.requested_index = index
        self.frame_label.set(f'{index+1}/{len(self.files)}: {self.files[index].name}')
        if self.frame_timer is not None:
            self.after_cancel(self.frame_timer)
        self.frame_timer = self.after(100,lambda:self.request_frame(index))

    def request_frame(self,index):
        self.frame_timer = None
        self.requested_index = index
        if self.loading:
            return
        self.loading = True
        source = self.files[index]
        self.frame_label.set(f'{index+1}/{len(self.files)}: {source.name}')
        self.status.set(f'Reading {source.name}…')
        def worker():
            try:
                vectors = parse_ovf_vectors(source)
                result = (vectors,ovf_mesh_coordinates(source,vectors.shape),read_ovf_time_ns(source),None)
            except Exception as exc:
                result = (None,None,None,str(exc))
            if not self.gui.closed:
                self.gui.events.put((self.loaded,(source,index,result)))
        threading.Thread(target=worker,daemon=True).start()

    def loaded(self,source,index,result):
        self.loading = False
        if not self.files:
            return
        if index!=self.requested_index or source!=self.files[index]:
            self.request_frame(self.requested_index)
            return
        self.vectors,self.mesh,self.time_ns,error = result
        self.loaded_index = index
        self.updating = True
        self.frame.set(index+1)
        self.updating = False
        if error:
            self.canvas.delete('all')
            self.rendered_image = None
            self.status.set(f'Unreadable frame {source.name}: {error}. Select another frame.')
            return
        self.update_slice()

    def update_slice(self):
        if self.vectors is None:
            return
        count = self.vectors.shape[2 if self.axis.get()=='X' else 1]
        self.updating = True
        self.slice_slider.configure(from_=1,to=max(1,count))
        self.slice.set(min(count,max(1,round(float(self.slice.get())))))
        self.updating = False
        self.schedule_render()

    def slice_changed(self,value):
        if not self.updating:
            self.schedule_render()

    def schedule_render(self):
        if self.gui.closed:
            return
        if self.render_timer is not None:
            self.after_cancel(self.render_timer)
        self.render_timer = self.after(60,self.render)

    def render(self):
        self.render_timer = None
        if self.vectors is None or self.gui.closed:
            return
        try:
            axis = self.axis.get()
            count = self.vectors.shape[2 if axis=='X' else 1]
            index = min(count-1,max(0,round(float(self.slice.get()))-1))
            plane,horizontal,pair,normal = extract_cross_section(self.vectors,axis,index)
            info = self.mesh[axis.lower()]
            self.slice_label.set(f'{axis} = {info["centers"][index]:.3f} {info["unit"]} | cell {index+1}/{count}')
            low,high = float(self.low.get()),float(self.high.get())
            density,length,head = int(self.density.get()),float(self.length.get()),float(self.head.get())
            if density<1 or not all(math.isfinite(v) and v>0 for v in (length,head)):
                raise ValueError('Arrow density, length and head size must be positive.')
            comp_list = ['mx','my','mz'] if self.component.get()=='ALL' else [self.component.get()]
            width,height = max(400,self.canvas.winfo_width()),max(220,self.canvas.winfo_height())
            panel_w = max(160,width//len(comp_list)-80)
            panel_h = max(100,height-110)
            panels = []
            font = annotation_font(12)
            for component in comp_list:
                image = scalar_slice_image(plane[:,:,{'mx':0,'my':1,'mz':2}[component]],self.colors.get(),low,high)
                w,h = panel_w,panel_h
                if not self.stretch.get():
                    horizontal_span = abs(self.mesh[horizontal]['edges'][1]-self.mesh[horizontal]['edges'][0])
                    vertical_span = abs(self.mesh['z']['edges'][1]-self.mesh['z']['edges'][0])
                    ratio = min(panel_w/max(horizontal_span,1e-30),panel_h/max(vertical_span,1e-30))
                    w,h = max(1,round(horizontal_span*ratio)),max(1,round(vertical_span*ratio))
                image = image.resize((w,h),Image.Resampling.NEAREST)
                if self.arrows.get():
                    image = cross_section_arrows(image,plane,axis,self.arrow_component.get(),self.arrow_mode.get(),density,length,head)
                image = decorate_image(image,dict(colorbar=self.colorbar.get(),colorbar_size=self.gui.colorbar_size.get(),
                           colorbar_position=self.gui.colorbar_position.get(),colorbar_decimals=self.gui.colorbar_decimals.get(),
                           vmin=low,vmax=high,font_size=12),colors=self.colors.get(),component=component)
                image = annotation_text_panel(image,f'{component} | {horizontal.upper()}Z plane','Top center',font,4)
                hor,z = self.mesh[horizontal],self.mesh['z']
                image = annotation_text_panel(image,f'{horizontal.upper()}: {hor["edges"][0]:.3f} – {hor["edges"][1]:.3f} {hor["unit"]}\nZ: {z["edges"][0]:.3f} – {z["edges"][1]:.3f} {z["unit"]}','Bottom center',font,4)
                panels.append(image)
            composite = Image.new('RGB',(sum(p.width for p in panels)+10*(len(panels)-1),max(p.height for p in panels)),'white')
            x = 0
            for panel in panels:
                composite.paste(panel,(x,0))
                x += panel.width+10
            if self.show_time.get():
                text = f't = {self.time_ns:.3f} ns' if self.time_ns is not None else 't = unavailable (OVF time missing)'
                composite = annotation_text_panel(composite,text,'Top left',font,4)
            self.rendered_image = composite
            ratio = min(width/composite.width,height/composite.height)
            shown = composite.resize((max(1,round(composite.width*ratio)),max(1,round(composite.height*ratio))),Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(shown,master=self.gui.root)
            self.canvas.delete('all')
            self.canvas.create_image(width//2,height//2,image=self.photo,anchor='center')
            self.status.set(f'{self.files[self.loaded_index].name} | mesh {self.vectors.shape[2]} × {self.vectors.shape[1]} × {self.vectors.shape[0]} | {"stretched display" if self.stretch.get() else "physical aspect"}')
        except (ValueError,OSError) as exc:
            self.status.set(f'Cross section: {exc}')

    def save(self):
        if self.rendered_image is None:
            self.status.set('Load a readable frame first.')
            return
        path = filedialog.asksaveasfilename(parent=self.gui.root,title='Save cross section',defaultextension='.png',filetypes=[('PNG','*.png')])
        if path:
            self.rendered_image.save(path)

    def close(self):
        for timer in (self.frame_timer,self.render_timer):
            if timer is not None:
                self.after_cancel(timer)


class LinkedCrossSectionBase(CrossSectionBase):
    """Linked XY overview and XY/YZ/XZ slice, with draggable reference lines."""
    def __init__(self,gui,parent):
        super().__init__(gui,parent)
        self.view = tk.StringVar(value='YZ')
        self.positions = {a:tk.DoubleVar(value=1) for a in 'xyz'}
        self.position_labels = {a:tk.StringVar() for a in 'xyz'}
        self.position_initialized = False
        self.top_data_rects = []
        self.bottom_data_rects = []
        self.display_transform = None
        self.drag_target = None
        self.slice_slider.master.grid_remove()
        controls = ttk.Frame(self)
        controls.grid(row=2,column=0,columnspan=3,sticky='ew',pady=4)
        controls.columnconfigure(2,weight=1)
        ttk.Label(controls,text='Cross-section view').grid(row=0,column=0,sticky='w',padx=4)
        box = ttk.Combobox(controls,textvariable=self.view,values=['XY','YZ','XZ'],state='readonly',width=7)
        box.grid(row=0,column=1,sticky='w',padx=4)
        box.bind('<<ComboboxSelected>>',lambda e:self.view_changed())
        ttk.Label(controls,text='Drag reference lines with the left mouse button in either view.').grid(row=0,column=2,columnspan=2,sticky='w',padx=6)
        self.position_sliders = {}
        for row,axis in enumerate('xyz',1):
            ttk.Label(controls,text=axis.upper()+' position').grid(row=row,column=0,sticky='w',padx=4)
            slider = ttk.Scale(controls,from_=1,to=1,variable=self.positions[axis],
                               command=lambda value,a=axis:self.position_changed(a,value))
            slider.grid(row=row,column=1,columnspan=2,sticky='ew',padx=6,pady=2)
            self.position_sliders[axis] = slider
            ttk.Label(controls,textvariable=self.position_labels[axis],width=29).grid(row=row,column=3,sticky='w')
        for child in self.winfo_children():
            if isinstance(child,ttk.Label) and int(child.grid_info().get('row',-1))==5:
                child.configure(text='Top: XY view at selected Z. Bottom: selected XY/YZ/XZ slice. Yellow = active cut; cyan = linked position. Normal components: ⊙ toward / ⊗ away. Stretch changes display aspect only.')
        self.canvas.bind('<ButtonPress-1>',self.drag_start)
        self.canvas.bind('<B1-Motion>',self.drag_move)
        self.canvas.bind('<ButtonRelease-1>',self.drag_end)

    def load_folder(self):
        if self.frame_timer is not None:
            self.after_cancel(self.frame_timer)
            self.frame_timer = None
        self.position_initialized = False
        super().load_folder()

    def request_frame(self,index):
        if self.files:
            super().request_frame(min(len(self.files)-1,max(0,index)))

    def view_changed(self):
        self.axis.set({'XY':'Z','YZ':'X','XZ':'Y'}[self.view.get()])
        self.update_slice()

    def update_slice(self):
        if self.vectors is None:
            return
        self.updating = True
        for axis,count in zip('xyz',(self.vectors.shape[2],self.vectors.shape[1],self.vectors.shape[0])):
            self.position_sliders[axis].configure(from_=1,to=max(1,count))
            value = (1 if axis=='z' else (count+1)//2) if not self.position_initialized else round(self.positions[axis].get())
            self.positions[axis].set(min(count,max(1,value)))
        self.position_initialized = True
        self.updating = False
        self.schedule_render()

    def position_changed(self,axis,value):
        if not self.updating:
            self.schedule_render()

    def schedule_render(self):
        # Throttle instead of debounce, so drawing continues during a mouse drag.
        if self.gui.closed or self.render_timer is not None:
            return
        self.render_timer = self.after(16,self.render)

    def indices(self):
        return {axis:min(count-1,max(0,round(self.positions[axis].get())-1))
                for axis,count in zip('xyz',(self.vectors.shape[2],self.vectors.shape[1],self.vectors.shape[0]))}

    def reference_lines(self,image,view,indices):
        horizontal,vertical = {'XY':('x','y'),'YZ':('y','z'),'XZ':('x','z')}[view]
        counts = dict(zip('xyz',(self.vectors.shape[2],self.vectors.shape[1],self.vectors.shape[0])))
        x = (indices[horizontal]+.5)*image.width/counts[horizontal]
        y = (counts[vertical]-indices[vertical]-.5)*image.height/counts[vertical]
        active = {'YZ':'x','XZ':'y','XY':'z'}[self.view.get()]
        draw = ImageDraw.Draw(image)
        for points,axis in (((x,0,x,image.height-1),horizontal),((0,y,image.width-1,y),vertical)):
            draw.line(points,fill='black',width=3)
            draw.line(points,fill=(255,235,0) if axis==active else (0,235,255),width=1)
        return image

    def panel(self,plane,view,component,title,width,height,indices,low,high,density,length,head):
        horizontal,vertical,normal = {'XY':('x','y','Z'),'YZ':('y','z','X'),'XZ':('x','z','Y')}[view]
        w,h = width,height
        if not self.stretch.get():
            h_span = abs(self.mesh[horizontal]['edges'][1]-self.mesh[horizontal]['edges'][0])
            v_span = abs(self.mesh[vertical]['edges'][1]-self.mesh[vertical]['edges'][0])
            ratio = min(width/max(h_span,1e-30),height/max(v_span,1e-30))
            w,h = max(1,round(h_span*ratio)),max(1,round(v_span*ratio))
        image = scalar_slice_image(plane[:,:,{'mx':0,'my':1,'mz':2}[component]],self.colors.get(),low,high)
        image = image.resize((w,h),Image.Resampling.NEAREST)
        if self.arrows.get():
            image = cross_section_arrows(image,plane,normal,self.arrow_component.get(),self.arrow_mode.get(),density,length,head)
        image = self.reference_lines(image,view,indices)
        image = decorate_image(image,dict(colorbar=self.colorbar.get(),colorbar_size=self.gui.colorbar_size.get(),
                     colorbar_position=self.gui.colorbar_position.get(),colorbar_decimals=self.gui.colorbar_decimals.get(),
                     vmin=low,vmax=high,font_size=12),colors=self.colors.get(),component=component)
        rect = [0,0,w,h]
        font = annotation_font(11)
        previous = image.size
        image = annotation_text_panel(image,f'{title} | {component} | {view}','Top center',font,3)
        rect[0] += (image.width-previous[0])//2
        rect[2] += (image.width-previous[0])//2
        rect[1] += image.height-previous[1]
        rect[3] += image.height-previous[1]
        a,b = self.mesh[horizontal],self.mesh[vertical]
        previous = image.size
        label = f'{horizontal.upper()}: {a["edges"][0]:.3f}–{a["edges"][1]:.3f} {a["unit"]}; {vertical.upper()}: {b["edges"][0]:.3f}–{b["edges"][1]:.3f} {b["unit"]}'
        if not hasattr(self,'show_scales') or self.show_scales.get():
            image = annotation_text_panel(image,label,'Bottom center',font,3)
        rect[0] += (image.width-previous[0])//2
        rect[2] += (image.width-previous[0])//2
        return image,rect

    def render(self):
        self.render_timer = None
        if self.vectors is None or self.gui.closed:
            return
        try:
            indices = self.indices()
            for axis,count in zip('xyz',(self.vectors.shape[2],self.vectors.shape[1],self.vectors.shape[0])):
                info = self.mesh[axis]
                self.position_labels[axis].set(f'{info["centers"][indices[axis]]:.3f} {info["unit"]} | cell {indices[axis]+1}/{count}')
            view = self.view.get()
            self.axis.set({'XY':'Z','YZ':'X','XZ':'Y'}[view])
            self.slice.set(indices[self.axis.get().lower()]+1)
            top = self.vectors[indices['z'],:,:,:]
            bottom,_,_,_ = extract_cross_section(self.vectors,self.axis.get(),indices[self.axis.get().lower()])
            self.last_top_plane,self.last_bottom_plane = top,bottom
            self.last_indices = indices
            low,high = float(self.low.get()),float(self.high.get())
            density,length,head = int(self.density.get()),float(self.length.get()),float(self.head.get())
            if density<1 or not all(math.isfinite(v) and v>0 for v in (length,head)):
                raise ValueError('Arrow density, length and head size must be positive.')
            components = ['mx','my','mz'] if self.component.get()=='ALL' else [self.component.get()]
            width,height = self.canvas.winfo_width(),self.canvas.winfo_height()
            width,height = width if width>20 else 900,height if height>20 else 500
            panel_width = max(80,width//len(components)-60)
            panel_height = max(40,(height-130)//2)
            rows = []
            row_rects = []
            for plane,plane_view,title in ((top,'XY','Top view'),(bottom,view,'Cross section')):
                panels = [self.panel(plane,plane_view,component,title,panel_width,panel_height,indices,low,high,density,length,head) for component in components]
                row = Image.new('RGB',(sum(p.width for p,r in panels)+8*(len(panels)-1),max(p.height for p,r in panels)),'white')
                rects = []
                x = 0
                for panel,rect in panels:
                    row.paste(panel,(x,0))
                    rects.append((rect[0]+x,rect[1],rect[2]+x,rect[3]))
                    x += panel.width+8
                rows.append(row)
                row_rects.append(rects)
            composite = Image.new('RGB',(max(r.width for r in rows),sum(r.height for r in rows)+12),'#dddddd')
            rect_groups = []
            y = 0
            for row,rects in zip(rows,row_rects):
                x = (composite.width-row.width)//2
                composite.paste(row,(x,y))
                rect_groups.append([(a+x,b+y,c+x,d+y) for a,b,c,d in rects])
                y += row.height+12
            if self.show_time.get():
                text = f't = {self.time_ns:.3f} ns' if self.time_ns is not None else 't = unavailable (OVF time missing)'
                previous = composite.size
                composite = annotation_text_panel(composite,text,'Top left',annotation_font(12),3)
                dx,dy = (composite.width-previous[0])//2,composite.height-previous[1]
                rect_groups = [[(a+dx,b+dy,c+dx,d+dy) for a,b,c,d in rects] for rects in rect_groups]
            self.top_data_rects,self.bottom_data_rects = rect_groups
            self.rendered_image = composite
            ratio = min(width/composite.width,height/composite.height)
            shown_w,shown_h = max(1,round(composite.width*ratio)),max(1,round(composite.height*ratio))
            shown = composite.resize((shown_w,shown_h),Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(shown,master=self.gui.root)
            self.canvas.delete('all')
            self.canvas.create_image(width//2,height//2,image=self.photo,anchor='center')
            self.display_transform = (width//2-shown_w/2,height//2-shown_h/2,shown_w/composite.width,shown_h/composite.height)
            self.status.set(f'{self.files[self.loaded_index].name} | top XY at Z cell {indices["z"]+1} | bottom {view} | drag to move linked reference lines | {"stretched display" if self.stretch.get() else "physical aspect"}')
        except (ValueError,OSError) as exc:
            self.status.set(f'Cross section: {exc}')

    def pointer_in_image(self,event):
        if self.display_transform is None:
            return None
        x,y,sx,sy = self.display_transform
        return (event.x-x)/sx,(event.y-y)/sy

    def drag_start(self,event):
        point = self.pointer_in_image(event)
        if point is None or self.vectors is None:
            return
        x,y = point
        for view,rects in (('XY',self.top_data_rects),(self.view.get(),self.bottom_data_rects)):
            for rect in rects:
                a,b,c,d = rect
                if a<=x<c and b<=y<d:
                    self.drag_target = (view,rect)
                    self.canvas.configure(cursor='crosshair')
                    self.drag_move(event)
                    return

    def drag_move(self,event):
        point = self.pointer_in_image(event)
        if self.drag_target is None or point is None or self.vectors is None:
            return
        view,(a,b,c,d) = self.drag_target
        horizontal,vertical = {'XY':('x','y'),'YZ':('y','z'),'XZ':('x','z')}[view]
        counts = dict(zip('xyz',(self.vectors.shape[2],self.vectors.shape[1],self.vectors.shape[0])))
        px = min(1.,max(0.,(point[0]-a)/max(1,c-a)))
        py = min(1.,max(0.,(point[1]-b)/max(1,d-b)))
        self.updating = True
        self.positions[horizontal].set(min(counts[horizontal],int(px*counts[horizontal])+1))
        self.positions[vertical].set(min(counts[vertical],int((1-py)*counts[vertical])+1))
        self.updating = False
        self.schedule_render()

    def drag_end(self,event):
        if self.drag_target is not None:
            self.drag_move(event)
        self.drag_target = None
        self.canvas.configure(cursor='')


def solid_arrow_faces(origin, vector, length, radius, head_ratio=.3, sides=6, head_radius=None):
    magnitude = float(np.linalg.norm(vector))
    if magnitude<1e-9 or length<=0:
        return []
    direction = np.asarray(vector,dtype=float)/magnitude
    helper = np.array([0.,0.,1.]) if abs(direction[2])<.9 else np.array([0.,1.,0.])
    u = np.cross(direction,helper)
    u /= np.linalg.norm(u)
    v = np.cross(direction,u)
    start = np.asarray(origin,dtype=float)-direction*length/2
    tip = start+direction*length
    shoulder = start+direction*length*(1-head_ratio)
    angles = np.arange(sides)*2*np.pi/sides
    ring = np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v
    base,neck,head = start+ring*radius,shoulder+ring*radius,shoulder+ring*(radius*2.5 if head_radius is None else head_radius)
    faces = [base[::-1],head[::-1]]
    for i in range(sides):
        j = (i+1)%sides
        faces.extend([[base[i],base[j],neck[j],neck[i]],[head[i],head[j],tip]])
    return faces


class InteractiveCrossSectionBase(LinkedCrossSectionBase):
    """Left controls, linked overview (25%) and interactive solid-glyph scene (75%)."""
    def __init__(self,gui,parent):
        super().__init__(gui,parent)
        for child in self.winfo_children():
            child.destroy()
        for row in range(12):
            self.rowconfigure(row,weight=0)
        for column in range(5):
            self.columnconfigure(column,weight=0)
        self.rowconfigure(0,weight=1)
        self.columnconfigure(1,weight=1)
        self.glyph_budget = 600
        self.arrows.set(True)
        self.zoom = 1.
        self.scene_view = None
        self.base_limits = None
        self.top_export_image = None
        self.divider = ttk.Panedwindow(self,orient='horizontal')
        self.divider.grid(row=0,column=0,columnspan=2,sticky='nsew')
        self.sidebar_panel = ttk.Frame(self.divider)
        self.sidebar_panel.columnconfigure(0,weight=1)
        self.sidebar_panel.rowconfigure(0,weight=1)
        self.sidebar_canvas = tk.Canvas(self.sidebar_panel,width=280,highlightthickness=0)
        self.sidebar_canvas.grid(row=0,column=0,sticky='nsew')
        scrollbar = ttk.Scrollbar(self.sidebar_panel,orient='vertical',command=self.sidebar_canvas.yview)
        scrollbar.grid(row=0,column=1,sticky='ns')
        self.sidebar_canvas.configure(yscrollcommand=scrollbar.set)
        self.controls = ttk.Frame(self.sidebar_canvas,padding=6)
        self.sidebar_canvas.create_window((0,0),window=self.controls,anchor='nw',width=264)
        self.controls.bind('<Configure>',lambda e:self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox('all')))
        self.figures = ttk.Frame(self)
        self.figures.grid(row=0,column=1,sticky='nsew',padx=(8,0))
        self.figures.columnconfigure(0,weight=1)
        self.figures.rowconfigure(0,weight=1,uniform='figure_split')
        self.figures.rowconfigure(1,weight=3,uniform='figure_split')
        self.fig1_frame,self.fig2_frame = ttk.Frame(self.figures),ttk.Frame(self.figures)
        self.fig1_frame.grid(row=0,column=0,sticky='nsew')
        self.fig2_frame.grid(row=1,column=0,sticky='nsew')
        self.fig1_frame.grid_propagate(False)
        self.fig2_frame.grid_propagate(False)
        self.canvas = tk.Canvas(self.fig1_frame,background='#252525',highlightthickness=0)
        self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<Configure>',lambda e:self.schedule_render())
        self.canvas.bind('<ButtonPress-1>',self.drag_start)
        self.canvas.bind('<B1-Motion>',self.drag_move)
        self.canvas.bind('<ButtonRelease-1>',self.drag_end)
        self.figure = Figure(figsize=(6,5),dpi=100,facecolor='white')
        self.ax3d = self.figure.add_subplot(111,projection='3d')
        self.figure.subplots_adjust(left=.01,right=.89,bottom=.05,top=.91)
        self.plot_canvas = FigureCanvasTkAgg(self.figure,master=self.fig2_frame)
        self.plot_canvas.get_tk_widget().pack(fill='both',expand=True)
        self.ax3d.mouse_init(rotate_btn=2,pan_btn=1,zoom_btn=3)
        self.mpl_callbacks = [self.plot_canvas.mpl_connect('scroll_event',self.scroll_zoom),
                              self.plot_canvas.mpl_connect('key_press_event',self.camera_key)]
        self.status.set('Load a vector-field folder. Figure 1: drag slice lines. Figure 2: middle-drag rotate; left-drag pan; wheel zoom.')
        self.build_sidebar()

    def group(self,title):
        group = ttk.LabelFrame(self.controls,text=title,padding=6)
        group.pack(fill='x',pady=4)
        return group

    def picker(self,parent,variable):
        row = ttk.Frame(parent)
        row.pack(fill='x',pady=3)
        for name,icon in zip(('mx','my','mz'),self.gui.component_icons):
            ttk.Radiobutton(row,image=icon,variable=variable,value=name,command=self.schedule_render).pack(side='left',padx=2)
        ttk.Radiobutton(row,text='ALL',variable=variable,value='ALL',command=self.schedule_render).pack(side='left',padx=2)

    def dropdown(self,parent,label,variable,values):
        ttk.Label(parent,text=label).pack(anchor='w')
        box = ttk.Combobox(parent,textvariable=variable,values=values,state='readonly')
        box.pack(fill='x',pady=2)
        box.bind('<<ComboboxSelected>>',lambda e:self.schedule_render())
        return box

    def build_sidebar(self):
        files = self.group('OVF frames')
        ttk.Entry(files,textvariable=self.folder).pack(fill='x')
        row = ttk.Frame(files)
        row.pack(fill='x',pady=3)
        ttk.Button(row,text='Browse…',command=self.browse).pack(side='left',padx=2)
        ttk.Button(row,text='Load frames',command=self.load_folder).pack(side='left',padx=2)
        self.frame_slider = ttk.Scale(files,from_=1,to=1,variable=self.frame,command=self.frame_changed)
        self.frame_slider.pack(fill='x')
        ttk.Label(files,textvariable=self.frame_label,wraplength=235).pack(anchor='w')
        positions = self.group('Cross-section position')
        box = self.dropdown(positions,'Figure 2 plane',self.view,['XY','YZ','XZ'])
        box.bind('<<ComboboxSelected>>',lambda e:self.view_changed())
        self.position_sliders = {}
        for axis in 'xyz':
            ttk.Label(positions,text=axis.upper()+' position').pack(anchor='w')
            scale = ttk.Scale(positions,from_=1,to=1,variable=self.positions[axis],command=lambda value,a=axis:self.position_changed(a,value))
            scale.pack(fill='x')
            self.position_sliders[axis] = scale
            ttk.Label(positions,textvariable=self.position_labels[axis],wraplength=235).pack(anchor='w')
        glyphs = self.group('3D arrows')
        ttk.Checkbutton(glyphs,text='Show solid 3D arrows',variable=self.arrows,command=self.schedule_render).pack(anchor='w')
        self.picker(glyphs,self.arrow_component)
        self.dropdown(glyphs,'Arrow scaling',self.arrow_mode,['Magnitude','Normalized'])
        for label,var in (('Density',self.density),('Length scale',self.length),('Head size',self.head)):
            row = ttk.Frame(glyphs)
            row.pack(fill='x',pady=1)
            ttk.Label(row,text=label,width=12).pack(side='left')
            entry = ttk.Entry(row,textvariable=var,width=9)
            entry.pack(side='left')
            entry.bind('<Return>',lambda e:self.schedule_render())
        ttk.Label(glyphs,text='Dense fields are sampled to at most 600 solid arrows for responsive interaction.',wraplength=235).pack(anchor='w',pady=3)
        colors = self.group('Colors and labels')
        ttk.Label(colors,text='Color component').pack(anchor='w')
        self.picker(colors,self.component)
        ttk.Label(colors,text='Color map').pack(anchor='w')
        self.color_box = ttk.Combobox(colors,textvariable=self.colors,values=STREAM_COLOR_MAPS,state='normal')
        self.color_box.pack(fill='x',pady=3)
        self.color_box.bind('<<ComboboxSelected>>',lambda e:self.schedule_render())
        self.color_box.bind('<Return>',lambda e:self.schedule_render())
        self.color_preset_button = ttk.Button(colors,text='Choose preset…',command=self.choose_color_preset)
        self.color_preset_button.pack(fill='x',pady=(0,3))
        row = ttk.Frame(colors)
        row.pack(fill='x')
        for label,var in (('Min',self.low),('Max',self.high)):
            ttk.Label(row,text=label).pack(side='left',padx=2)
            entry = ttk.Entry(row,textvariable=var,width=7)
            entry.pack(side='left')
            entry.bind('<Return>',lambda e:self.schedule_render())
        ttk.Checkbutton(colors,text='Display time (ns)',variable=self.show_time,command=self.schedule_render).pack(anchor='w')
        ttk.Checkbutton(colors,text='Color bar scale',variable=self.colorbar,command=self.schedule_render).pack(anchor='w')
        self.dropdown(colors,'Color-bar size',self.gui.colorbar_size,['Small','Medium','Bigger'])
        self.dropdown(colors,'Color-bar position',self.gui.colorbar_position,['Top','Middle','Bottom'])
        ttk.Checkbutton(colors,text='Decimals (3)',variable=self.gui.colorbar_decimals,command=self.schedule_render).pack(anchor='w')
        ttk.Checkbutton(colors,text='Expand Z thickness (keep XY aspect)',variable=self.stretch,command=self.schedule_render).pack(anchor='w')
        actions = self.group('View and export')
        ttk.Button(actions,text='Reset view',command=self.reset_view).pack(fill='x',pady=2)
        ttk.Button(actions,text='Refresh',command=self.schedule_render).pack(fill='x',pady=2)
        ttk.Button(actions,text='Export Figure',command=self.export_figures).pack(fill='x',pady=2)
        ttk.Label(actions,text='Exports Figure 1 and Figure 2 separately as 300 PPI PNGs.\n\nFigure 1: left-drag reference lines.\nFigure 2: middle-drag rotates; left-drag pans; wheel zooms. R resets; X/Y/Z aligns camera.',wraplength=235).pack(anchor='w',pady=4)
        ttk.Label(self.controls,textvariable=self.status,wraplength=240).pack(fill='x',pady=5)
        def wheel(event):
            self.sidebar_canvas.yview_scroll(-int(event.delta/120),'units')
        def bind_children(widget):
            widget.bind('<MouseWheel>',wheel,add='+')
            for child in widget.winfo_children():
                bind_children(child)
        bind_children(self.controls)
        self.sidebar_canvas.bind('<MouseWheel>',wheel)

    def view_changed(self):
        self.scene_view = None
        self.zoom = 1.
        super().view_changed()

    def render(self):
        self.render_timer = None
        if self.vectors is None or self.gui.closed:
            return
        try:
            indices = self.indices()
            self.last_indices = indices
            for axis,count in zip('xyz',(self.vectors.shape[2],self.vectors.shape[1],self.vectors.shape[0])):
                info = self.mesh[axis]
                self.position_labels[axis].set(f'{info["centers"][indices[axis]]:.3f} {info["unit"]} | cell {indices[axis]+1}/{count}')
            low,high = float(self.low.get()),float(self.high.get())
            density,length,head = int(self.density.get()),float(self.length.get()),float(self.head.get())
            if not math.isfinite(low) or not math.isfinite(high) or low>=high or density<1 or not all(math.isfinite(v) and v>0 for v in (length,head)):
                raise ValueError('Use min < max and positive finite arrow settings.')
            top = self.vectors[indices['z']]
            normal = {'XY':'Z','YZ':'X','XZ':'Y'}[self.view.get()]
            bottom,_,_,_ = extract_cross_section(self.vectors,normal,indices[normal.lower()])
            self.last_top_plane,self.last_bottom_plane = top,bottom
            components = ['mx','my','mz'] if self.component.get()=='ALL' else [self.component.get()]
            width,height = self.canvas.winfo_width(),self.canvas.winfo_height()
            width,height = width if width>20 else 700,height if height>20 else 180
            panels = [self.panel(top,'XY',component,'Figure 1: top view',max(60,width//len(components)-50),max(20,height-75),indices,low,high,density,length,head) for component in components]
            image = Image.new('RGB',(sum(p.width for p,r in panels)+8*(len(panels)-1),max(p.height for p,r in panels)),'white')
            rects,x = [],0
            for panel,(a,b,c,d) in panels:
                image.paste(panel,(x,0))
                rects.append((a+x,b,c+x,d))
                x += panel.width+8
            self.top_data_rects,self.bottom_data_rects = rects,[]
            self.top_export_image = self.rendered_image = image
            ratio = min(width/image.width,height/image.height)
            w,h = max(1,round(image.width*ratio)),max(1,round(image.height*ratio))
            self.photo = ImageTk.PhotoImage(image.resize((w,h),Image.Resampling.LANCZOS),master=self.gui.root)
            self.canvas.delete('all')
            self.canvas.create_image(width//2,height//2,image=self.photo,anchor='center')
            self.display_transform = (width//2-w/2,height//2-h/2,w/image.width,h/image.height)
            self.render_scene(bottom,indices,normal,components,low,high,density,length,head)
            self.status.set(f'{self.files[self.loaded_index].name}\nFigure 1: XY at Z cell {indices["z"]+1}\nFigure 2: {self.view.get()} | {self.glyph_count} solid arrows')
        except (ValueError,OSError) as exc:
            self.status.set(f'Cross section: {exc}')

    def render_scene(self,plane,indices,normal,components,low,high,density,length,head):
        old_angles = (self.ax3d.elev,self.ax3d.azim)
        new_view = self.scene_view!=self.view.get()
        self.figure.clear()
        count = len(components)
        self.axes3d = []
        horizontal,vertical = {'X':('y','z'),'Y':('x','z'),'Z':('x','y')}[normal]
        cmap = stream_color_map(self.colors.get())
        norm = Normalize(low,high)
        rows,cols = plane.shape[:2]
        row_edges = np.unique(np.r_[np.arange(0,rows,max(1,math.ceil(rows/32))),rows])
        col_edges = np.unique(np.r_[np.arange(0,cols,max(1,math.ceil(cols/48))),cols])
        h_edges,v_edges = self.mesh[horizontal]['edges'],self.mesh[vertical]['edges']
        hh = h_edges[0]+col_edges/cols*(h_edges[1]-h_edges[0])
        vv = v_edges[0]+row_edges/rows*(v_edges[1]-v_edges[0])
        hs,vs = np.meshgrid(hh,vv)
        coordinates = {horizontal:hs,vertical:vs,normal.lower():np.full_like(hs,self.mesh[normal.lower()]['centers'][indices[normal.lower()]])}
        stride = max(1,round(cols/max(1,density)))
        row_stride = max(1,round(rows/max(1,density)))
        sample_rows = np.arange(0,rows,row_stride)
        sample_cols = np.arange(0,cols,stride)
        while len(sample_rows)*len(sample_cols)>max(1,self.glyph_budget//count):
            if len(sample_cols)>=len(sample_rows):
                sample_cols = sample_cols[::2]
            else:
                sample_rows = sample_rows[::2]
        h_step = abs(h_edges[1]-h_edges[0])/max(1,len(sample_cols))
        v_step = abs(v_edges[1]-v_edges[0])/max(1,len(sample_rows))
        reference_length = max(1e-6,min(h_step,v_step)*.7*length)
        selected = {'mx':0,'my':1,'mz':2}.get(self.arrow_component.get())
        normal_padding = max(reference_length*1.5,max(abs(h_edges[1]-h_edges[0]),abs(v_edges[1]-v_edges[0]))*.035)
        scene_spans = np.array([2*normal_padding if axis==normal.lower() else
                               abs(self.mesh[axis]['edges'][1]-self.mesh[axis]['edges'][0]) for axis in 'xyz'])
        box_aspect = np.ones(3) if self.stretch.get() else scene_spans
        physical_per_display = np.maximum(scene_spans/box_aspect,1e-12)
        visual_spacing = min(h_step/physical_per_display['xyz'.index(horizontal)],
                             v_step/physical_per_display['xyz'.index(vertical)])
        self.glyph_count = 0
        for plot_index,component in enumerate(components,1):
            ax = self.figure.add_subplot(1,count,plot_index,projection='3d')
            self.axes3d.append(ax)
            component_index = {'mx':0,'my':1,'mz':2}[component]
            values = plane[:,:,component_index]
            face_values = values[np.ix_(np.minimum(row_edges[:-1],rows-1),np.minimum(col_edges[:-1],cols-1))]
            ax.plot_surface(coordinates['x'],coordinates['y'],coordinates['z'],facecolors=cmap(norm(face_values)),
                            rstride=1,cstride=1,shade=False,linewidth=0,antialiased=False,alpha=.8)
            polygons,colors = [],[]
            if self.arrows.get():
                for row in sample_rows:
                    for col in sample_cols:
                        vector = plane[row,col].astype(float).copy()
                        if selected is not None:
                            vector[np.arange(3)!=selected] = 0
                        magnitude = float(np.linalg.norm(vector))
                        if magnitude<1e-8:
                            continue
                        origin = np.zeros(3)
                        origin['xyz'.index(horizontal)] = self.mesh[horizontal]['centers'][col]
                        origin['xyz'.index(vertical)] = self.mesh[vertical]['centers'][row]
                        origin['xyz'.index(normal.lower())] = self.mesh[normal.lower()]['centers'][indices[normal.lower()]]
                        arrow_length = reference_length*(magnitude if self.arrow_mode.get()=='Magnitude' else 1.)
                        # Make cone/shaft widths legible under expanded aspect while
                        # preserving the physical start/end points and component direction.
                        display_delta = vector/magnitude*arrow_length/physical_per_display
                        display_length = float(np.linalg.norm(display_delta))
                        radius = max(1e-6,min(display_length*.07*min(head,6),visual_spacing*.18*min(head,3)))
                        faces = solid_arrow_faces(origin/physical_per_display,vector/physical_per_display,
                                                  display_length,radius)
                        faces = [np.asarray(face)*physical_per_display for face in faces]
                        polygons.extend(faces)
                        color = cmap(norm(values[row,col]))
                        # Darken glyphs relative to the plane so their solid geometry remains visible.
                        colors.extend([tuple(np.array(color[:3])*.65)+(1.,)]*len(faces))
                        self.glyph_count += 1
                if polygons:
                    ax.add_collection3d(Poly3DCollection(polygons,facecolors=colors,edgecolors=(.1,.1,.1,.25),linewidths=.15,shade=True))
            self.base_limits = {}
            for axis in 'xyz':
                lo,hi = self.mesh[axis]['edges']
                if axis==normal.lower():
                    center = self.mesh[axis]['centers'][indices[axis]]
                    padding = max(reference_length*1.5,max(abs(h_edges[1]-h_edges[0]),abs(v_edges[1]-v_edges[0]))*.035)
                    lo,hi = center-padding,center+padding
                self.base_limits[axis] = (lo,hi)
                getattr(ax,'set_'+axis+'label')(axis.upper()+' ('+self.mesh[axis]['unit']+')',fontsize=8,labelpad=0)
                axis_object = getattr(ax,axis+'axis')
                axis_object.set_major_locator(MaxNLocator(4))
                formatter = ScalarFormatter(useOffset=False)
                formatter.set_scientific(False)
                axis_object.set_major_formatter(formatter)
            ax.tick_params(labelsize=7,pad=0)
            title = '$m_{'+component[-1]+'}$ | '+self.view.get()
            if getattr(self,'xy_top_camera',False):
                title += f' | Z = {self.mesh["z"]["centers"][indices["z"]]:.3f} {self.mesh["z"]["unit"]}'
            if self.show_time.get():
                title += f' | {self.time_ns:.3f} ns' if self.time_ns is not None else ' | time unavailable'
            ax.set_title(title,fontsize=10,pad=3)
            ax.mouse_init(rotate_btn=2,pan_btn=1,zoom_btn=3)
            if new_view:
                elev,azim = {'YZ':(15,10),'XZ':(15,-80),'XY':(65,-90)}[self.view.get()]
            else:
                elev,azim = old_angles
            ax.view_init(elev=elev,azim=azim)
            if self.colorbar.get():
                fraction = {'Small':.45,'Medium':.7,'Bigger':1.}[self.gui.colorbar_size.get()]
                bottom = {'Top':.9-.76*fraction,'Middle':.52-.38*fraction,'Bottom':.14}[self.gui.colorbar_position.get()]
                left = (plot_index-1)/count+.90/count
                bar_axis = self.figure.add_axes([left,bottom,.018/count,.76*fraction])
                bar = self.figure.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=bar_axis,ticks=[low,(low+high)/2,high])
                bar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f' if self.gui.colorbar_decimals.get() else '%.0f'))
                bar.ax.tick_params(labelsize=7,pad=2)
        self.ax3d = self.axes3d[0]
        self.scene_view = self.view.get()
        self.figure.subplots_adjust(left=.02,right=.78,bottom=.08,top=.90,wspace=.45)
        self.apply_zoom()
        self.plot_canvas.draw_idle()

    def apply_zoom(self):
        if not self.base_limits:
            return
        spans = np.array([self.base_limits[a][1]-self.base_limits[a][0] for a in 'xyz'])
        # Expanded display gives each coordinate dimension equal visual space;
        # tick coordinates stay in nm. Unchecked preserves physical aspect.
        aspect = thin_dimension_aspect(spans,self.stretch.get())
        for ax in self.axes3d:
            for axis in 'xyz':
                lo,hi = self.base_limits[axis]
                center = (lo+hi)/2
                half = (hi-lo)/(2*self.zoom)
                getattr(ax,'set_'+axis+'lim')(center-half,center+half)
            ax.set_box_aspect(aspect)

    def scroll_zoom(self,event):
        if event.inaxes not in getattr(self,'axes3d',[]) or self.base_limits is None:
            return
        factor = 1.2 if event.button=='up' or getattr(event,'step',0)>0 else 1/1.2
        self.zoom = min(30.,max(.1,self.zoom*factor))
        self.apply_zoom()
        self.plot_canvas.draw_idle()

    def reset_view(self):
        self.zoom = 1.
        self.scene_view = None
        self.schedule_render()

    def camera_key(self,event):
        key = event.key or ''
        if key.lower()=='r':
            self.reset_view()
        elif key.lower() in 'xyz' and len(key)==1 and self.base_limits is not None:
            elev,azim = {'x':(0,0),'y':(0,90),'z':(90,-90)}[key.lower()]
            if key.isupper():
                elev,azim = (-90,-90) if key=='Z' else (elev,azim+180)
            for ax in self.axes3d:
                ax.view_init(elev=elev,azim=azim)
            self.plot_canvas.draw_idle()

    def export_to_folder(self,folder):
        if self.top_export_image is None or self.vectors is None:
            raise ValueError('Load a readable frame first.')
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError('Choose an existing export folder.')
        stem = safe_name(self.files[self.loaded_index].stem)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        fig1 = folder/f'{stem}_figure1_XY_{stamp}.png'
        fig2 = folder/f'{stem}_figure2_{self.view.get()}_{stamp}.png'
        image = self.top_export_image
        image.resize((image.width*3,image.height*3),Image.Resampling.LANCZOS).save(fig1,dpi=(300,300))
        self.figure.savefig(fig2,format='png',dpi=300,facecolor=self.figure.get_facecolor())
        return fig1,fig2

    def export_figures(self):
        if self.top_export_image is None:
            self.status.set('Load a readable frame first.');return
        if self.loading or self.refining:self.status.set('Wait for the current frame to finish loading.');return
        parent = filedialog.askdirectory(parent=self.gui.root,title='Choose output folder for Figure 1 and Figure 2 PNGs')
        if not parent:return
        folder = None
        try:
            folder = auto_export_folder(parent,3,self.files[self.loaded_index],'Images')
            paths = self.export_to_folder(folder)
            self.status.set('Saved two separate 300 PPI PNGs in '+str(folder))
            open_folder(folder)
        except Exception as exc:
            discard_empty_export_folder(folder)
            self.status.set(f'Image export failed: {exc}')

    def close(self):
        super().close()
        for callback in self.mpl_callbacks:
            self.plot_canvas.mpl_disconnect(callback)
        idle = getattr(self.plot_canvas,'_idle_draw_id',None)
        if idle:
            self.plot_canvas.get_tk_widget().after_cancel(idle)
        self.figure.clear()


class ROICrossSectionBase(InteractiveCrossSectionBase):
    """ROI-linked slices and cursor-anchored 3D zoom; the overview stays undistorted."""
    def __init__(self,gui,parent):
        self.roi = None
        self.roi_selecting = False
        self.roi_drag = None
        self.roi_overlay = None
        self.camera_limits = None
        self.camera_base_centers = None
        super().__init__(gui,parent)
        self.roi_enabled = tk.BooleanVar(value=False)
        group = self.group('Region of interest (Figure 1)')
        # Keep ROI near the slice controls rather than at the bottom of the sidebar.
        groups = self.controls.winfo_children()
        if len(groups)>2:
            group.pack_configure(before=groups[2])
        ttk.Checkbutton(group,text='Enable ROI',variable=self.roi_enabled,command=self.toggle_roi).pack(anchor='w')
        ttk.Button(group,text='Select / change ROI',command=self.select_roi).pack(fill='x',pady=2)
        ttk.Button(group,text='Clear ROI',command=self.clear_roi).pack(fill='x',pady=2)
        ttk.Label(group,text='Drag a rectangle on Figure 1. Slice analysis uses only its X/Y region, through all Z layers. Figure 1 keeps its physical aspect.',wraplength=235).pack(anchor='w')

    def panel(self,plane,view,component,title,width,height,indices,low,high,density,length,head):
        # Expand thin dimension is exclusively a Figure 2 setting.
        previous = self.stretch.get()
        try:
            self.stretch.set(False)
            return super().panel(plane,view,component,title,width,height,indices,low,high,density,length,head)
        finally:
            self.stretch.set(previous)

    def load_folder(self):
        self.roi = None
        self.roi_drag = None
        self.camera_limits = self.camera_base_centers = None
        self.roi_selecting = self.roi_enabled.get()
        super().load_folder()

    def valid_roi(self):
        if not self.roi_enabled.get() or self.roi is None or self.vectors is None:
            return None
        nx,ny = self.vectors.shape[2],self.vectors.shape[1]
        x0,x1,y0,y1 = self.roi
        x0,x1 = max(0,min(nx-1,x0)),max(1,min(nx,x1))
        y0,y1 = max(0,min(ny-1,y0)),max(1,min(ny,y1))
        if x0>=x1 or y0>=y1:
            return None
        return x0,x1,y0,y1

    def indices(self):
        result = super().indices()
        roi = self.valid_roi()
        if roi:
            x0,x1,y0,y1 = roi
            result['x'] = min(x1-1,max(x0,result['x']))
            result['y'] = min(y1-1,max(y0,result['y']))
        return result

    def update_slice(self):
        super().update_slice()
        if self.vectors is None:
            return
        roi = self.valid_roi()
        if roi:
            x0,x1,y0,y1 = roi
            self.updating = True
            for axis,low,high in (('x',x0+1,x1),('y',y0+1,y1)):
                self.position_sliders[axis].configure(from_=low,to=high)
                self.positions[axis].set(min(high,max(low,self.positions[axis].get())))
            self.updating = False

    def toggle_roi(self):
        self.roi_selecting = self.roi_enabled.get() and self.roi is None
        self.camera_limits = self.camera_base_centers = None
        self.zoom = 1.
        self.scene_view = None
        self.update_slice()
        if self.roi_selecting:
            self.status.set('ROI enabled: left-drag a rectangle inside Figure 1.')

    def select_roi(self):
        self.roi_enabled.set(True)
        self.roi_selecting = True
        self.status.set('Left-drag a rectangle inside Figure 1 to select the ROI.')

    def clear_roi(self):
        self.roi = self.roi_drag = None
        self.roi_enabled.set(False)
        self.roi_selecting = False
        self.camera_limits = self.camera_base_centers = None
        self.zoom = 1.
        self.scene_view = None
        self.update_slice()

    def reference_lines(self,image,view,indices):
        image = super().reference_lines(image,view,indices)
        roi = self.valid_roi()
        if view=='XY' and roi:
            x0,x1,y0,y1 = roi
            nx,ny = self.vectors.shape[2],self.vectors.shape[1]
            rectangle = (x0*image.width/nx,(ny-y1)*image.height/ny,
                         min(image.width-1,x1*image.width/nx),min(image.height-1,(ny-y0)*image.height/ny))
            draw = ImageDraw.Draw(image)
            draw.rectangle(rectangle,outline='black',width=3)
            draw.rectangle(rectangle,outline=(255,0,255),width=1)
        return image

    def drag_start(self,event):
        if not self.roi_selecting:
            return super().drag_start(event)
        point = self.pointer_in_image(event)
        if point is None or self.vectors is None:
            return
        for rect in self.top_data_rects:
            a,b,c,d = rect
            if a<=point[0]<c and b<=point[1]<d:
                self.roi_drag = (rect,point,point)
                self.canvas.configure(cursor='crosshair')
                return

    def drag_move(self,event):
        if self.roi_drag is None:
            return super().drag_move(event)
        point = self.pointer_in_image(event)
        if point is None:
            return
        rect,start,_ = self.roi_drag
        a,b,c,d = rect
        end = (min(c,max(a,point[0])),min(d,max(b,point[1])))
        self.roi_drag = (rect,start,end)
        ox,oy,sx,sy = self.display_transform
        if self.roi_overlay is not None:
            self.canvas.delete(self.roi_overlay)
        self.roi_overlay = self.canvas.create_rectangle(ox+start[0]*sx,oy+start[1]*sy,
                              ox+end[0]*sx,oy+end[1]*sy,outline='#ff00ff',width=2,dash=(4,2))

    def drag_end(self,event):
        if self.roi_drag is None:
            return super().drag_end(event)
        self.drag_move(event)
        (a,b,c,d),start,end = self.roi_drag
        nx,ny = self.vectors.shape[2],self.vectors.shape[1]
        def cell(point):
            return (min(nx-1,max(0,int((point[0]-a)/(c-a)*nx))),
                    min(ny-1,max(0,int((1-(point[1]-b)/(d-b))*ny))))
        x0,y0 = cell(start)
        x1,y1 = cell(end)
        self.roi = min(x0,x1),max(x0,x1)+1,min(y0,y1),max(y0,y1)+1
        self.roi_drag = None
        self.roi_selecting = False
        self.roi_overlay = None
        self.canvas.configure(cursor='')
        self.camera_limits = self.camera_base_centers = None
        self.scene_view = None
        self.zoom = 1.
        self.update_slice()

    def render_scene(self,plane,indices,normal,components,low,high,density,length,head):
        original_mesh = self.mesh
        roi = self.valid_roi()
        if roi:
            x0,x1,y0,y1 = roi
            if normal=='X':
                plane = plane[:,y0:y1,:]
                slices = {'y':(y0,y1)}
            elif normal=='Y':
                plane = plane[:,x0:x1,:]
                slices = {'x':(x0,x1)}
            else:
                plane = plane[y0:y1,x0:x1,:]
                slices = {'x':(x0,x1),'y':(y0,y1)}
            self.mesh = dict(original_mesh)
            for axis,(first,last) in slices.items():
                info = original_mesh[axis]
                step = (info['edges'][1]-info['edges'][0])/len(info['centers'])
                self.mesh[axis] = dict(info,centers=info['centers'][first:last],
                                      edges=(info['edges'][0]+first*step,info['edges'][0]+last*step))
        self.last_analysis_plane = plane
        self.last_analysis_mesh = self.mesh
        if self.camera_limits is not None and self.scene_view==self.view.get():
            # Keep manual panning/rotation and cursor-centered zoom across refreshes.
            self.camera_limits = {a:getattr(self.ax3d,'get_'+a+'lim')() for a in 'xyz'}
        try:
            super().render_scene(plane,indices,normal,components,low,high,density,length,head)
        finally:
            self.mesh = original_mesh

    def apply_zoom(self):
        if not self.base_limits:
            return
        centers = {a:sum(self.base_limits[a])/2 for a in 'xyz'}
        if self.camera_limits is None:
            super().apply_zoom()
        else:
            normal = {'XY':'z','YZ':'x','XZ':'y'}[self.view.get()]
            if self.camera_base_centers is not None:
                shift = centers[normal]-self.camera_base_centers[normal]
                self.camera_limits[normal] = tuple(v+shift for v in self.camera_limits[normal])
            spans = np.array([self.base_limits[a][1]-self.base_limits[a][0] for a in 'xyz'])
            for ax in self.axes3d:
                for axis in 'xyz':
                    getattr(ax,'set_'+axis+'lim')(*self.camera_limits[axis])
                ax.set_box_aspect(thin_dimension_aspect(spans,self.stretch.get()))
        self.camera_base_centers = centers

    def cursor_anchor(self,event):
        """Intersect the mouse's projected camera ray with the selected slice plane."""
        from mpl_toolkits.mplot3d import proj3d
        ax = event.inaxes
        if getattr(event,'xdata',None) is None or getattr(event,'ydata',None) is None:
            return np.array([sum(getattr(ax,'get_'+a+'lim')())/2 for a in 'xyz'])
        inverse = np.linalg.inv(ax.get_proj())
        near = np.array(proj3d.inv_transform(event.xdata,event.ydata,-1,inverse))
        far = np.array(proj3d.inv_transform(event.xdata,event.ydata,1,inverse))
        normal = {'XY':2,'YZ':0,'XZ':1}[self.view.get()]
        coordinate = self.mesh['xyz'[normal]]['centers'][self.indices()['xyz'[normal]]]
        direction = far-near
        if abs(direction[normal])<1e-12:
            return np.array([sum(getattr(ax,'get_'+a+'lim')())/2 for a in 'xyz'])
        anchor = near+direction*((coordinate-near[normal])/direction[normal])
        if not np.isfinite(anchor).all():
            return np.array([sum(getattr(ax,'get_'+a+'lim')())/2 for a in 'xyz'])
        return anchor

    def scroll_zoom(self,event):
        if event.inaxes not in getattr(self,'axes3d',[]) or self.base_limits is None:
            return
        anchor = self.cursor_anchor(event)
        factor = 1.2 if event.button=='up' or getattr(event,'step',0)>0 else 1/1.2
        new_zoom = min(30.,max(.1,self.zoom*factor))
        factor = new_zoom/self.zoom
        self.zoom = new_zoom
        self.camera_limits = {}
        for index,axis in enumerate('xyz'):
            low,high = getattr(event.inaxes,'get_'+axis+'lim')()
            self.camera_limits[axis] = (anchor[index]+(low-anchor[index])/factor,
                                        anchor[index]+(high-anchor[index])/factor)
        self.apply_zoom()
        self.plot_canvas.draw_idle()

    def reset_view(self):
        self.camera_limits = self.camera_base_centers = None
        super().reset_view()

    def view_changed(self):
        self.camera_limits = self.camera_base_centers = None
        super().view_changed()



# Optional acceleration; the NumPy renderer remains available in any kernel.
try:
    from numba import njit as _mumax_njit
except Exception:
    _mumax_njit = None
_MUMAX_FAST_KERNEL = None
_MUMAX_WARMING = False
_MUMAX_KERNEL_ERROR = None


def _mumax_hit_scalar(ox,oy,oz,dx,dy,dz,cx,cy,cz,ux,uy,uz,length,radius,head_radius,ratio):
    vx,vy,vz = ox-cx+ux*length/2,oy-cy+uy*length/2,oz-cz+uz*length/2
    so,sd = vx*ux+vy*uy+vz*uz,dx*ux+dy*uy+dz*uz
    px,py,pz = vx-so*ux,vy-so*uy,vz-so*uz
    qx,qy,qz = dx-sd*ux,dy-sd*uy,dz-sd*uz
    a = qx*qx+qy*qy+qz*qz
    b = 2*(px*qx+py*qy+pz*qz)
    c = px*px+py*py+pz*pz
    bound = max(radius,head_radius)
    if (a>1e-15 and b*b-4*a*(c-bound*bound)<0) or (a<=1e-15 and c>bound*bound):
        return np.inf,0.,0.,0.
    best,nx,ny,nz = np.inf,0.,0.,0.
    shoulder = length*(1-ratio)
    k2 = (head_radius/max(length*ratio,1e-15))**2
    for surface in range(2):
        if surface==0:
            aa,bb,cc = a,b,c-radius*radius
        else:
            q = length-so
            aa,bb,cc = a-k2*sd*sd,b+2*k2*q*sd,c-k2*q*q
        disc = bb*bb-4*aa*cc
        if disc<0:continue
        if abs(aa)>1e-16:
            sq = math.sqrt(max(0.,disc))
            t1,t2 = (-bb-sq)/(2*aa),(-bb+sq)/(2*aa)
        elif abs(bb)>1e-16:
            t1,t2 = -cc/bb,-cc/bb
        else:
            continue
        for which in range(2):
            t = t1 if which==0 else t2
            s = so+t*sd
            valid = (0<=s<=shoulder) if surface==0 else (shoulder<=s<=length)
            if valid and 1e-10<t<best:
                best = t
                nx,ny,nz = px+t*qx,py+t*qy,pz+t*qz
                if surface==1:
                    cone = k2*(length-s)
                    nx,ny,nz = nx+cone*ux,ny+cone*uy,nz+cone*uz
    if abs(sd)>1e-14:
        for cap in range(2):
            s = 0. if cap==0 else shoulder
            r = radius if cap==0 else head_radius
            t = (s-so)/sd
            rx,ry,rz = px+t*qx,py+t*qy,pz+t*qz
            if 1e-10<t<best and rx*rx+ry*ry+rz*rz<=r*r:
                best,nx,ny,nz = t,-ux,-uy,-uz
    norm = max(math.sqrt(nx*nx+ny*ny+nz*nz),1e-15)
    return best,nx/norm,ny/norm,nz/norm


def _mumax_pixels_compiled(image,depth,centers,units,lengths,radii,head_radii,colors,
                          ratio,ambient,half,is_cube,lowx,lowy,highx,highy,visible,
                          inverse,metric,xstart,ystart,xstep,ystep,perspective,light_key,light_fill,light_intensity,specular=0.,specular_power=40.):
    height,width = image.shape[:2]
    if perspective:
        eye_x = inverse[0,2]/inverse[3,2]/metric[0]
        eye_y = inverse[1,2]/inverse[3,2]/metric[1]
        eye_z = inverse[2,2]/inverse[3,2]/metric[2]
    else:
        direction_x,direction_y,direction_z = inverse[0,2]/metric[0],inverse[1,2]/metric[1],inverse[2,2]/metric[2]
        norm = math.sqrt(direction_x**2+direction_y**2+direction_z**2)
        direction_x,direction_y,direction_z = direction_x/norm,direction_y/norm,direction_z/norm
    for glyph in visible:
        cx,cy,cz = centers[glyph]
        ux,uy,uz = units[glyph]
        for y in range(lowy[glyph],highy[glyph]):
            yd = ystart+y*ystep
            for x in range(lowx[glyph],highx[glyph]):
                xd = xstart+x*xstep
                plane = -1. if perspective else -10.
                w = xd*inverse[3,0]+yd*inverse[3,1]+plane*inverse[3,2]+inverse[3,3]
                wx = (xd*inverse[0,0]+yd*inverse[0,1]+plane*inverse[0,2]+inverse[0,3])/w/metric[0]
                wy = (xd*inverse[1,0]+yd*inverse[1,1]+plane*inverse[1,2]+inverse[1,3])/w/metric[1]
                wz = (xd*inverse[2,0]+yd*inverse[2,1]+plane*inverse[2,2]+inverse[2,3])/w/metric[2]
                if perspective:
                    ox,oy,oz = eye_x,eye_y,eye_z
                    dx,dy,dz = wx-ox,wy-oy,wz-oz
                    norm = max(math.sqrt(dx*dx+dy*dy+dz*dz),1e-15)
                    dx,dy,dz = dx/norm,dy/norm,dz/norm
                else:
                    ox,oy,oz = wx,wy,wz
                    dx,dy,dz = direction_x,direction_y,direction_z
                if is_cube:
                    entry,exit = -np.inf,np.inf
                    en_axis,ex_axis,en_sign,ex_sign = 0,0,0.,0.
                    hit = True
                    for axis in range(3):
                        o = ox-cx if axis==0 else oy-cy if axis==1 else oz-cz
                        d = dx if axis==0 else dy if axis==1 else dz
                        if abs(d)<1e-14:
                            if abs(o)>half[axis]:hit = False
                            continue
                        a,b = (-half[axis]-o)/d,(half[axis]-o)/d
                        near,far = min(a,b),max(a,b)
                        sign = 1. if d>=0 else -1.
                        if near>entry:entry,en_axis,en_sign = near,axis,-sign
                        if far<exit:exit,ex_axis,ex_sign = far,axis,sign
                    t = exit if entry<0 else entry
                    axis = ex_axis if entry<0 else en_axis
                    sign = ex_sign if entry<0 else en_sign
                    nx,ny,nz = (sign if axis==0 else 0.),(sign if axis==1 else 0.),(sign if axis==2 else 0.)
                    if not hit or exit<entry or t<=0:continue
                else:
                    t,nx,ny,nz = _mumax_hit_jit(ox,oy,oz,dx,dy,dz,cx,cy,cz,ux,uy,uz,
                                               lengths[glyph],radii[glyph],head_radii[glyph],ratio)
                pixel = y*width+x
                if not np.isfinite(t) or t<=0 or t>=depth[pixel]:continue
                depth[pixel] = t
                key = max(0.,nx*light_key[0]+ny*light_key[1]+nz*light_key[2])
                fill = max(0.,nx*light_fill[0]+ny*light_fill[1]+nz*light_fill[2])
                light = ambient+(1-ambient)*(.1+.75*key+.15*fill)*light_intensity
                hx,hy,hz=light_key[0]-dx,light_key[1]-dy,light_key[2]-dz
                hn=max(math.sqrt(hx*hx+hy*hy+hz*hz),1e-15)
                shine=max(0.,(nx*hx+ny*hy+nz*hz)/hn)**specular_power
                highlight=specular*shine*light_intensity
                for channel in range(3):
                    image[y,x,channel] = min(255,max(0,int((colors[glyph,channel]*light+highlight)*255)))
                image[y,x,3] = min(255,max(0,int(colors[glyph,3]*255)))


if _mumax_njit is not None:
    _mumax_hit_jit = _mumax_njit(inline='always')(_mumax_hit_scalar)
    _mumax_render_jit = _mumax_njit(nogil=True)(_mumax_pixels_compiled)


def warm_mumax_renderer():
    """Compile once in a background thread; keep the GUI available meanwhile."""
    global _MUMAX_WARMING
    if _mumax_njit is None or _MUMAX_WARMING or _MUMAX_FAST_KERNEL is not None:return
    # Flush retired Tk widgets/images on the GUI thread before background JIT work.
    import gc
    gc.collect()
    _MUMAX_WARMING = True
    def worker():
        global _MUMAX_FAST_KERNEL,_MUMAX_KERNEL_ERROR,_MUMAX_WARMING
        try:
            _mumax_render_jit(np.zeros((2,2,4),np.uint8),np.full(4,np.inf,np.float32),
               np.zeros((1,3)),np.array([[0.,0.,1.]]),np.ones(1),np.full(1,.1),np.full(1,.2),
               np.ones((1,4)),.3,.45,np.ones(3),False,np.zeros(1,np.int64),np.zeros(1,np.int64),
               np.ones(1,np.int64),np.ones(1,np.int64),np.zeros(1,np.int64),np.eye(4),np.ones(3),0.,0.,1.,1.,False,np.array([1.,-1.,2.])/math.sqrt(6.),np.array([-1.,-.5,1.])/1.5,1.,0.,40.)
            _MUMAX_FAST_KERNEL = _mumax_render_jit
        except Exception as exc:
            _MUMAX_KERNEL_ERROR = str(exc)
        finally:
            _MUMAX_WARMING = False
    threading.Thread(target=worker,daemon=True).start()


from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.projections import register_projection


class FullViewportAxes3D(Axes3D):
    """Rectangular 3D viewport with equal screen units and automatic initial fit."""
    name = 'mumax_full_3d'

    def apply_aspect(self,position=None):
        self._set_position(self.get_position(original=True) if position is None else position,'active')

    def draw(self,renderer):
        from matplotlib.ticker import NullLocator
        # Z projects to a point in the exact XY view; avoid stacked tick labels.
        top = abs(np.cos(np.deg2rad(self.elev)))<1e-3
        saved = getattr(self,'_hidden_z_scale',None)
        if top and saved is None:
            self._hidden_z_scale = (self.zaxis.get_major_locator(),self.get_zlabel())
            self.zaxis.set_major_locator(NullLocator())
            self.set_zlabel('')
        elif not top and saved is not None:
            self.zaxis.set_major_locator(saved[0])
            self.set_zlabel(saved[1])
            self._hidden_z_scale = None
        return super().draw(renderer)

    def get_proj(self):
        matrix = super().get_proj()
        width,height = max(1,self.bbox.width),max(1,self.bbox.height)
        matrix[0] *= height/width
        pinned = getattr(self,'_camera_fit_scale',None)
        if pinned is not None:
            matrix[:2] *= pinned
            return matrix
        key = (round(width),round(height),tuple(np.round(self._box_aspect,5)))
        if getattr(self,'_fit_key',None)!=key:
            self._fit_key = key
            corners = np.array([[x,y,z,1.] for x in self.get_xlim3d()
                                for y in self.get_ylim3d() for z in self.get_zlim3d()])
            clip = corners@matrix.T
            xy = clip[:,:2]/clip[:,3,None]
            pixel = self.transData.transform(xy)
            span = np.ptp(pixel,axis=0)
            self._fit_scale = min(3.,.88*width/max(span[0],1),.85*height/max(span[1],1))
        matrix[:2] *= self._fit_scale
        return matrix


register_projection(FullViewportAxes3D)


def thin_dimension_aspect(spans,expand):
    result = np.maximum(np.asarray(spans,dtype=float),1e-15).copy()
    if expand:
        result[2] = max(result[2],min(result[0],result[1]))
    return result


def ray_solid_hits(origins,directions,centers,units,lengths,radii,head_radii,head_ratio,
                   cuboid_half=None):
    """Analytic solid intersections. All inputs use the same display metric."""
    count = len(origins)
    best = np.full(count,np.inf)
    normals = np.zeros((count,3))
    if cuboid_half is not None:
        v = origins-centers
        safe = np.where(abs(directions)>1e-14,directions,1e-14)
        a,b = (-cuboid_half-v)/safe,(cuboid_half-v)/safe
        near,far = np.minimum(a,b),np.maximum(a,b)
        entry,exit = near.max(axis=1),far.min(axis=1)
        inside = entry<0
        t = np.where(inside,exit,entry)
        good = (exit>=entry)&(t>0)&np.isfinite(t)
        axis = np.where(inside,far.argmin(axis=1),near.argmax(axis=1))
        normals[np.arange(count),axis] = np.where(inside,1.,-1.)*np.sign(directions[np.arange(count),axis])
        best[good] = t[good]
        return best,normals
    v = origins-(centers-units*lengths[:,None]/2)
    axial_o = np.sum(v*units,axis=1)
    axial_d = np.sum(directions*units,axis=1)
    radial_o = v-axial_o[:,None]*units
    radial_d = directions-axial_d[:,None]*units
    aa = np.sum(radial_d**2,axis=1)
    bb = 2*np.sum(radial_o*radial_d,axis=1)
    cc = np.sum(radial_o**2,axis=1)
    # Reject rays outside the enclosing cylinder before solving every surface.
    bound_r = np.maximum(radii,head_radii)
    disc = bb*bb-4*aa*(cc-bound_r*bound_r)
    possible = np.where(aa>1e-15,disc>=0,cc<=bound_r*bound_r)
    keep = np.flatnonzero(possible)
    if not len(keep):return best,normals
    full_best,full_normals = best,normals
    if len(keep)<count:
        units,lengths,radii,head_radii = units[keep],lengths[keep],radii[keep],head_radii[keep]
        radial_o,radial_d = radial_o[keep],radial_d[keep]
        axial_o,axial_d = axial_o[keep],axial_d[keep]
        aa,bb,cc = aa[keep],bb[keep],cc[keep]
        count = len(keep)
        best,normals = np.full(count,np.inf),np.zeros((count,3))
    shoulder = lengths*(1-head_ratio)

    def accept(t,good,normal):
        valid = good&(t>1e-10)&(t<best)&np.isfinite(t)
        if np.any(valid):
            best[valid] = t[valid]
            normals[valid] = normal[valid]

    def roots(a,b,c):
        disc = b*b-4*a*c
        good = disc>=0
        sqrt = np.sqrt(np.maximum(disc,0))
        safe = np.where(abs(a)>1e-16,2*a,1.)
        linear = -c/np.where(abs(b)>1e-16,b,1.)
        return [np.where(abs(a)>1e-16,(-b-sqrt)/safe,linear),
                np.where(abs(a)>1e-16,(-b+sqrt)/safe,linear)],good&((abs(a)>1e-16)|(abs(b)>1e-16))

    candidates,real = roots(aa,bb,cc-radii*radii)
    for t in candidates:
        s = axial_o+t*axial_d
        accept(t,real&(s>=0)&(s<=shoulder),radial_o+t[:,None]*radial_d)
    k2 = (head_radii/np.maximum(lengths*head_ratio,1e-15))**2
    q = lengths-axial_o
    candidates,real = roots(aa-k2*axial_d**2,bb+2*k2*q*axial_d,cc-k2*q*q)
    for t in candidates:
        s = axial_o+t*axial_d
        normal = radial_o+t[:,None]*radial_d+(k2*(lengths-s))[:,None]*units
        accept(t,real&(s>=shoulder)&(s<=lengths),normal)
    safe_d = np.where(abs(axial_d)>1e-14,axial_d,1.)
    for s,r in ((np.zeros(count),radii),(shoulder,head_radii)):
        t = (s-axial_o)/safe_d
        radial = radial_o+t[:,None]*radial_d
        accept(t,(abs(axial_d)>1e-14)&(np.sum(radial**2,axis=1)<=r*r),-units)
    normals /= np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-15)
    if len(keep)<len(full_best):
        full_best[keep],full_normals[keep] = best,normals
        return full_best,full_normals
    return best,normals


class SmoothSolidGlyphArtist(Artist):
    """Depth-buffered, smoothly lit cylinders/cones; work scales with visible pixels."""
    def __init__(self,ax,centers,vectors,lengths,metric,colors,shaft,head,head_ratio,
                 ambient,quality,cuboid_dimensions=None):
        super().__init__()
        self.ax = ax
        self.centers = np.asarray(centers,dtype=float)
        self.metric = np.asarray(metric,dtype=float)
        direction = np.asarray(vectors,dtype=float)/self.metric
        self.units = direction/np.maximum(np.linalg.norm(direction,axis=1)[:,None],1e-15)
        self.lengths = np.asarray(lengths,dtype=float)
        self.radii = np.asarray(shaft,dtype=float)
        self.head_radii = np.asarray(head,dtype=float)
        self.head_ratio,self.ambient = head_ratio,ambient
        self.colors = np.asarray(colors,dtype=float)
        self.quality = quality
        self.interactive = False
        self.half = None if cuboid_dimensions is None else np.asarray(cuboid_dimensions,dtype=float)/self.metric/2
        self.cache_key,self.cached = None,None
        self.last_draw_seconds = 0.
        self.set_zorder(3)
        self.set_clip_on(True)

    def raster(self,width,height,matrix,box):
        # Pixel row zero is the bottom of the Matplotlib display viewport.
        depth = np.full(width*height,np.inf,dtype=np.float32)
        if getattr(self,'fill_planes',None):self.fill_depth=depth
        image = np.zeros((width*height,4),dtype=np.uint8)
        specular,power,opacity=glyph_reflection_settings(self)
        translucent=opacity<1.
        self._cpu_transparency_composited=translucent
        fragment_pixels,fragment_depth,fragment_colors=[],[],[]
        if not len(self.centers):
            if translucent:
                append_cpu_layer_fragments(self,fragment_pixels,fragment_depth,fragment_colors,width,height,matrix,box)
                return composite_cpu_fragments(fragment_pixels,fragment_depth,fragment_colors,width,height)
            return image.reshape(height,width,4)
        if self.half is None:
            # Circumscribed rings give much tighter bounds than a world-space box.
            reference = np.zeros_like(self.units)
            reference[np.arange(len(reference)),np.argmin(abs(self.units),axis=1)] = 1.
            basis = np.cross(self.units,reference)
            basis /= np.maximum(np.linalg.norm(basis,axis=1)[:,None],1e-15)
            other = np.cross(self.units,basis)
            angles = np.arange(16)*2*np.pi/16
            rings = basis[:,None,:]*np.cos(angles)[None,:,None]+other[:,None,:]*np.sin(angles)[None,:,None]
            start = self.centers/self.metric-self.units*self.lengths[:,None]/2
            shoulder = start+self.units*(self.lengths*(1-self.head_ratio))[:,None]
            tip = start+self.units*self.lengths[:,None]
            pad = 1/np.cos(np.pi/16)
            corners = np.concatenate((start[:,None,:]+rings*(self.radii*pad)[:,None,None],
                       shoulder[:,None,:]+rings*(np.maximum(self.radii,self.head_radii)*pad)[:,None,None],
                       tip[:,None,:]),axis=1)*self.metric
        else:
            signs = np.array([[a,b,c] for a in (-1,1) for b in (-1,1) for c in (-1,1)])
            corners = self.centers[:,None,:]+(self.half*self.metric)[None,None,:]*signs
        homogeneous = np.concatenate((corners,np.ones((*corners.shape[:2],1))),axis=2)
        clip = homogeneous@matrix.T
        projected = clip[:,:,:2]/clip[:,:,3,None]
        # transData is affine; convert projected coordinates to display pixels.
        trans = self.ax.transData.get_affine().get_matrix()
        px = projected[:,:,0]*trans[0,0]+trans[0,2]
        py = projected[:,:,1]*trans[1,1]+trans[1,2]
        px = (px-box.x0)*width/box.width
        py = (py-box.y0)*height/box.height
        lowx = np.clip(np.floor(px.min(axis=1))-1,0,width).astype(int)
        lowy = np.clip(np.floor(py.min(axis=1))-1,0,height).astype(int)
        highx = np.clip(np.ceil(px.max(axis=1))+1,0,width).astype(int)
        highy = np.clip(np.ceil(py.max(axis=1))+1,0,height).astype(int)
        sizes = (highx-lowx)*(highy-lowy)
        visible = np.flatnonzero((sizes>0)&np.isfinite(projected).all(axis=(1,2)))
        inverse = np.linalg.inv(matrix)
        inv_data = self.ax.transData.inverted().get_affine().get_matrix()
        lighting=artist_lighting(self)
        key,fill=lighting['directions'];intensity=lighting['intensity']
        if _MUMAX_FAST_KERNEL is not None and not translucent:
            self.last_engine = 'compiled'
            xstart = (box.x0+.5*box.width/width)*inv_data[0,0]+inv_data[0,2]
            ystart = (box.y0+.5*box.height/height)*inv_data[1,1]+inv_data[1,2]
            _MUMAX_FAST_KERNEL(image.reshape(height,width,4),depth,self.centers/self.metric,self.units,
                self.lengths,self.radii,self.head_radii,self.colors,float(self.head_ratio),float(self.ambient),
                np.ones(3) if self.half is None else self.half,self.half is not None,
                lowx,lowy,highx,highy,visible,inverse,self.metric,xstart,ystart,
                box.width/width*inv_data[0,0],box.height/height*inv_data[1,1],not np.isinf(self.ax._focal_length),key,fill,intensity,specular,power)
            return image.reshape(height,width,4)
        self.last_engine = 'numpy'
        # Bound all intermediate arrays, including close-up/high-PPI exports.
        batch_budget = 120000
        batches = []
        current,total = [],0
        for glyph in visible:
            if total+sizes[glyph]>batch_budget and current:
                batches.append(current);current,total = [],0
            current.append(glyph);total += sizes[glyph]
        if current:batches.append(current)
        for batch in batches:
            # Giant single-glyph rectangles are divided into row strips.
            pieces = []
            for glyph in batch:
                w = highx[glyph]-lowx[glyph]
                rows = max(1,batch_budget//max(1,w))
                for first in range(lowy[glyph],highy[glyph],rows):
                    pieces.append((glyph,lowx[glyph],highx[glyph],first,min(highy[glyph],first+rows)))
            # Normal batches share one vectorized intersection operation.
            work = [pieces] if len(pieces)==len(batch) else [[piece] for piece in pieces]
            for group in work:
                xs,ys,ids = [],[],[]
                for glyph,x0,x1,y0,y1 in group:
                    yy,xx = np.mgrid[y0:y1,x0:x1]
                    xs.append(xx.ravel());ys.append(yy.ravel())
                    ids.append(np.full(xx.size,glyph,dtype=int))
                xx,yy,gids = np.concatenate(xs),np.concatenate(ys),np.concatenate(ids)
                screen_x = box.x0+(xx+.5)*box.width/width
                screen_y = box.y0+(yy+.5)*box.height/height
                xd = screen_x*inv_data[0,0]+inv_data[0,2]
                yd = screen_y*inv_data[1,1]+inv_data[1,2]
                # Avoid threaded BLAS for tiny 4-column ray transforms.
                if np.isinf(self.ax._focal_length):
                    near = xd[:,None]*inverse[:,0]+yd[:,None]*inverse[:,1]-10*inverse[:,2]+inverse[:,3]
                    origins = near[:,:3]/near[:,3,None]/self.metric
                    direction = inverse[:3,2]/self.metric
                    direction /= max(np.linalg.norm(direction),1e-15)
                    directions = np.broadcast_to(direction,origins.shape)
                else:
                    eye = inverse[:3,2]/inverse[3,2]/self.metric
                    point = xd[:,None]*inverse[:,0]+yd[:,None]*inverse[:,1]-inverse[:,2]+inverse[:,3]
                    directions = point[:,:3]/point[:,3,None]/self.metric-eye
                    directions /= np.maximum(np.linalg.norm(directions,axis=1)[:,None],1e-15)
                    origins = np.broadcast_to(eye,directions.shape)
                centers = self.centers[gids]/self.metric
                half = None if self.half is None else np.broadcast_to(self.half,(len(gids),3))
                t,normals = ray_solid_hits(origins,directions,centers,self.units[gids],self.lengths[gids],
                                          self.radii[gids],self.head_radii[gids],self.head_ratio,half)
                good = np.isfinite(t)
                if not np.any(good):continue
                pixel = (yy*width+xx)[good]
                td = t[good].astype(np.float32)
                if translucent:
                    gids,normals,view_rays=gids[good],normals[good],directions[good]
                else:
                    np.minimum.at(depth,pixel,td)
                    winner = td<=depth[pixel]
                    view_rays=directions[good][winner]
                    pixel,gids,normals = pixel[winner],gids[good][winner],normals[good][winner]
                brightness = self.ambient+(1-self.ambient)*(.1+.75*np.maximum(np.sum(normals*key,axis=1),0)+.15*np.maximum(np.sum(normals*fill,axis=1),0))*intensity
                rgba = self.colors[gids].copy()
                halfway=key-view_rays
                halfway/=np.maximum(np.linalg.norm(halfway,axis=1)[:,None],1e-15)
                shine=np.maximum(np.sum(normals*halfway,axis=1),0)**power
                rgba[:,:3]=np.clip(rgba[:,:3]*brightness[:,None]+(specular*shine*intensity)[:,None],0,1)
                rgba[:,3]*=opacity
                if translucent:
                    keep=rgba[:,3]>0
                    fragment_pixels.append(pixel[keep]);fragment_depth.append(td[keep]);fragment_colors.append(rgba[keep])
                else:image[pixel] = np.clip(rgba*255,0,255).astype(np.uint8)
        if translucent:
            append_cpu_layer_fragments(self,fragment_pixels,fragment_depth,fragment_colors,width,height,matrix,box)
            return composite_cpu_fragments(fragment_pixels,fragment_depth,fragment_colors,width,height)
        return image.reshape(height,width,4)

    def draw(self,renderer):
        import time
        if not self.get_visible():return
        started = time.perf_counter()
        box = self.ax.bbox.frozen()
        width,height = max(1,round(box.width)),max(1,round(box.height))
        # Supersampling smooths tiny edges; curved surfaces themselves are analytic.
        ss = {'Fast':1.,'Smooth':1.5,'High':2.,'Fast (16)':1.,'Smooth (64)':1.5,'High (96)':2.}.get(self.quality,1.5)
        if self.interactive or self.figure.dpi>=200:ss = 1.
        sw,sh = max(1,round(width*ss)),max(1,round(height*ss))
        matrix = self.ax.get_proj()
        cache_key = (sw,sh,tuple(box.bounds),matrix.tobytes(),float(self.ambient),json.dumps(getattr(self,'light_settings',None),sort_keys=True),json.dumps(getattr(self,'reflection_settings',None),sort_keys=True))
        if cache_key!=self.cache_key:
            raster = self.raster(sw,sh,matrix,box)
            if (sw,sh)!=(width,height):
                raster = np.array(Image.fromarray(raster).resize((width,height),Image.Resampling.LANCZOS))
            self.cached,self.cache_key = raster,cache_key
        gc = renderer.new_gc()
        self._set_gc_clip(gc)
        renderer.draw_image(gc,round(box.x0),round(box.y0),self.cached)
        gc.restore()
        self.last_draw_seconds = time.perf_counter()-started
        if width*height>6000000:
            self.cached,self.cache_key = None,None
        self.stale = False



REFLECTION_DEFAULTS=dict(enabled=False,layer=25.,arrow=25.,sharpness=40.,transparency=0.)


def reflection_control_settings(view):
    names={'layer':('layer_reflection','Layer reflection'),
           'arrow':('arrow_reflection','Arrow reflection'),
           'sharpness':('reflection_sharpness','Highlight sharpness'),
           'transparency':('arrow_transparency','Arrow transparency')}
    values=dict(REFLECTION_DEFAULTS)
    variable=getattr(view,'reflection_enabled',None)
    if variable is not None:values['enabled']=bool(variable.get())
    for key,(attribute,label) in names.items():
        variable=getattr(view,attribute,None)
        if variable is None:continue
        try:value=float(variable.get())
        except (ValueError,TypeError,tk.TclError):raise ValueError(label+' must be a finite number.')
        low,high=(1.,128.) if key=='sharpness' else (0.,100.)
        if not math.isfinite(value) or not low<=value<=high:
            raise ValueError(f'{label} must be {low:g}–{high:g}.')
        values[key]=value
    return values


def add_reflection_controls(view,parent,changed):
    view.reflection_enabled=tk.BooleanVar(master=parent,value=False)
    for attribute,initial in (('layer_reflection','25'),('arrow_reflection','25'),
                               ('reflection_sharpness','40'),('arrow_transparency','0')):
        setattr(view,attribute,tk.StringVar(master=parent,value=initial))
    frame=ttk.LabelFrame(parent,text='Reflection / transparency',padding=5)
    frame.pack(fill='x',pady=5)
    ttk.Checkbutton(frame,text='Light reflection',variable=view.reflection_enabled,command=changed).pack(anchor='w')
    for label,attribute,low,high in (('Layer reflection (%)','layer_reflection',0,100),
        ('Arrow reflection (%)','arrow_reflection',0,100),('Highlight sharpness','reflection_sharpness',1,128),
        ('Arrow transparency (%)','arrow_transparency',0,100)):
        ttk.Label(frame,text=label).pack(anchor='w')
        row=ttk.Frame(frame);row.pack(fill='x')
        variable=getattr(view,attribute)
        ttk.Scale(row,from_=low,to=high,variable=variable,command=lambda value:changed()).pack(side='left',fill='x',expand=True)
        spin=ttk.Spinbox(row,from_=low,to=high,increment=1,textvariable=variable,width=6,command=changed)
        spin.pack(side='right',padx=2)
        spin.bind('<Return>',lambda event:changed());spin.bind('<FocusOut>',lambda event:changed())
    view.reflection_controls=frame
    return frame


def reflection_specular(settings,target,ambient):
    if not settings or not settings['enabled']:return 0.
    return (1-float(ambient))*float(settings[target])/100


def reflected_layer_record(record,settings):
    # A copy keeps independently selected layer finishes/colors/transparency intact.
    result=dict(record)
    if settings is not None:result['_reflection']=dict(settings)
    return result


def reflect_layer_planes(planes,settings):
    result=[]
    for plane in planes:
        updated=dict(plane,record=reflected_layer_record(plane['record'],settings))
        if plane.get('parts'):updated['parts']=reflect_layer_planes(plane['parts'],settings)
        result.append(updated)
    return result


def glyph_reflection_settings(artist):
    settings=getattr(artist,'reflection_settings',None)
    ambient=artist.ambient
    return (reflection_specular(settings,'arrow',ambient),
            float(settings['sharpness']) if settings else 40.,
            1-float(settings['transparency'])/100 if settings else 1.)


def apply_stream_reflection(view):
    settings=view.style.get('reflection')
    if settings is None:return
    ambient=view.style.get('ambient_light',.35)
    opacity=1-settings['transparency']/100
    for kind,item in view.actor_cache.items():
        if item is None:continue
        actor,_=item
        target='arrow' if kind in ('arrows','streams') else 'layer'
        if kind=='arrows':actor.prop.opacity=opacity
        if actor.prop.lighting:
            actor.prop.specular=reflection_specular(settings,target,ambient)
            actor.prop.specular_power=settings['sharpness']
    records={r['index']:r for r in layer_fill_effective_state(view.style.get('layer_fill',{})).get('layers',[])}
    for index,actor in view.layer_fill_actors.items():
        if index in records:
            apply_layer_finish(actor,reflected_layer_record(records[index],settings),ambient)
    renderer=view.plotter.renderers[0]
    renderer.SetUseDepthPeeling(True);renderer.SetMaximumNumberOfPeels(16);renderer.SetOcclusionRatio(0.)


def composite_cpu_fragments(pixels,distances,rgba,width,height):
    """Composite all visible glyphs far-to-near, including arrows behind arrows."""
    image=np.zeros((height*width,4),float)
    if not pixels:return np.uint8(image.reshape(height,width,4))
    pixels=np.concatenate(pixels);distances=np.concatenate(distances);colors=np.concatenate(rgba)
    if not len(pixels):return np.uint8(image.reshape(height,width,4))
    order=np.lexsort((-distances,pixels));pixels=pixels[order];colors=colors[order]
    unique,starts,counts=np.unique(pixels,return_index=True,return_counts=True)
    for rank in range(int(counts.max())):
        active=counts>rank;dest=unique[active];source=colors[starts[active]+rank]
        alpha=source[:,3]
        image[dest,:3]=source[:,:3]*alpha[:,None]+image[dest,:3]*(1-alpha[:,None])
        image[dest,3]=alpha+image[dest,3]*(1-alpha)
    alpha=image[:,3]
    image[:,:3]=np.divide(image[:,:3],alpha[:,None],out=np.zeros((len(image),3)),where=alpha[:,None]>1e-12)
    return np.uint8(np.clip(np.rint(image.reshape(height,width,4)*255),0,255))


def append_cpu_layer_fragments(artist,pixels,distances,colors,width,height,matrix,box):
    """Keep layer thickness/footprints when compositing translucent CPU glyphs."""
    planes=[]
    for group in getattr(artist,'fill_planes',[]):
        bounds=(*group['xy'],*group['z_bounds'])
        planes.extend(dict(part,merge_bounds=bounds) for part in group.get('parts',[group]))
    if not planes:return
    inverse=np.linalg.inv(matrix);inv_data=artist.ax.transData.inverted().get_affine().get_matrix()
    metric=artist.metric;lighting=artist_lighting(artist)
    for first in range(0,width*height,40000):
        ids=np.arange(first,min(width*height,first+40000));yy,xx=np.divmod(ids,width)
        xd=(box.x0+(xx+.5)*box.width/width)*inv_data[0,0]+inv_data[0,2]
        yd=(box.y0+(yy+.5)*box.height/height)*inv_data[1,1]+inv_data[1,2]
        if np.isinf(artist.ax._focal_length):
            near=xd[:,None]*inverse[:,0]+yd[:,None]*inverse[:,1]-10*inverse[:,2]+inverse[:,3]
            origins=near[:,:3]/near[:,3,None]/metric
            ray=inverse[:3,2]/metric;ray/=max(np.linalg.norm(ray),1e-15)
            directions=np.broadcast_to(ray,origins.shape)
        else:
            eye=inverse[:3,2]/inverse[3,2]/metric
            point=xd[:,None]*inverse[:,0]+yd[:,None]*inverse[:,1]-inverse[:,2]+inverse[:,3]
            directions=point[:,:3]/point[:,3,None]/metric-eye
            directions/=np.maximum(np.linalg.norm(directions,axis=1)[:,None],1e-15)
            origins=np.broadcast_to(eye,directions.shape)
        for plane in planes:
            if plane['opacity']<=0:continue
            x0,x1,y0,y1=plane['xy'];z0,z1=plane['z_bounds']
            lower=np.array([x0,y0,z0])/metric;upper=np.array([x1,y1,z1])/metric
            parallel=np.abs(directions)<1e-13
            a=np.divide(lower-origins,directions,out=np.full_like(origins,-np.inf),where=~parallel)
            b=np.divide(upper-origins,directions,out=np.full_like(origins,np.inf),where=~parallel)
            enter=np.minimum(a,b);leave=np.maximum(a,b)
            start=np.maximum(0,enter.max(axis=1));stop=leave.min(axis=1)
            outside=np.any(parallel&((origins<lower)|(origins>upper)),axis=1)
            selected=np.flatnonzero(~outside&(stop>start+1e-12))
            if not len(selected):continue
            ray=directions[selected];current=stop[selected].copy();begin=start[selected]
            path=np.maximum(current-begin,1e-15)
            normal=np.zeros_like(ray);axis=enter[selected].argmax(axis=1)
            normal[np.arange(len(selected)),axis]=-np.sign(ray[np.arange(len(selected)),axis])
            if 'merge_bounds' in plane:
                gx0,gx1,gy0,gy1,gz0,gz1=plane['merge_bounds']
                ga=np.divide(np.array([gx0,gy0,gz0])/metric-origins[selected],ray,
                    out=np.full_like(ray,-np.inf),where=np.abs(ray)>1e-13)
                gb=np.divide(np.array([gx1,gy1,gz1])/metric-origins[selected],ray,
                    out=np.full_like(ray,np.inf),where=np.abs(ray)>1e-13)
                group_enter=np.minimum(ga,gb);group_leave=np.maximum(ga,gb)
                path=np.maximum(group_leave.min(axis=1)-np.maximum(0,group_enter.max(axis=1)),1e-15)
                normal[:]=0;axis=group_enter.argmax(axis=1)
                normal[np.arange(len(selected)),axis]=-np.sign(ray[np.arange(len(selected)),axis])
            h,w=plane['rgba'].shape[:2];cell=np.array([(x1-x0)/w,(y1-y0)/h])/metric[:2]
            position=origins[selected,:2]+ray[:,:2]*(current-1e-10*path)[:,None]
            ij=np.floor((position-lower[:2])/cell).astype(int);ij[:,0]=ij[:,0].clip(0,w-1);ij[:,1]=ij[:,1].clip(0,h-1)
            active=np.arange(len(selected));transparency=1-plane['opacity']
            for _ in range(w+h+2):
                if not len(active):break
                rr=ray[active,:2];boundary=lower[:2]+(ij[active]+(rr<0))*cell
                crossing=np.divide(boundary-origins[selected[active],:2],rr,out=np.full_like(rr,-np.inf),where=np.abs(rr)>1e-13)
                next_t=np.minimum(current[active],np.maximum(begin[active],crossing.max(axis=1)))
                rgba=plane['rgba'][ij[active,1],ij[active,0]].astype(float)/255
                amount=(1-transparency**((current[active]-next_t)/path[active]))*rgba[:,3]
                rgba[:,:3]=shade_layer_fill(rgba[:,:3],ray[active],plane['record'],artist.ambient,normal[active],lighting)
                # Split absorption between the front/back of each traversed cell,
                # so glyphs inside a layer remain visible through its front half.
                rgba[:,3]=1-np.sqrt(np.maximum(0,1-amount))
                keep=rgba[:,3]>0
                for distance in (current[active],next_t):
                    pixels.append(ids[selected[active]][keep]);distances.append(distance[keep]);colors.append(rgba[keep].copy())
                hit=np.abs(crossing-next_t[:,None])<=1e-9*np.maximum(1,path[active,None])
                ij[active]-=(np.sign(rr)*hit).astype(int);current[active]=next_t
                active=active[(current[active]>begin[active]+1e-12)&(ij[active,0]>=0)&(ij[active,0]<w)&(ij[active,1]>=0)&(ij[active,1]<h)]



LIGHT_DEFAULTS=dict(x_angle=0.,y_angle=0.,height_m=1e-6,intensity=1.)


def light_control_settings(view):
    values={}
    for name,attribute,label in (('x_angle','light_x_angle','X light angle'),('y_angle','light_y_angle','Y light angle'),
        ('height_m','light_z_height','Z light height'),('intensity','light_intensity','Light intensity')):
        variable=getattr(view,attribute,None)
        try:value=float(variable.get()) if variable is not None else LIGHT_DEFAULTS[name]
        except (ValueError,TypeError,tk.TclError):raise ValueError(label+' must be a finite number.')
        if not math.isfinite(value):raise ValueError(label+' must be a finite number.')
        values[name]=value
    if not 0<=values['intensity']<=5:raise ValueError('Light intensity must be 0–5.')
    values['x_angle']%=360;values['y_angle']%=360
    return values


def add_light_controls(view,parent,changed):
    view.light_x_angle=tk.StringVar(master=parent,value='0')
    view.light_y_angle=tk.StringVar(master=parent,value='0')
    view.light_z_height=tk.StringVar(master=parent,value='1e-6')
    view.light_intensity=tk.StringVar(master=parent,value='1')
    frame=ttk.Frame(parent);frame.pack(fill='x',pady=(5,2))
    ttk.Label(frame,text='Light Angle').pack(anchor='w')
    for label,variable in (('X angle (°)',view.light_x_angle),('Y angle (°)',view.light_y_angle)):
        ttk.Label(frame,text=label).pack(anchor='w')
        row=ttk.Frame(frame);row.pack(fill='x')
        ttk.Scale(row,from_=0,to=360,variable=variable,command=lambda value:changed()).pack(side='left',fill='x',expand=True)
        spin=ttk.Spinbox(row,from_=0,to=360,increment=1,textvariable=variable,width=8,command=changed)
        spin.pack(side='right',padx=3);spin.bind('<Return>',lambda event:changed());spin.bind('<FocusOut>',lambda event:changed())
    ttk.Label(frame,text='Z light height (m)').pack(anchor='w')
    spin=ttk.Spinbox(frame,from_=-1000000,to=1000000,increment=1e-7,textvariable=view.light_z_height,command=changed)
    spin.pack(fill='x');spin.bind('<Return>',lambda event:changed());spin.bind('<FocusOut>',lambda event:changed())
    ttk.Label(frame,text='Light intensity (×)').pack(anchor='w')
    row=ttk.Frame(frame);row.pack(fill='x')
    ttk.Scale(row,from_=0,to=5,variable=view.light_intensity,command=lambda value:changed()).pack(side='left',fill='x',expand=True)
    spin=ttk.Spinbox(row,from_=0,to=5,increment=.05,textvariable=view.light_intensity,width=8,command=changed)
    spin.pack(side='right',padx=3);spin.bind('<Return>',lambda event:changed());spin.bind('<FocusOut>',lambda event:changed())
    view.light_controls=frame
    return frame


def light_unit_metres(unit):
    return {'m':1.,'cm':1e-2,'mm':1e-3,'um':1e-6,'µm':1e-6,'μm':1e-6,'nm':1e-9,'pm':1e-12}.get(unit,1.)


def light_geometry(bounds,settings,unit='nm',metric=None):
    """World-space directional lamps aimed at the volume center.

    Height sets the vertical offset in meters before rotation about world X then
    world Y. Use XY's physical scale so Z stretching does not stretch the lamp.
    """
    bounds=np.asarray(bounds,float).reshape(3,2);center=bounds.mean(axis=1)
    spans=bounds[:,1]-bounds[:,0];radius=max(float(np.linalg.norm(spans[:2])),1e-12)
    scale=1. if metric is None else math.sqrt(float(metric[0]*metric[1]))
    metres_per_display=max(light_unit_metres(unit)*scale,1e-300)
    height=float(np.clip(settings['height_m']/metres_per_display,-radius*1e12,radius*1e12))
    ax=math.radians(settings.get('x_angle',0.));ay=math.radians(settings.get('y_angle',0.))
    cx,sx=math.cos(ax),math.sin(ax);cy,sy=math.cos(ay),math.sin(ay)
    rotation_x=np.array([[1.,0,0],[0,cx,-sx],[0,sx,cx]])
    rotation_y=np.array([[cy,0,sy],[0,1.,0],[-sy,0,cy]])
    horizontal=np.array([radius/math.sqrt(2.),-radius/math.sqrt(2.),0.])
    offsets=np.array([horizontal+[0,0,height],-horizontal+[0,0,height*.5]])@rotation_x.T@rotation_y.T
    directions=offsets/np.maximum(np.linalg.norm(offsets,axis=1)[:,None],1e-15)
    return dict(focal=center,positions=center+offsets,directions=directions,intensity=settings['intensity'])


def artist_lighting(artist):
    settings=getattr(artist,'light_settings',None)
    if settings is None:
        key=np.array([1.,-1.,2.]);fill=np.array([-1.,-.5,1.])
        return dict(directions=np.array([key/np.linalg.norm(key),fill/np.linalg.norm(fill)]),intensity=1.)
    bounds=getattr(artist,'light_bounds',None)
    if bounds is None:
        points=[]
        if len(artist.centers):points.append(artist.centers/artist.metric)
        for plane in getattr(artist,'fill_planes',[]):
            x0,x1,y0,y1=plane['xy'];z0,z1=plane['z_bounds']
            points.append(np.array([[x0,y0,z0],[x1,y1,z1]])/artist.metric)
        values=np.concatenate(points) if points else np.array([[0,0,0],[1,1,1.]])
        bounds=np.column_stack((values.min(axis=0),values.max(axis=0)))
    return light_geometry(bounds,settings,getattr(artist,'light_unit','nm'),artist.metric)


def apply_stream_light_controls(view):
    import vtk
    plotter=view.plotter
    renderer=plotter.renderers[0]
    if getattr(view,'scene_light_plotter',None) is not plotter:
        renderer.RemoveAllLights();renderer.AutomaticLightCreationOff()
        view.scene_lights=[];view.scene_light_plotter=plotter
        for _ in range(2):
            light=vtk.vtkLight();light.SetLightTypeToSceneLight();light.SetPositional(False)
            renderer.AddLight(light);view.scene_lights.append(light)
    values=light_geometry(view.result['display_bounds'],view.style.get('light',LIGHT_DEFAULTS),view.mesh['x']['unit'])
    for light,position,intensity in zip(view.scene_lights,values['positions'],(.85,.2)):
        light.SetPosition(*position);light.SetFocalPoint(*values['focal']);light.SetIntensity(intensity*values['intensity'])


_MUMAX_GPU_ERROR = None


def close_mumax_gpu_scenes(figure):
    for scene in getattr(figure,'_mumax_gpu_scenes',{}).values():scene.close()
    figure._mumax_gpu_scenes={}


class MumaxGPUScene:
    """Persistent OpenGL buffers and instanced glyphs; Matplotlib keeps the axes/UI."""
    def __init__(self):
        import vtk
        self.vtk=vtk;self.pv=require_tab4_backend()
        self.window=vtk.vtkRenderWindow();self.window.SetOffScreenRendering(True)
        self.window.SetAlphaBitPlanes(1);self.window.SetMultiSamples(0)
        self.renderer=vtk.vtkRenderer();self.renderer.SetBackground(0,0,0);self.renderer.SetBackgroundAlpha(0)
        self.renderer.AutomaticLightCreationOff();self.renderer.SetUseDepthPeeling(True)
        self.renderer.SetMaximumNumberOfPeels(12);self.renderer.SetOcclusionRatio(.01)
        self.window.AddRenderer(self.renderer);self.camera=vtk.vtkCamera();self.renderer.SetActiveCamera(self.camera)
        self.lights=[]
        for position,intensity in (((1,-1,2),.85),((-1,-.5,1),.2)):
            light=vtk.vtkLight();light.SetLightTypeToSceneLight();light.SetPosition(*position);light.SetFocalPoint(0,0,0)
            light.SetPositional(False);light.SetIntensity(intensity);self.renderer.AddLight(light);self.lights.append(light)
        self.capture=vtk.vtkWindowToImageFilter();self.capture.SetInput(self.window)
        self.capture.SetInputBufferTypeToRGBA();self.capture.ReadFrontBufferOff();self.capture.ShouldRerenderOff()
        self.geometry_key=None;self.glyph_actors=[];self.fill_actors=[];self.closed=False;self.device=None

    def close(self):
        if not self.closed:
            self.window.Finalize();self.closed=True

    def signature(self,artist):
        import hashlib
        digest=hashlib.blake2b(digest_size=16)
        for value in (artist.centers,artist.units,artist.lengths,artist.radii,artist.head_radii,artist.colors,artist.metric):
            digest.update(np.ascontiguousarray(value).view(np.uint8))
        digest.update(repr((artist.head_ratio,artist.quality,None if artist.half is None else tuple(artist.half))).encode())
        for plane in getattr(artist,'fill_planes',[]):
            parts=plane.get('parts',[plane]);digest.update(repr(len(parts)).encode())
            for part in parts:
                digest.update(repr((part['xy'],part['z_bounds'],part['rgba'].shape)).encode())
                digest.update(np.ascontiguousarray(part['rgba']).view(np.uint8))
        return digest.digest()

    def glyph_actor(self,points,units,scales,colors,source,orient=True):
        vtk=self.vtk;data=self.pv.PolyData(points)
        data['direction']=units;data['scale']=scales;data['rgba']=np.uint8(np.clip(np.rint(colors*255),0,255))
        mapper=vtk.vtkGlyph3DMapper();mapper.SetInputData(data);mapper.SetSourceData(source)
        mapper.SetOrientationArray('direction');mapper.SetOrientationModeToDirection();mapper.SetOrient(orient)
        mapper.SetScaleArray('scale');mapper.SetScaleModeToScaleByVectorComponents();mapper.ScalingOn()
        mapper.SetScalarModeToUsePointFieldData();mapper.SelectColorArray('rgba');mapper.SetColorModeToDirectScalars()
        actor=vtk.vtkActor();actor.SetMapper(mapper);actor.GetProperty().SetInterpolationToPhong()
        self.renderer.AddActor(actor);self.glyph_actors.append(actor)

    def prepare(self,artist):
        key=self.signature(artist)
        if key!=self.geometry_key:
            self.renderer.RemoveAllViewProps();self.glyph_actors=[];self.fill_actors=[]
            sides={'Fast':16,'Fast (16)':16,'Smooth':64,'Smooth (64)':64,'High':96,'High (96)':96}.get(artist.quality,64)
            if len(artist.centers):
                centers=artist.centers/artist.metric
                if artist.half is not None:
                    self.glyph_actor(centers,artist.units,np.broadcast_to(artist.half*2,centers.shape).copy(),artist.colors,self.pv.Cube(),orient=False)
                else:
                    shaft=artist.lengths*(1-artist.head_ratio);head=artist.lengths*artist.head_ratio
                    cylinder=self.pv.Cylinder(direction=(1,0,0),radius=1.,height=1.,resolution=sides)
                    cone=self.pv.Cone(direction=(1,0,0),radius=1.,height=1.,resolution=sides).compute_normals(
                        split_vertices=True,feature_angle=60,consistent_normals=True)
                    self.glyph_actor(centers-artist.units*(head/2)[:,None],artist.units,
                        np.column_stack((shaft,artist.radii,artist.radii)),artist.colors,cylinder)
                    self.glyph_actor(centers+artist.units*(shaft/2)[:,None],artist.units,
                        np.column_stack((head,artist.head_radii,artist.head_radii)),artist.colors,cone)
            for plane in getattr(artist,'fill_planes',[]):
                data=vtk_layer_fill_mesh(plane)
                mapper=self.vtk.vtkDataSetMapper();mapper.SetInputData(data)
                mapper.SetScalarModeToUseCellFieldData();mapper.SelectColorArray('layer_rgba');mapper.SetColorModeToDirectScalars()
                actor=self.vtk.vtkActor();actor.SetMapper(mapper);actor.SetScale(*(1/artist.metric))
                prop=actor.GetProperty();settings=layer_finish_material(plane['record'],artist.ambient)
                prop.SetAmbient(settings['ambient']);prop.SetDiffuse(settings['diffuse']);prop.SetSpecular(settings['specular'])
                prop.SetSpecularPower(settings['specular_power']);prop.SetOpacity(plane['opacity'])
                finish=LAYER_FILL_FINISHES.get(plane['record'].get('finish','Matte'),LAYER_FILL_FINISHES['Matte'])
                if finish['metal'] and plane['record']['mode']=='Solid color':
                    from matplotlib.colors import to_rgb
                    prop.SetSpecularColor(*to_rgb(plane['record']['color']))
                self.renderer.AddActor(actor);self.fill_actors.append((actor,plane))
            bounds=np.asarray(self.renderer.ComputeVisiblePropBounds()).reshape(3,2)
            if not np.isfinite(bounds).all() or np.any(bounds[:,0]>bounds[:,1]):bounds=np.array([[0,1],[0,1],[0,1]],float)
            self.corners=np.array([[x,y,z,1.] for x in bounds[0] for y in bounds[1] for z in bounds[2]])
            self.geometry_key=key
        for actor in self.glyph_actors:
            specular,power,opacity=glyph_reflection_settings(artist)
            prop=actor.GetProperty();prop.SetAmbient(artist.ambient);prop.SetDiffuse(1-artist.ambient)
            prop.SetSpecular(specular);prop.SetSpecularPower(power);prop.SetOpacity(opacity)
        for (actor,_),plane in zip(self.fill_actors,getattr(artist,'fill_planes',[])):
            values=layer_finish_material(plane['record'],artist.ambient);prop=actor.GetProperty()
            prop.SetAmbient(values['ambient']);prop.SetDiffuse(values['diffuse']);prop.SetSpecular(values['specular'])
            prop.SetSpecularPower(values['specular_power']);prop.SetOpacity(plane['opacity'])
            finish=LAYER_FILL_FINISHES.get(plane['record'].get('finish','Matte'),LAYER_FILL_FINISHES['Matte'])
            from matplotlib.colors import to_rgb
            prop.SetSpecularColor(*(to_rgb(plane['record']['color']) if finish['metal'] and plane['record']['mode']=='Solid color' else (1.,1.,1.)))

    def render(self,artist,width,height,matrix,box,tile=None,prepared=False):
        from vtkmodules.util.numpy_support import vtk_to_numpy
        if not prepared:self.prepare(artist)
        if getattr(artist,'light_settings',None) is not None:
            lighting=artist_lighting(artist)
            for light,position,power in zip(self.lights,lighting['positions'],(.85,.2)):
                light.SetPosition(*position);light.SetFocalPoint(*lighting['focal']);light.SetIntensity(power*lighting['intensity'])
        else:
            for light,position,power in zip(self.lights,((1,-1,2),(-1,-.5,1)),(.85,.2)):
                light.SetPosition(*position);light.SetFocalPoint(0,0,0);light.SetIntensity(power)
        self.renderer.SetMaximumNumberOfPeels(6 if artist.interactive else 16)
        transform=artist.ax.transData.get_affine().get_matrix()
        projection=matrix@np.diag(np.r_[artist.metric,1.])
        projection[0]=2*transform[0,0]/box.width*projection[0]+(2*(transform[0,2]-box.x0)/box.width-1)*projection[3]
        projection[1]=2*transform[1,1]/box.height*projection[1]+(2*(transform[1,2]-box.y0)/box.height-1)*projection[3]
        render_width,render_height=width,height
        if tile is not None:
            left,bottom,render_width,render_height=tile
            projection[0]=width/render_width*projection[0]+((width-2*left)/render_width-1)*projection[3]
            projection[1]=height/render_height*projection[1]+((height-2*bottom)/render_height-1)*projection[3]
        inverse=np.linalg.inv(projection)
        center=inverse@np.array([0.,0.,-1.,1.]);center=center[:3]/center[3]
        top=inverse@np.array([0.,1.,-1.,1.]);up=top[:3]/top[3]-center
        if np.isinf(artist.ax._focal_length):
            direction=inverse[:3,2];direction/=max(np.linalg.norm(direction),1e-15)
            eye=center-direction*max(10.,np.linalg.norm(np.ptp(self.corners[:,:3],axis=0))*10)
            self.camera.ParallelProjectionOn()
        else:
            eye=inverse[:3,2]/inverse[3,2];self.camera.ParallelProjectionOff()
        self.camera.SetPosition(*eye);self.camera.SetFocalPoint(*center);self.camera.SetViewUp(*up)
        self.renderer.ResetCameraClippingRange()
        clip=self.corners@projection.T;valid=clip[:,3]>1e-10
        if not valid.any():return np.zeros((render_height,render_width,4),np.uint8)
        z=clip[valid,2]/clip[valid,3];lo,hi=z.min(),z.max();pad=max((hi-lo)*.02,1e-5);lo-=pad;hi+=pad
        projection[2]=2/(hi-lo)*projection[2]-(1+2*lo/(hi-lo))*projection[3]
        view=self.camera.GetModelViewTransformMatrix();view=np.array([[view.GetElement(r,c) for c in range(4)] for r in range(4)])
        explicit=projection@np.linalg.inv(view);vtkmatrix=self.vtk.vtkMatrix4x4();vtkmatrix.DeepCopy(explicit.ravel())
        self.camera.SetExplicitProjectionTransformMatrix(vtkmatrix);self.camera.UseExplicitProjectionTransformMatrixOn()
        self.window.SetSize(render_width,render_height);self.window.Render()
        if self.device is None:
            report=self.window.ReportCapabilities();self.device=next((line.split(':',1)[1].strip() for line in report.splitlines() if line.startswith('OpenGL renderer string:')),'OpenGL')
            if any(word in self.device.lower() for word in ('llvmpipe','softpipe','software','gdi generic')):
                raise RuntimeError('Software OpenGL; use the compiled CPU renderer.')
        self.capture.Modified();self.capture.Update()
        rgba=vtk_to_numpy(self.capture.GetOutput().GetPointData().GetScalars()).reshape(render_height,render_width,4).copy()
        # The OpenGL framebuffer is premultiplied; Matplotlib expects straight alpha.
        alpha=rgba[:,:,3].astype(float)/255
        rgb=np.divide(rgba[:,:,:3],alpha[:,:,None],out=np.zeros((*rgba.shape[:2],3)),where=alpha[:,:,None]>0)
        rgba[:,:,:3]=np.uint8(np.clip(np.rint(rgb),0,255))
        artist.last_engine='GPU';artist.gpu_device=self.device
        return rgba


def mumax_gpu_raster(artist,width,height,matrix,box):
    global _MUMAX_GPU_ERROR
    if _MUMAX_GPU_ERROR is not None or getattr(artist,'force_cpu',False):return None
    try:
        figure=artist.figure
        scenes=getattr(figure,'_mumax_gpu_scenes',None)
        if scenes is None:scenes={};figure._mumax_gpu_scenes=scenes
        slot=artist.ax.get_subplotspec().num1
        if slot not in scenes:scenes[slot]=MumaxGPUScene()
        scene=scenes[slot]
        if width*height<=4000000 and max(width,height)<=4096:
            return scene.render(artist,width,height,matrix,box)
        # Render large PNGs in tiles, retaining every output pixel while keeping
        # depth/transparency buffers within the laptop GPU's memory budget.
        scene.prepare(artist);result=np.empty((height,width,4),np.uint8)
        for bottom in range(0,height,1536):
            for left in range(0,width,1536):
                w,h=min(1536,width-left),min(1536,height-bottom)
                result[bottom:bottom+h,left:left+w]=scene.render(artist,width,height,matrix,box,(left,bottom,w,h),prepared=True)
        return result
    except Exception as exc:
        _MUMAX_GPU_ERROR=str(exc);artist.gpu_error=str(exc)
        if getattr(artist,'figure',None) is not None:close_mumax_gpu_scenes(artist.figure)
        warm_mumax_renderer()
        return None


class LayerFilledGlyphArtist(SmoothSolidGlyphArtist):
    def raster(self,width,height,matrix,box):
        gpu=mumax_gpu_raster(self,width,height,matrix,box)
        if gpu is not None:return gpu
        image=super().raster(width,height,matrix,box)
        if not getattr(self,'fill_planes',None) or getattr(self,'_cpu_transparency_composited',False):
            self.fill_depth=None
            return image
        try:return composite_layer_fill_rays(self,image,self.fill_depth,width,height,matrix,box)
        finally:self.fill_depth=None


class BandCrossSectionBase(ROICrossSectionBase):
    """Inclusive X/Y slabs, arrow-only scalar colors, and optional nm coordinate scales."""
    def __init__(self,gui,parent):
        self.band_initialized = False
        self.band_drag = None
        self.show_scales = tk.BooleanVar(value=True)
        super().__init__(gui,parent)
        self.band_ends = {a:tk.DoubleVar(value=1) for a in 'xy'}
        self.end_labels = {a:tk.StringVar() for a in 'xy'}
        self.end_sliders = {}
        position_group = self.position_sliders['x'].master
        for child in position_group.winfo_children():
            child.destroy()
        box = self.dropdown(position_group,'Figure 2 view',self.view,['XY','YZ','XZ'])
        box.bind('<<ComboboxSelected>>',lambda e:self.view_changed())
        self.position_sliders = {}
        for axis in 'xy':
            for suffix,variable,label_var in (('1 start',self.positions[axis],self.position_labels[axis]),
                                               ('2 end',self.band_ends[axis],self.end_labels[axis])):
                ttk.Label(position_group,text=axis.upper()+suffix).pack(anchor='w')
                is_end = suffix.startswith('2')
                slider = ttk.Scale(position_group,from_=1,to=1,variable=variable,
                                   command=lambda value,a=axis,e=is_end:self.band_changed(a,e))
                slider.pack(fill='x')
                (self.end_sliders if is_end else self.position_sliders)[axis] = slider
                ttk.Label(position_group,textvariable=label_var,wraplength=235).pack(anchor='w')
        ttk.Label(position_group,text='Z (XY layer)').pack(anchor='w')
        self.position_sliders['z'] = ttk.Scale(position_group,from_=1,to=1,variable=self.positions['z'],command=lambda v:self.position_changed('z',v))
        self.position_sliders['z'].pack(fill='x')
        ttk.Label(position_group,textvariable=self.position_labels['z'],wraplength=235).pack(anchor='w')
        ttk.Label(position_group,text='Bounds are inclusive. X1 = X2 gives a single X slice; Y1 = Y2 gives a single Y slice. XY uses the selected Z layer.',wraplength=235).pack(anchor='w',pady=4)
        scales = self.group('Coordinate scales')
        ttk.Checkbutton(scales,text='Show XYZ scales (nm)',variable=self.show_scales,command=self.schedule_render).pack(anchor='w')
        ttk.Label(scales,text='Arrow colors show the selected component: blue −1, white 0, red +1 with the default map. Figure 2 has no colored surface mask.',wraplength=235).pack(anchor='w',pady=3)

    def load_folder(self):
        self.band_initialized = False
        super().load_folder()

    def update_slice(self):
        super().update_slice()
        if self.vectors is None:
            return
        self.updating = True
        roi = self.valid_roi()
        for axis,count in (('x',self.vectors.shape[2]),('y',self.vectors.shape[1])):
            first,last = (roi[0]+1,roi[1]) if roi and axis=='x' else (roi[2]+1,roi[3]) if roi else (1,count)
            if not self.band_initialized:
                start,end = first,last
            else:
                start = min(last,max(first,round(self.positions[axis].get())))
                end = min(last,max(start,round(self.band_ends[axis].get())))
            self.position_sliders[axis].configure(from_=first,to=last)
            self.end_sliders[axis].configure(from_=first,to=last)
            self.positions[axis].set(start)
            self.band_ends[axis].set(end)
        self.band_initialized = True
        self.updating = False

    def bounds(self):
        result = {}
        roi = self.valid_roi()
        for axis,count in (('x',self.vectors.shape[2]),('y',self.vectors.shape[1])):
            a,b = sorted((round(self.positions[axis].get())-1,round(self.band_ends[axis].get())-1))
            low,high = max(0,a),min(count,b+1)
            if roi:
                rlow,rhigh = (roi[0],roi[1]) if axis=='x' else (roi[2],roi[3])
                low,high = max(rlow,low),min(rhigh,high)
            result[axis] = (low,max(low+1,high))
        z = self.indices()['z']
        result['z'] = (z,z+1) if self.view.get()=='XY' else (0,self.vectors.shape[0])
        return result

    def band_changed(self,axis,is_end):
        if self.updating:
            return
        start,end = self.positions[axis].get(),self.band_ends[axis].get()
        if start>end:
            (self.positions[axis] if is_end else self.band_ends[axis]).set(end if is_end else start)
        self.camera_limits = self.camera_base_centers = None
        self.zoom = 1.
        self.schedule_render()

    def toggle_roi(self):
        if not self.roi_enabled.get():
            self.band_initialized = False
        super().toggle_roi()

    def clear_roi(self):
        self.band_initialized = False
        super().clear_roi()

    def reference_lines(self,image,view,indices):
        if view!='XY':
            return image
        # Draw bounds without tinting/masking any region.
        draw = ImageDraw.Draw(image)
        nx,ny = self.vectors.shape[2],self.vectors.shape[1]
        bounds = self.bounds()
        font = annotation_font(8)
        for axis,count,size in (('x',nx,image.width),('y',ny,image.height)):
            first,last = bounds[axis][0],bounds[axis][1]-1
            lines = [(first,axis.upper()+'1',0),(last,axis.upper()+'2',1)] if first!=last else [(first,axis.upper()+'1='+axis.upper()+'2',0)]
            for cell,label,which in lines:
                coord = (cell+.5)*size/count if axis=='x' else (count-cell-.5)*size/count
                color = ((255,235,0),(255,150,0))[which] if axis=='x' else ((0,235,255),(255,0,255))[which]
                points = (coord,0,coord,image.height-1) if axis=='x' else (0,coord,image.width-1,coord)
                draw.line(points,fill='black',width=3)
                draw.line(points,fill=color,width=1)
                box = font.getbbox(label)
                tx,ty = (min(max(0,coord+2),max(0,image.width-(box[2]-box[0])-1)),1) if axis=='x' else (1,min(max(0,coord+1),max(0,image.height-(box[3]-box[1])-1)))
                draw.text((tx,ty),label,font=font,fill='black',stroke_width=1,stroke_fill='white')
        return image

    def render(self):
        super().render()
        if self.vectors is not None:
            for axis,count in (('x',self.vectors.shape[2]),('y',self.vectors.shape[1])):
                index = min(count-1,max(0,round(self.band_ends[axis].get())-1))
                info = self.mesh[axis]
                self.end_labels[axis].set(f'{info["centers"][index]:.3f} {info["unit"]} | cell {index+1}/{count}')

    def drag_start(self,event):
        if self.roi_selecting:
            return super().drag_start(event)
        point = self.pointer_in_image(event)
        if point is None or self.vectors is None:
            return
        counts = {'x':self.vectors.shape[2],'y':self.vectors.shape[1]}
        for rect in self.top_data_rects:
            a,b,c,d = rect
            if a<=point[0]<c and b<=point[1]<d:
                distances = []
                for axis in 'xy':
                    for is_end,var in ((False,self.positions[axis]),(True,self.band_ends[axis])):
                        fraction = (round(var.get())-.5)/counts[axis]
                        coord = a+fraction*(c-a) if axis=='x' else b+(1-fraction)*(d-b)
                        distance = abs(point[0 if axis=='x' else 1]-coord)
                        distances.append((distance,axis,is_end))
                _,axis,is_end = min(distances)
                self.band_drag = rect,axis,is_end
                self.canvas.configure(cursor='crosshair')
                self.drag_move(event)
                return

    def drag_move(self,event):
        if self.roi_drag is not None:
            return super().drag_move(event)
        if self.band_drag is None:
            return
        point = self.pointer_in_image(event)
        if point is None:
            return
        (a,b,c,d),axis,is_end = self.band_drag
        count = self.vectors.shape[2 if axis=='x' else 1]
        fraction = (point[0]-a)/(c-a) if axis=='x' else 1-(point[1]-b)/(d-b)
        cell = min(count,max(1,int(min(1,max(0,fraction))*count)+1))
        (self.band_ends if is_end else self.positions)[axis].set(cell)
        self.band_changed(axis,is_end)

    def drag_end(self,event):
        if self.roi_drag is not None:
            return super().drag_end(event)
        if self.band_drag is not None:
            self.drag_move(event)
        self.band_drag = None
        self.canvas.configure(cursor='')

    def render_scene(self,plane,indices,normal,components,low,high,density,length,head):
        from matplotlib.ticker import ScalarFormatter
        light_settings=light_control_settings(self)
        bounds = self.bounds()
        self.last_band_bounds = bounds
        ranges = {a:np.arange(*bounds[a]) for a in 'xyz'}
        self.last_analysis_volume = self.vectors[slice(*bounds['z']),slice(*bounds['y']),slice(*bounds['x']),:]
        self.last_analysis_mesh = {}
        for axis in 'xyz':
            first,last = bounds[axis]
            info = self.mesh[axis]
            step = (info['edges'][1]-info['edges'][0])/len(info['centers'])
            self.last_analysis_mesh[axis] = dict(info,centers=info['centers'][first:last],edges=(info['edges'][0]+first*step,info['edges'][0]+last*step))
        old_angles = self.ax3d.elev,self.ax3d.azim
        old_roll = float(getattr(self.ax3d,'roll',0.))
        new_view = self.scene_view!=self.view.get()
        if self.camera_limits is not None and not new_view:
            self.camera_limits = {a:getattr(self.ax3d,'get_'+a+'lim')() for a in 'xyz'}
        self.figure.clear()
        self.figure.set_facecolor(self.background_color.get())
        self.figure.subplots_adjust(left=.035,right=.86 if self.colorbar.get() else .985,bottom=.055,top=.90 if self.show_time.get() else .94,wspace=.15)
        cmap = stream_color_map(self.colors.get())
        norm = Normalize(low,high)
        layer_step = int(self.z_layer_step.get())
        if layer_step<1:
            raise ValueError('Z layer step must be at least 1.')
        refined_z_step = layer_step*getattr(self,'applied_factor',1)
        sample_points,samples,spacings = sliced_arrow_lattice(self.vectors.shape,self.mesh,bounds,density,refined_z_step)
        self.last_sample_points = sample_points
        self.requested_sample_count = len(sample_points)*len(components)
        self.base_limits = {a:self.last_analysis_mesh[a]['edges'] for a in 'xyz'}
        spans = np.array([self.base_limits[a][1]-self.base_limits[a][0] for a in 'xyz'])
        physical_per_display = spans/thin_dimension_aspect(spans,self.stretch.get())
        # Arrow size follows XY pitch; changing Z sampling cannot shrink arrows.
        reference_length = float(spacings[:2].min())*.7*length
        visual_spacing = float(np.min(spacings[:2]/physical_per_display[:2]))
        self.last_sample_axes = samples
        self.last_sample_spacings = spacings.copy()
        selected = {'mx':0,'my':1,'mz':2}.get(self.arrow_component.get())
        self.axes3d,self.glyph_origins_by_axes = [],{}
        self.glyph_count = 0
        self.last_glyph_colors = []
        self.last_display_lengths = []
        self.glyph_artists = []
        for number,component in enumerate(components,1):
            ax = self.figure.add_subplot(1,len(components),number,projection='mumax_full_3d')
            self.axes3d.append(ax)
            component_index = {'mx':0,'my':1,'mz':2}[component]
            origins = np.empty((0,3));artist=None
            if self.arrows.get():
                x,y,z = sample_points.T
                full = np.asarray(self.vectors[z,y,x],dtype=float)
                vectors = full.copy()
                if selected is not None and self.glyph_type.get()=='Arrows':
                    vectors[:,np.arange(3)!=selected] = 0
                magnitudes = np.linalg.norm(vectors,axis=1)
                valid = np.isfinite(vectors).all(axis=1)&(magnitudes>=1e-8)
                origins = np.column_stack([self.mesh[a]['centers'][v] for a,v in zip('xyz',(x,y,z))])[valid]
                full,vectors,magnitudes = full[valid],vectors[valid],magnitudes[valid]
                lengths = np.full(len(origins),visual_spacing*.7*length)
                if self.arrow_mode.get()=='Magnitude':lengths *= magnitudes
                colors = cmap(norm(full[:,component_index]))
                self.last_display_lengths.extend(lengths)
                self.last_glyph_colors.extend(zip(full[:,component_index],colors))
                self.glyph_count += len(origins)
                if len(origins):
                    radii = np.maximum(1e-6,np.minimum(lengths*self.shaft_radius.get(),visual_spacing*.2))
                    head_radii = np.maximum(1e-6,np.minimum(lengths*self.head_radius.get(),visual_spacing*.48))
                    dimensions = spacings*self.cuboid_scale.get() if self.glyph_type.get()=='Cuboids' else None
                    artist = LayerFilledGlyphArtist(ax,origins,vectors,lengths,physical_per_display,colors,
                                 radii,head_radii,self.head_ratio.get(),self.ambient_light.get(),
                                 self.surface_quality.get(),dimensions)
                    ax.add_artist(artist)
                    self.glyph_artists.append(artist)
            fill_planes=build_layer_fill_planes(self.vectors,self.mesh,layer_fill_source_z(self),
                getattr(self,'layer_fill_state',{}),component,self.colors.get(),(low,high),bounds,
                slice_z=self.mesh['z']['centers'][indices['z']] if self.view.get()=='XY' else None,native=layer_fill_native_data(self))
            if fill_planes:
                if artist is None:
                    artist=LayerFilledGlyphArtist(ax,np.empty((0,3)),np.empty((0,3)),np.empty(0),physical_per_display,
                        np.empty((0,4)),np.empty(0),np.empty(0),self.head_ratio.get(),self.ambient_light.get(),self.surface_quality.get())
                    ax.add_artist(artist);self.glyph_artists.append(artist)
                artist.fill_planes=fill_planes
            if artist is not None:
                artist.reflection_settings=reflection_control_settings(self)
                artist.fill_planes=reflect_layer_planes(getattr(artist,'fill_planes',[]),artist.reflection_settings)
                artist.light_settings=dict(light_settings);artist.light_unit=self.mesh['x']['unit']
                artist.light_bounds=np.array([self.base_limits[a] for a in 'xyz'])/physical_per_display[:,None]
            self.glyph_origins_by_axes[ax] = origins
            for axis in 'xyz':
                getattr(ax,'set_'+axis+'label')(axis.upper()+' ('+self.mesh[axis]['unit']+')',fontsize=8,labelpad=0)
                axis_object = getattr(ax,axis+'axis')
                axis_object.set_major_locator(MaxNLocator(4))
                formatter = ScalarFormatter(useOffset=False)
                formatter.set_scientific(False)
                axis_object.set_major_formatter(formatter)
            ax.tick_params(labelsize=7,pad=0)
            title = '$m_{'+component[-1]+'}$'
            if self.show_time.get():
                title += f'\nt = {self.time_ns:.3f} ns' if self.time_ns is not None else '\nt = unavailable'
            ax.set_title(title,fontsize=10,pad=3,loc='center',multialignment='center')
            ax.mouse_init(rotate_btn=2,pan_btn=1,zoom_btn=3)
            elev,azim = {'YZ':(15,10),'XZ':(15,-80),'XY':(65,-90)}[self.view.get()] if new_view else old_angles
            if getattr(self,'xy_top_camera',False) and new_view:
                elev,azim = 90,-90
            ax.set_proj_type('ortho' if getattr(self,'xy_top_camera',False) else 'persp')
            ax.view_init(elev=elev,azim=azim,roll=0. if new_view else old_roll)
            if getattr(self,'movie_camera_fit',None) is not None:ax._camera_fit_scale = self.movie_camera_fit
            if not self.show_scales.get():
                ax.set_axis_off()
            self.decorate_scene(ax)
            if self.colorbar.get():
                fraction = {'Small':.45,'Medium':.7,'Bigger':1}[self.gui.colorbar_size.get()]
                bottom = {'Top':.9-.76*fraction,'Middle':.52-.38*fraction,'Bottom':.14}[self.gui.colorbar_position.get()]
                bar_axis = self.figure.add_axes([(number-1)/len(components)+.9/len(components),bottom,.018/len(components),.76*fraction])
                bar = self.figure.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=bar_axis,ticks=[low,(low+high)/2,high])
                bar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f' if self.gui.colorbar_decimals.get() else '%.0f'))
                bar.ax.tick_params(labelsize=7,pad=2)
                self.decorate_colorbar(bar)
        self.ax3d = self.axes3d[0]
        self.scene_view = self.view.get()
        self.apply_zoom()
        if getattr(self,'active_figure',2)==2:
            self.plot_canvas.draw_idle()

    def cursor_anchor(self,event):
        from mpl_toolkits.mplot3d import proj3d
        origins = getattr(self,'glyph_origins_by_axes',{}).get(event.inaxes)
        if origins is not None and len(origins) and getattr(event,'xdata',None) is not None and getattr(event,'ydata',None) is not None:
            xx,yy,zz = proj3d.proj_transform(origins[:,0],origins[:,1],origins[:,2],event.inaxes.get_proj())
            closest = int(np.argmin((xx-event.xdata)**2+(yy-event.ydata)**2))
            return np.array(proj3d.inv_transform(event.xdata,event.ydata,zz[closest],np.linalg.inv(event.inaxes.get_proj())))
        return super().cursor_anchor(event)


def cuboid_faces(center,dimensions):
    center = np.asarray(center,dtype=float)
    half = np.asarray(dimensions,dtype=float)/2
    vertices = np.array([center+half*np.array(signs) for signs in
                        ((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                         (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1))])
    return [vertices[list(indices)] for indices in ((3,2,1,0),(4,5,6,7),(0,1,5,4),
                                                   (1,2,6,5),(2,3,7,6),(3,0,4,7))]


def shade_glyph_faces(faces,colors,ambient=.45,physical_per_display=None):
    """Ambient + key/fill directional lighting; ambient=1 preserves map RGB exactly."""
    rgba = np.asarray(colors,dtype=float).copy()
    if not len(faces):
        return rgba
    ambient = float(np.clip(ambient,0,1))
    first_vertices = np.array([np.asarray(face)[:3] for face in faces],dtype=float)
    if physical_per_display is not None:
        first_vertices /= np.asarray(physical_per_display)
    normals = np.cross(first_vertices[:,1]-first_vertices[:,0],first_vertices[:,2]-first_vertices[:,0])
    lengths = np.linalg.norm(normals,axis=1)
    normals /= np.maximum(lengths[:,None],1e-15)
    key = np.array([1.,-1.,2.])
    fill = np.array([-1.,-.5,1.])
    key /= np.linalg.norm(key)
    fill /= np.linalg.norm(fill)
    diffuse = .1+.75*np.maximum(np.sum(normals*key,axis=1),0)+.15*np.maximum(np.sum(normals*fill,axis=1),0)
    brightness = ambient+(1-ambient)*diffuse
    rgba[:,:3] *= brightness[:,None]
    return np.clip(rgba,0,1)


class SmoothGlyphCrossSectionBase(BandCrossSectionBase):
    def __init__(self,gui,parent):
        super().__init__(gui,parent)
        self.glyph_type = tk.StringVar(value='Arrows')
        self.surface_quality = tk.StringVar(value='Smooth')
        self.head_ratio = tk.DoubleVar(value=.3)
        self.shaft_radius = tk.DoubleVar(value=.045)
        self.head_radius = tk.DoubleVar(value=.12)
        self.cuboid_scale = tk.DoubleVar(value=.65)
        self.ambient_light = tk.DoubleVar(value=.45)
        self.background_color = tk.StringVar(value='#ffffff')
        self.outline_color = tk.StringVar(value='#303030')
        self.show_outline = tk.BooleanVar(value=True)
        self.last_lit_colors = None
        group = next(child for child in self.controls.winfo_children()
                     if isinstance(child,ttk.LabelFrame) and child.cget('text')=='3D arrows')
        for child in group.winfo_children():
            child.destroy()
        group.configure(text='Glyphs: arrows / cuboids')
        ttk.Checkbutton(group,text='Show glyphs',variable=self.arrows,command=self.schedule_render).pack(anchor='w')
        row = ttk.Frame(group)
        row.pack(fill='x',pady=3)
        for name in ('Arrows','Cuboids'):
            ttk.Radiobutton(row,text=name,variable=self.glyph_type,value=name,command=self.glyph_type_changed).pack(side='left',padx=5)
        self.arrow_geometry = ttk.Frame(group)
        self.arrow_geometry.pack(fill='x')
        self.parameter_slider(self.arrow_geometry,'Scaling factor',self.length,.1,5)
        self.parameter_slider(self.arrow_geometry,'Head ratio',self.head_ratio,.05,.8)
        self.parameter_slider(self.arrow_geometry,'Shaft radius',self.shaft_radius,.005,.2)
        self.parameter_slider(self.arrow_geometry,'Head radius',self.head_radius,.01,.4)
        self.dropdown(self.arrow_geometry,'Preview quality',self.surface_quality,['Fast','Smooth','High'])
        self.cuboid_geometry = ttk.Frame(group)
        self.cuboid_geometry.pack(fill='x')
        self.parameter_slider(self.cuboid_geometry,'Cuboid scale',self.cuboid_scale,.05,1.5)
        ttk.Label(group,text='Radii are relative to arrow length. Cuboid size is relative to sampled-cell spacing.',wraplength=235).pack(anchor='w',pady=3)
        ttk.Label(group,text='Arrow components').pack(anchor='w')
        self.picker(group,self.arrow_component)
        self.dropdown(group,'Arrow scaling',self.arrow_mode,['Magnitude','Normalized'])
        row = ttk.Frame(group)
        row.pack(fill='x',pady=3)
        ttk.Label(row,text='XY density across X').pack(side='left',padx=3)
        entry = ttk.Entry(row,textvariable=self.density,width=8)
        entry.pack(side='left',padx=3)
        entry.bind('<Return>',lambda e:self.schedule_render())
        row = ttk.Frame(group)
        row.pack(fill='x',pady=3)
        ttk.Label(row,text='Z layer step').pack(side='left',padx=3)
        ttk.Spinbox(row,from_=1,to=1000,textvariable=self.z_layer_step,width=8,command=self.schedule_render).pack(side='left',padx=3)
        ttk.Label(group,text='XY density matches Figure 1. Z step is independent: 1 samples each original Z layer. Upscaling preserves this physical spacing. Dense 3D views may render more slowly.',wraplength=235).pack(anchor='w')
        appearance = self.group('Ambient light and appearance')
        appearance.pack_configure(after=group)
        self.parameter_slider(appearance,'Ambient light',self.ambient_light,0,1)
        ttk.Label(appearance,text='Lower = shaded faces; 1 = original flat color-map colors.',wraplength=235).pack(anchor='w',pady=3)
        add_light_controls(self,appearance,self.schedule_render)
        add_reflection_controls(self,appearance,self.schedule_render)
        self.background_button = ttk.Button(appearance,text='Background color…',command=lambda:self.pick_color(self.background_color))
        self.background_button.pack(fill='x',pady=2)
        ttk.Checkbutton(appearance,text='Show outline',variable=self.show_outline,command=self.schedule_render).pack(anchor='w')
        ttk.Button(appearance,text='Outline color…',command=lambda:self.pick_color(self.outline_color)).pack(fill='x',pady=2)
        self.glyph_type_changed()

    def render(self):
        super().render()
        if self.vectors is not None:
            self.status.set(self.status.get().replace('solid arrows',self.glyph_type.get().lower()))

    def parameter_slider(self,parent,label,variable,low,high):
        row = ttk.Frame(parent)
        row.pack(fill='x',pady=(4,0))
        ttk.Label(row,text=label).pack(side='left')
        value = tk.StringVar(value=f'{float(variable.get()):.3g}')
        ttk.Label(row,textvariable=value).pack(side='right')
        def update(number):
            value.set(f'{float(number):.3g}')
            self.schedule_render()
        ttk.Scale(parent,from_=low,to=high,variable=variable,command=update).pack(fill='x')

    def glyph_type_changed(self):
        for child in self.arrow_geometry.winfo_children():
            if isinstance(child,ttk.Scale):
                child.configure(state='normal' if self.glyph_type.get()=='Arrows' else 'disabled')
        for child in self.cuboid_geometry.winfo_children():
            if isinstance(child,ttk.Scale):
                child.configure(state='normal' if self.glyph_type.get()=='Cuboids' else 'disabled')
        self.schedule_render()

    def pick_color(self,variable):
        from tkinter import colorchooser
        color = colorchooser.askcolor(color=variable.get(),parent=self.gui.root,title='Choose 3D view color')[1]
        if color:
            variable.set(color)
            self.schedule_render()

    def glyph_faces(self,origin,vector,display_length,physical_per_display,visual_spacing,spacings):
        if self.glyph_type.get()=='Cuboids':
            return cuboid_faces(origin,np.asarray(spacings)*float(self.cuboid_scale.get()))
        radius = max(1e-6,min(display_length*self.shaft_radius.get(),visual_spacing*.2))
        cone_radius = max(1e-6,min(display_length*self.head_radius.get(),visual_spacing*.48))
        faces = solid_arrow_faces(origin/physical_per_display,vector/physical_per_display,display_length,radius,
                                 head_ratio=self.head_ratio.get(),
                                 sides={'Fast':16,'Smooth':64,'High':96,'Fast (16)':16,'Smooth (64)':64,'High (96)':96}[self.surface_quality.get()],
                                 head_radius=cone_radius)
        return [np.asarray(face)*physical_per_display for face in faces]

    def glyph_collection(self,polygons,face_colors,physical_per_display):
        lit = shade_glyph_faces(polygons,face_colors,self.ambient_light.get(),physical_per_display)
        self.last_lit_colors = lit
        self.last_base_face_colors = np.asarray(face_colors)
        # No per-face edge strokes: they reveal the mesh and create dark stripes.
        # Show outline controls the region boundary separately.
        return Poly3DCollection(polygons,facecolors=lit,edgecolors='none',linewidths=0,
                                antialiaseds=False,shade=False)

    def decorate_scene(self,ax):
        from matplotlib.colors import to_rgb,to_rgba
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        background = self.background_color.get()
        ax.set_facecolor(background)
        brightness = np.dot(to_rgb(background),(.2126,.7152,.0722))
        contrast = 'black' if brightness>.45 else 'white'
        ax.tick_params(colors=contrast)
        ax.title.set_color(contrast)
        for axis in 'xyz':
            axis_object = getattr(ax,axis+'axis')
            axis_object.label.set_color(contrast)
            axis_object.line.set_color(contrast)
            axis_object.set_pane_color(to_rgba(background))
        if self.show_outline.get():
            limits = self.base_limits
            corners = np.array([[limits['x'][x],limits['y'][y],limits['z'][z]]
                                for x,y,z in ((0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1))])
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            ax.add_collection3d(Line3DCollection([corners[[a,b]] for a,b in edges],colors=self.outline_color.get(),linewidths=.6))

    def decorate_colorbar(self,bar):
        from matplotlib.colors import to_rgb
        contrast = 'black' if np.dot(to_rgb(self.background_color.get()),(.2126,.7152,.0722))>.45 else 'white'
        bar.ax.set_facecolor(self.background_color.get())
        bar.ax.tick_params(colors=contrast)
        bar.outline.set_edgecolor(contrast)


def xy_arrow_lattice(shape,mesh,density):
    """Approximately density arrows across X with uniform physical XY spacing."""
    ny,nx = shape[:2]
    dx = abs(mesh['x']['edges'][1]-mesh['x']['edges'][0])/nx
    dy = abs(mesh['y']['edges'][1]-mesh['y']['edges'][0])/ny
    sx = max(1,round(nx/max(1,int(density))))
    sy = max(1,round(sx*dx/max(dy,1e-30)))
    xs = np.arange(min(sx//2,nx-1),nx,sx,dtype=int)
    ys = np.arange(min(sy//2,ny-1),ny,sy,dtype=int)
    return xs,ys,(sx*dx,sy*dy)


def sliced_arrow_lattice(shape,mesh,bounds,density,z_step):
    xs,ys,xy_pitch = xy_arrow_lattice((shape[1],shape[2]),mesh,density)
    samples = {}
    for axis,values in (('x',xs),('y',ys)):
        low,high = bounds[axis]
        clipped = values[(values>=low)&(values<high)]
        # A single plane or a narrow band must remain visible even between grid lines.
        samples[axis] = clipped if len(clipped) else np.array([(low+high-1)//2])
    low,high = bounds['z']
    z_step = max(1,int(z_step))
    zs = np.arange(min(z_step//2,shape[0]-1),shape[0],z_step,dtype=int)
    zs = zs[(zs>=low)&(zs<high)]
    samples['z'] = zs if len(zs) else np.array([(low+high-1)//2])
    grid = np.meshgrid(samples['x'],samples['y'],samples['z'],indexing='ij')
    points = np.column_stack([a.ravel() for a in grid])
    dz = abs(mesh['z']['edges'][1]-mesh['z']['edges'][0])/shape[0]*z_step
    return points,samples,np.array([*xy_pitch,dz])


def sampled_voxel_points(bounds,budget):
    """Sample a complete regular lattice; never remove isolated grid points."""
    sizes = np.array([bounds[a][1]-bounds[a][0] for a in 'xyz'],dtype=int)
    budget = max(1,min(int(budget),int(np.prod(sizes))))
    minimum = np.minimum(sizes,2)
    if np.prod(minimum)>budget:
        minimum = np.ones(3,dtype=int)
    best = None
    for nz in range(int(minimum[2]),min(sizes[2],budget)+1):
        for ny in range(int(minimum[1]),min(sizes[1],budget//nz)+1):
            nx = min(sizes[0],budget//(ny*nz))
            if nx<minimum[0]:
                continue
            counts = np.array([nx,ny,nz])
            spacing = sizes/counts
            balance = np.var(np.log(spacing))
            score = (int(np.prod(counts)),-balance)
            if best is None or score>best[0]:
                best = score,counts
    samples = [np.linspace(bounds[a][0],bounds[a][1]-1,int(n),dtype=int)
               if n>1 else np.array([(bounds[a][0]+bounds[a][1]-1)//2])
               for a,n in zip('xyz',best[1])]
    grid = np.meshgrid(*samples,indexing='ij')
    return np.column_stack([g.ravel() for g in grid])


def refine_vector_grid(vectors,mesh,factor):
    """Cell-centred trilinear refinement, preserving physical bounds and vacuum."""
    if factor==1:
        return vectors,mesh
    target = np.array(vectors.shape[:3],dtype=np.int64)*factor
    if int(np.prod(target))*12>1536*1024**2:
        raise ValueError('Refined vector grid exceeds 1.5 GiB. Choose a smaller upscaling factor.')
    data = np.asarray(vectors,dtype=np.float32)
    for axis in (2,1,0):
        count = data.shape[axis]
        positions = np.clip((np.arange(count*factor)+.5)/factor-.5,0,count-1)
        lower = np.floor(positions).astype(int)
        upper = np.minimum(lower+1,count-1)
        shape = list(data.shape)
        shape[axis] *= factor
        refined = np.empty(shape,dtype=np.float32)
        block = max(1,min(8,2000000//max(1,data.size//count)))
        for first in range(0,len(positions),block):
            last = min(len(positions),first+block)
            weight_shape = [1]*4
            weight_shape[axis] = last-first
            w = (positions[first:last]-lower[first:last]).astype(np.float32).reshape(weight_shape)
            low = np.take(data,lower[first:last],axis=axis)
            high = np.take(data,upper[first:last],axis=axis)
            section = [slice(None)]*4
            section[axis] = slice(first,last)
            refined[tuple(section)] = low+(high-low)*w
        data = refined
    occupied = np.any(vectors!=0,axis=-1)
    for axis in range(3):
        occupied = np.repeat(occupied,factor,axis=axis)
    data[~occupied] = 0
    new_mesh = {}
    for axis,n in zip('xyz',target[::-1]):
        info = mesh[axis]
        lo,hi = info['edges']
        new_mesh[axis] = dict(info,centers=lo+(np.arange(n)+.5)*(hi-lo)/n)
    return data,new_mesh


def set_png_ppi(path,ppi=300):
    """Update PNG density without decoding a large upscaled export."""
    import zlib
    path = Path(path)
    data = path.read_bytes()
    position = 8
    output = bytearray(data[:8])
    payload = struct.pack('>IIB',round(ppi/.0254),round(ppi/.0254),1)
    chunk = struct.pack('>I',len(payload))+b'pHYs'+payload+struct.pack('>I',zlib.crc32(b'pHYs'+payload)&0xffffffff)
    written = False
    while position<len(data):
        size = struct.unpack('>I',data[position:position+4])[0]
        kind = data[position+4:position+8]
        if kind==b'pHYs':
            if not written:
                output.extend(chunk)
                written = True
        else:
            output.extend(data[position:position+size+12])
            if kind==b'IHDR' and not written:
                output.extend(chunk)
                written = True
        position += size+12
    path.write_bytes(output)


class OverviewCrossSectionBase(SmoothGlyphCrossSectionBase):
    def __init__(self,gui,parent):
        self.fig1_zoom = 1.
        self.fig1_center = (.5,.5)
        self.fig1_pan = None
        self.figure_upscale = tk.StringVar(value='1x')
        self.fig1_zoom_label = tk.StringVar(value='Figure 1: 1.00×')
        self.fig1_nm_view = tk.StringVar()
        super().__init__(gui,parent)
        old_canvas = self.plot_canvas
        for callback in self.mpl_callbacks:
            old_canvas.mpl_disconnect(callback)
        idle = getattr(old_canvas,'_idle_draw_id',None)
        if idle:
            old_canvas.get_tk_widget().after_cancel(idle)
        self.figure.clear()
        self.figures.destroy()
        self.fig1_window = None
        self.figures = ttk.Frame(self.divider)
        toolbar = ttk.Frame(self.figures,padding=5)
        toolbar.pack(fill='x')
        self.view_title = tk.StringVar(value='Figure 1 — XY overview / ROI selection')
        ttk.Label(toolbar,textvariable=self.view_title).pack(side='left')
        ttk.Button(toolbar,text='Figure 2 — 3D view',command=lambda:self.open_figure(2)).pack(side='right')
        ttk.Button(toolbar,text='Figure 1 — 2D view',command=lambda:self.open_figure(1)).pack(side='right')
        ttk.Button(toolbar,text='XY top view',command=self.align_xy_top).pack(side='right')
        ttk.Label(self.figures,textvariable=self.status,wraplength=600,padding=5).pack(side='bottom',fill='x')
        self.fig2_window = tk.Toplevel(gui.root)
        self.fig2_window.title('Figure 2 — Interactive 3D slice analysis')
        self.fig2_window.geometry('1100x800')
        self.fig2_window.minsize(500,400)
        self.fig1_frame = ttk.Frame(self.figures)
        self.fig1_frame.pack(fill='both',expand=True)
        self.canvas = tk.Canvas(self.fig1_frame,background='#252525',highlightthickness=0)
        self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<Configure>',lambda e:self.draw_fig1_view())
        self.canvas.bind('<ButtonPress-1>',self.drag_start)
        self.canvas.bind('<B1-Motion>',self.drag_move)
        self.canvas.bind('<ButtonRelease-1>',self.drag_end)
        self.canvas.bind('<MouseWheel>',self.fig1_scroll)
        self.canvas.bind('<Button-4>',lambda e:self.change_fig1_zoom(1.2,e))
        self.canvas.bind('<Button-5>',lambda e:self.change_fig1_zoom(1/1.2,e))
        self.canvas.bind('<ButtonPress-2>',self.fig1_pan_start)
        self.canvas.bind('<B2-Motion>',self.fig1_pan_move)
        self.canvas.bind('<ButtonRelease-2>',lambda e:setattr(self,'fig1_pan',None))
        self.fig2_frame = ttk.Frame(self.figures)
        self.fig2_frame.pack(fill='both',expand=True)
        self.figure = Figure(figsize=(10,7),dpi=100,facecolor=self.background_color.get())
        self.ax3d = self.figure.add_subplot(111,projection='3d')
        self.plot_canvas = FigureCanvasTkAgg(self.figure,master=self.fig2_frame)
        self.plot_canvas.get_tk_widget().pack(fill='both',expand=True)
        self.ax3d.mouse_init(rotate_btn=2,pan_btn=1,zoom_btn=3)
        self.mpl_callbacks = [self.plot_canvas.mpl_connect('scroll_event',self.scroll_zoom),
                              self.plot_canvas.mpl_connect('key_press_event',self.camera_key)]
        self.fig2_window.protocol('WM_DELETE_WINDOW',self.fig2_window.withdraw)
        self.fig2_window.withdraw()
        # Main panel holds shared controls on the right; plots have independent windows.
        self.columnconfigure(0,weight=1)
        self.columnconfigure(1,weight=0)
        self.sidebar_canvas.grid_configure(column=0,sticky='nsew')
        for child in self.winfo_children():
            if isinstance(child,ttk.Scrollbar):
                child.grid_configure(column=1,sticky='nse')
        zoom_group = self.group('Figure 1 zoom / output resolution')
        for child in self.controls.winfo_children():
            if isinstance(child,ttk.LabelFrame) and child.cget('text').startswith('Region of interest'):
                zoom_group.pack_configure(before=child)
                break
        row = ttk.Frame(zoom_group)
        row.pack(fill='x',pady=3)
        ttk.Button(row,text='Zoom +',command=lambda:self.change_fig1_zoom(1.25)).pack(side='left',padx=2)
        ttk.Button(row,text='Zoom −',command=lambda:self.change_fig1_zoom(.8)).pack(side='left',padx=2)
        ttk.Button(row,text='Reset',command=self.reset_fig1_zoom).pack(side='left',padx=2)
        ttk.Label(zoom_group,textvariable=self.fig1_zoom_label).pack(anchor='w')
        ttk.Label(zoom_group,textvariable=self.fig1_nm_view,wraplength=235).pack(anchor='w')
        ttk.Label(zoom_group,text='Wheel: zoom around cursor. Middle-drag: pan. Left-drag: ROI or bound lines.',wraplength=235).pack(anchor='w',pady=3)
        self.dropdown(zoom_group,'Upscaling',self.figure_upscale,['1x','2x','4x','8x','10x'])
        ttk.Label(zoom_group,text='Refines XYZ vector data before slicing. Physical nm bounds stay unchanged. PNG exports retain 300 PPI.',wraplength=235).pack(anchor='w')
        self.master.bind('<<NotebookTabChanged>>',self.tab_selected,add='+')

    def tab_selected(self,event=None):
        if not self.gui.closed and self.master.select()==str(self):
            self.open_figures()

    def open_figure(self,number):
        if number==1:
            self.draw_fig1_view()
            return
        self.fig2_window.deiconify()
        self.fig2_window.lift()
        self.plot_canvas.draw_idle()

    def open_figures(self):
        self.open_figure(1)
        self.open_figure(2)

    def load_folder(self):
        self.fig1_zoom,self.fig1_center = 1.,(.5,.5)
        self.open_figures()
        super().load_folder()

    def figure_scale(self):
        scale = int(self.figure_upscale.get().replace('x',''))
        if scale not in (1,2,4,8,10):
            raise ValueError('Choose a valid upscaling factor.')
        return scale

    def panel(self,plane,view,component,title,width,height,indices,low,high,density,length,head):
        scale = self.figure_scale()*3 if getattr(self,'export_render',False) else 1
        horizontal,vertical = {'XY':('x','y'),'YZ':('y','z'),'XZ':('x','z')}[view]
        h_span = abs(self.mesh[horizontal]['edges'][1]-self.mesh[horizontal]['edges'][0])
        v_span = abs(self.mesh[vertical]['edges'][1]-self.mesh[vertical]['edges'][0])
        ratio = min(width/max(h_span,1e-30),height/max(v_span,1e-30))
        w,h = max(1,round(h_span*ratio))*scale,max(1,round(v_span*ratio))*scale
        image = scalar_slice_image(plane[:,:,{'mx':0,'my':1,'mz':2}[component]],self.colors.get(),low,high)
        image = image.resize((w,h),Image.Resampling.NEAREST)
        if view=='XY':
            image=layer_fill_2d(image,plane,self.mesh['z']['centers'][indices['z']],layer_fill_source_z(self),
                getattr(self,'layer_fill_state',{}),component,self.colors.get(),(low,high),native=layer_fill_native_data(self))
        sampling = None
        if view=='XY':
            xs,ys,_ = xy_arrow_lattice(plane.shape,self.mesh,density)
            self.last_fig1_sample_axes = {'x':xs,'y':ys}
            sampling = xs,ys
        if self.arrows.get() and getattr(self,'export_render',False):
            image = cross_section_arrows(image,plane,{'XY':'Z','YZ':'X','XZ':'Y'}[view],self.arrow_component.get(),self.arrow_mode.get(),density,length,head*scale,sample_indices=sampling,opacity=1-reflection_control_settings(self)['transparency']/100)
        image = self.reference_lines(image,view,indices)
        image = decorate_image(image,dict(colorbar=self.colorbar.get(),colorbar_size=self.gui.colorbar_size.get(),
                        colorbar_position=self.gui.colorbar_position.get(),colorbar_decimals=self.gui.colorbar_decimals.get(),
                        vmin=low,vmax=high,font_size=12*scale),colors=self.colors.get(),component=component)
        rect = [0,0,w,h]
        font,pad = annotation_font(11*scale),3*scale
        previous = image.size
        header=component
        if self.show_time.get():header+=f'\nt = {self.time_ns:.3f} ns' if self.time_ns is not None else '\nt = unavailable'
        image = annotation_text_panel(image,header,'Top center',font,pad,alignment='Center')
        dx,dy = (image.width-previous[0])//2,image.height-previous[1]
        rect = [rect[0]+dx,rect[1]+dy,rect[2]+dx,rect[3]+dy]
        if self.show_scales.get():
            a,b = self.mesh[horizontal],self.mesh[vertical]
            previous = image.size
            label = f'{horizontal.upper()}: {a["edges"][0]:.3f}–{a["edges"][1]:.3f} {a["unit"]}; {vertical.upper()}: {b["edges"][0]:.3f}–{b["edges"][1]:.3f} {b["unit"]}'
            image = annotation_text_panel(image,label,'Bottom center',font,pad)
            dx = (image.width-previous[0])//2
            rect[0],rect[2] = rect[0]+dx,rect[2]+dx
        return image,rect

    def render(self):
        super().render()
        self.draw_fig1_view()
        if self.vectors is not None and hasattr(self,'requested_sample_count'):
            self.status.set(self.status.get()+f'\nFigure 2 grid: {len(self.last_sample_axes["x"])} × {len(self.last_sample_axes["y"])} × {len(self.last_sample_axes["z"])}; zero vectors omitted.'+
                f'\nXY pitch: {self.last_sample_spacings[0]:.3f} × {self.last_sample_spacings[1]:.3f} {self.mesh["x"]["unit"]}. Figure 1 shows the full field; Figure 2 shows selected bounds.')

    def draw_fig1_view(self):
        image = self.top_export_image
        if image is None or self.gui.closed:
            return
        width,height = self.canvas.winfo_width(),self.canvas.winfo_height()
        if width<2 or height<2:
            return
        scale = min(width/image.width,height/image.height)*self.fig1_zoom
        ox = width/2-self.fig1_center[0]*image.width*scale
        oy = height/2-self.fig1_center[1]*image.height*scale
        self.display_transform = ox,oy,scale,scale
        left = max(0,math.floor(-ox/scale))
        top = max(0,math.floor(-oy/scale))
        right = min(image.width,math.ceil((width-ox)/scale))
        bottom = min(image.height,math.ceil((height-oy)/scale))
        self.canvas.delete('all')
        if left<right and top<bottom:
            # Crop before resizing: deep zoom never allocates a giant off-screen bitmap.
            crop = image.crop((left,top,right,bottom))
            crop = crop.resize((max(1,round((right-left)*scale)),max(1,round((bottom-top)*scale))),Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(crop,master=self.gui.root)
            self.canvas.create_image(ox+left*scale,oy+top*scale,image=self.photo,anchor='nw')
        self.fig1_zoom_label.set(f'Figure 1: {self.fig1_zoom:.2f}×')
        self.update_fig1_nm_view((left,top,right,bottom))

    def update_fig1_nm_view(self,viewport):
        self.fig1_nm_view.set('')
        if not self.show_scales.get() or self.mesh is None:
            return
        for a,b,c,d in self.top_data_rects:
            left,top,right,bottom = max(a,viewport[0]),max(b,viewport[1]),min(c,viewport[2]),min(d,viewport[3])
            if left>=right or top>=bottom:
                continue
            xe,ye = self.mesh['x']['edges'],self.mesh['y']['edges']
            x0 = xe[0]+(left-a)/(c-a)*(xe[1]-xe[0])
            x1 = xe[0]+(right-a)/(c-a)*(xe[1]-xe[0])
            y0 = ye[0]+(d-bottom)/(d-b)*(ye[1]-ye[0])
            y1 = ye[0]+(d-top)/(d-b)*(ye[1]-ye[0])
            self.fig1_nm_view.set(f'Visible X: {x0:.3f}–{x1:.3f} {self.mesh["x"]["unit"]}\nVisible Y: {y0:.3f}–{y1:.3f} {self.mesh["y"]["unit"]}')
            break

    def change_fig1_zoom(self,factor,event=None):
        image = self.top_export_image
        if image is None:
            return
        width,height = self.canvas.winfo_width(),self.canvas.winfo_height()
        px,py = (event.x,event.y) if event is not None else (width/2,height/2)
        point = self.pointer_in_image(SimplePointer(px,py))
        if point is None:
            return
        self.fig1_zoom = min(40.,max(1.,self.fig1_zoom*factor))
        scale = min(width/image.width,height/image.height)*self.fig1_zoom
        ox,oy = px-point[0]*scale,py-point[1]*scale
        self.fig1_center = ((width/2-ox)/(image.width*scale),(height/2-oy)/(image.height*scale))
        if self.fig1_zoom==1:
            self.fig1_center = (.5,.5)
        self.draw_fig1_view()

    def fig1_scroll(self,event):
        self.change_fig1_zoom(1.2 if event.delta>0 else 1/1.2,event)
        return 'break'

    def reset_fig1_zoom(self):
        self.fig1_zoom,self.fig1_center = 1.,(.5,.5)
        self.draw_fig1_view()

    def fig1_pan_start(self,event):
        self.fig1_pan = event.x,event.y,self.fig1_center

    def fig1_pan_move(self,event):
        if self.fig1_pan is None or self.top_export_image is None or self.display_transform is None:
            return
        x,y,center = self.fig1_pan
        image = self.top_export_image
        scale = self.display_transform[2]
        self.fig1_center = (center[0]-(event.x-x)/(image.width*scale),center[1]-(event.y-y)/(image.height*scale))
        self.draw_fig1_view()

    def export_to_folder(self,folder):
        if self.top_export_image is None or self.vectors is None:
            raise ValueError('Load a readable frame first.')
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError('Choose an existing export folder.')
        factor = self.figure_scale()
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        stem = safe_name(self.files[self.loaded_index].stem)
        paths = (folder/f'{stem}_figure1_XY_{factor}x_{stamp}.png',folder/f'{stem}_figure2_{self.view.get()}_{factor}x_{stamp}.png')
        image = self.top_export_image
        image.save(paths[0],dpi=(300,300))
        self.figure.savefig(paths[1],format='png',dpi=300*factor,facecolor=self.figure.get_facecolor())
        set_png_ppi(paths[1],300)
        return paths

    def close(self):
        super().close()
        for window in (self.fig2_window,):
            if window.winfo_exists():
                window.destroy()


class SimplePointer:
    def __init__(self,x,y):
        self.x,self.y = x,y



class PlaybackCrossSectionBase(OverviewCrossSectionBase):
    """Shared plot viewport, resizable sidebar, and refined analysis data."""
    def __init__(self,gui,parent):
        self.native_result = None
        self.refine_generation = 0
        self.refining = False
        self.applied_factor = 1
        self.active_figure = 1
        self.z_layer_step = tk.StringVar(value='1')
        self.xy_top_camera = False
        self.export_render = False
        self.camera_refine_timer = None
        self.live_traces = []
        super().__init__(gui,parent)
        self.fig1_frame.pack_forget()
        self.fig2_frame.pack_forget()
        self.divider.add(self.figures,weight=4)
        self.divider.add(self.sidebar_panel,weight=1)
        self.divider.grid(row=0,column=0,columnspan=2,sticky='nsew')
        self.columnconfigure(1,weight=0)
        self.sidebar_canvas.bind('<Configure>',self.resize_controls)
        self.after_idle(self.initial_divider)
        self.live_traces.append((self.figure_upscale,self.figure_upscale.trace_add('write',self.upscale_changed)))
        for name in ('density','arrow_mode','length','component','arrow_component','glyph_type',
                     'head_ratio','shaft_radius','head_radius','ambient_light','cuboid_scale','surface_quality','z_layer_step'):
            variable = getattr(self,name,None)
            if isinstance(variable,tk.Variable):
                token = variable.trace_add('write',lambda *args:self.schedule_render())
                self.live_traces.append((variable,token))
        self.open_figure(1)
        self.mpl_callbacks.extend([
            self.plot_canvas.mpl_connect('button_press_event',self.begin_camera_interaction),
            self.plot_canvas.mpl_connect('button_release_event',self.end_camera_interaction)])

    def begin_camera_interaction(self,event=None):
        if event is not None and event.inaxes not in getattr(self,'axes3d',[]):return
        if self.camera_refine_timer is not None:
            self.after_cancel(self.camera_refine_timer)
            self.camera_refine_timer = None
        for artist in getattr(self,'glyph_artists',[]):artist.interactive = True

    def end_camera_interaction(self,event=None):
        if self.camera_refine_timer is not None:
            self.after_cancel(self.camera_refine_timer)
            self.camera_refine_timer = None
        for artist in getattr(self,'glyph_artists',[]):artist.interactive = False
        if not self.gui.closed and self.active_figure==2:self.plot_canvas.draw_idle()

    def scroll_zoom(self,event):
        self.begin_camera_interaction(event)
        super().scroll_zoom(event)
        self.camera_refine_timer = self.after(220,self.end_camera_interaction)

    def align_xy_top(self):
        if self.vectors is None:
            self.status.set('Load a readable vector field first.')
            return
        self.view.set('XY')
        self.view_changed()
        self.xy_top_camera = True
        self.scene_view = None
        self.open_figure(2)
        self.schedule_render()

    def reset_view(self):
        self.xy_top_camera = False
        super().reset_view()

    def view_changed(self):
        self.xy_top_camera = False
        super().view_changed()

    def schedule_render(self):
        if self.gui.closed or self.render_timer is not None:
            return
        self.render_timer = self.after(75,self.render)

    def export_to_folder(self,folder):
        self.export_render = True
        try:
            self.render()
            return super().export_to_folder(folder)
        finally:
            self.export_render = False
            self.schedule_render()

    def resize_controls(self,event):
        for item in self.sidebar_canvas.find_all():
            if self.sidebar_canvas.type(item)=='window':
                self.sidebar_canvas.itemconfigure(item,width=max(100,event.width))

    def initial_divider(self):
        if self.winfo_exists():
            width = self.divider.winfo_width()
            if width>350:
                self.divider.sashpos(0,max(150,width-300))

    def open_figure(self,number):
        self.active_figure = number
        self.fig1_frame.pack_forget()
        self.fig2_frame.pack_forget()
        frame = self.fig1_frame if number==1 else self.fig2_frame
        frame.pack(fill='both',expand=True)
        self.view_title.set('Figure 1 — XY overview / ROI selection' if number==1 else 'Figure 2 — Interactive 3D slice analysis')
        if number==1:
            self.draw_fig1_view()
        else:
            self.plot_canvas.draw_idle()

    def open_figures(self):
        self.open_figure(self.active_figure)

    def loaded(self,source,index,result):
        if not self.files or index!=self.requested_index or source!=self.files[index] or result[3]:
            return super().loaded(source,index,result)
        self.native_result = source,index,result
        self.begin_refinement()

    def upscale_changed(self,*args):
        if self.native_result is not None and not self.gui.closed:
            source,index,_ = self.native_result
            if self.files and index==self.requested_index and source==self.files[index]:
                self.begin_refinement()

    def begin_refinement(self):
        source,index,result = self.native_result
        factor = self.figure_scale()
        self.refine_generation += 1
        token = self.refine_generation
        self.refining = self.loading = True
        self.status.set(f'Refining vector data to {factor}x for both figures…')
        def worker():
            try:
                vectors,mesh = refine_vector_grid(result[0],result[1],factor)
                refined = vectors,mesh,result[2],None
            except Exception as exc:
                refined = None,None,result[2],str(exc)
            if not self.gui.closed:
                self.gui.events.put((self.refined_loaded,(source,index,token,factor,refined)))
        threading.Thread(target=worker,daemon=True).start()

    def refined_loaded(self,source,index,token,factor,result):
        if token!=self.refine_generation:
            return
        self.refining = False
        if self.files and index==self.requested_index and source==self.files[index] and not result[3]:
            ratio = factor/self.applied_factor
            if ratio!=1 and self.vectors is not None:
                self.updating = True
                for axis in 'xy':
                    start,end = self.positions[axis].get(),self.band_ends[axis].get()
                    # Equal boundaries remain a single cell after refinement.
                    new_start = round((start-1)*ratio)+1
                    new_end = new_start if start==end else round(end*ratio)
                    self.positions[axis].set(new_start)
                    self.band_ends[axis].set(new_end)
                self.positions['z'].set(round((self.positions['z'].get()-.5)*ratio+.5))
                if self.roi is not None:
                    self.roi = tuple(round(v*ratio) for v in self.roi)
                self.updating = False
            self.applied_factor = factor
            self.camera_limits = self.camera_base_centers = None
            self.scene_view = None
        super().loaded(source,index,result)

    def render(self):
        if self.refining:
            self.render_timer = None
            return
        super().render()

    def close(self):
        self.refine_generation += 1
        if self.camera_refine_timer is not None:
            self.after_cancel(self.camera_refine_timer)
            self.camera_refine_timer = None
        for variable,token in self.live_traces:
            variable.trace_remove('write',token)
        self.live_traces.clear()
        super().close()



class FrozenViewValue:
    def __init__(self,value):self.value = value
    def get(self):return self.value
    def set(self,value):self.value = value


def draw_xy_vector_view(image,plane,mesh,rect,clip_rect,density,length,head,mode,selection,zoom,opacity=1.):
    """Rasterize visible vector geometry at final viewport resolution."""
    width,height = image.size
    aa = max(1,min(4,int(math.sqrt(16_000_000/max(1,width*height)))))
    layer = Image.new('RGBA',(width*aa,height*aa),(0,0,0,0))
    draw = ImageDraw.Draw(layer)
    left,top,right,bottom = rect
    ny,nx = plane.shape[:2]
    xs,ys,_ = xy_arrow_lattice(plane.shape,mesh,density)
    sx,sy = (right-left)/nx,(bottom-top)/ny
    stride_x = int(xs[1]-xs[0]) if len(xs)>1 else max(1,round(nx/density))
    stride_y = int(ys[1]-ys[0]) if len(ys)>1 else max(1,round(stride_x*sx/sy))
    maxlen = max(2.*zoom,min(sx*stride_x,sy*stride_y)*.42*length)
    base_head = min(max(1.,head)*zoom,maxlen*.45)
    padding = maxlen+base_head*2
    xs = xs[(left+(xs+.5)*sx>=-padding)&(left+(xs+.5)*sx<=width+padding)]
    ys = ys[(top+(ny-ys-.5)*sy>=-padding)&(top+(ny-ys-.5)*sy<=height+padding)]
    def polygon(points):draw.polygon([(x*aa,y*aa) for x,y in points],fill=(15,15,15,255))
    count = 0
    for row in ys:
        for col in xs:
            vx = float(plane[row,col,0]) if selection in ('ALL','mx') else 0.
            vy = float(plane[row,col,1]) if selection in ('ALL','my') else 0.
            magnitude = math.hypot(vx,vy)
            if not math.isfinite(magnitude) or magnitude<1e-8:continue
            factor = magnitude if mode=='Magnitude' else 1.
            arrow_len,arrow_head = maxlen*factor,base_head*factor
            if arrow_len<.2:continue
            ux,uy = vx/magnitude,-vy/magnitude
            cx,cy = left+(col+.5)*sx,top+(ny-row-.5)*sy
            x1,y1 = cx-ux*arrow_len/2,cy-uy*arrow_len/2
            x2,y2 = cx+ux*arrow_len/2,cy+uy*arrow_len/2
            half = max(.25*zoom*factor,arrow_head/8)
            back = arrow_head*1.7
            polygon(solid_arrow_polygon_2d((x1,y1),(x2,y2),half,
                                          back*math.cos(.55),back*math.sin(.55)))
            count += 1
    if aa>1:layer = layer.resize((width,height),Image.Resampling.LANCZOS)
    if opacity<1:layer.putalpha(layer.getchannel("A").point(lambda v:round(v*max(0.,opacity))))
    x0,y0,x1,y1 = clip_rect
    # Clip glyphs to this component's data panel, preserving captions/color bars.
    result = Image.alpha_composite(image.crop(clip_rect).convert('RGBA'),layer.crop(clip_rect))
    image.paste(result.convert('RGB'),(x0,y0))
    return count,aa


class CrossSectionTab(PlaybackCrossSectionBase):
    def __init__(self,gui,parent):
        self.playing = False
        self.play_timer = None
        self.play_deadline = 0.
        self.mp4_busy = False
        self.mp4_stop = threading.Event()
        self.mp4_job = None
        self.mp4_thread = None
        self.play_fps = tk.StringVar(value='24')
        self.movie_view = tk.IntVar(value=2)
        self.range_start = tk.StringVar(value='1')
        self.range_end = tk.StringVar(value='1')
        self.xy_scalar_cache = {}
        self.xy_scalar_key = None
        super().__init__(gui,parent)
        # Keep field labels/readouts; move explanations and duplicate status to footer.
        def tidy(widget):
            for child in list(widget.winfo_children()):
                if isinstance(child,ttk.Label):
                    text = str(child.cget('text'))
                    variable = str(child.cget('textvariable'))
                    if len(text)>70 or '\n' in text or text.startswith(('Lower =','Radii are','Arrow colors','Up to','Bounds are')) or variable in (str(self.status),str(self.fig1_zoom_label),str(self.fig1_nm_view)):
                        child.destroy();continue
                if isinstance(child,ttk.Button) and str(child.cget('text')) in ('Zoom +','Zoom −'):
                    child.destroy();continue
                tidy(child)
        tidy(self.controls)
        for child in list(self.figures.winfo_children()):
            if isinstance(child,ttk.Label) and str(child.cget('textvariable'))==str(self.status):child.destroy()
        self.footer = ttk.Label(self,textvariable=self.status,wraplength=900,padding=(5,3))
        self.footer.grid(row=1,column=0,columnspan=2,sticky='ew')
        self.bind('<Configure>',lambda e:self.footer.configure(wraplength=max(200,e.width-20)),add='+')
        files = self.frame_slider.master
        # Upscaling belongs immediately below the source-frame slider/readout.
        zoom_group = next((w for w in self.controls.winfo_children() if isinstance(w,ttk.LabelFrame)
                          and w.cget('text')=='Figure 1 zoom / output resolution'),None)
        if zoom_group is not None:
            for child in list(zoom_group.winfo_children()):
                if isinstance(child,ttk.Combobox) and str(child.cget('textvariable'))==str(self.figure_upscale):child.destroy()
                elif isinstance(child,ttk.Label) and str(child.cget('text'))=='Upscaling':child.destroy()
            zoom_group.configure(text='2D view')
        self.dropdown(files,'Upscaling',self.figure_upscale,['1x','2x','4x','8x','10x'])
        selection = ttk.Frame(files);selection.pack(fill='x',pady=2)
        ttk.Checkbutton(selection,text='2D',variable=self.movie_view,onvalue=1,offvalue=2,command=self.movie_view_changed).pack(side='left',padx=3)
        ttk.Checkbutton(selection,text='3D',variable=self.movie_view,onvalue=2,offvalue=1,command=self.movie_view_changed).pack(side='left',padx=3)
        playback = ttk.Frame(files)
        playback.pack(fill='x',pady=(5,2))
        self.play_button = ttk.Button(playback,text='Play',command=self.toggle_play)
        self.play_button.pack(side='left',padx=2)
        ttk.Label(playback,text='FPS').pack(side='left',padx=(5,2))
        self.fps_spin = ttk.Spinbox(playback,from_=1,to=120,textvariable=self.play_fps,width=5)
        self.fps_spin.pack(side='left')
        row = ttk.Frame(files);row.pack(fill='x',pady=2)
        ttk.Label(row,text='Start').pack(side='left',padx=2)
        self.start_spin = ttk.Spinbox(row,from_=1,to=1,textvariable=self.range_start,width=6)
        self.start_spin.pack(side='left',padx=2)
        ttk.Label(row,text='End').pack(side='left',padx=2)
        self.end_spin = ttk.Spinbox(row,from_=1,to=1,textvariable=self.range_end,width=6)
        self.end_spin.pack(side='left',padx=2)
        row = ttk.Frame(files);row.pack(fill='x',pady=2)
        self.mp4_button = ttk.Button(row,text='Export MP4',command=self.export_mp4)
        self.mp4_button.pack(side='left',padx=2)
        self.mp4_stop_button = ttk.Button(row,text='Stop',command=self.stop_mp4,state='disabled')
        self.mp4_stop_button.pack(side='left',padx=2)
        self.status.set('Load a vector-field folder. Wheel: zoom; middle-drag: pan (2D) / rotate (3D).')

        self.layer_fill_state=dict(enabled=False,layers=[])
        self.layer_fill_control=LayerFillControl(self.controls,self.schedule_render,self.status)
        self.layer_fill_control.pack(fill='x',pady=5)

    def choose_color_preset(self):
        self.color_preset_picker = show_shared_color_presets(self.gui.root,self,self.colors,self.schedule_render)

    def movie_view_changed(self):
        self.open_figure(1 if self.movie_view.get()==1 else 2)

    def frame_range(self):
        if not self.files:raise ValueError('Load an OVF folder first.')
        first,last = int(self.range_start.get()),int(self.range_end.get())
        if not 1<=first<=last<=len(self.files):
            raise ValueError(f'Use 1 ≤ Start ≤ End ≤ {len(self.files)}.')
        return first-1,last-1

    def fps_value(self):
        fps = float(self.play_fps.get())
        if not math.isfinite(fps) or not 1<=fps<=120:raise ValueError('FPS must be between 1 and 120.')
        return fps

    def load_folder(self):
        if self.mp4_busy:return
        self.pause_play()
        super().load_folder()
        if self.files:
            self.range_start.set('1');self.range_end.set(str(len(self.files)))
            self.start_spin.configure(to=len(self.files));self.end_spin.configure(to=len(self.files))

    def toggle_play(self):
        if self.playing:self.pause_play();return
        if self.mp4_busy:return
        try:first,last = self.frame_range();self.fps_value()
        except (ValueError,TypeError) as exc:self.status.set(str(exc));return
        self.playing = True
        self.play_button.configure(text='Pause')
        self.open_figure(1 if self.movie_view.get()==1 else 2)
        current = self.loaded_index if self.loaded_index is not None else first
        self.play_next = first if not first<=current<=last else current
        self.play_deadline = time.monotonic()
        self.play_timer = self.after(1,self.play_tick)

    def pause_play(self):
        self.playing = False
        if self.play_timer is not None:self.after_cancel(self.play_timer);self.play_timer = None
        if hasattr(self,'play_button'):self.play_button.configure(text='Play')

    def play_tick(self):
        self.play_timer = None
        if not self.playing or self.gui.closed:return
        if self.loading or self.refining or self.render_timer is not None:
            self.play_timer = self.after(10,self.play_tick);return
        try:first,last = self.frame_range();fps = self.fps_value()
        except (ValueError,TypeError) as exc:self.pause_play();self.status.set(str(exc));return
        target = self.play_next
        if not first<=target<=last:target = first
        self.play_next = first if target==last else target+1
        self.request_frame(target)
        self.play_deadline = max(self.play_deadline+1/fps,time.monotonic())
        self.play_timer = self.after(max(1,round((self.play_deadline-time.monotonic())*1000)),self.play_tick)

    def loaded(self,source,index,result):
        if result[3] and index==self.requested_index:
            if self.playing:
                self.loading = False
                self.status.set(f'Skipped unreadable frame: {source.name}')
                return
            self.top_export_image = None
            self.xy_scalar_cache = {}
            self.figure.clear();self.plot_canvas.draw_idle()
        super().loaded(source,index,result)

    def frame_changed(self,value):
        if self.mp4_busy:return
        if self.playing and not self.updating:self.pause_play()
        super().frame_changed(value)

    def open_figure(self,number):
        if self.playing and number!=self.movie_view.get():self.pause_play()
        super().open_figure(number)

    def draw_fig1_view(self):
        image = self.top_export_image
        if image is None or self.gui.closed:return
        width,height = self.canvas.winfo_width(),self.canvas.winfo_height()
        if width<2 or height<2:return
        scale = min(width/image.width,height/image.height)*self.fig1_zoom
        ox,oy = width/2-self.fig1_center[0]*image.width*scale,height/2-self.fig1_center[1]*image.height*scale
        self.display_transform = ox,oy,scale,scale
        left,top = max(0,math.floor(-ox/scale)),max(0,math.floor(-oy/scale))
        right,bottom = min(image.width,math.ceil((width-ox)/scale)),min(image.height,math.ceil((height-oy)/scale))
        view = Image.new('RGB',(width,height),'#252525')
        header=['mx | my | mz' if self.component.get()=='ALL' else self.component.get()]
        if self.show_time.get():header.append(f't = {self.time_ns:.3f} ns' if self.time_ns is not None else 't = unavailable')
        header_height=figure_header_height(header,11)
        # Keep the viewport header fixed while data zoom/pan uses physical coordinates.
        display_image=image.copy()
        if self.top_data_rects:
            first_data_y=min(rect[1] for rect in self.top_data_rects)
            ImageDraw.Draw(display_image).rectangle((0,0,image.width,first_data_y-1),fill='white')
        if left<right and top<bottom:
            crop = display_image.crop((left,top,right,bottom)).resize((max(1,round((right-left)*scale)),max(1,round((bottom-top)*scale))),Image.Resampling.LANCZOS)
            view.paste(crop,(round(ox+left*scale),round(oy+top*scale)))
        if self.vectors is not None and hasattr(self,'last_top_plane'):
            components = ['mx','my','mz'] if self.component.get()=='ALL' else [self.component.get()]
            key = (id(self.vectors),self.loaded_index,self.indices()['z'],self.colors.get(),self.low.get(),self.high.get())
            if self.xy_scalar_key!=key:self.xy_scalar_cache = {};self.xy_scalar_key = key
            self.last_sharp_arrow_count = 0
            for component,(a,b,c,d) in zip(components,self.top_data_rects):
                rect = (ox+a*scale,oy+b*scale,ox+c*scale,oy+d*scale)
                x0,y0 = max(0,math.ceil(rect[0])),max(header_height,math.ceil(rect[1]))
                x1,y1 = min(width,math.floor(rect[2])),min(height,math.floor(rect[3]))
                if x0>=x1 or y0>=y1:continue
                if component not in self.xy_scalar_cache:
                    self.xy_scalar_cache[component] = scalar_slice_image(self.last_top_plane[:,:,{'mx':0,'my':1,'mz':2}[component]],self.colors.get(),float(self.low.get()),float(self.high.get()))
                scalar = self.xy_scalar_cache[component]
                sw,sh = scalar.size
                extent = ((x0-rect[0])*sw/(rect[2]-rect[0]),(y0-rect[1])*sh/(rect[3]-rect[1]),
                          (x1-rect[0])*sw/(rect[2]-rect[0]),(y1-rect[1])*sh/(rect[3]-rect[1]))
                patch = scalar.transform((x1-x0,y1-y0),Image.Transform.EXTENT,extent,resample=Image.Resampling.BILINEAR)
                patch=layer_fill_2d(patch,self.last_top_plane,self.mesh['z']['centers'][self.indices()['z']],layer_fill_source_z(self),
                    getattr(self,'layer_fill_state',{}),component,self.colors.get(),(float(self.low.get()),float(self.high.get())),extent,layer_fill_native_data(self))
                view.paste(patch,(x0,y0))
                if self.arrows.get():
                    count,aa = draw_xy_vector_view(view,self.last_top_plane,self.mesh,rect,(x0,y0,x1,y1),
                        int(self.density.get()),float(self.length.get()),float(self.head.get()),self.arrow_mode.get(),self.arrow_component.get(),scale,1-reflection_control_settings(self)['transparency']/100)
                    self.last_sharp_arrow_count += count;self.last_sharp_aa = aa
                self.draw_view_bounds(view,rect,(x0,y0,x1,y1))
        draw_figure_header(view,header,11)
        self.fig1_view_image = view
        if not getattr(self,'export_2d_only',False):
            self.photo = ImageTk.PhotoImage(view,master=self.gui.root)
            self.canvas.delete('all');self.canvas.create_image(0,0,image=self.photo,anchor='nw')
        self.fig1_zoom_label.set(f'Figure 1: {self.fig1_zoom:.2f}×')
        self.update_fig1_nm_view((left,top,right,bottom))

    def draw_view_bounds(self,image,rect,clip):
        draw = ImageDraw.Draw(image)
        nx,ny = self.vectors.shape[2],self.vectors.shape[1]
        bounds = self.bounds();font = annotation_font(9)
        x0,y0,x1,y1 = clip
        for axis,count in (('x',nx),('y',ny)):
            first,last = bounds[axis][0],bounds[axis][1]-1
            for which,cell in enumerate((first,last) if first!=last else (first,)):
                coordinate = rect[0]+(cell+.5)*(rect[2]-rect[0])/count if axis=='x' else rect[1]+(count-cell-.5)*(rect[3]-rect[1])/count
                color = ((255,235,0),(255,150,0))[which] if axis=='x' else ((0,235,255),(255,0,255))[which]
                if axis=='x' and x0<=coordinate<x1:
                    points = (coordinate,y0,coordinate,y1-1);position = (min(x1-20,coordinate+3),y0+2)
                elif axis=='y' and y0<=coordinate<y1:
                    points = (x0,coordinate,x1-1,coordinate);position = (x0+2,min(y1-15,coordinate+2))
                else:continue
                draw.line(points,fill='black',width=3);draw.line(points,fill=color,width=1)
                draw.text(position,axis.upper()+str(which+1),font=font,fill='black',stroke_width=1,stroke_fill='white')

    def render(self):
        try:light_control_settings(self);reflection_control_settings(self)
        except ValueError as exc:self.status.set(str(exc));return
        if self.vectors is not None and hasattr(self,'layer_fill_control'):
            self.layer_fill_control.sync_layers(layer_fill_source_z(self))
            self.layer_fill_state=self.layer_fill_control.snapshot()
        super().render()
        if self.vectors is None or self.refining or self.mp4_busy:return
        if self.status.get().startswith('Cross section:'):return
        name = self.files[self.loaded_index].name if self.files and self.loaded_index is not None else ''
        text = f'{name} | Frame {self.loaded_index+1}/{len(self.files)}' if name else ''
        if self.show_time.get() and self.time_ns is not None:text += f' | {self.time_ns:.3f} ns'
        if hasattr(self,'last_sample_spacings'):
            p = self.last_sample_spacings
            text += f' | XY pitch {p[0]:.3f} × {p[1]:.3f} {self.mesh["x"]["unit"]}'
        if self.playing:text += f' | Playing ({self.play_fps.get()} FPS target)'
        if self.files and self.loaded_index is not None:
            note = field_source_note(self.files[self.loaded_index])
            if note:text += ' | '+note
        self.status.set(text)

    def export_mp4(self):
        if self.mp4_busy:return
        try:
            first,last=self.frame_range();fps=self.fps_value()
            if self.loading or self.refining:raise ValueError('Wait for the current frame to finish loading.')
            if self.vectors is None:raise ValueError('Load a readable OVF frame first.')
        except (ValueError,TypeError) as exc:self.status.set(str(exc));return
        parent=filedialog.askdirectory(parent=self.gui.root,title='Choose output folder for Tab 3 MP4')
        if not parent:return
        folder=None
        try:
            view='Figure1_XY_2D' if self.movie_view.get()==1 else 'Figure2_'+self.view.get()+'_3D'
            source=self.files[first]
            folder=auto_export_folder(parent,3,source,'MP4')
            filename=auto_export_filename(source,view,self.component.get(),self.figure_scale(),'mp4',first,last,fps)
            self.start_mp4_export(folder/filename,auto_folder=folder)
        except Exception as exc:
            discard_empty_export_folder(folder);self.status.set('MP4 export failed: '+str(exc))

    def set_movie_controls(self,busy):
        if busy:
            self.movie_widget_states = []
            def disable(widget):
                for child in widget.winfo_children():
                    if child is self.mp4_stop_button:continue
                    if isinstance(child,(ttk.Button,ttk.Entry,ttk.Combobox,ttk.Spinbox,ttk.Scale,ttk.Checkbutton,ttk.Radiobutton)):
                        self.movie_widget_states.append((child,child.cget('state')))
                        child.configure(state='disabled')
                    disable(child)
            disable(self.controls)
            self.mp4_stop_button.configure(state='normal')
        else:
            for widget,state in getattr(self,'movie_widget_states',[]):
                if widget.winfo_exists():widget.configure(state=state)
            self.mp4_stop_button.configure(state='disabled')

        if hasattr(self,'layer_fill_control'):self.layer_fill_control.set_busy(busy)

    def start_mp4_export(self,path,auto_folder=None):
        import copy
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        if self.mp4_busy:raise ValueError('A video export is already running.')
        first,last = self.frame_range();fps = self.fps_value()
        if self.vectors is None:raise ValueError('Load a readable OVF frame first.')
        light_control_settings(self)
        self.pause_play()
        movie_view = 1 if self.movie_view.get()==1 else 2
        self.open_figure(movie_view);self.gui.root.update_idletasks()
        widget = self.canvas if movie_view==1 else self.plot_canvas.get_tk_widget()
        width = max(2,widget.winfo_width()//2*2)
        height = max(2,widget.winfo_height()//2*2)
        target = Path(path)
        if not target.parent.is_dir():raise ValueError('Choose an existing output folder.')
        partial = target.with_name(target.stem+'.partial-'+str(time.time_ns())+'.mp4')
        clone = copy.copy(self)
        clone.layer_fill_state=copy.deepcopy(self.layer_fill_control.snapshot())
        for key,value in vars(self).items():
            if isinstance(value,tk.Variable):setattr(clone,key,FrozenViewValue(value.get()))
        clone.positions = {a:FrozenViewValue(v.get()) for a,v in self.positions.items()}
        clone.band_ends = {a:FrozenViewValue(v.get()) for a,v in self.band_ends.items()}
        clone.gui = copy.copy(self.gui)
        for key,value in vars(self.gui).items():
            if isinstance(value,tk.Variable):setattr(clone.gui,key,FrozenViewValue(value.get()))
        clone.figure = Figure(figsize=(width/100,height/100),dpi=100)
        clone.plot_canvas = FigureCanvasAgg(clone.figure)
        clone.ax3d = clone.figure.add_subplot(111,projection='mumax_full_3d')
        clone.ax3d.view_init(elev=self.ax3d.elev,azim=self.ax3d.azim,roll=self.ax3d.roll)
        clone.movie_camera_fit = getattr(self.ax3d,'_fit_scale',None)
        clone.scene_view = self.view.get()
        clone.axes3d = []
        clone.camera_limits = copy.deepcopy(self.camera_limits)
        clone.camera_base_centers = copy.deepcopy(self.camera_base_centers)
        clone.active_figure = movie_view
        clone.export_2d_only = movie_view==1
        clone.canvas = SimpleNamespace(winfo_width=lambda:width,winfo_height=lambda:height)
        clone.xy_scalar_cache = {};clone.xy_scalar_key = None
        self.mp4_stop.clear();self.mp4_busy = True
        job = dict(token=object(),clone=clone,first=first,last=last,fps=fps,path=target,partial=partial,
                   size=(width,height),view=movie_view,factor=self.figure_scale(),files=list(self.files),encoded=0,skipped=[],error=None,saved=False,auto_folder=auto_folder)
        self.mp4_job = job
        self.set_movie_controls(True)
        self.status.set(f'Exporting frames {first+1}–{last+1} at {fps:g} FPS ({width}×{height})…')
        def worker():
            writer = None
            try:
                writer = cv2.VideoWriter(str(partial),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
                if not writer.isOpened():raise ValueError('Could not open the MP4 encoder for this video size.')
                for index in range(first,last+1):
                    if self.mp4_stop.is_set() or self.gui.closed:break
                    source = job['files'][index]
                    try:
                        vectors = parse_ovf_vectors(source)
                        mesh = ovf_mesh_coordinates(source,vectors.shape)
                        native_fill=(vectors,mesh)
                        vectors,mesh = refine_vector_grid(vectors,mesh,job['factor'])
                        timestamp = read_ovf_time_ns(source)
                    except Exception as exc:
                        job['skipped'].append((source.name,str(exc)));continue
                    ack = threading.Event();reply = {}
                    self.gui.events.put((self.render_movie_frame,(job,index,(vectors,mesh,timestamp,native_fill),ack,reply)))
                    while not ack.wait(.05):
                        if self.mp4_stop.is_set() or self.gui.closed:break
                    if self.mp4_stop.is_set() or self.gui.closed:break
                    if 'error' in reply:raise ValueError(reply['error'])
                    writer.write(reply['frame']);job['encoded'] += 1
                writer.release();writer = None
                if self.mp4_stop.is_set() or self.gui.closed:
                    partial.unlink(missing_ok=True)
                elif not job['encoded']:
                    raise ValueError('No readable OVF frames in the selected range.')
                else:os.replace(partial,target);job['saved']=True
            except Exception as exc:
                job['error'] = str(exc)
                if writer is not None:writer.release()
                partial.unlink(missing_ok=True)
            finally:
                if not self.gui.closed:self.gui.events.put((self.finish_mp4,(job,)))
        self.mp4_thread = threading.Thread(target=worker,daemon=True)
        self.mp4_thread.start()

    def prepare_2d_movie_view(self,size,indices,components):
        width,height=size
        top=self.vectors[indices['z']]
        self.last_top_plane=top
        low,high=float(self.low.get()),float(self.high.get())
        density,length,head=int(self.density.get()),float(self.length.get()),float(self.head.get())
        panels=[self.panel(top,'XY',component,'Figure 1: top view',max(60,width//len(components)-50),
                           max(20,height-75),indices,low,high,density,length,head) for component in components]
        image=Image.new('RGB',(sum(p.width for p,r in panels)+8*(len(panels)-1),max(p.height for p,r in panels)),'white')
        rects=[];x=0
        for panel,(a,b,c,d) in panels:
            image.paste(panel,(x,0));rects.append((a+x,b,c+x,d));x+=panel.width+8
        self.top_data_rects,self.bottom_data_rects=rects,[]
        self.top_export_image=self.rendered_image=image

    def render_movie_frame(self,job,index,result,ack,reply):
        try:
            if self.gui.closed or self.mp4_stop.is_set() or self.mp4_job is not job:return
            clone = job['clone']
            clone.vectors,clone.mesh,clone.time_ns = result[:3]
            if len(result)>3:clone.layer_fill_native_data=result[3]
            for axis,count in zip('xyz',clone.vectors.shape[:3][::-1]):
                start = min(count,max(1,round(clone.positions[axis].get())))
                clone.positions[axis].set(start)
                if axis in 'xy':clone.band_ends[axis].set(min(count,max(start,round(clone.band_ends[axis].get()))))
            indices = clone.indices()
            normal = {'XY':'Z','YZ':'X','XZ':'Y'}[clone.view.get()]
            plane,_,_,_ = extract_cross_section(clone.vectors,normal,indices[normal.lower()])
            components = ['mx','my','mz'] if clone.component.get()=='ALL' else [clone.component.get()]
            if job['view']==1:
                clone.prepare_2d_movie_view(job['size'],indices,components)
                clone.draw_fig1_view()
                reply['frame'] = np.ascontiguousarray(cv2.cvtColor(np.asarray(clone.fig1_view_image),cv2.COLOR_RGB2BGR))
            else:
                clone.render_scene(plane,indices,normal,components,float(clone.low.get()),float(clone.high.get()),
                                   int(clone.density.get()),float(clone.length.get()),float(clone.head.get()))
                clone.plot_canvas.draw()
                rgba = np.asarray(clone.plot_canvas.buffer_rgba())
                # Agg may round a fractional inch dimension down by one pixel.
                if rgba.shape[1::-1]!=job['size']:
                    rgba = cv2.resize(rgba,job['size'],interpolation=cv2.INTER_AREA)
                reply['frame'] = np.ascontiguousarray(cv2.cvtColor(rgba,cv2.COLOR_RGBA2BGR))
            self.status.set(f'Exporting frame {index+1}/{len(job["files"])} | Range {job["first"]+1}–{job["last"]+1} | {job["fps"]:g} FPS')
        except Exception as exc:reply['error'] = str(exc)
        finally:ack.set()

    def stop_mp4(self):
        if self.mp4_busy:self.mp4_stop.set();self.status.set('Stopping video export…')

    def finish_mp4(self,job):
        if self.mp4_job is not job:return
        close_mumax_gpu_scenes(job['clone'].figure)
        self.mp4_busy = False;self.mp4_job = None
        self.set_movie_controls(False)
        job['clone'].figure.clear()
        if job.get('auto_folder'):
            if job['saved']:open_folder(job['auto_folder'])
            else:discard_empty_export_folder(job['auto_folder'])
        if job['error']:self.status.set('MP4 export failed: '+job['error'])
        elif self.mp4_stop.is_set():self.status.set('MP4 export stopped. Partial video removed.')
        else:self.status.set(f'Saved {job["path"]} | {job["encoded"]} frames | Skipped {len(job["skipped"])} unreadable')

    def close(self):
        picker=getattr(self,'color_preset_picker',None)
        if picker is not None and picker.winfo_exists():picker.destroy()
        self.pause_play();self.mp4_stop.set()
        if self.mp4_job is not None:close_mumax_gpu_scenes(self.mp4_job['clone'].figure)
        close_mumax_gpu_scenes(self.figure)
        super().close()




class Tab4DependencyError(RuntimeError):
    pass


def require_tab4_backend():
    try:
        import pyvista as pv
        import vtk
        return pv
    except ImportError as exc:
        raise Tab4DependencyError(
            'Missing 3D viewer dependencies in this notebook kernel. Run %pip install pyvista vtk numba, '
            'then restart the kernel. Python: '+sys.executable) from exc


def find_mumax_vtk_converter():
    configured = os.environ.get('MUMAX3_CONVERT','').strip()
    candidates = [configured,shutil.which('mumax3-convert'),shutil.which('mumax3-convert.exe'),
                  str(Path(sys.executable).resolve().parent/'mumax3-convert.exe') if getattr(sys,'frozen',False) else None]
    for value in candidates:
        if value and Path(value).is_file():return str(Path(value).resolve())
    raise FileNotFoundError('VTK fallback requires mumax3-convert. Add it to PATH or set MUMAX3_CONVERT to its executable path.')


def vtk_point_vectors(dataset):
    dimensions = tuple(int(n) for n in getattr(dataset,'dimensions',()))
    if len(dimensions)!=3 or min(dimensions)<1:
        raise ValueError('Converted VTK is not a rectangular 3D grid.')
    candidates = ['m',getattr(dataset,'active_vectors_name',None),*dataset.point_data.keys()]
    for name in candidates:
        if not name or name not in dataset.point_data:continue
        data = np.asarray(dataset.point_data[name])
        if data.ndim==2 and data.shape==(int(np.prod(dimensions)),3):
            vectors = np.array(data,dtype=np.float32).reshape((*dimensions[::-1],3))
            if not np.isfinite(vectors).all():raise ValueError('Converted VTK contains non-finite vectors.')
            return vectors
    raise ValueError('Converted VTK does not contain a three-component magnetization field.')


def converted_vtk_mesh(dataset,vectors,source):
    """Use OVF cell-centre metadata; MuMax VTS coordinates otherwise start at zero."""
    mesh = _native_ovf_mesh_coordinates(source,vectors.shape)
    points = np.asarray(dataset.points).reshape((*vectors.shape[:3],3))
    with Path(source).open('rb') as stream:header = stream.read(65536).decode('latin1')
    match = re.search(r'^\s*#\s*meshunit\s*:\s*(\S+)',header,re.I|re.M)
    unit = match.group(1) if match else 'm'
    factor = {'m':1e9,'nm':1.,'um':1e3,'µm':1e3}.get(unit,1.)
    shown_unit = 'nm' if unit in ('m','nm','um','µm') else unit
    for index,axis in enumerate('xyz'):
        if mesh[axis]['unit']!='cell':continue
        centers = [points[0,0,:,0],points[0,:,0,1],points[:,0,0,2]][index]*factor
        if len(centers)>1:
            differences = np.diff(centers)
            step = float(np.median(differences))
            if step<=0 or not np.allclose(differences,step,rtol=1e-4,atol=max(1e-12,abs(step)*1e-5)):
                raise ValueError('Converted VTK coordinates are not a uniform rectangular mesh.')
        else:
            value = re.search(r'^\s*#\s*'+axis+r'stepsize\s*:\s*([-+0-9.eE]+)',header,re.I|re.M)
            if value is None:raise ValueError('Cannot determine spacing of the converted single-layer VTK.')
            step = float(value.group(1))*factor
        mesh[axis] = dict(centers=centers.copy(),edges=(centers[0]-step/2,centers[-1]+step/2),unit=shown_unit)
    return mesh


def read_cached_mumax_vtk(path):
    return read_field_file(path)


# Shared field I/O for every view, preview, image export, and movie export.
FIELD_EXTS = frozenset({'.ovf', '.omf', '.ohf', '.obf', '.oef',
                        '.vtk', '.vti', '.vts', '.vtr', '.vtu', '.vtp'})
VTK_FIELD_EXTS = frozenset({'.vtk', '.vti', '.vts', '.vtr', '.vtu', '.vtp'})
FIELD_FORMAT_LABEL = 'OVF / OOMMF / VTK'
_FIELD_LOCAL = threading.local()
_FIELD_NOTES = {}
_FIELD_NOTE_LOCK = threading.RLock()
_native_parse_ovf_vectors = parse_ovf_vectors
_native_ovf_mesh_coordinates = ovf_mesh_coordinates
_native_read_ovf_time_ns = read_ovf_time_ns


def field_files(folder):
    return sorted((p for p in Path(folder).iterdir()
                   if p.is_file() and p.suffix.lower() in FIELD_EXTS), key=natural_key)


def video_source_files(folder):
    images = sorted((p for p in Path(folder).iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=natural_key)
    return images or field_files(folder)


def field_render_settings(gui):
    """Snapshot Tk variables on the main thread before starting any worker."""
    return dict(component=gui.video_component.get(),
                upscale=int(gui.upscale.get().replace('x', '')),
                arrows=bool(gui.arrows.get()), density=int(gui.arrow_density.get()),
                length=float(gui.arrow_length.get()), head=float(gui.arrow_head.get()),
                mode=gui.arrow_mode.get())


def field_source_note(path):
    with _FIELD_NOTE_LOCK:
        return _FIELD_NOTES.get(str(Path(path).resolve()), '')


def field_unit(unit):
    unit = str(unit).strip().strip('{}"').lower().replace('μ', 'µ')
    factors = {'m': 1e9, 'meter': 1e9, 'metre': 1e9, 'nm': 1.,
               'nanometer': 1., 'nanometre': 1., 'um': 1e3, 'µm': 1e3,
               'mm': 1e6, 'cm': 1e7, 'pm': .001}
    return (factors[unit], 'nm') if unit in factors else (1., unit or 'VTK units')


def field_axis(centers, unit, step=None, edges=None):
    centers = np.asarray(centers, dtype=float).copy()
    if centers.size < 1 or not np.isfinite(centers).all():
        raise ValueError('The field contains invalid mesh coordinates.')
    if step is None:
        step = float((centers[-1]-centers[0])/(len(centers)-1)) if len(centers) > 1 else 1.
    if not math.isfinite(step) or step <= 0:
        raise ValueError('The field has invalid cell spacing.')
    return dict(centers=centers, edges=tuple(edges) if edges is not None else
                (float(centers[0]-step/2), float(centers[-1]+step/2)), unit=unit)


def choose_field_vectors(dataset):
    """Prefer active vectors, then conventional field names, then a 3-vector array."""
    active = getattr(dataset, 'active_vectors_info', None)
    association = getattr(getattr(active, 'association', None), 'value', -1)
    active_name = getattr(active, 'name', None)
    for where, arrays, count in (('point', dataset.point_data, dataset.n_points),
                                 ('cell', dataset.cell_data, dataset.n_cells)):
        preferred = [active_name] if association == (0 if where == 'point' else 1) else []
        names = list(dict.fromkeys(preferred + ['m', 'M', 'magnetization', 'Magnetization',
                                               'H', 'B', 'vectors'] + list(arrays.keys())))
        for name in names:
            if name is None or name not in arrays or str(name).lower() in ('normals', 'tcoords'):
                continue
            values = np.asarray(arrays[name])
            if values.shape == (count, 3):
                if not np.isfinite(values).all():
                    raise ValueError(f'VTK vector array {name!r} contains non-finite values.')
                return where, name, np.asarray(values, dtype=np.float32)
    raise ValueError('This VTK file has no three-component vector field (point or cell data).')


def field_mesh_from_axes(axes, factor, unit, steps=None, edges=None):
    return {a: field_axis(np.asarray(axes[i])*factor, unit,
                          None if steps is None else abs(float(steps[i]))*factor,
                          None if edges is None else np.asarray(edges[i])*factor)
            for i, a in enumerate('xyz')}


def uniform_field_axes(vectors, mesh):
    """Keep uniform grids exact; regularize nonuniform axes for the analysis filters."""
    changed = False
    for a, dimension in zip('xyz', (2, 1, 0)):
        old = mesh[a]['centers']
        if len(old) < 3:
            continue
        new = np.linspace(old[0], old[-1], len(old))
        tolerance = max(abs(old[-1]-old[0])*1e-7, np.finfo(float).eps)
        if np.allclose(old, new, rtol=0, atol=tolerance):
            continue
        # Interpolate the vectors, preserving their magnitude and physical extent.
        indices = np.clip(np.searchsorted(old, new)-1, 0, len(old)-2)
        weight = (new-old[indices])/(old[indices+1]-old[indices])
        shape = [1]*4; shape[dimension] = len(new)
        vectors = (np.take(vectors, indices, axis=dimension)*(1-weight.reshape(shape)) +
                   np.take(vectors, indices+1, axis=dimension)*weight.reshape(shape)).astype(np.float32)
        mesh[a] = field_axis(new, mesh[a]['unit'], edges=mesh[a]['edges'])
        changed = True
    return vectors, mesh, changed


def scattered_field_grid(points, values, factor=1., unit='nm', bounds=None):
    """Recover Cartesian point lattices exactly; sample scattered nodes conservatively."""
    points = np.asarray(points, dtype=float)*factor
    values = np.asarray(values, dtype=np.float32)
    if points.shape != values.shape or not np.isfinite(points).all():
        raise ValueError('Invalid field node positions.')
    axes = [np.unique(points[:, i]) for i in range(3)]
    dimensions = tuple(len(a) for a in axes)
    # A complete Cartesian lattice is common in OOMMF irregular-mesh exports.
    if math.prod(dimensions) == len(points):
        indices = [np.searchsorted(axes[i], points[:, i]) for i in range(3)]
        flat = np.ravel_multi_index(indices[::-1], dimensions[::-1])
        if len(np.unique(flat)) == len(points):
            vectors = np.empty((*dimensions[::-1], 3), dtype=np.float32)
            vectors.reshape(-1, 3)[flat] = values
            mesh = field_mesh_from_axes(axes, 1., unit)
            vectors, mesh, changed = uniform_field_axes(vectors, mesh)
            return vectors, mesh, 'Nonuniform node coordinates resampled to a uniform grid.' if changed else ''
    # Keep vacuum outside a local node neighborhood empty, rather than extending
    # a nearest-node field across the full bounding box.
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    distances = tree.query(points, k=min(2, len(points)))[0]
    if len(points) < 2:
        raise ValueError('An irregular field needs at least two distinct nodes.')
    nearest = distances[:, 1]
    nearest = nearest[nearest > 0]
    if not len(nearest):
        raise ValueError('The irregular field contains only duplicate node positions.')
    pitch = float(np.median(nearest))
    lo, hi = points.min(axis=0), points.max(axis=0)
    dimensions = np.maximum(1, np.rint((hi-lo)/pitch).astype(int)+1)
    scale = max(1., (math.prod(int(n) for n in dimensions)/2_000_000)**(1/3))
    dimensions = np.maximum(1, np.ceil(dimensions/scale).astype(int))
    axes = [np.linspace(lo[i], hi[i], dimensions[i]) for i in range(3)]
    vectors = np.zeros((*tuple(dimensions[::-1]), 3), dtype=np.float32)
    yy, xx = np.meshgrid(axes[1], axes[0], indexing='ij')
    for z, coordinate in enumerate(axes[2]):
        samples = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, coordinate)))
        distance, index = tree.query(samples, workers=-1)
        valid = distance <= pitch*.9
        layer = vectors[z].reshape(-1, 3); layer[valid] = values[index[valid]]
    mesh = field_mesh_from_axes(axes, 1., unit, steps=[
        (a[-1]-a[0])/(len(a)-1) if len(a)>1 else pitch for a in axes])
    return vectors, mesh, 'Irregular node field sampled onto a uniform analysis grid; empty regions retained.'


def read_oommf_irregular(path, header):
    raw = Path(path).read_bytes()
    count_match = re.search(r'^\s*#\s*pointcount\s*:\s*(\d+)', header, re.I | re.M)
    dimension = re.search(r'^\s*#\s*valuedim\s*:\s*(\d+)', header, re.I | re.M)
    if dimension and int(dimension.group(1)) != 3:
        raise ValueError('Select a three-component OOMMF vector field; scalar tables/fields are not supported.')
    if not count_match or int(count_match.group(1)) < 1:
        raise ValueError('OOMMF irregular mesh is missing a valid pointcount.')
    count = int(count_match.group(1))*6
    start = re.search(rb'#\s*Begin:\s*Data\s+(Text|Binary\s+[48])\s*\r?\n', raw, re.I)
    if not start:
        raise ValueError('Unsupported OOMMF vector data block.')
    kind = start.group(1).lower(); offset = start.end()
    if kind == b'text':
        end = re.search(rb'#\s*End:\s*Data\s+Text', raw[offset:], re.I)
        if not end: raise ValueError('OOMMF text data block has no end marker.')
        block = raw[offset:offset+end.start()].decode('ascii')
        data = np.asarray(' '.join(line.split('#', 1)[0] for line in block.splitlines()).split(), dtype=float)
    else:
        width = int(kind[-1:]); check = 1234567. if width == 4 else 123456789012345.
        endian = next((e for e in ('<', '>') if len(raw) >= offset+width and
                       struct.unpack(e+('f' if width==4 else 'd'), raw[offset:offset+width])[0] == check), None)
        if endian is None or len(raw) < offset+width*(count+1):
            raise ValueError('Incomplete OOMMF binary block or invalid check value.')
        data = np.frombuffer(raw, dtype=endian+f'f{width}', count=count, offset=offset+width).astype(float)
        if not re.match(rb'\s*#\s*End:\s*Data\s+Binary\s+'+str(width).encode(), raw[offset+width*(count+1):], re.I):
            raise ValueError('OOMMF binary data size does not match its header.')
    if data.size != count or not np.isfinite(data).all():
        raise ValueError('OOMMF irregular data size or values are invalid.')
    data = data.reshape(-1, 6)
    unit = re.search(r'^\s*#\s*meshunit\s*:\s*(\S+)', header, re.I|re.M)
    factor, shown = field_unit(unit.group(1) if unit else 'm')
    multiplier = re.search(r'^\s*#\s*valuemultiplier\s*:\s*([-+0-9.eE]+)', header, re.I|re.M)
    values = data[:, 3:] * (float(multiplier.group(1)) if multiplier else 1.)
    vectors, mesh, note = scattered_field_grid(data[:, :3], values, factor, shown)
    return (vectors, mesh, _native_read_ovf_time_ns(path), None), note


def read_vtk_field(path):
    pv = require_tab4_backend()
    dataset = pv.read(str(path))
    if isinstance(dataset, pv.MultiBlock):
        dataset = dataset.combine()
    where, name, values = choose_field_vectors(dataset)
    sidecar = Path(path).with_suffix('.json')
    metadata = {}
    if sidecar.is_file():
        try:
            candidate = json.loads(sidecar.read_text(encoding='utf-8'))
            if isinstance(candidate, dict): metadata = candidate
        except (ValueError, OSError): pass
    unit = metadata.get('meshunit', metadata.get('units', ''))
    for key in ('meshunit', 'length_unit', 'length_units', 'units'):
        if not unit and key in dataset.field_data:
            unit = str(np.asarray(dataset.field_data[key]).ravel()[0])
    factors, shown = field_unit(unit)
    axis_units = metadata.get('axis_units')
    if axis_units:
        converted = [field_unit(u) for u in axis_units]
        if len(set(converted)) != 1:
            raise ValueError('VTK coordinate axes must use the same length unit.')
        factors, shown = converted[0]
    timestamp = metadata.get('time_ns')
    if timestamp is None:
        for key, factor in (('time_ns', 1.), ('TimeValue', 1e9), ('TIME', 1e9), ('time', 1e9)):
            if key in dataset.field_data:
                timestamp = float(np.asarray(dataset.field_data[key]).ravel()[0])*factor; break
    if timestamp is not None:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp): timestamp = None
    note = '' if unit or axis_units else 'VTK has no length unit metadata; coordinates retained in VTK units.'
    note = (str(metadata.get('note', ''))+' '+note).strip()
    dimensions = tuple(int(n) for n in getattr(dataset, 'dimensions', ()))
    if len(dimensions) == 3 and min(dimensions) > 0:
        if where == 'cell':
            dimensions = tuple(max(1, n-1) for n in dimensions)
            points = np.asarray(dataset.cell_centers().points)
        else:
            points = np.asarray(dataset.points)
        if len(points) == int(np.prod(dimensions)):
            coords = points.reshape((*dimensions[::-1], 3))
            axes = [coords[0, 0, :, 0], coords[0, :, 0, 1], coords[:, 0, 0, 2]]
            expected = np.broadcast_arrays(axes[0][None, None, :], axes[1][None, :, None], axes[2][:, None, None])
            # Curvilinear structured grids follow the unstructured sampling path.
            if all(np.allclose(coords[..., i], expected[i], rtol=1e-7,
                               atol=max(np.ptp(points[:, i])*1e-8, 1e-30)) for i in range(3)):
                vectors = values.reshape((*dimensions[::-1], 3)).copy()
                for i, a in enumerate(axes):
                    if len(a)>1 and a[-1] < a[0]:
                        axes[i] = a[::-1]; vectors = np.flip(vectors, axis=2-i).copy()
                spacing = getattr(dataset, 'spacing', (1., 1., 1.))
                mesh = field_mesh_from_axes(axes, factors, shown, steps=[
                    (a[-1]-a[0])/(len(a)-1) if len(a)>1 else abs(spacing[i]) for i, a in enumerate(axes)])
                edge_bounds = metadata.get('edge_bounds')
                if edge_bounds is not None and len(edge_bounds)==6:
                    for i, a in enumerate('xyz'):
                        mesh[a]['edges'] = tuple(float(x)*factors for x in edge_bounds[2*i:2*i+2])
                vectors, mesh, changed = uniform_field_axes(vectors, mesh)
                if changed: note += ' Nonuniform axes resampled to a uniform analysis grid.'
                return (vectors, mesh, timestamp, None), note.strip()
    # Sample real cell domains for unstructured/curvilinear meshes. Outside cells
    # stays zero, so cavities and the external material outline are preserved.
    if dataset.n_cells and not isinstance(dataset, pv.PolyData):
        lo = np.asarray(dataset.bounds)[::2]; hi = np.asarray(dataset.bounds)[1::2]
        extent = hi-lo; active = extent>max(float(np.max(extent))*1e-10, 1e-30)
        dimension = max(1, int(active.sum()))
        volume = float(np.prod(extent[active]))
        pitch = (volume/max(dataset.n_points, 1))**(1/dimension)
        dimensions = np.where(active, np.maximum(2, np.ceil(extent/max(pitch, 1e-30)).astype(int)+1), 1)
        dimensions = np.maximum(1, np.ceil(dimensions/max(1., (int(np.prod(dimensions))/2_000_000)**(1/3))).astype(int))
        spacing = np.where(active, extent/np.maximum(1, dimensions-1), pitch)
        grid = pv.ImageData(dimensions=tuple(dimensions), spacing=tuple(spacing), origin=tuple(lo))
        sampled = grid.sample(dataset)
        arrays = sampled.point_data
        if name not in arrays: raise ValueError('VTK resampling did not produce the selected vector field.')
        data = np.array(arrays[name], dtype=np.float32)
        if 'vtkValidPointMask' in arrays: data[np.asarray(arrays['vtkValidPointMask'])==0] = 0
        vectors = data.reshape((*tuple(dimensions[::-1]), 3))
        mesh = field_mesh_from_axes([lo[i]+np.arange(dimensions[i])*spacing[i] for i in range(3)], factors, shown, steps=spacing)
        return (vectors, mesh, timestamp, None), (note+' Unstructured VTK sampled onto a uniform analysis grid.').strip()
    points = np.asarray(dataset.points if where=='point' else dataset.cell_centers().points)
    vectors, mesh, extra = scattered_field_grid(points, values, factors, shown)
    return (vectors, mesh, timestamp, None), (note+' '+extra).strip()


def read_field_file(path):
    path = Path(path).resolve(); stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = getattr(_FIELD_LOCAL, 'cached', None)
    if cached is not None and cached[0] == key:
        return cached[1]
    if path.suffix.lower() in VTK_FIELD_EXTS:
        result, note = read_vtk_field(path)
    else:
        with path.open('rb') as stream:
            header = stream.read(65536).decode('latin1')
        if re.search(r'^\s*#\s*meshtype\s*:\s*irregular', header, re.I|re.M) or re.search(r'^#\s*OOMMF:\s*irregular', header, re.I):
            result, note = read_oommf_irregular(path, header)
        else:
            vectors = _native_parse_ovf_vectors(path)
            multiplier = re.search(r'^\s*#\s*valuemultiplier\s*:\s*([-+0-9.eE]+)', header, re.I|re.M)
            if multiplier:
                value = float(multiplier.group(1))
                if not math.isfinite(value): raise ValueError('Invalid OOMMF value multiplier.')
                vectors = (vectors*value).astype(np.float32)
            if not np.isfinite(vectors).all(): raise ValueError('Field vector magnitudes overflow float32.')
            mesh = _native_ovf_mesh_coordinates(path, vectors.shape)
            # OOMMF may omit base while specifying its bounding box and step.
            for a, n in zip('xyz', vectors.shape[:3][::-1]):
                if mesh[a]['unit'] != 'cell': continue
                step = re.search(r'^\s*#\s*'+a+r'stepsize\s*:\s*([-+0-9.eE]+)', header, re.I|re.M)
                low = re.search(r'^\s*#\s*'+a+r'min\s*:\s*([-+0-9.eE]+)', header, re.I|re.M)
                unit = re.search(r'^\s*#\s*meshunit\s*:\s*(\S+)', header, re.I|re.M)
                if step and low:
                    factor, shown = field_unit(unit.group(1) if unit else 'm')
                    delta = float(step.group(1)); base = float(low.group(1))+delta/2
                    mesh[a] = field_axis((base+np.arange(n)*delta)*factor, shown, delta*factor)
            result = vectors, mesh, _native_read_ovf_time_ns(path), None
            note = ''
    _FIELD_LOCAL.cached = key, result
    with _FIELD_NOTE_LOCK:
        if len(_FIELD_NOTES) > 1024: _FIELD_NOTES.clear()
        _FIELD_NOTES[str(path)] = note
    return result


# Existing tools retain their public names, using the same reader internally.
def parse_ovf_vectors(path):
    return read_field_file(path)[0]


def ovf_mesh_coordinates(path, shape):
    result = read_field_file(path)
    if tuple(result[0].shape) != tuple(shape):
        raise ValueError('Requested mesh shape does not match the source field.')
    return result[1]


def read_ovf_time_ns(path):
    return read_field_file(path)[2]


def render_field_image(path, component='mz', colors='blue,white,red', upscale=1,
                       arrows=False, density=60, length=2.5, head=8,
                       mode='Normalized', annotations=None, cancel_event=None):
    check_cancel(cancel_event)
    vectors, mesh, timestamp, _ = read_field_file(path)
    chosen = {'mx': 0, 'my': 1, 'mz': 2}.get(component)
    if chosen is None: raise ValueError('Choose mx, my, or mz for a field movie frame.')
    options = dict(annotations or {})
    low, high = float(options.get('vmin', -1)), float(options.get('vmax', 1))
    image = scalar_slice_image(vectors[0, ..., chosen], colors, low, high)
    factor = int(upscale)
    if factor != 1:
        image = image.resize((image.width*factor, image.height*factor), Image.Resampling.LANCZOS)
    if arrows:
        image = overlay_arrows(image, vectors, int(density), float(length), float(head)*factor,
                               mode=mode, cancel_event=cancel_event)
    check_cancel(cancel_event)
    return decorate_image(image, options, colors=colors, component=component, time_ns=timestamp, scale=factor)



class Tab4VTKCache:
    """Thread-safe, owned temporary VTK output; input files are never changed."""
    def __init__(self):
        self.temp = None; self.directory = None; self.entries = {}
        self.lock = threading.RLock(); self.active = 0; self.closed = False
        self.cancel = threading.Event(); self.conversions = 0; self.source_locks = {}

    def fingerprint(self, source):
        import hashlib
        source = Path(source).resolve(); info = source.stat()
        value = str(source)+'|'+str(info.st_size)+'|'+str(info.st_mtime_ns)
        return hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]

    def ensure_directory(self):
        with self.lock:
            if self.closed: raise ConversionCancelled()
            if self.temp is None:
                self.temp = tempfile.TemporaryDirectory(prefix='mumax-tab4-vtk-')
                self.directory = Path(self.temp.name)
            return self.directory

    def read_source(self, source):
        key = self.fingerprint(source)
        with self.lock:
            cached = self.entries.get(key)
        if cached is not None and cached.is_file():
            return read_cached_mumax_vtk(cached), cached
        try:
            return read_field_file(source), None
        except (ValueError, OSError, UnicodeError) as primary_error:
            if Path(source).suffix.lower() in VTK_FIELD_EXTS: raise
            try: return self.convert_and_read(source)
            except (ConversionCancelled, Tab4DependencyError): raise
            except Exception as fallback_error:
                raise ValueError(f'OVF read failed: {primary_error}. VTK fallback failed: {fallback_error}') from fallback_error

    def convert_and_read(self, source, cancel_event=None):
        from types import SimpleNamespace
        pv = require_tab4_backend(); source = Path(source).resolve()
        cancel = SimpleNamespace(is_set=lambda: self.cancel.is_set() or
                                  (cancel_event is not None and cancel_event.is_set()))
        key = self.fingerprint(source)
        with self.lock:
            if self.closed: raise ConversionCancelled()
            self.active += 1
            source_lock = self.source_locks.setdefault(key, threading.Lock())
        try:
            # Both automatic fallback and the explicit batch may request a frame.
            while not source_lock.acquire(timeout=.1): check_cancel(cancel)
            try:
                check_cancel(cancel)
                cached = self.entries.get(key)
                if cached is not None and cached.is_file(): return read_cached_mumax_vtk(cached), cached
                directory = self.ensure_directory()
                with tempfile.TemporaryDirectory(prefix='.work-', dir=directory) as temporary:
                    work = Path(temporary)
                    try:
                        native = read_field_file(source)
                    except (ValueError, OSError, UnicodeError):
                        if source.suffix.lower() in VTK_FIELD_EXTS: raise
                        converter = find_mumax_vtk_converter()
                        try:
                            run_converter([converter, '-vtk', 'binary', '-o', str(work), str(source)],
                                          cwd=work, cancel_event=cancel)
                        except subprocess.CalledProcessError as exc:
                            detail = (exc.stderr or exc.stdout or str(exc))[-1500:]
                            raise ValueError('mumax3-convert could not decode this vector field: '+detail) from exc
                        check_cancel(cancel)
                        outputs = sorted(p for p in work.iterdir() if p.suffix.lower() in VTK_FIELD_EXTS)
                        if not outputs: raise ValueError('mumax3-convert produced no VTK dataset.')
                        converted = pv.read(outputs[0])
                        vectors = vtk_point_vectors(converted)
                        mesh = converted_vtk_mesh(converted, vectors, source)
                        native = vectors, mesh, _native_read_ovf_time_ns(source), None
                    vectors, mesh, timestamp, _ = native
                    dimensions = tuple(vectors.shape[:3][::-1])
                    spacing = [(mesh[a]['edges'][1]-mesh[a]['edges'][0])/n for a, n in zip('xyz', dimensions)]
                    dataset = pv.ImageData(dimensions=dimensions, spacing=spacing,
                                           origin=[mesh[a]['centers'][0] for a in 'xyz'])
                    dataset.point_data['m'] = vectors.reshape(-1, 3); dataset.set_active_vectors('m')
                    stem = safe_name(source.stem)+'_'+key
                    vtk_path = directory/(stem+'.vtk')
                    staged = work/(stem+'.vtk')
                    dataset.save(staged, binary=True)
                    metadata = {'axis_units': [mesh[a]['unit'] for a in 'xyz'],
                                'edge_bounds': [float(v) for a in 'xyz' for v in mesh[a]['edges']],
                                'source': str(source), 'time_ns': timestamp,
                                'note': field_source_note(source)}
                    staged.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf-8')
                    check_cancel(cancel)
                    if self.fingerprint(source) != key:
                        raise ValueError('The source changed during VTK conversion; convert it again.')
                    os.replace(staged.with_suffix('.json'), vtk_path.with_suffix('.json'))
                    os.replace(staged, vtk_path)
                    self.entries[key] = vtk_path; self.conversions += 1
                    return read_cached_mumax_vtk(vtk_path), vtk_path
            finally:
                source_lock.release()
        finally:
            with self.lock:
                self.active -= 1
                if self.closed and self.active == 0: self.cleanup()

    def cleanup(self):
        if self.temp is None: return
        target = self.directory.resolve(); base = Path(tempfile.gettempdir()).resolve()
        if target.parent != base or not target.name.startswith('mumax-tab4-vtk-'):
            raise RuntimeError('Refusing to clean a directory outside the owned VTK cache.')
        self.temp.cleanup(); self.temp = None; self.entries.clear(); self.source_locks.clear()

    def close(self):
        self.cancel.set()
        with self.lock:
            self.closed = True
            if self.active == 0: self.cleanup()




def physical_tab3_roi(source):
    roi = source.valid_roi()
    if roi is None or source.mesh is None:return None
    x0,x1,y0,y1 = roi
    result = {'unit':source.mesh['x']['unit']}
    for axis,first,last,count in (('x',x0,x1,source.vectors.shape[2]),('y',y0,y1,source.vectors.shape[1])):
        lo,hi = source.mesh[axis]['edges']
        step = (hi-lo)/count
        result[axis] = (lo+first*step,lo+last*step)
    return result


def crop_stream_roi(vectors,mesh,roi):
    if roi is None:return vectors,mesh
    counts = dict(zip('xyz',vectors.shape[:3][::-1]))
    bounds = {'z':(0,counts['z'])}
    cropped_mesh = dict(mesh)
    for axis in 'xy':
        if mesh[axis]['unit']!=roi['unit']:raise ValueError('ROI and field use different coordinate units.')
        centers = mesh[axis]['centers']
        lo,hi = roi[axis]
        first,last = np.searchsorted(centers,[lo,hi],side='left')
        first,last = max(0,int(first)),min(len(centers),int(last))
        if last<=first:raise ValueError('The Tab 3 ROI contains no samples in this volume. Check the ROI and upscaling.')
        bounds[axis] = first,last
        low,high = mesh[axis]['edges'];step = (high-low)/counts[axis]
        cropped_mesh[axis] = dict(mesh[axis],centers=centers[first:last].copy(),edges=(low+first*step,low+last*step))
    data = vectors[:,slice(*bounds['y']),slice(*bounds['x']),:].copy()
    return data,cropped_mesh


_MUMAX_VTK_CPU = None

STREAM_COLOR_MAPS=tuple(COLOR_PRESETS)+('viridis','plasma','inferno','magma','cividis','turbo','coolwarm','seismic','bwr','RdBu','RdBu_r','jet','gray','hsv')


# ParaView transfer-function presets: bundled snapshot, 2026-10-16.
# https://github.com/Kitware/ParaView/blob/master/Remoting/Views/ColorMaps.json
# Legacy names from ParaView 5.10 and the 5.9.1 official Python documentation.
PARAVIEW_PRESET_LICENSE = 'Copyright (c) 2000 Kitware Inc.\n\nRedistribution and use in source and binary forms, with or without modification,\nare permitted provided that the following conditions are met:\n\n* Redistributions of source code must retain the above copyright notice, this list\n  of conditions and the following disclaimer.\n\n* Redistributions in binary form must reproduce the above copyright notice, this\n  list of conditions and the following disclaimer in the documentation and/or\n  other materials provided with the distribution.\n\n* Neither the name of Kitware nor the names of any contributors may be used to\n  endorse or promote products derived from this software without specific prior\n  written permission.\n\nTHIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND\nANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED\nWARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE\nDISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY\nDIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES\n(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;\nLOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON\nANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT\n(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS\nSOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\n'
PARAVIEW_COLOR_PRESETS=json.loads('{"Cool to Warm":{"ColorSpace":"Diverging","Name":"Cool to Warm","NanColor":[1,1,0],"DefaultMap":true,"RGBPoints":[0,0.23137254902,0.298039215686,0.752941176471,0.5,0.865,0.865,0.865,1,0.705882352941,0.0156862745098,0.149019607843]},"Cool to Warm (Extended)":{"ColorSpace":"Lab","Creator":"Francesca Samsel","Name":"Cool to Warm (Extended)","DefaultMap":true,"NanColor":[0.25,0,0],"RGBPoints":[0,0,0,0.34902,0.03125,0.039216,0.062745,0.380392,0.0625,0.062745,0.117647,0.411765,0.09375,0.090196,0.184314,0.45098,0.125,0.12549,0.262745,0.501961,0.15625,0.160784,0.337255,0.541176,0.1875,0.2,0.396078,0.568627,0.21875,0.239216,0.454902,0.6,0.25,0.286275,0.521569,0.65098,0.28125,0.337255,0.592157,0.701961,0.3125,0.388235,0.654902,0.74902,0.34375,0.466667,0.737255,0.819608,0.375,0.572549,0.819608,0.878431,0.40625,0.654902,0.866667,0.909804,0.4375,0.752941,0.917647,0.941176,0.46875,0.823529,0.956863,0.968627,0.5,0.988235,0.960784,0.901961,0.5,0.941176,0.984314,0.988235,0.52,0.988235,0.945098,0.85098,0.54,0.980392,0.898039,0.784314,0.5625,0.968627,0.835294,0.698039,0.59375,0.94902,0.733333,0.588235,0.625,0.929412,0.65098,0.509804,0.65625,0.909804,0.564706,0.435294,0.6875,0.878431,0.458824,0.352941,0.71875,0.839216,0.388235,0.286275,0.75,0.760784,0.294118,0.211765,0.78125,0.701961,0.211765,0.168627,0.8125,0.65098,0.156863,0.129412,0.84375,0.6,0.094118,0.094118,0.875,0.54902,0.066667,0.098039,0.90625,0.501961,0.05098,0.12549,0.9375,0.45098,0.054902,0.172549,0.96875,0.4,0.054902,0.192157,1,0.34902,0.070588,0.211765]},"Black-Body Radiation":{"ColorSpace":"RGB","Name":"Black-Body Radiation","DefaultMap":true,"NanColor":[0,0.498039215686,1],"RGBPoints":[0,0,0,0,0.4,0.901960784314,0,0,0.8,0.901960784314,0.901960784314,0,1,1,1,1]},"X Ray":{"ColorSpace":"RGB","Name":"X Ray","DefaultMap":true,"NanColor":[1,0,0],"RGBPoints":[0,1,1,1,1,0,0,0]},"Inferno":{"ColorSpace":"Lab","Name":"Inferno","NanColor":[0,1,0],"Source":"https://github.com/BIDS/colormap/blob/master/colormaps.py","License":"CC0","Creator":"Nathaniel J. Smith & Stefan van der Walt","DefaultMap":true,"RGBPoints":[0.0,0.001462,0.000466,0.013866,0.003922,0.002267,0.00127,0.01857,0.007843,0.003299,0.002249,0.024239,0.011765,0.004547,0.003392,0.030909,0.015686,0.006006,0.004692,0.038558,0.019608,0.007676,0.006136,0.046836,0.023529,0.009561,0.007713,0.055143,0.027451,0.011663,0.009417,0.06346,0.031373,0.013995,0.011225,0.071862,0.035294,0.016561,0.013136,0.080282,0.039216,0.019373,0.015133,0.088767,0.043137,0.022447,0.017199,0.097327,0.047059,0.025793,0.019331,0.10593,0.05098,0.029432,0.021503,0.114621,0.054902,0.033385,0.023702,0.123397,0.058824,0.037668,0.025921,0.132232,0.062745,0.042253,0.028139,0.141141,0.066667,0.046915,0.030324,0.150164,0.070588,0.051644,0.032474,0.159254,0.07451,0.056449,0.034569,0.168414,0.078431,0.06134,0.03659,0.177642,0.082353,0.066331,0.038504,0.186962,0.086275,0.071429,0.040294,0.196354,0.090196,0.076637,0.041905,0.205799,0.094118,0.081962,0.043328,0.215289,0.098039,0.087411,0.044556,0.224813,0.101961,0.09299,0.045583,0.234358,0.105882,0.098702,0.046402,0.243904,0.109804,0.104551,0.047008,0.25343,0.113725,0.110536,0.047399,0.262912,0.117647,0.116656,0.047574,0.272321,0.121569,0.122908,0.047536,0.281624,0.12549,0.129285,0.047293,0.290788,0.129412,0.135778,0.046856,0.299776,0.133333,0.142378,0.046242,0.308553,0.137255,0.149073,0.045468,0.317085,0.141176,0.15585,0.044559,0.325338,0.145098,0.162689,0.043554,0.333277,0.14902,0.169575,0.042489,0.340874,0.152941,0.176493,0.041402,0.348111,0.156863,0.183429,0.040329,0.354971,0.160784,0.190367,0.039309,0.361447,0.164706,0.197297,0.0384,0.367535,0.168627,0.204209,0.037632,0.373238,0.172549,0.211095,0.03703,0.378563,0.176471,0.217949,0.036615,0.383522,0.180392,0.224763,0.036405,0.388129,0.184314,0.231538,0.036405,0.3924,0.188235,0.238273,0.036621,0.396353,0.192157,0.244967,0.037055,0.400007,0.196078,0.25162,0.037705,0.403378,0.2,0.258234,0.038571,0.406485,0.203922,0.26481,0.039647,0.409345,0.207843,0.271347,0.040922,0.411976,0.211765,0.27785,0.042353,0.414392,0.215686,0.284321,0.043933,0.416608,0.219608,0.290763,0.045644,0.418637,0.223529,0.297178,0.04747,0.420491,0.227451,0.303568,0.049396,0.422182,0.231373,0.309935,0.051407,0.423721,0.235294,0.316282,0.05349,0.425116,0.239216,0.32261,0.055634,0.426377,0.243137,0.328921,0.057827,0.427511,0.247059,0.335217,0.06006,0.428524,0.25098,0.3415,0.062325,0.429425,0.254902,0.347771,0.064616,0.430217,0.258824,0.354032,0.066925,0.430906,0.262745,0.360284,0.069247,0.431497,0.266667,0.366529,0.071579,0.431994,0.270588,0.372768,0.073915,0.4324,0.27451,0.379001,0.076253,0.432719,0.278431,0.385228,0.078591,0.432955,0.282353,0.391453,0.080927,0.433109,0.286275,0.397674,0.083257,0.433183,0.290196,0.403894,0.08558,0.433179,0.294118,0.410113,0.087896,0.433098,0.298039,0.416331,0.090203,0.432943,0.301961,0.422549,0.092501,0.432714,0.305882,0.428768,0.09479,0.432412,0.309804,0.434987,0.097069,0.432039,0.313725,0.441207,0.099338,0.431594,0.317647,0.447428,0.101597,0.43108,0.321569,0.453651,0.103848,0.430498,0.32549,0.459875,0.106089,0.429846,0.329412,0.4661,0.108322,0.429125,0.333333,0.472328,0.110547,0.428334,0.337255,0.478558,0.112764,0.427475,0.341176,0.484789,0.114974,0.426548,0.345098,0.491022,0.117179,0.425552,0.34902,0.497257,0.119379,0.424488,0.352941,0.503493,0.121575,0.423356,0.356863,0.50973,0.123769,0.422156,0.360784,0.515967,0.12596,0.420887,0.364706,0.522206,0.12815,0.419549,0.368627,0.528444,0.130341,0.418142,0.372549,0.534683,0.132534,0.416667,0.376471,0.54092,0.134729,0.415123,0.380392,0.547157,0.136929,0.413511,0.384314,0.553392,0.139134,0.411829,0.388235,0.559624,0.141346,0.410078,0.392157,0.565854,0.143567,0.408258,0.396078,0.572081,0.145797,0.406369,0.4,0.578304,0.148039,0.404411,0.403922,0.584521,0.150294,0.402385,0.407843,0.590734,0.152563,0.40029,0.411765,0.59694,0.154848,0.398125,0.415686,0.603139,0.157151,0.395891,0.419608,0.60933,0.159474,0.393589,0.423529,0.615513,0.161817,0.391219,0.427451,0.621685,0.164184,0.388781,0.431373,0.627847,0.166575,0.386276,0.435294,0.633998,0.168992,0.383704,0.439216,0.640135,0.171438,0.381065,0.443137,0.64626,0.173914,0.378359,0.447059,0.652369,0.176421,0.375586,0.45098,0.658463,0.178962,0.372748,0.454902,0.66454,0.181539,0.369846,0.458824,0.670599,0.184153,0.366879,0.462745,0.676638,0.186807,0.363849,0.466667,0.682656,0.189501,0.360757,0.470588,0.688653,0.192239,0.357603,0.47451,0.694627,0.195021,0.354388,0.478431,0.700576,0.197851,0.351113,0.482353,0.7065,0.200728,0.347777,0.486275,0.712396,0.203656,0.344383,0.490196,0.718264,0.206636,0.340931,0.494118,0.724103,0.20967,0.337424,0.498039,0.729909,0.212759,0.333861,0.501961,0.735683,0.215906,0.330245,0.505882,0.741423,0.219112,0.326576,0.509804,0.747127,0.222378,0.322856,0.513725,0.752794,0.225706,0.319085,0.517647,0.758422,0.229097,0.315266,0.521569,0.76401,0.232554,0.311399,0.52549,0.769556,0.236077,0.307485,0.529412,0.775059,0.239667,0.303526,0.533333,0.780517,0.243327,0.299523,0.537255,0.785929,0.247056,0.295477,0.541176,0.791293,0.250856,0.29139,0.545098,0.796607,0.254728,0.287264,0.54902,0.801871,0.258674,0.283099,0.552941,0.807082,0.262692,0.278898,0.556863,0.812239,0.266786,0.274661,0.560784,0.817341,0.270954,0.27039,0.564706,0.822386,0.275197,0.266085,0.568627,0.827372,0.279517,0.26175,0.572549,0.832299,0.283913,0.257383,0.576471,0.837165,0.288385,0.252988,0.580392,0.841969,0.292933,0.248564,0.584314,0.846709,0.297559,0.244113,0.588235,0.851384,0.30226,0.239636,0.592157,0.855992,0.307038,0.235133,0.596078,0.860533,0.311892,0.230606,0.6,0.865006,0.316822,0.226055,0.603922,0.869409,0.321827,0.221482,0.607843,0.873741,0.326906,0.216886,0.611765,0.878001,0.33206,0.212268,0.615686,0.882188,0.337287,0.207628,0.619608,0.886302,0.342586,0.202968,0.623529,0.890341,0.347957,0.198286,0.627451,0.894305,0.353399,0.193584,0.631373,0.898192,0.358911,0.18886,0.635294,0.902003,0.364492,0.184116,0.639216,0.905735,0.37014,0.17935,0.643137,0.90939,0.375856,0.174563,0.647059,0.912966,0.381636,0.169755,0.65098,0.916462,0.387481,0.164924,0.654902,0.919879,0.393389,0.16007,0.658824,0.923215,0.399359,0.155193,0.662745,0.92647,0.405389,0.150292,0.666667,0.929644,0.411479,0.145367,0.670588,0.932737,0.417627,0.140417,0.67451,0.935747,0.423831,0.13544,0.678431,0.938675,0.430091,0.130438,0.682353,0.941521,0.436405,0.125409,0.686275,0.944285,0.442772,0.120354,0.690196,0.946965,0.449191,0.115272,0.694118,0.949562,0.45566,0.110164,0.698039,0.952075,0.462178,0.105031,0.701961,0.954506,0.468744,0.099874,0.705882,0.956852,0.475356,0.094695,0.709804,0.959114,0.482014,0.089499,0.713725,0.961293,0.488716,0.084289,0.717647,0.963387,0.495462,0.079073,0.721569,0.965397,0.502249,0.073859,0.72549,0.967322,0.509078,0.068659,0.729412,0.969163,0.515946,0.063488,0.733333,0.970919,0.522853,0.058367,0.737255,0.97259,0.529798,0.053324,0.741176,0.974176,0.53678,0.048392,0.745098,0.975677,0.543798,0.043618,0.74902,0.977092,0.55085,0.03905,0.752941,0.978422,0.557937,0.034931,0.756863,0.979666,0.565057,0.031409,0.760784,0.980824,0.572209,0.028508,0.764706,0.981895,0.579392,0.02625,0.768627,0.982881,0.586606,0.024661,0.772549,0.983779,0.593849,0.02377,0.776471,0.984591,0.601122,0.023606,0.780392,0.985315,0.608422,0.024202,0.784314,0.985952,0.61575,0.025592,0.788235,0.986502,0.623105,0.027814,0.792157,0.986964,0.630485,0.030908,0.796078,0.987337,0.63789,0.034916,0.8,0.987622,0.64532,0.039886,0.803922,0.987819,0.652773,0.045581,0.807843,0.987926,0.66025,0.05175,0.811765,0.987945,0.667748,0.058329,0.815686,0.987874,0.675267,0.065257,0.819608,0.987714,0.682807,0.072489,0.823529,0.987464,0.690366,0.07999,0.827451,0.987124,0.697944,0.087731,0.831373,0.986694,0.70554,0.095694,0.835294,0.986175,0.713153,0.103863,0.839216,0.985566,0.720782,0.112229,0.843137,0.984865,0.728427,0.120785,0.847059,0.984075,0.736087,0.129527,0.85098,0.983196,0.743758,0.138453,0.854902,0.982228,0.751442,0.147565,0.858824,0.981173,0.759135,0.156863,0.862745,0.980032,0.766837,0.166353,0.866667,0.978806,0.774545,0.176037,0.870588,0.977497,0.782258,0.185923,0.87451,0.976108,0.789974,0.196018,0.878431,0.974638,0.797692,0.206332,0.882353,0.973088,0.805409,0.216877,0.886275,0.971468,0.813122,0.227658,0.890196,0.969783,0.820825,0.238686,0.894118,0.968041,0.828515,0.249972,0.898039,0.966243,0.836191,0.261534,0.901961,0.964394,0.843848,0.273391,0.905882,0.962517,0.851476,0.285546,0.909804,0.960626,0.859069,0.29801,0.913725,0.95872,0.866624,0.31082,0.917647,0.956834,0.874129,0.323974,0.921569,0.954997,0.881569,0.337475,0.92549,0.953215,0.888942,0.351369,0.929412,0.951546,0.896226,0.365627,0.933333,0.950018,0.903409,0.380271,0.937255,0.948683,0.910473,0.395289,0.941176,0.947594,0.917399,0.410665,0.945098,0.946809,0.924168,0.426373,0.94902,0.946392,0.930761,0.442367,0.952941,0.946403,0.937159,0.458592,0.956863,0.946903,0.943348,0.47497,0.960784,0.947937,0.949318,0.491426,0.964706,0.949545,0.955063,0.50786,0.968627,0.95174,0.960587,0.524203,0.972549,0.954529,0.965896,0.540361,0.976471,0.957896,0.971003,0.556275,0.980392,0.961812,0.975924,0.571925,0.984314,0.966249,0.980678,0.587206,0.988235,0.971162,0.985282,0.602154,0.992157,0.976511,0.989753,0.61676,0.996078,0.982257,0.994109,0.631017,1.0,0.988362,0.998364,0.644924]},"Black, Blue and White":{"ColorSpace":"RGB","Name":"Black, Blue and White","NanColor":[1,1,0],"DefaultMap":true,"RGBPoints":[0,0,0,0,0.333,0,0,0.501960784314,0.666,0,0.501960784314,1,1,1,1,1]},"Blue Orange (divergent)":{"ColorSpace":"Lab","Creator":"Francesca Samsel","Name":"Blue Orange (divergent)","DefaultMap":true,"RGBPoints":[-30.3399130649763,0.0862745098039216,0.00392156862745098,0.298039215686275,-29.3502559661865,0.113725,0.0235294,0.45098,-28.5283393859863,0.105882,0.0509804,0.509804,-27.958028793335,0.0392157,0.0392157,0.560784,-27.4044914245605,0.0313725,0.0980392,0.6,-26.8677291870117,0.0431373,0.164706,0.639216,-26.096134185791,0.054902,0.243137,0.678431,-25.0729293823242,0.054902,0.317647,0.709804,-23.8148933330084,0.0509804,0.396078,0.741176,-22.9992658665124,0.0392157,0.466667,0.768627,-22.1836384000164,0.0313725,0.537255,0.788235,-21.3323650360107,0.0313725,0.615686,0.811765,-20.4601268768311,0.0235294,0.709804,0.831373,-19.5878868103027,0.0509804,0.8,0.85098,-18.8666133880615,0.0705882,0.854902,0.870588,-18.1956596374512,0.262745,0.901961,0.862745,-17.6085758209229,0.423529,0.941176,0.87451,-16.7027893066406,0.572549,0.964706,0.835294,-16.0989303588867,0.658824,0.980392,0.843137,-15.6628112792969,0.764706,0.980392,0.866667,-15.1931447982788,0.827451,0.980392,0.886275,-14.2705841064453,0.913725,0.988235,0.937255,-13.9854288101196,1,1,0.972549019607843,-13.7002735137939,0.988235,0.980392,0.870588,-13.2809276580811,0.992156862745098,0.972549019607843,0.803921568627451,-12.9622249603271,0.992157,0.964706,0.713725,-12.4254627227783,0.988235,0.956863,0.643137,-11.5699977874756,0.980392,0.917647,0.509804,-10.8487224578857,0.968627,0.87451,0.407843,-10.1106739044189,0.94902,0.823529,0.321569,-9.57391166687012,0.929412,0.776471,0.278431,-8.78554153442383,0.909804,0.717647,0.235294,-8.08104133605957,0.890196,0.658824,0.196078,-7.50234400308847,0.878431,0.619608,0.168627,-6.68671653659248,0.870588,0.54902,0.156863,-5.87108907009648,0.85098,0.47451,0.145098,-5.05546160360049,0.831373,0.411765,0.133333,-4.23983413710449,0.811765,0.345098,0.113725,-3.4242066706085,0.788235,0.266667,0.0941176,-2.6085792041125,0.741176,0.184314,0.0745098,-1.79295173761651,0.690196,0.12549,0.0627451,-0.977324271120517,0.619608,0.0627451,0.0431373,-0.214114964008331,0.54902,0.027451,0.0705882,0.456838220357895,0.470588,0.0156863,0.0901961,1.21166050434113,0.4,0.00392157,0.101961,2.28518559486346,0.188235294117647,0,0.0705882352941176]},"Viridis":{"ColorSpace":"Lab","Name":"Viridis","NanColor":[1,0,0],"Source":"https://github.com/BIDS/colormap/blob/master/colormaps.py","License":"CC0","Creator":"Eric Firing","DefaultMap":true,"RGBPoints":[0.0,0.267004,0.004874,0.329415,0.003922,0.26851,0.009605,0.335427,0.007843,0.269944,0.014625,0.341379,0.011765,0.271305,0.019942,0.347269,0.015686,0.272594,0.025563,0.353093,0.019608,0.273809,0.031497,0.358853,0.023529,0.274952,0.037752,0.364543,0.027451,0.276022,0.044167,0.370164,0.031373,0.277018,0.050344,0.375715,0.035294,0.277941,0.056324,0.381191,0.039216,0.278791,0.062145,0.386592,0.043137,0.279566,0.067836,0.391917,0.047059,0.280267,0.073417,0.397163,0.05098,0.280894,0.078907,0.402329,0.054902,0.281446,0.08432,0.407414,0.058824,0.281924,0.089666,0.412415,0.062745,0.282327,0.094955,0.417331,0.066667,0.282656,0.100196,0.42216,0.070588,0.28291,0.105393,0.426902,0.07451,0.283091,0.110553,0.431554,0.078431,0.283197,0.11568,0.436115,0.082353,0.283229,0.120777,0.440584,0.086275,0.283187,0.125848,0.44496,0.090196,0.283072,0.130895,0.449241,0.094118,0.282884,0.13592,0.453427,0.098039,0.282623,0.140926,0.457517,0.101961,0.28229,0.145912,0.46151,0.105882,0.281887,0.150881,0.465405,0.109804,0.281412,0.155834,0.469201,0.113725,0.280868,0.160771,0.472899,0.117647,0.280255,0.165693,0.476498,0.121569,0.279574,0.170599,0.479997,0.12549,0.278826,0.17549,0.483397,0.129412,0.278012,0.180367,0.486697,0.133333,0.277134,0.185228,0.489898,0.137255,0.276194,0.190074,0.493001,0.141176,0.275191,0.194905,0.496005,0.145098,0.274128,0.199721,0.498911,0.14902,0.273006,0.20452,0.501721,0.152941,0.271828,0.209303,0.504434,0.156863,0.270595,0.214069,0.507052,0.160784,0.269308,0.218818,0.509577,0.164706,0.267968,0.223549,0.512008,0.168627,0.26658,0.228262,0.514349,0.172549,0.265145,0.232956,0.516599,0.176471,0.263663,0.237631,0.518762,0.180392,0.262138,0.242286,0.520837,0.184314,0.260571,0.246922,0.522828,0.188235,0.258965,0.251537,0.524736,0.192157,0.257322,0.25613,0.526563,0.196078,0.255645,0.260703,0.528312,0.2,0.253935,0.265254,0.529983,0.203922,0.252194,0.269783,0.531579,0.207843,0.250425,0.27429,0.533103,0.211765,0.248629,0.278775,0.534556,0.215686,0.246811,0.283237,0.535941,0.219608,0.244972,0.287675,0.53726,0.223529,0.243113,0.292092,0.538516,0.227451,0.241237,0.296485,0.539709,0.231373,0.239346,0.300855,0.540844,0.235294,0.237441,0.305202,0.541921,0.239216,0.235526,0.309527,0.542944,0.243137,0.233603,0.313828,0.543914,0.247059,0.231674,0.318106,0.544834,0.25098,0.229739,0.322361,0.545706,0.254902,0.227802,0.326594,0.546532,0.258824,0.225863,0.330805,0.547314,0.262745,0.223925,0.334994,0.548053,0.266667,0.221989,0.339161,0.548752,0.270588,0.220057,0.343307,0.549413,0.27451,0.21813,0.347432,0.550038,0.278431,0.21621,0.351535,0.550627,0.282353,0.214298,0.355619,0.551184,0.286275,0.212395,0.359683,0.55171,0.290196,0.210503,0.363727,0.552206,0.294118,0.208623,0.367752,0.552675,0.298039,0.206756,0.371758,0.553117,0.301961,0.204903,0.375746,0.553533,0.305882,0.203063,0.379716,0.553925,0.309804,0.201239,0.38367,0.554294,0.313725,0.19943,0.387607,0.554642,0.317647,0.197636,0.391528,0.554969,0.321569,0.19586,0.395433,0.555276,0.32549,0.1941,0.399323,0.555565,0.329412,0.192357,0.403199,0.555836,0.333333,0.190631,0.407061,0.556089,0.337255,0.188923,0.41091,0.556326,0.341176,0.187231,0.414746,0.556547,0.345098,0.185556,0.41857,0.556753,0.34902,0.183898,0.422383,0.556944,0.352941,0.182256,0.426184,0.55712,0.356863,0.180629,0.429975,0.557282,0.360784,0.179019,0.433756,0.55743,0.364706,0.177423,0.437527,0.557565,0.368627,0.175841,0.44129,0.557685,0.372549,0.174274,0.445044,0.557792,0.376471,0.172719,0.448791,0.557885,0.380392,0.171176,0.45253,0.557965,0.384314,0.169646,0.456262,0.55803,0.388235,0.168126,0.459988,0.558082,0.392157,0.166617,0.463708,0.558119,0.396078,0.165117,0.467423,0.558141,0.4,0.163625,0.471133,0.558148,0.403922,0.162142,0.474838,0.55814,0.407843,0.160665,0.47854,0.558115,0.411765,0.159194,0.482237,0.558073,0.415686,0.157729,0.485932,0.558013,0.419608,0.15627,0.489624,0.557936,0.423529,0.154815,0.493313,0.55784,0.427451,0.153364,0.497,0.557724,0.431373,0.151918,0.500685,0.557587,0.435294,0.150476,0.504369,0.55743,0.439216,0.149039,0.508051,0.55725,0.443137,0.147607,0.511733,0.557049,0.447059,0.14618,0.515413,0.556823,0.45098,0.144759,0.519093,0.556572,0.454902,0.143343,0.522773,0.556295,0.458824,0.141935,0.526453,0.555991,0.462745,0.140536,0.530132,0.555659,0.466667,0.139147,0.533812,0.555298,0.470588,0.13777,0.537492,0.554906,0.47451,0.136408,0.541173,0.554483,0.478431,0.135066,0.544853,0.554029,0.482353,0.133743,0.548535,0.553541,0.486275,0.132444,0.552216,0.553018,0.490196,0.131172,0.555899,0.552459,0.494118,0.129933,0.559582,0.551864,0.498039,0.128729,0.563265,0.551229,0.501961,0.127568,0.566949,0.550556,0.505882,0.126453,0.570633,0.549841,0.509804,0.125394,0.574318,0.549086,0.513725,0.124395,0.578002,0.548287,0.517647,0.123463,0.581687,0.547445,0.521569,0.122606,0.585371,0.546557,0.52549,0.121831,0.589055,0.545623,0.529412,0.121148,0.592739,0.544641,0.533333,0.120565,0.596422,0.543611,0.537255,0.120092,0.600104,0.54253,0.541176,0.119738,0.603785,0.5414,0.545098,0.119512,0.607464,0.540218,0.54902,0.119423,0.611141,0.538982,0.552941,0.119483,0.614817,0.537692,0.556863,0.119699,0.61849,0.536347,0.560784,0.120081,0.622161,0.534946,0.564706,0.120638,0.625828,0.533488,0.568627,0.12138,0.629492,0.531973,0.572549,0.122312,0.633153,0.530398,0.576471,0.123444,0.636809,0.528763,0.580392,0.12478,0.640461,0.527068,0.584314,0.126326,0.644107,0.525311,0.588235,0.128087,0.647749,0.523491,0.592157,0.130067,0.651384,0.521608,0.596078,0.132268,0.655014,0.519661,0.6,0.134692,0.658636,0.517649,0.603922,0.137339,0.662252,0.515571,0.607843,0.14021,0.665859,0.513427,0.611765,0.143303,0.669459,0.511215,0.615686,0.146616,0.67305,0.508936,0.619608,0.150148,0.676631,0.506589,0.623529,0.153894,0.680203,0.504172,0.627451,0.157851,0.683765,0.501686,0.631373,0.162016,0.687316,0.499129,0.635294,0.166383,0.690856,0.496502,0.639216,0.170948,0.694384,0.493803,0.643137,0.175707,0.6979,0.491033,0.647059,0.180653,0.701402,0.488189,0.65098,0.185783,0.704891,0.485273,0.654902,0.19109,0.708366,0.482284,0.658824,0.196571,0.711827,0.479221,0.662745,0.202219,0.715272,0.476084,0.666667,0.20803,0.718701,0.472873,0.670588,0.214,0.722114,0.469588,0.67451,0.220124,0.725509,0.466226,0.678431,0.226397,0.728888,0.462789,0.682353,0.232815,0.732247,0.459277,0.686275,0.239374,0.735588,0.455688,0.690196,0.24607,0.73891,0.452024,0.694118,0.252899,0.742211,0.448284,0.698039,0.259857,0.745492,0.444467,0.701961,0.266941,0.748751,0.440573,0.705882,0.274149,0.751988,0.436601,0.709804,0.281477,0.755203,0.432552,0.713725,0.288921,0.758394,0.428426,0.717647,0.296479,0.761561,0.424223,0.721569,0.304148,0.764704,0.419943,0.72549,0.311925,0.767822,0.415586,0.729412,0.319809,0.770914,0.411152,0.733333,0.327796,0.77398,0.40664,0.737255,0.335885,0.777018,0.402049,0.741176,0.344074,0.780029,0.397381,0.745098,0.35236,0.783011,0.392636,0.74902,0.360741,0.785964,0.387814,0.752941,0.369214,0.788888,0.382914,0.756863,0.377779,0.791781,0.377939,0.760784,0.386433,0.794644,0.372886,0.764706,0.395174,0.797475,0.367757,0.768627,0.404001,0.800275,0.362552,0.772549,0.412913,0.803041,0.357269,0.776471,0.421908,0.805774,0.35191,0.780392,0.430983,0.808473,0.346476,0.784314,0.440137,0.811138,0.340967,0.788235,0.449368,0.813768,0.335384,0.792157,0.458674,0.816363,0.329727,0.796078,0.468053,0.818921,0.323998,0.8,0.477504,0.821444,0.318195,0.803922,0.487026,0.823929,0.312321,0.807843,0.496615,0.826376,0.306377,0.811765,0.506271,0.828786,0.300362,0.815686,0.515992,0.831158,0.294279,0.819608,0.525776,0.833491,0.288127,0.823529,0.535621,0.835785,0.281908,0.827451,0.545524,0.838039,0.275626,0.831373,0.555484,0.840254,0.269281,0.835294,0.565498,0.84243,0.262877,0.839216,0.575563,0.844566,0.256415,0.843137,0.585678,0.846661,0.249897,0.847059,0.595839,0.848717,0.243329,0.85098,0.606045,0.850733,0.236712,0.854902,0.616293,0.852709,0.230052,0.858824,0.626579,0.854645,0.223353,0.862745,0.636902,0.856542,0.21662,0.866667,0.647257,0.8584,0.209861,0.870588,0.657642,0.860219,0.203082,0.87451,0.668054,0.861999,0.196293,0.878431,0.678489,0.863742,0.189503,0.882353,0.688944,0.865448,0.182725,0.886275,0.699415,0.867117,0.175971,0.890196,0.709898,0.868751,0.169257,0.894118,0.720391,0.87035,0.162603,0.898039,0.730889,0.871916,0.156029,0.901961,0.741388,0.873449,0.149561,0.905882,0.751884,0.874951,0.143228,0.909804,0.762373,0.876424,0.137064,0.913725,0.772852,0.877868,0.131109,0.917647,0.783315,0.879285,0.125405,0.921569,0.79376,0.880678,0.120005,0.92549,0.804182,0.882046,0.114965,0.929412,0.814576,0.883393,0.110347,0.933333,0.82494,0.88472,0.106217,0.937255,0.83527,0.886029,0.102646,0.941176,0.845561,0.887322,0.099702,0.945098,0.85581,0.888601,0.097452,0.94902,0.866013,0.889868,0.095953,0.952941,0.876168,0.891125,0.09525,0.956863,0.886271,0.892374,0.095374,0.960784,0.89632,0.893616,0.096335,0.964706,0.906311,0.894855,0.098125,0.968627,0.916242,0.896091,0.100717,0.972549,0.926106,0.89733,0.104071,0.976471,0.935904,0.89857,0.108131,0.980392,0.945636,0.899815,0.112838,0.984314,0.9553,0.901065,0.118128,0.988235,0.964894,0.902323,0.123941,0.992157,0.974417,0.90359,0.130215,0.996078,0.983868,0.904867,0.136897,1.0,0.993248,0.906157,0.143936]},"Gray and Red":{"ColorSpace":"Lab","Name":"Gray and Red","RGBPoints":[-1,0.101961,0.101961,0.101961,-0.87451,0.227451,0.227451,0.227451,-0.74902,0.359939,0.359939,0.359939,-0.623529,0.502653,0.502653,0.502653,-0.498039,0.631373,0.631373,0.631373,-0.372549,0.749865,0.749865,0.749865,-0.247059,0.843368,0.843368,0.843368,-0.121569,0.926105,0.926105,0.926105,0.00392157,0.999846,0.997232,0.995694,0.129412,0.994925,0.908651,0.857901,0.254902,0.982468,0.800692,0.706113,0.380392,0.960323,0.66782,0.536332,0.505882,0.894579,0.503806,0.399769,0.631373,0.81707,0.33218,0.281046,0.756863,0.728489,0.155017,0.197386,0.882353,0.576932,0.055363,0.14925,1,0.403922,0,0.121569]},"Linear Green (Gr4L)":{"ColorSpace":"Lab","Creator":"Francesca Samsel","Name":"Linear Green (Gr4L)","NanColor":[0.25,0,0],"DefaultMap":true,"RGBPoints":[0,0.054902,0.109804,0.121569,0.05,0.07451,0.172549,0.180392,0.1,0.086275,0.231373,0.219608,0.15,0.094118,0.278431,0.25098,0.2,0.109804,0.34902,0.278431,0.25,0.113725,0.4,0.278431,0.3,0.117647,0.45098,0.270588,0.35,0.117647,0.490196,0.243137,0.4,0.113725,0.521569,0.203922,0.45,0.109804,0.54902,0.152941,0.5,0.082353,0.588235,0.082353,0.55,0.109804,0.631373,0.05098,0.6,0.211765,0.678431,0.082353,0.65,0.317647,0.721569,0.113725,0.7,0.431373,0.760784,0.160784,0.75,0.556863,0.8,0.239216,0.8,0.666667,0.839216,0.294118,0.85,0.784314,0.878431,0.396078,0.9,0.886275,0.921569,0.533333,0.95,0.960784,0.94902,0.670588,1,1,0.984314,0.901961]},"Cold and Hot":{"ColorSpace":"RGB","Name":"Cold and Hot","NanColor":[1,1,0],"DefaultMap":true,"RGBPoints":[0,0,1,1,0.45,0,0,1,0.5,0,0,0.501960784314,0.55,1,0,0,1,1,1,0]},"Blue - Green - Orange":{"ColorSpace":"CIELAB","Name":"Blue - Green - Orange","DefaultMap":true,"NanColor":[0.25,0.0,0.0],"RGBPoints":[0.0,0.831373,0.909804,0.980392,0.0125,0.74902,0.862745,0.960784,0.025,0.694118,0.827451,0.941176,0.05,0.568627,0.760784,0.921569,0.075,0.45098,0.705882,0.901961,0.1,0.345098,0.643137,0.858824,0.125,0.247059,0.572549,0.819608,0.15,0.180392,0.521569,0.780392,0.16,0.14902,0.490196,0.74902,0.18,0.129412,0.447059,0.709804,0.2,0.101961,0.427451,0.690196,0.21,0.094118,0.403922,0.658824,0.22,0.090196,0.392157,0.639216,0.23,0.082353,0.368627,0.619608,0.24,0.070588,0.352941,0.6,0.25,0.066667,0.329412,0.568627,0.26,0.07451,0.313725,0.541176,0.27,0.086275,0.305882,0.509804,0.28,0.094118,0.286275,0.478431,0.29,0.101961,0.278431,0.45098,0.3,0.109804,0.266667,0.411765,0.31,0.113725,0.258824,0.380392,0.32,0.113725,0.25098,0.34902,0.33,0.109804,0.266667,0.321569,0.34,0.105882,0.301961,0.262745,0.35,0.094118,0.309804,0.243137,0.36,0.082353,0.321569,0.227451,0.37,0.07451,0.341176,0.219608,0.38,0.070588,0.360784,0.211765,0.39,0.066667,0.380392,0.215686,0.4,0.062745,0.4,0.176471,0.425,0.07451,0.419608,0.145098,0.45,0.086275,0.439216,0.117647,0.475,0.121569,0.470588,0.117647,0.5,0.184314,0.501961,0.14902,0.525,0.254902,0.541176,0.188235,0.55,0.32549,0.580392,0.231373,0.575,0.403922,0.619608,0.278431,0.6,0.501961,0.670588,0.333333,0.63,0.592157,0.729412,0.4,0.65,0.741176,0.788235,0.490196,0.67,0.858824,0.858824,0.603922,0.7,0.921569,0.835294,0.580392,0.75,0.901961,0.729412,0.494118,0.8,0.858824,0.584314,0.388235,0.85,0.8,0.439216,0.321569,0.9,0.678431,0.298039,0.203922,0.95,0.54902,0.168627,0.109804,0.975,0.478431,0.082353,0.047059,1.0,0.45098,0.007843,0.0]},"Rainbow Desaturated":{"ColorSpace":"RGB","Name":"Rainbow Desaturated","DefaultMap":true,"NanColor":[1,1,0],"RGBPoints":[0,0.278431372549,0.278431372549,0.858823529412,0.143,0,0,0.360784313725,0.285,0,1,1,0.429,0,0.501960784314,0,0.571,1,1,0,0.714,1,0.380392156863,0,0.857,0.419607843137,0,0,1,0.878431372549,0.301960784314,0.301960784314]},"Yellow - Gray - Blue":{"ColorSpace":"Lab","Name":"Yellow - Gray - Blue","Creator":"Francesca Samsel","DefaultMap":true,"RGBPoints":[0.0,0.301961,0.047059,0.090196,0.01639024615287782,0.396078431372549,0.0392156862745098,0.058823529411764705,0.03278049230575558,0.49411764705882355,0.054901960784313725,0.03529411764705882,0.049170738458633456,0.5882352941176471,0.11372549019607843,0.023529411764705882,0.06556098461151122,0.6627450980392157,0.16862745098039217,0.01568627450980392,0.08195123076438904,0.7411764705882353,0.22745098039215686,0.00392156862745098,0.09834147691726686,0.788235294117647,0.2901960784313726,0.0,0.11473172307014465,0.8627450980392157,0.3803921568627451,0.011764705882352941,0.13112196922302247,0.9019607843137255,0.4588235294117647,0.027450980392156862,0.14751221537590026,0.9176470588235294,0.5215686274509804,0.047058823529411764,0.16390246152877808,0.9254901960784314,0.5803921568627451,0.0784313725490196,0.18029270768165587,0.9372549019607843,0.6431372549019608,0.12156862745098039,0.19668295383453369,0.9450980392156862,0.7098039215686275,0.1843137254901961,0.2130731999874115,0.9529411764705882,0.7686274509803922,0.24705882352941178,0.2294634461402893,0.9647058823529412,0.8274509803921568,0.3254901960784314,0.24585369229316711,0.9686274509803922,0.8784313725490196,0.4235294117647059,0.26224393844604493,0.9725490196078431,0.9176470588235294,0.5137254901960784,0.27863418459892275,0.9803921568627451,0.9490196078431372,0.596078431372549,0.2950244307518005,0.9803921568627451,0.9725490196078431,0.6705882352941176,0.31141467690467833,0.9882352941176471,0.9882352941176471,0.7568627450980392,0.32714931321144103,0.984313725490196,0.9882352941176471,0.8549019607843137,0.32780492305755615,0.9882352941176471,0.9882352941176471,0.8588235294117647,0.32781492305755616,0.9529411764705882,0.9529411764705882,0.8941176470588236,0.32781492305755616,0.9529411764705882,0.9529411764705882,0.8941176470588236,0.34483569985938073,0.8901960784313725,0.8901960784313725,0.807843137254902,0.3618564766612053,0.8274509803921568,0.8235294117647058,0.7372549019607844,0.37887725346302986,0.7764705882352941,0.7647058823529411,0.6784313725490196,0.39589803026485443,0.7254901960784313,0.7137254901960784,0.6274509803921569,0.412918807066679,0.6784313725490196,0.6627450980392157,0.5803921568627451,0.42993958386850356,0.6313725490196078,0.6078431372549019,0.5333333333333333,0.4469603606703281,0.5803921568627451,0.5568627450980392,0.48627450980392156,0.46398113747215275,0.5372549019607843,0.5058823529411764,0.44313725490196076,0.4810019142739773,0.4980392156862745,0.4588235294117647,0.40784313725490196,0.4980226910758019,0.4627450980392157,0.4196078431372549,0.37254901960784315,0.5150434678776264,0.43137254901960786,0.38823529411764707,0.34509803921568627,0.532064244679451,0.403921568627451,0.3568627450980392,0.3176470588235294,0.5490850214812756,0.37254901960784315,0.3215686274509804,0.29411764705882354,0.5661057982831001,0.34509803921568627,0.29411764705882354,0.26666666666666666,0.5831265750849247,0.3176470588235294,0.2627450980392157,0.23921568627450981,0.6001473518867493,0.28627450980392155,0.23137254901960785,0.21176470588235294,0.6171681286885738,0.2549019607843137,0.2,0.1843137254901961,0.6341889054903984,0.23137254901960785,0.17254901960784313,0.16470588235294117,0.651209682292223,0.2,0.1450980392156863,0.13725490196078433,0.6682404590940476,0.14902,0.196078,0.278431,0.7014164131846428,0.2,0.2549019607843137,0.34509803921568627,0.734592367275238,0.24705882352941178,0.3176470588235294,0.41568627450980394,0.7677683213658333,0.3058823529411765,0.38823529411764707,0.49411764705882355,0.8009442754564285,0.37254901960784315,0.4588235294117647,0.5686274509803921,0.8341202295470238,0.44313725490196076,0.5333333333333333,0.6431372549019608,0.867296183637619,0.5176470588235295,0.615686274509804,0.7254901960784313,0.9004721377282143,0.6,0.6980392156862745,0.8,0.9336480918188095,0.6862745098039216,0.7843137254901961,0.8705882352941177,0.9668240459094047,0.7607843137254902,0.8588235294117647,0.9294117647058824,0.9834120229547024,0.807843137254902,0.9019607843137255,0.9607843137254902,1.0,0.8901960784313725,0.9568627450980393,0.984313725490196]},"Rainbow Uniform":{"ColorSpace":"RGB","Name":"Rainbow Uniform","Creator":"Colin Ware","DefaultMap":true,"NanColor":[1,0,0],"RGBPoints":[0.0,0.02,0.3813,0.9981,0.023809523809523808,0.02000006,0.424267768,0.96906969,0.047619047619047616,0.02,0.467233763,0.940033043,0.07142857142857142,0.02,0.5102,0.911,0.09523809523809523,0.02000006,0.546401494,0.872669438,0.11904761904761904,0.02,0.582600362,0.83433295,0.14285714285714285,0.02,0.6188,0.796,0.16666666666666666,0.02000006,0.652535156,0.749802434,0.19047619047619047,0.02,0.686267004,0.703599538,0.21428571428571427,0.02,0.72,0.6574,0.23809523809523808,0.02000006,0.757035456,0.603735359,0.2619047619047619,0.02,0.794067037,0.55006613,0.2857142857142857,0.02,0.8311,0.4964,0.30952380952380953,0.021354336738172372,0.8645368555261631,0.4285579460761159,0.3333333333333333,0.023312914349117714,0.897999359924484,0.36073871343115577,0.35714285714285715,0.015976108242848862,0.9310479513349017,0.2925631815088092,0.38095238095238093,0.27421074700988196,0.952562960995083,0.15356836602739213,0.40476190476190477,0.4933546281681699,0.9619038625309482,0.11119493614749336,0.42857142857142855,0.6439,0.9773,0.0469,0.4523809523809524,0.762401813,0.984669591,0.034600153,0.47619047619047616,0.880901185,0.992033407,0.022299877,0.5,0.9995285432627147,0.9995193706781492,0.0134884641450013,0.5238095238095238,0.999402998,0.955036376,0.079066628,0.5476190476190477,0.9994,0.910666223,0.148134024,0.5714285714285714,0.9994,0.8663,0.2172,0.5952380952380952,0.999269665,0.818035981,0.217200652,0.6190476190476191,0.999133332,0.769766184,0.2172,0.6428571428571429,0.999,0.7215,0.2172,0.6666666666666666,0.99913633,0.673435546,0.217200652,0.6904761904761905,0.999266668,0.625366186,0.2172,0.7142857142857143,0.9994,0.5773,0.2172,0.7380952380952381,0.999402998,0.521068455,0.217200652,0.7619047619047619,0.9994,0.464832771,0.2172,0.7857142857142857,0.9994,0.4086,0.2172,0.8095238095238095,0.9947599917687346,0.33177297300202935,0.2112309638520206,0.8333333333333334,0.9867129505479589,0.2595183410914934,0.19012239549291934,0.8571428571428571,0.9912458875646419,0.14799417507952672,0.21078892136920357,0.8809523809523809,0.949903037,0.116867171,0.252900603,0.9047619047619048,0.903199533,0.078432949,0.291800389,0.9285714285714286,0.8565,0.04,0.3307,0.9523809523809523,0.798902627,0.04333345,0.358434298,0.9761904761904762,0.741299424,0.0466667,0.386166944,1.0,0.6837,0.05,0.4139]},"Turbo":{"ColorSpace":"RGB","Name":"Turbo","Creator":"Anton Mikhailov","DefaultMap":true,"RGBPoints":[0,0.18995,0.07176,0.23217,0.0039216,0.19483,0.08339,0.26149,0.0078431,0.19956,0.09498,0.29024,0.011765,0.20415,0.10652,0.31844,0.015686,0.2086,0.11802,0.34607,0.019608,0.21291,0.12947,0.37314,0.023529,0.21708,0.14087,0.39964,0.027451,0.22111,0.15223,0.42558,0.031373,0.225,0.16354,0.45096,0.035294,0.22875,0.17481,0.47578,0.039216,0.23236,0.18603,0.50004,0.043137,0.23582,0.1972,0.52373,0.047059,0.23915,0.20833,0.54686,0.05098,0.24234,0.21941,0.56942,0.054902,0.24539,0.23044,0.59142,0.058824,0.2483,0.24143,0.61286,0.062745,0.25107,0.25237,0.63374,0.066667,0.25369,0.26327,0.65406,0.070588,0.25618,0.27412,0.67381,0.07451,0.25853,0.28492,0.693,0.078431,0.26074,0.29568,0.71162,0.082353,0.2628,0.30639,0.72968,0.086275,0.26473,0.31706,0.74718,0.090196,0.26652,0.32768,0.76412,0.094118,0.26816,0.33825,0.7805,0.098039,0.26967,0.34878,0.79631,0.10196,0.27103,0.35926,0.81156,0.10588,0.27226,0.3697,0.82624,0.1098,0.27334,0.38008,0.84037,0.11373,0.27429,0.39043,0.85393,0.11765,0.27509,0.40072,0.86692,0.12157,0.27576,0.41097,0.87936,0.12549,0.27628,0.42118,0.89123,0.12941,0.27667,0.43134,0.90254,0.13333,0.27691,0.44145,0.91328,0.13725,0.27701,0.45152,0.92347,0.14118,0.27698,0.46153,0.93309,0.1451,0.2768,0.47151,0.94214,0.14902,0.27648,0.48144,0.95064,0.15294,0.27603,0.49132,0.95857,0.15686,0.27543,0.50115,0.96594,0.16078,0.27469,0.51094,0.97275,0.16471,0.27381,0.52069,0.97899,0.16863,0.27273,0.5304,0.98461,0.17255,0.27106,0.54015,0.9893,0.17647,0.26878,0.54995,0.99303,0.18039,0.26592,0.55979,0.99583,0.18431,0.26252,0.56967,0.99773,0.18824,0.25862,0.57958,0.99876,0.19216,0.25425,0.5895,0.99896,0.19608,0.24946,0.59943,0.99835,0.2,0.24427,0.60937,0.99697,0.20392,0.23874,0.61931,0.99485,0.20784,0.23288,0.62923,0.99202,0.21176,0.22676,0.63913,0.98851,0.21569,0.22039,0.64901,0.98436,0.21961,0.21382,0.65886,0.97959,0.22353,0.20708,0.66866,0.97423,0.22745,0.20021,0.67842,0.96833,0.23137,0.19326,0.68812,0.9619,0.23529,0.18625,0.69775,0.95498,0.23922,0.17923,0.70732,0.94761,0.24314,0.17223,0.7168,0.93981,0.24706,0.16529,0.7262,0.93161,0.25098,0.15844,0.73551,0.92305,0.2549,0.15173,0.74472,0.91416,0.25882,0.14519,0.75381,0.90496,0.26275,0.13886,0.76279,0.8955,0.26667,0.13278,0.77165,0.8858,0.27059,0.12698,0.78037,0.8759,0.27451,0.12151,0.78896,0.86581,0.27843,0.11639,0.7974,0.85559,0.28235,0.11167,0.80569,0.84525,0.28627,0.10738,0.81381,0.83484,0.2902,0.10357,0.82177,0.82437,0.29412,0.10026,0.82955,0.81389,0.29804,0.0975,0.83714,0.80342,0.30196,0.09532,0.84455,0.79299,0.30588,0.09377,0.85175,0.78264,0.3098,0.09287,0.85875,0.7724,0.31373,0.09267,0.86554,0.7623,0.31765,0.0932,0.87211,0.75237,0.32157,0.09451,0.87844,0.74265,0.32549,0.09662,0.88454,0.73316,0.32941,0.09958,0.8904,0.72393,0.33333,0.10342,0.896,0.715,0.33725,0.10815,0.90142,0.70599,0.34118,0.11374,0.90673,0.69651,0.3451,0.12014,0.91193,0.6866,0.34902,0.12733,0.91701,0.67627,0.35294,0.13526,0.92197,0.66556,0.35686,0.14391,0.9268,0.65448,0.36078,0.15323,0.93151,0.64308,0.36471,0.16319,0.93609,0.63137,0.36863,0.17377,0.94053,0.61938,0.37255,0.18491,0.94484,0.60713,0.37647,0.19659,0.94901,0.59466,0.38039,0.20877,0.95304,0.58199,0.38431,0.22142,0.95692,0.56914,0.38824,0.23449,0.96065,0.55614,0.39216,0.24797,0.96423,0.54303,0.39608,0.2618,0.96765,0.52981,0.4,0.27597,0.97092,0.51653,0.40392,0.29042,0.97403,0.50321,0.40784,0.30513,0.97697,0.48987,0.41176,0.32006,0.97974,0.47654,0.41569,0.33517,0.98234,0.46325,0.41961,0.35043,0.98477,0.45002,0.42353,0.36581,0.98702,0.43688,0.42745,0.38127,0.98909,0.42386,0.43137,0.39678,0.99098,0.41098,0.43529,0.41229,0.99268,0.39826,0.43922,0.42778,0.99419,0.38575,0.44314,0.44321,0.99551,0.37345,0.44706,0.45854,0.99663,0.3614,0.45098,0.47375,0.99755,0.34963,0.4549,0.48879,0.99828,0.33816,0.45882,0.50362,0.99879,0.32701,0.46275,0.51822,0.9991,0.31622,0.46667,0.53255,0.99919,0.30581,0.47059,0.54658,0.99907,0.29581,0.47451,0.56026,0.99873,0.28623,0.47843,0.57357,0.99817,0.27712,0.48235,0.58646,0.99739,0.26849,0.48627,0.59891,0.99638,0.26038,0.4902,0.61088,0.99514,0.2528,0.49412,0.62233,0.99366,0.24579,0.49804,0.63323,0.99195,0.23937,0.50196,0.64362,0.98999,0.23356,0.50588,0.65394,0.98775,0.22835,0.5098,0.66428,0.98524,0.2237,0.51373,0.67462,0.98246,0.2196,0.51765,0.68494,0.97941,0.21602,0.52157,0.69525,0.9761,0.21294,0.52549,0.70553,0.97255,0.21032,0.52941,0.71577,0.96875,0.20815,0.53333,0.72596,0.9647,0.2064,0.53725,0.7361,0.96043,0.20504,0.54118,0.74617,0.95593,0.20406,0.5451,0.75617,0.95121,0.20343,0.54902,0.76608,0.94627,0.20311,0.55294,0.77591,0.94113,0.2031,0.55686,0.78563,0.93579,0.20336,0.56078,0.79524,0.93025,0.20386,0.56471,0.80473,0.92452,0.20459,0.56863,0.8141,0.91861,0.20552,0.57255,0.82333,0.91253,0.20663,0.57647,0.83241,0.90627,0.20788,0.58039,0.84133,0.89986,0.20926,0.58431,0.8501,0.89328,0.21074,0.58824,0.85868,0.88655,0.2123,0.59216,0.86709,0.87968,0.21391,0.59608,0.8753,0.87267,0.21555,0.6,0.88331,0.86553,0.21719,0.60392,0.89112,0.85826,0.2188,0.60784,0.8987,0.85087,0.22038,0.61176,0.90605,0.84337,0.22188,0.61569,0.91317,0.83576,0.22328,0.61961,0.92004,0.82806,0.22456,0.62353,0.92666,0.82025,0.2257,0.62745,0.93301,0.81236,0.22667,0.63137,0.93909,0.80439,0.22744,0.63529,0.94489,0.79634,0.228,0.63922,0.95039,0.78823,0.22831,0.64314,0.9556,0.78005,0.22836,0.64706,0.96049,0.77181,0.22811,0.65098,0.96507,0.76352,0.22754,0.6549,0.96931,0.75519,0.22663,0.65882,0.97323,0.74682,0.22536,0.66275,0.97679,0.73842,0.22369,0.66667,0.98,0.73,0.22161,0.67059,0.98289,0.7214,0.21918,0.67451,0.98549,0.7125,0.2165,0.67843,0.98781,0.7033,0.21358,0.68235,0.98986,0.69382,0.21043,0.68627,0.99163,0.68408,0.20706,0.6902,0.99314,0.67408,0.20348,0.69412,0.99438,0.66386,0.19971,0.69804,0.99535,0.65341,0.19577,0.70196,0.99607,0.64277,0.19165,0.70588,0.99654,0.63193,0.18738,0.7098,0.99675,0.62093,0.18297,0.71373,0.99672,0.60977,0.17842,0.71765,0.99644,0.59846,0.17376,0.72157,0.99593,0.58703,0.16899,0.72549,0.99517,0.57549,0.16412,0.72941,0.99419,0.56386,0.15918,0.73333,0.99297,0.55214,0.15417,0.73725,0.99153,0.54036,0.1491,0.74118,0.98987,0.52854,0.14398,0.7451,0.98799,0.51667,0.13883,0.74902,0.9859,0.50479,0.13367,0.75294,0.9836,0.49291,0.12849,0.75686,0.98108,0.48104,0.12332,0.76078,0.97837,0.4692,0.11817,0.76471,0.97545,0.4574,0.11305,0.76863,0.97234,0.44565,0.10797,0.77255,0.96904,0.43399,0.10294,0.77647,0.96555,0.42241,0.09798,0.78039,0.96187,0.41093,0.0931,0.78431,0.95801,0.39958,0.08831,0.78824,0.95398,0.38836,0.08362,0.79216,0.94977,0.37729,0.07905,0.79608,0.94538,0.36638,0.07461,0.8,0.94084,0.35566,0.07031,0.80392,0.93612,0.34513,0.06616,0.80784,0.93125,0.33482,0.06218,0.81176,0.92623,0.32473,0.05837,0.81569,0.92105,0.31489,0.05475,0.81961,0.91572,0.3053,0.05134,0.82353,0.91024,0.29599,0.04814,0.82745,0.90463,0.28696,0.04516,0.83137,0.89888,0.27824,0.04243,0.83529,0.89298,0.26981,0.03993,0.83922,0.88691,0.26152,0.03753,0.84314,0.88066,0.25334,0.03521,0.84706,0.87422,0.24526,0.03297,0.85098,0.8676,0.2373,0.03082,0.8549,0.86079,0.22945,0.02875,0.85882,0.8538,0.2217,0.02677,0.86275,0.84662,0.21407,0.02487,0.86667,0.83926,0.20654,0.02305,0.87059,0.83172,0.19912,0.02131,0.87451,0.82399,0.19182,0.01966,0.87843,0.81608,0.18462,0.01809,0.88235,0.80799,0.17753,0.0166,0.88627,0.79971,0.17055,0.0152,0.8902,0.79125,0.16368,0.01387,0.89412,0.7826,0.15693,0.01264,0.89804,0.77377,0.15028,0.01148,0.90196,0.76476,0.14374,0.01041,0.90588,0.75556,0.13731,0.00942,0.9098,0.74617,0.13098,0.00851,0.91373,0.73661,0.12477,0.00769,0.91765,0.72686,0.11867,0.00695,0.92157,0.71692,0.11268,0.00629,0.92549,0.7068,0.1068,0.00571,0.92941,0.6965,0.10102,0.00522,0.93333,0.68602,0.09536,0.00481,0.93725,0.67535,0.0898,0.00449,0.94118,0.66449,0.08436,0.00424,0.9451,0.65345,0.07902,0.00408,0.94902,0.64223,0.0738,0.00401,0.95294,0.63082,0.06868,0.00401,0.95686,0.61923,0.06367,0.0041,0.96078,0.60746,0.05878,0.00427,0.96471,0.5955,0.05399,0.00453,0.96863,0.58336,0.04931,0.00486,0.97255,0.57103,0.04474,0.00529,0.97647,0.55852,0.04028,0.00579,0.98039,0.54583,0.03593,0.00638,0.98431,0.53295,0.03169,0.00705,0.98824,0.51989,0.02756,0.0078,0.99216,0.50664,0.02354,0.00863,0.99608,0.49321,0.01963,0.00955,1,0.4796,0.01583,0.01055]},"Jet":{"ColorSpace":"RGB","Name":"Jet","DefaultMap":true,"RGBPoints":[-1,0,0,0.5625,-0.777778,0,0,1,-0.269841,0,1,1,-0.015873,0.5,1,0.5,0.238095,1,1,0,0.746032,1,0,0,1,0.5,0,0]},"Warm to Cool":{"ColorSpace":"Diverging","Name":"Warm to Cool","NanColor":[1,1,0],"RGBPoints":[0,0.705882352941,0.0156862745098,0.149019607843,0.5,0.865,0.865,0.865,1,0.23137254902,0.298039215686,0.752941176471]},"Warm to Cool (Extended)":{"ColorSpace":"Lab","Creator":"Francesca Samsel","Name":"Warm to Cool (Extended)","NanColor":[0.250004,0,0],"RGBPoints":[0,0.34902,0,0.129412,0.025,0.4,0.00392157,0.101961,0.05,0.470588,0.0156863,0.0901961,0.075,0.54902,0.027451,0.0705882,0.1,0.619608,0.0627451,0.0431373,0.125,0.690196,0.12549,0.0627451,0.15,0.741176,0.184314,0.0745098,0.175,0.788235,0.266667,0.0941176,0.2,0.811765,0.345098,0.113725,0.225,0.831373,0.411765,0.133333,0.25,0.85098,0.47451,0.145098,0.275,0.870588,0.54902,0.156863,0.3,0.878431,0.619608,0.168627,0.325,0.890196,0.658824,0.196078,0.35,0.909804,0.717647,0.235294,0.375,0.929412,0.776471,0.278431,0.395522,0.94902,0.823529,0.321569,0.418905,0.968627,0.87451,0.407843,0.444278,0.980392,0.917647,0.509804,0.470149,0.988235,0.956863,0.643137,0.483582,0.992157,0.964706,0.713725,0.499,0.988235,0.980392,0.870588,0.5,1,1,1,0.501,0.913725,0.988235,0.937255,0.516418,0.827451,0.980392,0.886275,0.531343,0.764706,0.980392,0.866667,0.546766,0.658824,0.980392,0.843137,0.564179,0.572549,0.964706,0.835294,0.587562,0.423529,0.941176,0.87451,0.60597,0.262745,0.901961,0.862745,0.629851,0.0705882,0.854902,0.870588,0.651741,0.0509804,0.8,0.85098,0.681592,0.0235294,0.709804,0.831373,0.712935,0.0313725,0.615686,0.811765,0.75,0.0313725,0.537255,0.788235,0.775,0.0392157,0.466667,0.768627,0.8,0.0509804,0.396078,0.741176,0.825,0.054902,0.317647,0.709804,0.85,0.054902,0.243137,0.678431,0.875,0.0431373,0.164706,0.639216,0.9,0.0313725,0.0980392,0.6,0.925,0.0392157,0.0392157,0.560784,0.95,0.105882,0.0509804,0.509804,0.975,0.113725,0.0235294,0.45098,1,0.12549,0,0.380392]},"Grayscale":{"ColorSpace":"RGB","Name":"Grayscale","NanColor":[1,0,0],"RGBPoints":[0,0,0,0,1,1,1,1]},"Black, Orange and White":{"ColorSpace":"RGB","Name":"Black, Orange and White","NanColor":[1,1,0],"RGBPoints":[0,0,0,0,0.333,0.501960784314,0,0,0.666,1,0.501960784314,0,1,1,1,1]},"Linear YGB 1211g":{"ColorSpace":"Lab","Creator":"Francesca Samsel","Name":"Linear YGB 1211g","NanColor":[0.25,0,0],"RGBPoints":[0,1,0.988235,0.968627,0.02,1,0.952941,0.878431,0.05,0.968627,0.905882,0.776471,0.1,0.94902,0.898039,0.647059,0.15,0.901961,0.878431,0.556863,0.2,0.847059,0.858824,0.482353,0.25,0.690196,0.819608,0.435294,0.3,0.513725,0.768627,0.384314,0.35,0.337255,0.721569,0.337255,0.4,0.278431,0.658824,0.392157,0.45,0.231373,0.639216,0.435294,0.5,0.203922,0.6,0.486275,0.55,0.172549,0.568627,0.537255,0.6,0.141176,0.517647,0.54902,0.65,0.133333,0.458824,0.541176,0.7,0.12549,0.396078,0.529412,0.75,0.117647,0.321569,0.521569,0.8,0.121569,0.258824,0.509804,0.85,0.133333,0.227451,0.501961,0.9,0.145098,0.192157,0.490196,0.95,0.188235,0.164706,0.470588,1,0.258824,0.196078,0.439216]},"Linear Blue (8_31f)":{"ColorSpace":"Lab","Creator":"Francesca Samsel","Name":"Linear Blue (8_31f)","NanColor":[0.25,0,0],"RGBPoints":[0,0.960784,1,0.980392,0.05,0.815686,0.960784,0.913725,0.1,0.670588,0.929412,0.870588,0.15,0.556863,0.901961,0.843137,0.2,0.478431,0.870588,0.823529,0.25,0.439216,0.831373,0.803922,0.3,0.4,0.8,0.788235,0.35,0.376471,0.768627,0.768627,0.4,0.34902,0.709804,0.729412,0.45,0.32549,0.654902,0.690196,0.5,0.301961,0.607843,0.658824,0.55,0.247059,0.545098,0.619608,0.6,0.239216,0.494118,0.580392,0.65,0.227451,0.439216,0.541176,0.7,0.227451,0.403922,0.521569,0.75,0.231373,0.368627,0.501961,0.8,0.227451,0.321569,0.470588,0.85,0.219608,0.282353,0.439216,0.9,0.192157,0.235294,0.4,0.95,0.160784,0.184314,0.34902,1,0.133333,0.12549,0.301961]},"Blue to Red Rainbow":{"ColorSpace":"HSV","Name":"Blue to Red Rainbow","NanColor":[0.498039215686,0.498039215686,0.498039215686],"RGBPoints":[0,0,0,1,1,1,0,0]},"Rainbow Blended White":{"ColorSpace":"RGB","Name":"Rainbow Blended White","NanColor":[1,1,0],"RGBPoints":[0,1,1,1,0.17,0,0,1,0.34,0,1,1,0.5,0,1,0,0.67,1,1,0,0.84,1,0,0,1,0.878431372549,0,1]},"Rainbow Blended Grey":{"ColorSpace":"RGB","Name":"Rainbow Blended Grey","NanColor":[1,1,0],"RGBPoints":[0,0.317647058824,0.341176470588,0.43137254902,0.17,0,0,1,0.34,0,1,1,0.5,0,1,0,0.67,1,1,0,0.84,1,0,0,1,0.878431372549,0,1]},"Rainbow Blended Black":{"ColorSpace":"RGB","Name":"Rainbow Blended Black","NanColor":[1,1,0],"RGBPoints":[0,0,0,0,0.17,0,0,1,0.34,0,1,1,0.5,0,1,0,0.67,1,1,0,0.84,1,0,0,1,0.878431372549,0,1]},"Blue to Yellow":{"ColorSpace":"RGB","Name":"Blue to Yellow","NanColor":[1,0,0],"RGBPoints":[0,0.0392156862745,0.0392156862745,0.949019607843,1,0.949019607843,0.949019607843,0.0392156862745]},"blot":{"ColorSpace":"HSV","Name":"blot","RGBPoints":[0,0,0,1,0.166,0,0,1,0.167,1,0,1,0.332,1,0,1,0.333,0,1,1,0.5,0,1,1,0.501,0,1,0,0.666,0,1,0,0.667,1,1,0,0.832,1,1,0,0.833,1,0,0,1,1,0,0]},"CIELab Blue to Red":{"ColorSpace":"Lab","Name":"CIELab Blue to Red","NanColor":[1,1,0],"RGBPoints":[0,0,0.6,0.749019607843,1,0.76862745098,0.466666666667,0.341176470588]},"erdc_rainbow_bright":{"ColorSpace":"Lab","Name":"erdc_rainbow_bright","RGBPoints":[-1,0.32549,0.14902,0.960784,-0.866221,0.297047,0.375586,0.963836,-0.732441,0.180302,0.536818,0.964627,-0.598662,0.1302,0.649207,0.929647,-0.464883,0.0445143,0.749654,0.855998,-0.331104,0.0271325,0.830713,0.721527,-0.197324,0.259504,0.866145,0.543555,-0.0635452,0.428364,0.890725,0.329819,0.0702341,0.568503,0.898508,0.187623,0.204013,0.738259,0.890317,0.0825461,0.337793,0.84546,0.86136,0.0147555,0.471572,0.912191,0.808018,0,0.605351,0.962848,0.710445,0,0.73913,0.999469,0.600258,0.0176284,0.87291,0.994156,0.445975,0.193912,1,0.980407,0.247105,0.262699]},"erdc_rainbow_dark":{"ColorSpace":"Lab","Name":"erdc_rainbow_dark","RGBPoints":[-1,0,0,0.423499,-0.866221,0,0.119346,0.529237,-0.732441,0,0.238691,0.634976,-0.598662,0,0.346852,0.68788,-0.464883,0,0.45022,0.718141,-0.331104,0,0.553554,0.664839,-0.197324,0,0.651082,0.519303,-0.0635452,0.115841,0.72479,0.352857,0.0702341,0.326771,0.781195,0.140187,0.204013,0.522765,0.798524,0.0284624,0.337793,0.703162,0.788685,0.00885756,0.471572,0.845118,0.751133,0,0.605351,0.955734,0.690825,0,0.73913,0.995402,0.567916,0.0618524,0.87291,0.987712,0.403398,0.164851,1,0.980407,0.247105,0.262699]},"nic_CubicL":{"ColorSpace":"Lab","Name":"nic_CubicL","RGBPoints":[-1,0.479965,0.0118108,0.5307,-0.87451,0.522213,0.0551282,0.706919,-0.74902,0.50839,0.237278,0.867764,-0.623529,0.451617,0.373834,0.987255,-0.498039,0.39365,0.497255,0.97506,-0.372549,0.328631,0.599639,0.891843,-0.247059,0.250043,0.690286,0.778553,-0.121569,0.249656,0.764905,0.645857,0.00392157,0.297954,0.821466,0.50449,0.129412,0.337509,0.872595,0.358447,0.254902,0.430011,0.913789,0.297079,0.380392,0.587191,0.931381,0.333353,0.505882,0.727937,0.93591,0.353742,0.631373,0.826403,0.921081,0.365066,0.756863,0.893201,0.846317,0.372662,0.882353,0.965347,0.73884,0.378506,1,0.983235,0.597451,0.366856]},"nic_CubicYF":{"ColorSpace":"Lab","Name":"nic_CubicYF","RGBPoints":[-1,0.5151,0.0482,0.6697,-0.87451,0.520711,0.168955,0.800574,-0.74902,0.493694,0.278596,0.911824,-0.623529,0.440026,0.369475,0.984978,-0.498039,0.398932,0.457593,0.987053,-0.372549,0.350651,0.540644,0.929608,-0.247059,0.298827,0.615625,0.857729,-0.121569,0.239928,0.685061,0.769531,0.00392157,0.228832,0.739349,0.673287,0.129412,0.263297,0.78608,0.569988,0.254902,0.298107,0.828337,0.460214,0.380392,0.33092,0.864071,0.352674,0.505882,0.38306,0.898169,0.287309,0.631373,0.49023,0.917481,0.307961,0.756863,0.62372,0.926026,0.332309,0.882353,0.717458,0.92527,0.342476,1,0.8,0.9255,0.3529]},"gist_earth":{"ColorSpace":"Lab","Name":"gist_earth","RGBPoints":[-1,0,0,0,-0.87451,0.239216,0.027451,0.415686,-0.74902,0.0901961,0.254902,0.556863,-0.623529,0.0941176,0.352941,0.54902,-0.498039,0.105882,0.435294,0.533333,-0.372549,0.12549,0.52549,0.501961,-0.247059,0.156863,0.596078,0.443137,-0.121569,0.196078,0.65098,0.380392,0.00392157,0.282353,0.717647,0.301961,0.129412,0.466667,0.772549,0.27451,0.254902,0.678431,0.784314,0.309804,0.380392,0.901961,0.756863,0.376471,0.505882,0.992157,0.705882,0.521569,0.631373,1,0.721569,0.701961,0.756863,1,0.784314,0.784314,0.882353,1,0.866667,0.866667,1,1,1,1]},"2hot":{"ColorSpace":"Lab","Name":"2hot","RGBPoints":[-1,0.0416667,0,0,-0.873016,0.208333,0,0,-0.746032,0.375,0,0,-0.619048,0.541667,0,0,-0.492063,0.708333,0,0,-0.365079,0.854137,0,0,-0.238095,0.937488,0.039062,0,-0.111111,1,0.208333,0,0.015873,1,0.375,0,0.142857,1,0.541667,0,0.269841,1,0.708333,0,0.396825,1,0.858805,0.03125,0.52381,1,0.947392,0.15625,0.650794,1,1,0.3125,0.777778,1,1,0.5625,0.904762,1,1,0.8125,1,1,1,1]},"erdc_red2yellow_BW":{"ColorSpace":"Lab","Name":"erdc_red2yellow_BW","RGBPoints":[-1,7.54296e-07,0,1.09827e-05,-0.87451,0.18285,0.0264094,0,-0.74902,0.3066,0,0,-0.623529,0.422841,0,0,-0.498039,0.522945,0,0,-0.372549,0.605721,0,0,-0.247059,0.672502,0.14168,0,-0.121569,0.728167,0.244025,0,0.00392157,0.781215,0.333454,0,0.129412,0.825,0.423586,0,0.254902,0.855893,0.516793,0,0.380392,0.880491,0.608846,0,0.505882,0.910305,0.695505,0,0.631373,0.94109,0.779067,0.223528,0.756863,0.967873,0.858572,0.473521,0.882353,0.986815,0.933211,0.751583,1,1,1,0.999997]},"erdc_marine2gold_BW":{"ColorSpace":"Lab","Name":"erdc_marine2gold_BW","RGBPoints":[-1,1.11641e-07,0,1.62551e-06,-0.87451,0.0413146,0.0619808,0.209857,-0.74902,0.0185557,0.101341,0.350684,-0.623529,0.00486405,0.149847,0.461054,-0.498039,0.0836345,0.210845,0.517906,-0.372549,0.173222,0.276134,0.541793,-0.247059,0.259857,0.343877,0.535869,-0.121569,0.362299,0.408124,0.504293,0.00392157,0.468266,0.468276,0.468257,0.129412,0.582781,0.527545,0.374914,0.254902,0.691591,0.585251,0.274266,0.380392,0.784454,0.645091,0.247332,0.505882,0.862299,0.710383,0.27518,0.631373,0.920863,0.782923,0.351563,0.756863,0.955792,0.859699,0.533541,0.882353,0.976162,0.93433,0.780671,1,1,1,0.999983]},"erdc_blue2gold_BW":{"ColorSpace":"Lab","Name":"erdc_blue2gold_BW","RGBPoints":[-1,0,0,0,-0.87451,0.0742735,0.0440331,0.230013,-0.74902,0.125276,0.0258685,0.415826,-0.623529,0.143879,0.0163031,0.591346,-0.498039,0.212261,0.0627855,0.705239,-0.372549,0.306048,0.141178,0.763636,-0.247059,0.391537,0.232286,0.773263,-0.121569,0.461734,0.336633,0.708321,0.00392157,0.54209,0.427581,0.590007,0.129412,0.61704,0.508623,0.460978,0.254902,0.702703,0.579586,0.309117,0.380392,0.790336,0.644811,0.170397,0.505882,0.870173,0.710733,0.117134,0.631373,0.93656,0.781991,0.157144,0.756863,0.965672,0.862068,0.409836,0.882353,0.985751,0.936296,0.714162,1,1,1,0.999999]},"erdc_sapphire2gold_BW":{"ColorSpace":"Lab","Name":"erdc_sapphire2gold_BW","RGBPoints":[-1,0.107704,0.107708,0.107694,-0.87451,0.1851,0.112354,0.308554,-0.74902,0.236782,0.114233,0.48788,-0.623529,0.28296,0.126187,0.639464,-0.498039,0.344787,0.171643,0.739713,-0.372549,0.413325,0.242371,0.76913,-0.247059,0.481863,0.3131,0.719841,-0.121569,0.550402,0.383829,0.612222,0.00392157,0.61894,0.454558,0.51126,0.129412,0.687478,0.525287,0.39993,0.254902,0.756017,0.596016,0.289923,0.380392,0.824555,0.666745,0.255498,0.505882,0.892979,0.736822,0.27696,0.631373,0.938851,0.804966,0.351734,0.756863,0.966491,0.874853,0.53572,0.882353,0.982105,0.94153,0.782579,1,1,1,0.999986]},"erdc_red2purple_BW":{"ColorSpace":"Lab","Name":"erdc_red2purple_BW","RGBPoints":[-1,0,0,0,-0.87451,0.167793,0.0166271,0.0431278,-0.74902,0.262608,0.0107595,0.0791181,-0.623529,0.351902,0.0101858,0.100926,-0.498039,0.441257,0.0160835,0.131919,-0.372549,0.5221,0.0555972,0.195625,-0.247059,0.593852,0.104294,0.310234,-0.121569,0.654628,0.158115,0.448486,0.00392157,0.707443,0.220914,0.570253,0.129412,0.749504,0.293268,0.67897,0.254902,0.781587,0.370517,0.779269,0.380392,0.809951,0.451099,0.855831,0.505882,0.84424,0.531462,0.900451,0.631373,0.865174,0.620901,0.91606,0.756863,0.875041,0.714054,0.910284,0.882353,0.880764,0.80554,0.896276,1,0.887572,0.887591,0.887556]},"erdc_purple2pink_BW":{"ColorSpace":"Lab","Name":"erdc_purple2pink_BW","RGBPoints":[-1,0,0,0,-0.87451,0.117562,0.0291202,0.175876,-0.74902,0.178368,0.0458476,0.285454,-0.623529,0.237731,0.0680173,0.387717,-0.498039,0.300877,0.0956291,0.484802,-0.372549,0.370929,0.136858,0.554985,-0.247059,0.449033,0.189273,0.58863,-0.121569,0.529971,0.245796,0.598587,0.00392157,0.609914,0.300643,0.610244,0.129412,0.697079,0.351286,0.616371,0.254902,0.785858,0.401991,0.617376,0.380392,0.862517,0.45745,0.64463,0.505882,0.91359,0.525462,0.705336,0.631373,0.932583,0.61064,0.767412,0.756863,0.922478,0.706966,0.817522,0.882353,0.901302,0.803071,0.856311,1,0.887571,0.887591,0.887549]},"erdc_pbj_lin":{"ColorSpace":"Lab","Name":"erdc_pbj_lin","RGBPoints":[-1,0,0,0,-0.87451,0.091821,0.0611476,0.10617,-0.74902,0.160311,0.0900022,0.192713,-0.623529,0.22484,0.12126,0.272128,-0.498039,0.291263,0.157469,0.340828,-0.372549,0.360015,0.200388,0.388903,-0.247059,0.437497,0.250058,0.387201,-0.121569,0.512636,0.304969,0.355955,0.00392157,0.582603,0.360874,0.33488,0.129412,0.655126,0.416374,0.306351,0.254902,0.725889,0.473329,0.279051,0.380392,0.778125,0.537928,0.302697,0.505882,0.815894,0.606931,0.382431,0.631373,0.839159,0.679308,0.497608,0.756863,0.854748,0.751666,0.631792,0.882353,0.869483,0.822508,0.768592,1,0.887572,0.887589,0.887565]},"erdc_blue2green_muted":{"ColorSpace":"Lab","Name":"erdc_blue2green_muted","RGBPoints":[-1,0.107704,0.107708,0.107695,-0.87451,0.141522,0.13066,0.270741,-0.74902,0.180123,0.146119,0.42308,-0.623529,0.210161,0.169674,0.551795,-0.498039,0.239701,0.212939,0.634969,-0.372549,0.253916,0.282947,0.653641,-0.247059,0.242791,0.366933,0.608521,-0.121569,0.226302,0.446776,0.52693,0.00392157,0.236237,0.514689,0.458798,0.129412,0.274641,0.577589,0.376069,0.254902,0.349625,0.633993,0.288131,0.380392,0.4437,0.683677,0.260497,0.505882,0.536247,0.731214,0.285424,0.631373,0.628472,0.777128,0.349151,0.756863,0.718259,0.819287,0.496825,0.882353,0.804768,0.856164,0.703299,1,0.887571,0.887591,0.887548]},"erdc_blue2green_BW":{"ColorSpace":"Lab","Name":"erdc_blue2green_BW","RGBPoints":[-1,3.63578e-07,0,5.29374e-06,-0.87451,0.0539915,0.0577948,0.212806,-0.74902,0.0620393,0.0758942,0.388959,-0.623529,0.0697499,0.102032,0.54177,-0.498039,0.113295,0.156156,0.64334,-0.372549,0.152047,0.243196,0.670283,-0.247059,0.158096,0.344084,0.622864,-0.121569,0.151142,0.43922,0.532767,0.00392157,0.17155,0.521588,0.457719,0.129412,0.225861,0.599141,0.363997,0.254902,0.32328,0.67007,0.259083,0.380392,0.442344,0.733697,0.223754,0.505882,0.558409,0.794941,0.257411,0.631373,0.673875,0.854344,0.340822,0.756863,0.787244,0.909326,0.524717,0.882353,0.896483,0.958063,0.775914,1,1,1,0.999982]},"GREEN-WHITE_LINEAR":{"ColorSpace":"Lab","Name":"GREEN-WHITE_LINEAR","RGBPoints":[-1,0,0,0,-0.87451,0,0.062745,0,-0.74902,0,0.12549,0,-0.623529,0,0.188235,0,-0.498039,0,0.25098,0,-0.372549,0,0.313725,0,-0.247059,0,0.376471,0,-0.121569,0.094118,0.439216,0,0.00392157,0.196078,0.501961,0,0.129412,0.294118,0.564706,0,0.254902,0.396078,0.627451,0,0.380392,0.498039,0.690196,0,0.505882,0.6,0.752941,0.145098,0.631373,0.701961,0.815686,0.364706,0.756863,0.8,0.878431,0.580392,0.882353,0.901961,0.941176,0.796078,1,1,1,1]},"erdc_green2yellow_BW":{"ColorSpace":"Lab","Name":"erdc_green2yellow_BW","RGBPoints":[-1,0,0,0,-0.87451,0,0.105542,0.0603919,-0.74902,0,0.159454,0.104148,-0.623529,0,0.219502,0.15542,-0.498039,0,0.282276,0.203811,-0.372549,0,0.346331,0.235652,-0.247059,0,0.411765,0.235428,-0.121569,0,0.477177,0.217977,0.00392157,0.0593644,0.541635,0.21361,0.129412,0.233081,0.604722,0.210591,0.254902,0.369803,0.664942,0.226536,0.380392,0.498446,0.722367,0.288237,0.505882,0.601929,0.782244,0.380815,0.631373,0.703207,0.840497,0.512134,0.756863,0.803186,0.896433,0.674462,0.882353,0.903834,0.950266,0.846715,1,1,1,0.999981]},"blue2cyan":{"ColorSpace":"Lab","Name":"blue2cyan","RGBPoints":[-1,0,0,0,-0.87451,0,0.152941,0.364706,-0.74902,0,0.254902,0.470588,-0.623529,0,0.34902,0.572549,-0.498039,0,0.443137,0.670588,-0.372549,0,0.537255,0.772549,-0.247059,0,0.627451,0.870588,-0.121569,0,0.717647,0.964706,0.00392157,0.0784314,0.772549,1,0.129412,0.207843,0.858824,1,0.254902,0.32549,0.941176,1,0.380392,0.45098,1,1,0.505882,0.560784,1,1,0.631373,0.662745,1,1,0.756863,0.760784,1,1,0.882353,0.870588,1,1,1,1,1,1]},"erdc_blue2cyan_BW":{"ColorSpace":"Lab","Name":"erdc_blue2cyan_BW","RGBPoints":[-1,4.05298e-07,0,5.9012e-06,-0.87451,0.0207526,0.0740933,0.18093,-0.74902,0,0.121033,0.30343,-0.623529,0,0.166892,0.416095,-0.498039,0,0.216768,0.524796,-0.372549,0.0164769,0.275471,0.608585,-0.247059,0.0544527,0.344824,0.659267,-0.121569,0.0880643,0.419118,0.688675,0.00392157,0.127938,0.492556,0.720256,0.129412,0.149476,0.566946,0.756918,0.254902,0.188961,0.641333,0.792122,0.380392,0.245482,0.715336,0.827609,0.505882,0.329216,0.786235,0.874761,0.631373,0.453558,0.852803,0.918466,0.756863,0.626281,0.910493,0.954,0.882353,0.82257,0.958709,0.980146,1,1,1,0.999989]},"erdc_blue_BW":{"ColorSpace":"Lab","Name":"erdc_blue_BW","RGBPoints":[-1,0,0,0,-0.87451,0.0425591,0.0763529,0.150682,-0.74902,0.0569472,0.119154,0.275403,-0.623529,0.0635978,0.164772,0.395427,-0.498039,0.0774342,0.213851,0.510014,-0.372549,0.106815,0.267034,0.615102,-0.247059,0.122093,0.324649,0.720068,-0.121569,0.160851,0.387068,0.806956,0.00392157,0.213754,0.453516,0.878012,0.129412,0.26722,0.524656,0.932436,0.254902,0.326844,0.599279,0.968038,0.380392,0.403403,0.674712,0.984784,0.505882,0.499703,0.745519,1,0.631373,0.615055,0.813983,1,0.756863,0.74405,0.879228,1,0.882353,0.877909,0.941913,1,1,1,1,0.999996]},"BLUE-WHITE":{"ColorSpace":"Lab","Name":"BLUE-WHITE","RGBPoints":[-1,0,0,0,-0.87451,0,0,0.082353,-0.74902,0,0,0.168627,-0.623529,0,0,0.254902,-0.498039,0,0,0.337255,-0.372549,0,0,0.423529,-0.247059,0,0,0.509804,-0.121569,0,0.101961,0.592157,0.00392157,0,0.203922,0.678431,0.129412,0,0.301961,0.764706,0.254902,0,0.403922,0.85098,0.380392,0,0.505882,0.933333,0.505882,0,0.603922,1,0.631373,0.254902,0.705882,1,0.756863,0.509804,0.807843,1,0.882353,0.764706,0.905882,1,1,1,1,1]},"erdc_purple_BW":{"ColorSpace":"Lab","Name":"erdc_purple_BW","RGBPoints":[-1,4.264e-08,0,6.20844e-07,-0.87451,0.100579,0.0593111,0.145666,-0.74902,0.167794,0.0889224,0.254953,-0.623529,0.231446,0.123339,0.360511,-0.498039,0.296699,0.163027,0.461278,-0.372549,0.363211,0.209286,0.55306,-0.247059,0.431136,0.260776,0.637195,-0.121569,0.498202,0.320012,0.705799,0.00392157,0.567456,0.380459,0.778091,0.129412,0.629381,0.445284,0.8448,0.254902,0.688373,0.517374,0.895694,0.380392,0.74891,0.590906,0.93976,0.505882,0.805017,0.667956,0.977626,0.631373,0.850914,0.752618,0.992396,0.756863,0.89724,0.838454,0.994093,0.882353,0.948461,0.922603,0.994449,1,1,1,0.999967]},"erdc_magenta_BW":{"ColorSpace":"Lab","Name":"erdc_magenta_BW","RGBPoints":[-1,0,0,2.54023e-05,-0.87451,0.128696,0.0456782,0.11635,-0.74902,0.228133,0.0476299,0.201452,-0.623529,0.327273,0.0374065,0.282107,-0.498039,0.420953,0.0408166,0.35709,-0.372549,0.511562,0.0642203,0.430511,-0.247059,0.599552,0.102686,0.504257,-0.121569,0.684646,0.150536,0.579429,0.00392157,0.765817,0.205978,0.656062,0.129412,0.839176,0.27229,0.731807,0.254902,0.89536,0.357594,0.797309,0.380392,0.930238,0.457825,0.846984,0.505882,0.945921,0.564536,0.880571,0.631373,0.948995,0.670753,0.902279,0.756863,0.947124,0.772819,0.918171,0.882353,0.947265,0.869424,0.934352,1,0.954719,0.95475,0.954726]},"magenta":{"ColorSpace":"Lab","Name":"magenta","RGBPoints":[-1,0,0,0,-0.87451,0.364706,0,0.152941,-0.74902,0.470588,0,0.254902,-0.623529,0.572549,0,0.34902,-0.498039,0.670588,0,0.443137,-0.372549,0.772549,0,0.537255,-0.247059,0.870588,0,0.627451,-0.121569,0.964706,0,0.717647,0.00392157,1,0.0784314,0.772549,0.129412,1,0.207843,0.858824,0.254902,1,0.32549,0.941176,0.380392,1,0.45098,1,0.505882,1,0.560784,1,0.631373,1,0.662745,1,0.756863,1,0.760784,1,0.882353,1,0.870588,1,1,1,1,1]},"RED-PURPLE":{"ColorSpace":"Lab","Name":"RED-PURPLE","RGBPoints":[-1,0,0,0,-0.87451,0.188235,0,0.007843,-0.74902,0.345098,0,0.035294,-0.623529,0.439216,0,0.098039,-0.498039,0.533333,0,0.152941,-0.372549,0.627451,0.015686,0.211765,-0.247059,0.721569,0.031373,0.266667,-0.121569,0.8,0.047059,0.329412,0.00392157,0.862745,0.047059,0.403922,0.129412,0.941176,0.062745,0.466667,0.254902,0.988235,0.078431,0.54902,0.380392,0.988235,0.141176,0.643137,0.505882,0.988235,0.25098,0.729412,0.631373,0.988235,0.376471,0.811765,0.756863,0.988235,0.54902,0.886275,0.882353,0.988235,0.752941,0.952941,1,0.996078,0.996078,0.996078]},"erdc_red_BW":{"ColorSpace":"Lab","Name":"erdc_red_BW","RGBPoints":[-1,0,0,0,-0.87451,0.147204,0.0480135,0.0401815,-0.74902,0.253411,0.0617478,0.0301333,-0.623529,0.356059,0.0746331,0.0446897,-0.498039,0.457731,0.0934935,0.0636931,-0.372549,0.557199,0.122714,0.0860013,-0.247059,0.665179,0.144238,0.105585,-0.121569,0.763833,0.187056,0.138326,0.00392157,0.847035,0.254558,0.189407,0.129412,0.905663,0.345937,0.258215,0.254902,0.941431,0.447111,0.346277,0.380392,0.962608,0.546927,0.457571,0.505882,0.987833,0.637276,0.569944,0.631373,0.994202,0.732176,0.687958,0.756863,0.993304,0.826268,0.800567,0.882353,0.994413,0.917205,0.906393,1,1,1,0.999979]},"RED_TEMPERATURE":{"ColorSpace":"Lab","Name":"RED_TEMPERATURE","RGBPoints":[-1,0,0,0,-0.87451,0.090196,0,0,-0.74902,0.180392,0,0,-0.623529,0.270588,0,0,-0.498039,0.360784,0,0,-0.372549,0.45098,0,0,-0.247059,0.545098,0,0,-0.121569,0.635294,0,0,0.00392157,0.72549,0.058824,0,0.129412,0.815686,0.176471,0,0.254902,0.905882,0.294118,0,0.380392,1,0.411765,0,0.505882,1,0.533333,0.027451,0.631373,1,0.65098,0.27451,0.756863,1,0.768627,0.521569,0.882353,1,0.886275,0.768627,1,1,1,1]},"erdc_orange_BW":{"ColorSpace":"Lab","Name":"erdc_orange_BW","RGBPoints":[-1,0,0,2.53806e-05,-0.87451,0.135871,0.0593824,0,-0.74902,0.224328,0.0907216,0,-0.623529,0.318083,0.119647,0,-0.498039,0.414443,0.150246,0,-0.372549,0.511077,0.184884,0,-0.247059,0.605501,0.226033,0,-0.121569,0.695274,0.275491,0,0.00392157,0.777826,0.334445,0,0.129412,0.851498,0.402441,0,0.254902,0.915899,0.47759,0.000602975,0.380392,0.971984,0.557882,0.0361443,0.505882,1,0.641287,0.135967,0.631373,1,0.725198,0.27997,0.756863,1,0.808205,0.438135,0.882353,1,0.89306,0.587036,1,1,0.977928,0.721599]},"heated_object":{"ColorSpace":"Lab","Name":"heated_object","RGBPoints":[-1,0,0,0,-0.87451,0.34902,0.0862745,0,-0.74902,0.45098,0.172549,0,-0.623529,0.52549,0.231373,0,-0.498039,0.580392,0.278431,0,-0.372549,0.623529,0.313725,0,-0.247059,0.670588,0.352941,0,-0.121569,0.717647,0.392157,0,0.00392157,0.772549,0.439216,0,0.129412,0.839216,0.494118,0,0.254902,0.901961,0.541176,0,0.380392,0.968627,0.6,0,0.505882,1,0.658824,0,0.631373,1,0.721569,0,0.756863,1,0.827451,0.298039,0.882353,1,0.976471,0.72549,1,1,1,1]},"erdc_gold_BW":{"ColorSpace":"Lab","Name":"erdc_gold_BW","RGBPoints":[-1,0,0,1.90933e-05,-0.87451,0.128363,0.0636265,0,-0.74902,0.193795,0.111057,0,-0.623529,0.25976,0.15987,0,-0.498039,0.328546,0.210589,0,-0.372549,0.399726,0.26332,0,-0.247059,0.472969,0.318261,0,-0.121569,0.546245,0.375827,0,0.00392157,0.61745,0.436719,0,0.129412,0.685545,0.501113,0,0.254902,0.749578,0.568799,0,0.380392,0.80962,0.6394,0,0.505882,0.865572,0.712699,0.10257,0.631373,0.917709,0.787569,0.233665,0.756863,0.966914,0.863138,0.369608,0.882353,1,0.939405,0.496104,1,0.999225,1,0.612275]},"erdc_brown_BW":{"ColorSpace":"Lab","Name":"erdc_brown_BW","RGBPoints":[-1,3.3216e-07,0,4.83629e-06,-0.87451,0.14693,0.0518172,0,-0.74902,0.225806,0.0814996,0,-0.623529,0.301681,0.111452,0,-0.498039,0.370487,0.150664,0,-0.372549,0.43108,0.199477,0,-0.247059,0.4849,0.255107,0,-0.121569,0.536798,0.313486,0,0.00392157,0.59286,0.371167,0,0.129412,0.653119,0.428135,0,0.254902,0.714589,0.485917,0.0379541,0.380392,0.774667,0.54565,0.116634,0.505882,0.831222,0.608047,0.183895,0.631373,0.880305,0.674199,0.260298,0.756863,0.922314,0.742472,0.367086,0.882353,0.959408,0.811222,0.497258,1,0.993548,0.875183,0.622093]},"copper_Matlab":{"ColorSpace":"Lab","Name":"copper_Matlab","RGBPoints":[-1,0,0,0,-0.87451,0.0784314,0.0501961,0.0313725,-0.74902,0.156863,0.100392,0.0627451,-0.623529,0.235294,0.150588,0.0941176,-0.498039,0.313725,0.200784,0.12549,-0.372549,0.392157,0.25098,0.156863,-0.247059,0.470588,0.301176,0.188235,-0.121569,0.54902,0.351373,0.219608,0.00392157,0.627451,0.401569,0.25098,0.129412,0.705882,0.451765,0.282353,0.254902,0.784314,0.501961,0.313725,0.380392,0.862745,0.552157,0.345098,0.505882,0.941176,0.602353,0.376471,0.631373,1,0.652549,0.407843,0.756863,1,0.702745,0.439216,0.882353,1,0.752941,0.470588,1,1,0.8,0.5]},"pink_Matlab":{"ColorSpace":"Lab","Name":"pink_Matlab","RGBPoints":[-1,0,0,0,-0.87451,0.312416,0.204524,0.204524,-0.74902,0.441822,0.289241,0.289241,-0.623529,0.54112,0.354246,0.354246,-0.498039,0.624831,0.409048,0.409048,-0.372549,0.698582,0.45733,0.45733,-0.247059,0.764404,0.502282,0.500979,-0.121569,0.791292,0.591516,0.54112,0.00392157,0.817297,0.66895,0.578481,0.129412,0.842499,0.738308,0.613572,0.254902,0.866968,0.801687,0.646762,0.380392,0.890766,0.86041,0.678329,0.505882,0.913944,0.913944,0.711254,0.631373,0.936549,0.936549,0.79459,0.756863,0.958621,0.958621,0.869979,0.882353,0.980196,0.980196,0.939336,1,1,1,1]},"bone_Matlab":{"ColorSpace":"Lab","Name":"bone_Matlab","RGBPoints":[-1,0,0,0,-0.87451,0.054902,0.054902,0.075817,-0.74902,0.109804,0.109804,0.151634,-0.623529,0.164706,0.164706,0.227451,-0.498039,0.219608,0.219608,0.303268,-0.372549,0.27451,0.27451,0.379085,-0.247059,0.329412,0.329902,0.454412,-0.121569,0.384314,0.405719,0.509314,0.00392157,0.439216,0.481536,0.564216,0.129412,0.494118,0.557353,0.619118,0.254902,0.54902,0.63317,0.67402,0.380392,0.603922,0.708987,0.728922,0.505882,0.660294,0.783824,0.783824,0.631373,0.746569,0.838725,0.838725,0.756863,0.832843,0.893627,0.893627,0.882353,0.919118,0.948529,0.948529,1,1,1,1]},"Purples":{"ColorSpace":"Lab","Name":"Purples","RGBPoints":[-1,0.247059,0,0.490196,-0.87451,0.288397,0.07677,0.525629,-0.74902,0.32975,0.153587,0.561092,-0.623529,0.373057,0.236263,0.600461,-0.498039,0.416363,0.319,0.639923,-0.372549,0.459669,0.405613,0.685198,-0.247059,0.503345,0.491534,0.730058,-0.121569,0.562399,0.54862,0.757616,0.00392157,0.621453,0.606075,0.785544,0.129412,0.680508,0.674971,0.824914,0.254902,0.739562,0.743406,0.863899,0.380392,0.798616,0.800492,0.893426,0.505882,0.85684,0.856655,0.922491,0.631373,0.898178,0.894056,0.942176,0.756863,0.938654,0.930919,0.961646,0.882353,0.964245,0.958478,0.977393,1,0.988235,0.984314,0.992157]},"Blues":{"ColorSpace":"Lab","Name":"Blues","RGBPoints":[-1,0.031373,0.188235,0.419608,-0.87451,0.031373,0.253195,0.516063,-0.74902,0.031757,0.318139,0.612149,-0.623529,0.080969,0.38113,0.661361,-0.498039,0.130427,0.444152,0.710327,-0.372549,0.195386,0.509112,0.743791,-0.247059,0.260715,0.573841,0.777209,-0.121569,0.341423,0.628958,0.808704,0.00392157,0.422745,0.684075,0.839892,0.129412,0.523137,0.739193,0.861546,0.254902,0.622684,0.793464,0.883429,0.380392,0.701423,0.826928,0.910988,0.505882,0.778685,0.8603,0.937993,0.631373,0.825928,0.891795,0.953741,0.756863,0.87328,0.923291,0.969489,0.882353,0.922491,0.954787,0.985236,1,0.968627,0.984314,1]},"Greens":{"ColorSpace":"Lab","Name":"Greens","RGBPoints":[-1,0,0.266667,0.105882,-0.87451,0,0.347374,0.139346,-0.74902,0.000538,0.427912,0.172933,-0.623529,0.069435,0.486967,0.222145,-0.498039,0.138178,0.546082,0.271326,-0.372549,0.197232,0.609073,0.31857,-0.247059,0.257255,0.671742,0.365859,-0.121569,0.357647,0.720953,0.415071,0.00392157,0.45767,0.769919,0.465021,0.129412,0.546251,0.811257,0.537855,0.254902,0.634295,0.852211,0.610688,0.380392,0.709097,0.883706,0.683522,0.505882,0.78316,0.914833,0.755894,0.631373,0.842215,0.938454,0.818885,0.756863,0.899977,0.961538,0.880692,0.882353,0.935409,0.975317,0.92203,1,0.968627,0.988235,0.960784]},"PuBu":{"ColorSpace":"Lab","Name":"PuBu","RGBPoints":[-1,0.301961,0,0.294118,-0.87451,0.404321,0.029527,0.390573,-0.74902,0.50599,0.059592,0.486782,-0.623529,0.519769,0.158016,0.551742,-0.498039,0.533456,0.256194,0.616301,-0.372549,0.54133,0.33887,0.655671,-0.247059,0.54902,0.421592,0.695087,-0.121569,0.54902,0.506236,0.736424,0.00392157,0.550127,0.590573,0.777701,0.129412,0.585559,0.665375,0.81707,0.254902,0.622145,0.739023,0.855825,0.380392,0.687105,0.784298,0.879446,0.505882,0.752065,0.829758,0.903253,0.631373,0.817024,0.87897,0.930811,0.756863,0.880907,0.927213,0.957832,0.882353,0.926182,0.958708,0.975548,1,0.968627,0.988235,0.992157]},"BuPu":{"ColorSpace":"Lab","Name":"BuPu","RGBPoints":[-1,0.007843,0.219608,0.345098,-0.87451,0.01178,0.286536,0.449427,-0.74902,0.015702,0.35328,0.553479,-0.623529,0.01767,0.396586,0.622376,-0.498039,0.021115,0.4402,0.690688,-0.372549,0.11757,0.503191,0.722184,-0.247059,0.214625,0.565859,0.753633,-0.121569,0.336671,0.615071,0.78316,0.00392157,0.457978,0.663975,0.812503,0.129412,0.556401,0.703345,0.836125,0.254902,0.65421,0.742714,0.859669,0.380392,0.736886,0.782084,0.881323,0.505882,0.81827,0.821638,0.903068,0.631373,0.873387,0.864944,0.92669,0.756863,0.927536,0.907605,0.949988,0.882353,0.964937,0.9391,0.967705,1,1,0.968627,0.984314]},"BuGn":{"ColorSpace":"Lab","Name":"BuGn","RGBPoints":[-1,0.031373,0.25098,0.505882,-0.87451,0.031373,0.329719,0.590527,-0.74902,0.031911,0.408397,0.674787,-0.623529,0.100807,0.479262,0.710219,-0.498039,0.169704,0.550219,0.745744,-0.372549,0.238601,0.62699,0.787082,-0.247059,0.307958,0.703114,0.826759,-0.121569,0.39654,0.752326,0.797232,0.00392157,0.485121,0.801046,0.767705,0.129412,0.573702,0.83451,0.738178,0.254902,0.661592,0.867743,0.711034,0.380392,0.732457,0.895302,0.74253,0.505882,0.801845,0.922307,0.774579,0.631373,0.841215,0.938055,0.817885,0.756863,0.880907,0.95391,0.861084,0.882353,0.926182,0.971626,0.902422,1,0.968627,0.988235,0.941176]},"GnBu":{"ColorSpace":"Lab","Name":"GnBu","RGBPoints":[-1,0,0.266667,0.105882,-0.87451,0,0.347374,0.139346,-0.74902,0.000538,0.427912,0.172933,-0.623529,0.069435,0.486967,0.222145,-0.498039,0.138178,0.546175,0.272095,-0.372549,0.197232,0.615071,0.368551,-0.247059,0.256609,0.683276,0.464867,-0.121569,0.329443,0.722645,0.555417,0.00392157,0.403137,0.762138,0.645413,0.129412,0.503529,0.805444,0.718247,0.254902,0.603922,0.848597,0.790465,0.380392,0.704314,0.887966,0.847551,0.505882,0.802307,0.926321,0.903714,0.631373,0.851519,0.944037,0.941115,0.756863,0.899977,0.961538,0.976901,0.882353,0.935409,0.975317,0.984775,1,0.968627,0.988235,0.992157]},"GnBuPu":{"ColorSpace":"Lab","Name":"GnBuPu","RGBPoints":[-1,0.003922,0.27451,0.211765,-0.87451,0.003922,0.349312,0.280661,-0.74902,0.003937,0.423852,0.349773,-0.623529,0.005905,0.46519,0.446228,-0.498039,0.009443,0.506344,0.542837,-0.372549,0.111803,0.535871,0.649135,-0.247059,0.214025,0.565859,0.753633,-0.121569,0.310481,0.615071,0.78316,0.00392157,0.407797,0.663975,0.812503,0.129412,0.531811,0.703345,0.836125,0.254902,0.65421,0.742714,0.859669,0.380392,0.736886,0.782084,0.881323,0.505882,0.81827,0.821176,0.902884,0.631373,0.873387,0.854641,0.922568,0.756863,0.927536,0.888535,0.942361,0.882353,0.964937,0.929873,0.964014,1,1,0.968627,0.984314]},"BuGnYl":{"ColorSpace":"Lab","Name":"BuGnYl","RGBPoints":[-1,0.031373,0.113725,0.345098,-0.87451,0.088458,0.159,0.463206,-0.74902,0.145052,0.204567,0.5807,-0.623529,0.139146,0.287243,0.620069,-0.498039,0.13318,0.370196,0.659562,-0.372549,0.123337,0.470588,0.706805,-0.247059,0.115386,0.570335,0.753126,-0.121569,0.186251,0.643168,0.761,0.00392157,0.258716,0.71514,0.768074,0.129412,0.380761,0.760415,0.750358,0.254902,0.503576,0.806075,0.732795,0.380392,0.645306,0.861192,0.719016,0.505882,0.783899,0.91511,0.705606,0.631373,0.858701,0.944637,0.6997,0.756863,0.931349,0.973303,0.698424,0.882353,0.966782,0.987082,0.777163,1,1,1,0.85098]},"PuRd":{"ColorSpace":"Lab","Name":"PuRd","RGBPoints":[-1,0.286275,0,0.415686,-0.87451,0.38273,0.001968,0.441276,-0.74902,0.479231,0.003922,0.466774,-0.623529,0.581592,0.003922,0.480554,-0.498039,0.683799,0.00549,0.494887,-0.372549,0.776317,0.105882,0.544098,-0.247059,0.867866,0.206321,0.592618,-0.121569,0.919047,0.308681,0.612303,0.00392157,0.968812,0.411226,0.632603,0.129412,0.974717,0.519493,0.671972,0.254902,0.980546,0.626451,0.71065,0.380392,0.984483,0.701253,0.732303,0.505882,0.988328,0.77504,0.755617,0.631373,0.990296,0.828189,0.812703,0.756863,0.992372,0.880907,0.869035,0.882353,0.996309,0.926182,0.912341,1,1,0.968627,0.952941]},"RdPu":{"ColorSpace":"Lab","Name":"RdPu","RGBPoints":[-1,0.403922,0,0.121569,-0.87451,0.500377,0,0.192434,-0.74902,0.596909,0.000277,0.263037,-0.623529,0.703206,0.035709,0.300438,-0.498039,0.808612,0.071296,0.338854,-0.372549,0.857824,0.116571,0.441215,-0.247059,0.905513,0.163552,0.54293,-0.121569,0.889765,0.281661,0.617732,0.00392157,0.873156,0.39897,0.691611,0.129412,0.82985,0.491488,0.736886,0.254902,0.789081,0.583237,0.781853,0.380392,0.810734,0.656071,0.819254,0.505882,0.833126,0.729181,0.85684,0.631373,0.870527,0.80792,0.898178,0.756863,0.907605,0.884398,0.938331,0.882353,0.9391,0.921799,0.958016,1,0.968627,0.956863,0.976471]},"Oranges":{"ColorSpace":"Lab","Name":"Oranges","RGBPoints":[-1,0.498039,0.152941,0.015686,-0.87451,0.57481,0.182468,0.013718,-0.74902,0.651765,0.212042,0.011734,-0.623529,0.752157,0.247474,0.007797,-0.498039,0.851719,0.283368,0.004475,-0.372549,0.898962,0.348328,0.039908,-0.247059,0.945652,0.413426,0.076401,-0.121569,0.969273,0.484291,0.157109,0.00392157,0.992157,0.554971,0.238185,0.129412,0.992157,0.619931,0.330704,0.254902,0.992157,0.684967,0.423837,0.380392,0.992157,0.751895,0.532103,0.505882,0.992249,0.817716,0.639354,0.631373,0.994218,0.861023,0.725967,0.756863,0.996186,0.903576,0.810965,0.882353,0.998155,0.933103,0.868051,1,1,0.960784,0.921569]},"Reds":{"ColorSpace":"Lab","Name":"Reds","RGBPoints":[-1,0.403922,0,0.05098,-0.87451,0.525967,0.029527,0.066728,-0.74902,0.647643,0.058962,0.082476,-0.623529,0.722445,0.076678,0.098224,-0.498039,0.797186,0.095194,0.114187,-0.372549,0.868051,0.164091,0.143714,-0.247059,0.937809,0.233541,0.173933,-0.121569,0.96143,0.326059,0.232987,0.00392157,0.984375,0.418147,0.292657,0.129412,0.986344,0.496886,0.371396,0.254902,0.988235,0.575702,0.450673,0.380392,0.988235,0.656409,0.543191,0.505882,0.98842,0.736747,0.635894,0.631373,0.992357,0.809581,0.732349,0.756863,0.996186,0.880692,0.826759,0.882353,0.998155,0.92203,0.885813,1,1,0.960784,0.941176]},"RdOr":{"ColorSpace":"Lab","Name":"RdOr","RGBPoints":[-1,0.498039,0,0,-0.87451,0.6004,0,0,-0.74902,0.702514,0.000738,0.000477,-0.623529,0.773379,0.095225,0.061499,-0.498039,0.843875,0.189865,0.12283,-0.372549,0.891119,0.294195,0.203537,-0.247059,0.937855,0.397924,0.283137,-0.121569,0.963445,0.476663,0.316601,0.00392157,0.988297,0.555771,0.351665,0.129412,0.990265,0.646321,0.436309,0.254902,0.992157,0.735256,0.519646,0.380392,0.992157,0.784468,0.570827,0.505882,0.992249,0.833218,0.623483,0.631373,0.994218,0.872587,0.706159,0.756863,0.996186,0.911419,0.788189,0.882353,0.998155,0.940946,0.859054,1,1,0.968627,0.92549]},"BrOrYl":{"ColorSpace":"Lab","Name":"BrOrYl","RGBPoints":[-1,0.4,0.145098,0.023529,-0.87451,0.500392,0.174625,0.019592,-0.74902,0.600784,0.204291,0.015656,-0.623529,0.701176,0.251534,0.011719,-0.498039,0.800984,0.299146,0.008397,-0.372549,0.863975,0.370012,0.043829,-0.247059,0.926321,0.441107,0.0794,-0.121569,0.961753,0.521815,0.120738,0.00392157,0.996078,0.602645,0.163122,0.129412,0.996078,0.68729,0.237924,0.254902,0.996078,0.771011,0.314879,0.380392,0.996078,0.832034,0.444798,0.505882,0.996171,0.892042,0.572595,0.631373,0.998139,0.931411,0.65724,0.756863,1,0.969489,0.741669,0.882353,1,0.985236,0.822376,1,1,1,0.898039]},"RdOrYl":{"ColorSpace":"Lab","Name":"RdOrYl","RGBPoints":[-1,0.501961,0,0.14902,-0.87451,0.622038,0,0.14902,-0.74902,0.741761,0.0004,0.148866,-0.623529,0.816563,0.05158,0.129181,-0.498039,0.890965,0.10356,0.110235,-0.372549,0.940177,0.205921,0.137793,-0.247059,0.988281,0.308789,0.165536,-0.121569,0.99025,0.432803,0.200969,0.00392157,0.992218,0.555217,0.236278,0.129412,0.994187,0.628051,0.267774,0.254902,0.996078,0.701038,0.301269,0.380392,0.996078,0.777809,0.383945,0.505882,0.996171,0.852826,0.466621,0.631373,0.998139,0.892195,0.549296,0.756863,1,0.931349,0.632188,0.882353,1,0.966782,0.7188,1,1,1,0.8]},"blue2yellow":{"ColorSpace":"Lab","Name":"blue2yellow","RGBPoints":[-1,0,0,1,0,0.5,0.5,0.5,1,1,1,0]},"erdc_blue2gold":{"ColorSpace":"Lab","Name":"erdc_blue2gold","RGBPoints":[-1,0.175119,0.0438468,1,-0.874016,0.22383,0.159771,0.94557,-0.748031,0.27254,0.233611,0.891216,-0.622047,0.321251,0.296526,0.836857,-0.496063,0.369962,0.354296,0.782359,-0.370079,0.418672,0.409139,0.72754,-0.244094,0.467383,0.462152,0.672148,-0.11811,0.51609,0.51396,0.615825,0.00787402,0.572863,0.55452,0.559172,0.133858,0.630269,0.593822,0.517729,0.259843,0.689588,0.624668,0.47446,0.385827,0.745394,0.656113,0.428638,0.511811,0.798624,0.688104,0.379105,0.637795,0.849926,0.720593,0.323834,0.76378,0.899765,0.753543,0.258657,0.889764,0.948487,0.78692,0.171778,1,0.990413,0.816451,0.00729848]},"erdc_blue2yellow":{"ColorSpace":"Lab","Name":"erdc_blue2yellow","RGBPoints":[-1,0.0830122,0,0.495617,-0.87451,0.141973,0.0551288,0.57363,-0.74902,0.193048,0.110258,0.604561,-0.623529,0.234231,0.165386,0.57643,-0.498039,0.275413,0.220515,0.548299,-0.372549,0.316596,0.275644,0.520169,-0.247059,0.357778,0.330773,0.492038,-0.121569,0.398961,0.385901,0.463908,0.00392157,0.449929,0.438487,0.426815,0.129412,0.511572,0.488299,0.379944,0.254902,0.581222,0.53603,0.325741,0.380392,0.650871,0.583761,0.271538,0.505882,0.720521,0.631493,0.217335,0.631373,0.79017,0.679224,0.163132,0.756863,0.85982,0.726955,0.108929,0.882353,0.910254,0.774159,0.14112,1,0.927513,0.81759,0.306289]},"erdc_cyan2orange":{"ColorSpace":"Lab","Name":"erdc_cyan2orange","RGBPoints":[-1,0.0471513,0.213874,0.414329,-0.87451,0.0674702,0.256648,0.439027,-0.74902,0.0959957,0.299331,0.462089,-0.623529,0.132428,0.341872,0.483212,-0.498039,0.188743,0.38277,0.500597,-0.372549,0.268511,0.420229,0.512179,-0.247059,0.352945,0.455602,0.519101,-0.121569,0.43893,0.489368,0.521538,0.00392157,0.522445,0.522495,0.522436,0.129412,0.600089,0.555682,0.53205,0.254902,0.67988,0.587981,0.539163,0.380392,0.761011,0.619586,0.544439,0.505882,0.84278,0.650741,0.548567,0.631373,0.910713,0.687347,0.557822,0.756863,0.952232,0.734972,0.577775,0.882353,0.975642,0.789858,0.604868,1,0.990752,0.843643,0.632857]},"erdc_purple2green":{"ColorSpace":"Lab","Name":"erdc_purple2green","RGBPoints":[-1,0.235006,0.0483128,0.530899,-0.87451,0.302968,0.108419,0.552391,-0.74902,0.360241,0.166059,0.569502,-0.623529,0.406746,0.226782,0.579373,-0.498039,0.444073,0.28964,0.582094,-0.372549,0.473648,0.353774,0.577947,-0.247059,0.497636,0.418154,0.567911,-0.121569,0.519086,0.481741,0.553968,0.00392157,0.542884,0.542914,0.542875,0.129412,0.566303,0.603989,0.527499,0.254902,0.595218,0.662965,0.516857,0.380392,0.628641,0.720701,0.510673,0.505882,0.665373,0.777849,0.508165,0.631373,0.704182,0.834921,0.508303,0.756863,0.743846,0.892328,0.50999,0.882353,0.783158,0.950422,0.512181,1,0.818617,1,0.513888]},"erdc_purple2green_dark":{"ColorSpace":"Lab","Name":"erdc_purple2green_dark","RGBPoints":[-1,0.107656,0,0.428682,-0.87451,0.1924,0,0.449799,-0.74902,0.255118,0.0648939,0.466726,-0.623529,0.304256,0.133066,0.476703,-0.498039,0.343202,0.19716,0.479793,-0.372549,0.373876,0.260353,0.476241,-0.247059,0.398497,0.322872,0.466953,-0.121569,0.420016,0.384252,0.453785,0.00392157,0.44319,0.443216,0.443186,0.129412,0.465553,0.502139,0.428233,0.254902,0.492959,0.559151,0.417591,0.380392,0.524654,0.615092,0.411016,0.505882,0.55959,0.670583,0.40779,0.631373,0.596614,0.726102,0.406948,0.756863,0.634544,0.782032,0.407439,0.882353,0.672183,0.838703,0.408237,1,0.706131,0.892759,0.408452]},"BuRd":{"ColorSpace":"Lab","Name":"BuRd","RGBPoints":[-1,0.019608,0.188235,0.380392,-0.87451,0.088504,0.321107,0.564937,-0.74902,0.163399,0.444983,0.697501,-0.623529,0.247059,0.555709,0.754095,-0.498039,0.420684,0.676432,0.818685,-0.372549,0.606459,0.789773,0.880277,-0.247059,0.761476,0.868512,0.924567,-0.121569,0.878047,0.925721,0.951942,0.00392157,0.969089,0.966474,0.964937,0.129412,0.983852,0.897578,0.846828,0.254902,0.982468,0.800692,0.706113,0.380392,0.960323,0.66782,0.536332,0.505882,0.894579,0.503806,0.399769,0.631373,0.81707,0.33218,0.281046,0.756863,0.728489,0.155017,0.197386,0.882353,0.576932,0.055363,0.14925,1,0.403922,0,0.121569]},"Spectral_lowBlue":{"ColorSpace":"Lab","Name":"Spectral_lowBlue","RGBPoints":[-1,0.368627,0.309804,0.635294,-0.87451,0.260361,0.450058,0.70173,-0.74902,0.248058,0.591311,0.717186,-0.623529,0.376009,0.734025,0.658132,-0.498039,0.537947,0.814764,0.64506,-0.372549,0.702345,0.879585,0.636678,-0.247059,0.84752,0.938639,0.607151,-0.121569,0.940408,0.976163,0.656055,0.00392157,0.999923,0.997616,0.745021,0.129412,0.997463,0.921338,0.61707,0.254902,0.995002,0.824606,0.499885,0.380392,0.992541,0.701576,0.39654,0.505882,0.973472,0.547405,0.318108,0.631373,0.937793,0.398539,0.270127,0.756863,0.861515,0.282891,0.299654,0.882353,0.746482,0.144637,0.288812,1,0.619608,0.003922,0.258824]},"GnRP":{"ColorSpace":"Lab","Name":"GnRP","RGBPoints":[-1,0,0.266667,0.105882,-0.87451,0.066436,0.394617,0.174779,-0.74902,0.168858,0.524567,0.25767,-0.623529,0.323875,0.657439,0.361015,-0.498039,0.504883,0.772318,0.506344,-0.372549,0.678431,0.870127,0.654902,-0.247059,0.803922,0.921799,0.780392,-0.121569,0.897116,0.951942,0.882814,0.00392157,0.967397,0.965936,0.967474,0.129412,0.928028,0.879815,0.930565,0.254902,0.866052,0.780777,0.882891,0.380392,0.77501,0.665129,0.821376,0.505882,0.675663,0.537024,0.737024,0.631373,0.57847,0.396155,0.645982,0.756863,0.492349,0.223914,0.547559,0.882353,0.375548,0.096886,0.423299,1,0.25098,0,0.294118]},"GYPi":{"ColorSpace":"Lab","Name":"GYPi","RGBPoints":[-1,0.152941,0.392157,0.098039,-0.87451,0.246444,0.505344,0.117724,-0.74902,0.351942,0.614533,0.161399,-0.623529,0.474971,0.717878,0.240138,-0.498039,0.611995,0.811226,0.392849,-0.372549,0.746328,0.893118,0.565321,-0.247059,0.859516,0.94233,0.747405,-0.121569,0.928105,0.96386,0.875663,0.00392157,0.969089,0.966859,0.968012,0.129412,0.983852,0.910265,0.948328,0.254902,0.979239,0.833218,0.914648,0.380392,0.949712,0.729873,0.862976,0.505882,0.905652,0.58293,0.763552,0.631373,0.85521,0.410073,0.652211,0.756863,0.793695,0.183699,0.531642,0.882353,0.683737,0.063899,0.420761,1,0.556863,0.003922,0.321569]},"GnYlRd":{"ColorSpace":"Lab","Name":"GnYlRd","RGBPoints":[-1,0,0.407843,0.215686,-0.87451,0.063975,0.525952,0.277201,-0.74902,0.177932,0.633064,0.332718,-0.623529,0.364937,0.724106,0.379469,-0.498039,0.527951,0.797155,0.40223,-0.372549,0.678431,0.862822,0.433449,-0.247059,0.803922,0.916955,0.514648,-0.121569,0.909419,0.961861,0.625067,0.00392157,0.999923,0.997616,0.745021,0.129412,0.997463,0.921338,0.61707,0.254902,0.995002,0.824606,0.499885,0.380392,0.992541,0.701576,0.39654,0.505882,0.973472,0.547405,0.318108,0.631373,0.939023,0.389927,0.245521,0.756863,0.867666,0.239831,0.176624,0.882353,0.762399,0.110727,0.151326,1,0.647059,0,0.14902]},"GBBr":{"ColorSpace":"Lab","Name":"GBBr","RGBPoints":[-1,0,0.235294,0.188235,-0.87451,0.002461,0.338639,0.301423,-0.74902,0.055902,0.448981,0.417609,-0.623529,0.183852,0.56955,0.538178,-0.498039,0.357785,0.700115,0.660746,-0.372549,0.540177,0.819531,0.77624,-0.247059,0.714879,0.890888,0.864821,-0.121569,0.851134,0.934564,0.922645,0.00392157,0.960861,0.959785,0.95694,0.129412,0.963322,0.927797,0.83391,0.254902,0.939946,0.868897,0.68935,0.380392,0.883353,0.775394,0.517109,0.505882,0.808074,0.625836,0.324106,0.631373,0.717647,0.476355,0.15494,0.756863,0.592157,0.358247,0.06882,0.882353,0.458593,0.26436,0.031142,1,0.329412,0.188235,0.019608]},"PuOr":{"ColorSpace":"Lab","Name":"PuOr","RGBPoints":[-1,0.498039,0.231373,0.031373,-0.87451,0.62599,0.30273,0.026451,-0.74902,0.746943,0.387082,0.037524,-0.623529,0.85767,0.490427,0.071972,-0.498039,0.936409,0.617762,0.236371,-0.372549,0.992695,0.743099,0.43291,-0.247059,0.995156,0.841523,0.63714,-0.121569,0.985313,0.913802,0.813687,0.00392157,0.966244,0.966398,0.967705,0.129412,0.889965,0.89504,0.938178,0.254902,0.806151,0.804306,0.894656,0.380392,0.712649,0.688658,0.833141,0.505882,0.594233,0.554325,0.744637,0.631373,0.474894,0.404229,0.652364,0.756863,0.366628,0.217224,0.563783,0.882353,0.266436,0.089965,0.434833,1,0.176471,0,0.294118]},"PRGn":{"ColorSpace":"Lab","Name":"PRGn","RGBPoints":[-1,0.25098,0,0.294118,-0.87451,0.383852,0.103345,0.431911,-0.74902,0.497732,0.234679,0.55371,-0.623529,0.583852,0.40692,0.652134,-0.498039,0.681968,0.545175,0.742561,-0.372549,0.7807,0.672357,0.825221,-0.247059,0.871742,0.788005,0.886736,-0.121569,0.930488,0.885198,0.932872,0.00392157,0.966321,0.968089,0.965859,0.129412,0.892503,0.950865,0.877278,0.254902,0.796078,0.91857,0.772549,0.380392,0.670588,0.866897,0.647059,0.505882,0.493195,0.765398,0.496655,0.631373,0.314187,0.649135,0.354556,0.756863,0.15917,0.516263,0.251211,0.882353,0.062284,0.386621,0.170473,1,0,0.266667,0.105882]},"PiYG":{"ColorSpace":"Lab","Name":"PiYG","RGBPoints":[-1,0.556863,0.003922,0.321569,-0.87451,0.692195,0.067897,0.427374,-0.74902,0.797539,0.197847,0.539177,-0.623529,0.859054,0.424221,0.659746,-0.498039,0.908574,0.592618,0.770319,-0.372549,0.951557,0.736332,0.866205,-0.247059,0.981084,0.839677,0.917878,-0.121569,0.98293,0.913802,0.949558,0.00392157,0.96732,0.968474,0.965629,0.129412,0.92549,0.963552,0.869666,0.254902,0.852441,0.939254,0.736025,0.380392,0.739254,0.890042,0.553941,0.505882,0.60323,0.805536,0.382238,0.631373,0.467282,0.711419,0.235217,0.756863,0.344252,0.608074,0.156478,0.882353,0.2406,0.49827,0.116494,1,0.152941,0.392157,0.098039]},"OrPu":{"ColorSpace":"Lab","Name":"OrPu","RGBPoints":[-1,0.176471,0,0.294118,-0.87451,0.272434,0.095963,0.444214,-0.74902,0.373395,0.228912,0.56932,-0.623529,0.481661,0.415917,0.657901,-0.498039,0.601922,0.562937,0.750481,-0.372549,0.718493,0.695886,0.836986,-0.247059,0.811995,0.811534,0.898501,-0.121569,0.894733,0.8995,0.940023,0.00392157,0.969166,0.966859,0.963629,0.129412,0.98639,0.910265,0.803691,0.254902,0.995002,0.835371,0.624375,0.380392,0.992541,0.736947,0.420146,0.505882,0.931949,0.609458,0.224221,0.631373,0.85075,0.483968,0.069819,0.756863,0.740023,0.380623,0.035371,0.882353,0.617993,0.29827,0.026759,1,0.498039,0.231373,0.031373]},"BrBG":{"ColorSpace":"Lab","Name":"BrBG","RGBPoints":[-1,0.329412,0.188235,0.019608,-0.87451,0.467205,0.269435,0.031911,-0.74902,0.6,0.365629,0.074202,-0.623529,0.72549,0.483737,0.160323,-0.498039,0.812995,0.635832,0.336409,-0.372549,0.88689,0.781238,0.527874,-0.247059,0.943483,0.87474,0.700115,-0.121569,0.963168,0.929796,0.841599,0.00392157,0.957247,0.959938,0.959554,0.129412,0.84406,0.932872,0.920185,0.254902,0.70396,0.886428,0.859285,0.380392,0.529258,0.815071,0.770704,0.505882,0.346251,0.691811,0.653057,0.631373,0.175855,0.562015,0.530642,0.756863,0.047905,0.441446,0.410073,0.882353,0.002307,0.33218,0.294348,1,0,0.235294,0.188235]},"erdc_divHi_purpleGreen":{"ColorSpace":"Lab","Name":"erdc_divHi_purpleGreen","RGBPoints":[-1,0.297553,0,0.489074,-0.87451,0.40259,0.151146,0.567754,-0.74902,0.516038,0.284843,0.658231,-0.623529,0.629783,0.423646,0.750938,-0.498039,0.735198,0.563697,0.835956,-0.372549,0.82408,0.695541,0.903582,-0.247059,0.889091,0.807454,0.944862,-0.121569,0.92334,0.886917,0.951839,0.00392157,0.921045,0.921084,0.921003,0.129412,0.877324,0.907455,0.845381,0.254902,0.797649,0.849713,0.734695,0.380392,0.691646,0.75964,0.600532,0.505882,0.568981,0.649159,0.453807,0.631373,0.438945,0.529756,0.304259,0.756863,0.30973,0.412001,0.158303,0.882353,0.187078,0.305111,0.00251458,1,0.101655,0.220836,0]},"erdc_divHi_purpleGreen_dim":{"ColorSpace":"Lab","Name":"erdc_divHi_purpleGreen_dim","RGBPoints":[-1,0.404088,0.131038,0.592767,-0.87451,0.486469,0.230957,0.651243,-0.74902,0.575165,0.339335,0.717723,-0.623529,0.662741,0.454332,0.784263,-0.498039,0.742071,0.570213,0.842918,-0.372549,0.806935,0.678992,0.886227,-0.247059,0.852219,0.771315,0.90763,-0.121569,0.873345,0.837327,0.901572,0.00392157,0.866783,0.86682,0.866745,0.129412,0.82839,0.858225,0.796812,0.254902,0.762578,0.814287,0.700202,0.380392,0.676429,0.744229,0.585735,0.505882,0.577033,0.65732,0.461526,0.631373,0.47128,0.562476,0.33476,0.756863,0.365461,0.467957,0.21076,0.882353,0.264758,0.381138,0.0878313,1,0.182591,0.312249,0]},"erdc_divLow_icePeach":{"ColorSpace":"Lab","Name":"erdc_divLow_icePeach","RGBPoints":[-1,0.480048,0.817441,0.998056,-0.87451,0.425898,0.726921,0.883187,-0.74902,0.366682,0.629445,0.761936,-0.623529,0.308756,0.531002,0.640217,-0.498039,0.258021,0.43705,0.523433,-0.372549,0.219244,0.352381,0.416348,-0.247059,0.195127,0.281032,0.322979,-0.121569,0.186286,0.22627,0.246525,0.00392157,0.192352,0.19236,0.192364,0.129412,0.255927,0.214469,0.191756,0.254902,0.340459,0.254426,0.206666,0.380392,0.444655,0.309315,0.234029,0.505882,0.565353,0.376004,0.270969,0.631373,0.697917,0.450748,0.314293,0.756863,0.836657,0.529064,0.360227,0.882353,0.972695,0.614884,0.413123,1,1,0.705904,0.472699]},"erdc_divLow_purpleGreen":{"ColorSpace":"Lab","Name":"erdc_divLow_purpleGreen","RGBPoints":[-1,0.956034,0.666487,0.952663,-0.87451,0.874457,0.572698,0.936352,-0.74902,0.753465,0.488253,0.909063,-0.623529,0.63309,0.413507,0.763833,-0.498039,0.514491,0.345878,0.620015,-0.372549,0.405008,0.288141,0.484376,-0.247059,0.311388,0.241986,0.363556,-0.121569,0.238722,0.209044,0.263449,0.00392157,0.192352,0.192366,0.192362,0.129412,0.200379,0.233201,0.168618,0.254902,0.230151,0.291737,0.165227,0.380392,0.279481,0.366076,0.178607,0.505882,0.344927,0.453267,0.205703,0.631373,0.421554,0.549449,0.242643,0.756863,0.503334,0.649999,0.284377,0.882353,0.583497,0.749672,0.324969,1,0.650705,0.837228,0.356264]},"Haze_green":{"ColorSpace":"Lab","Name":"Haze_green","RGBPoints":[-1,1,0.835294,0.886275,-0.87451,0.937255,0.756863,0.870443,-0.74902,0.875817,0.666376,0.857807,-0.623529,0.778359,0.583007,0.808134,-0.498039,0.676253,0.494118,0.745098,-0.372549,0.561365,0.390123,0.682353,-0.247059,0.438344,0.262745,0.621496,-0.121569,0.321133,0.141031,0.558751,0.00392157,0.203922,0.0217865,0.495861,0.129412,0.265505,0.129412,0.433261,0.254902,0.311692,0.255338,0.37008,0.380392,0.356282,0.377342,0.310821,0.505882,0.39971,0.488889,0.258243,0.631373,0.442556,0.604357,0.205519,0.756863,0.48671,0.71968,0.152941,0.882353,0.529847,0.830356,0.100944,1,0.572549,0.933333,0.054902]},"Haze_lime":{"ColorSpace":"Lab","Name":"Haze_lime","RGBPoints":[-1,0.704034,0.784196,1,-0.87451,0.633111,0.691418,0.956078,-0.74902,0.564021,0.600606,0.912157,-0.623529,0.496827,0.51189,0.868235,-0.498039,0.43157,0.425416,0.824314,-0.372549,0.368248,0.341347,0.780392,-0.247059,0.306767,0.259855,0.736471,-0.121569,0.246862,0.181069,0.692549,0.00392157,0.191619,0.109542,0.648627,0.129412,0.257404,0.194031,0.604706,0.254902,0.321794,0.278775,0.560784,0.380392,0.387909,0.364617,0.516863,0.505882,0.456569,0.451881,0.472941,0.631373,0.527424,0.540773,0.42902,0.756863,0.599759,0.631427,0.385098,0.882353,0.673065,0.723898,0.341176,1,0.742751,0.812252,0.3]},"Haze":{"ColorSpace":"RGB","Name":"Haze","RGBPoints":[-1,1,0.835294,0.996078,-0.00392157,0.023529,0.141176,0.498039,0.00392157,0.015686,0.137255,0.494118,1,0.984314,0.764706,0]},"Haze_cyan":{"ColorSpace":"Lab","Name":"Haze_cyan","RGBPoints":[-1,0.956863,1,0.835294,-0.87451,0.933188,0.921714,0.760784,-0.74902,0.870588,0.803486,0.671605,-0.623529,0.807843,0.684096,0.583297,-0.498039,0.745098,0.569208,0.494118,-0.372549,0.682353,0.437763,0.390123,-0.247059,0.621496,0.288163,0.262745,-0.121569,0.558751,0.144517,0.141031,0.00392157,0.495861,0.0217865,0.0413943,0.129412,0.433261,0.137255,0.129412,0.254902,0.37008,0.263181,0.255338,0.380392,0.306318,0.381845,0.372694,0.505882,0.243137,0.503994,0.494263,0.631373,0.180392,0.629484,0.619753,0.756863,0.117647,0.754975,0.747131,0.882353,0.054902,0.876398,0.866812,1,0,0.988235,0.976471]},"nic_Edge":{"ColorSpace":"Lab","Name":"nic_Edge","RGBPoints":[-1,0.191208,0.191208,0.191208,-0.87451,0.239484,0.00545035,0.614821,-0.74902,0.220593,0.0617459,0.863547,-0.623529,0.17509,0.278988,0.97794,-0.498039,0.143526,0.576069,0.998553,-0.372549,0.166456,0.871883,0.96594,-0.247059,0.376202,0.993555,0.981833,-0.121569,0.681996,0.991297,0.999239,0.00392157,0.954172,0.952734,0.94374,0.129412,0.999735,0.99301,0.662896,0.254902,0.979399,0.991466,0.357973,0.380392,0.968771,0.854967,0.162659,0.505882,0.999245,0.556697,0.144323,0.631373,0.973959,0.26223,0.177946,0.756863,0.852358,0.0526707,0.222974,0.882353,0.593889,0.00912724,0.238855,1,0.191208,0.191208,0.191208]},"erdc_iceFire_H":{"ColorSpace":"Lab","Name":"erdc_iceFire_H","RGBPoints":[-1,4.05432e-07,0,5.90122e-06,-0.87451,0,0.120401,0.302675,-0.74902,0,0.216583,0.524574,-0.623529,0.0552475,0.345025,0.6595,-0.498039,0.128047,0.492588,0.720288,-0.372549,0.188955,0.641309,0.792092,-0.247059,0.327673,0.784935,0.873434,-0.121569,0.60824,0.892164,0.935547,0.00392157,0.881371,0.912178,0.818099,0.129412,0.951407,0.835621,0.449279,0.254902,0.904481,0.690489,0,0.380392,0.85407,0.510864,0,0.505882,0.777093,0.33018,0.00088199,0.631373,0.672862,0.139087,0.00269398,0.756863,0.508815,0,0,0.882353,0.299417,0.000366289,0.000547829,1,0.0157519,0.00332021,4.55569e-08]},"erdc_iceFire_L":{"ColorSpace":"Lab","Name":"erdc_iceFire_L","RGBPoints":[-1,0.870485,0.913768,0.832905,-0.87451,0.586919,0.887865,0.934003,-0.74902,0.31583,0.776442,0.867858,-0.623529,0.18302,0.632034,0.787722,-0.498039,0.117909,0.484134,0.713825,-0.372549,0.0507239,0.335979,0.654741,-0.247059,0,0.209874,0.511832,-0.121569,0,0.114689,0.28935,0.00392157,0.0157519,0.00332021,4.55569e-08,0.129412,0.312914,0,0,0.254902,0.520865,0,0,0.380392,0.680105,0.15255,0.0025996,0.505882,0.785109,0.339479,0.000797922,0.631373,0.857354,0.522494,0,0.756863,0.910974,0.699774,0,0.882353,0.951921,0.842817,0.478545,1,0.881371,0.912178,0.818099]},"hsv":{"ColorSpace":"RGB","Name":"hsv","RGBPoints":[-1,1,0,0,-0.666666,1,0,1,-0.333333,0,0,1,0,0,1,1,0.33333,0,1,0,0.66666,1,1,0,1,1,0,0]},"hue_L60":{"ColorSpace":"Lab","Name":"hue_L60","RGBPoints":[-1,0.964784,0.400592,0.349549,-0.87451,0.964915,0.372498,0.53785,-0.74902,0.892353,0.401039,0.759569,-0.623529,0.79263,0.446956,0.903017,-0.498039,0.682208,0.49954,0.966673,-0.372549,0.56392,0.553082,0.968836,-0.247059,0.442031,0.606396,0.901601,-0.121569,0.305499,0.65701,0.765784,0.00392157,0.197251,0.687914,0.620914,0.129412,0.193882,0.701887,0.472654,0.254902,0.249866,0.706123,0.320005,0.380392,0.35132,0.697417,0.202919,0.505882,0.498097,0.669467,0.125232,0.631373,0.637477,0.626239,0.107431,0.756863,0.762115,0.56872,0.155812,0.882353,0.889434,0.481116,0.240445,1,0.964784,0.400592,0.349549]},"GyRd":{"ColorSpace":"Lab","Name":"GyRd","RGBPoints":[-1,0.101961,0.101961,0.101961,-0.87451,0.227451,0.227451,0.227451,-0.74902,0.359939,0.359939,0.359939,-0.623529,0.502653,0.502653,0.502653,-0.498039,0.631373,0.631373,0.631373,-0.372549,0.749865,0.749865,0.749865,-0.247059,0.843368,0.843368,0.843368,-0.121569,0.926105,0.926105,0.926105,0.00392157,0.999846,0.997232,0.995694,0.129412,0.994925,0.908651,0.857901,0.254902,0.982468,0.800692,0.706113,0.380392,0.960323,0.66782,0.536332,0.505882,0.894579,0.503806,0.399769,0.631373,0.81707,0.33218,0.281046,0.756863,0.728489,0.155017,0.197386,0.882353,0.576932,0.055363,0.14925,1,0.403922,0,0.121569]},"Muted Blue-Green":{"ColorSpace":"Lab","Creator":"Francesca Samsel","Name":"Muted Blue-Green","NanColor":[0.25,0,0],"RGBPoints":[0,0.109804,0.27451,0.301961,0.02,0.129412,0.309804,0.341176,0.05,0.14902,0.341176,0.380392,0.1,0.188235,0.403922,0.458824,0.15,0.227451,0.447059,0.521569,0.2,0.290196,0.494118,0.588235,0.25,0.368627,0.552941,0.670588,0.3,0.458824,0.619608,0.74902,0.35,0.588235,0.713725,0.85098,0.4,0.72549,0.815686,0.941176,0.45,0.831373,0.882353,0.980392,0.475,0.909804,0.933333,1,0.5,0.980392,0.984314,1,0.5,0.996078,1,0.94902,0.5,1,1,0.980392,0.5,0.980392,0.984314,1,0.525,0.972549,0.988235,0.890196,0.55,0.917647,0.960784,0.835294,0.6,0.835294,0.921569,0.772549,0.65,0.768627,0.901961,0.737255,0.7,0.670588,0.831373,0.654902,0.75,0.576471,0.760784,0.584314,0.8,0.498039,0.678431,0.521569,0.85,0.392157,0.560784,0.427451,0.9,0.294118,0.45098,0.333333,0.95,0.211765,0.34902,0.254902,1,0.152941,0.278431,0.196078]},"Green-Blue Asymmetric Divergent (62Blbc)":{"ColorSpace":"Lab","Creator":"Francesca Samsel","Name":"Green-Blue Asymmetric Divergent (62Blbc)","NanColor":[0.25,0,0],"RGBPoints":[0,0.121569,0.2,0.145098,0.05,0.196078,0.301961,0.223529,0.1,0.258824,0.4,0.278431,0.2,0.341176,0.54902,0.341176,0.25,0.419608,0.619608,0.376471,0.3,0.545098,0.701961,0.392157,0.35,0.643137,0.780392,0.403922,0.4,0.729412,0.819608,0.45098,0.45,0.811765,0.870588,0.521569,0.5,0.898039,0.909804,0.564706,0.55,0.941176,0.92549,0.686275,0.6,0.960784,0.94902,0.776471,0.64,1,1,1,0.65,0.890196,0.988235,0.972549,0.7,0.721569,0.894118,0.901961,0.75,0.631373,0.823529,0.839216,0.8,0.517647,0.662745,0.701961,0.85,0.384314,0.494118,0.54902,0.9,0.298039,0.360784,0.45098,0.95,0.223529,0.25098,0.34902,0.99,0.156863,0.172549,0.25098,1,0.137255,0.137255,0.188235]},"Asymmetrical Earth Tones (6_21b)":{"ColorSpace":"Lab","Creator":"Francesca Samsel","Name":"Asymmetrical Earth Tones (6_21b)","NanColor":[0.25,0,0],"RGBPoints":[0,0.141176,0.14902,0.2,0.05,0.215686,0.258824,0.321569,0.1,0.243137,0.368627,0.380392,0.15,0.27451,0.439216,0.4,0.2,0.32549,0.501961,0.384314,0.25,0.403922,0.6,0.419608,0.3,0.486275,0.701961,0.454902,0.35,0.556863,0.74902,0.494118,0.4,0.670588,0.8,0.545098,0.5,0.854902,0.901961,0.631373,0.55,0.92549,0.941176,0.694118,0.6,0.960784,0.94902,0.776471,0.65,0.988235,0.968627,0.909804,0.7,0.839216,0.815686,0.772549,0.75,0.701961,0.662745,0.615686,0.8,0.6,0.529412,0.478431,0.85,0.501961,0.403922,0.360784,0.9,0.439216,0.313725,0.290196,1,0.301961,0.164706,0.176471]},"Yellow 15":{"ColorSpace":"Lab","Creator":"Francesca Samsel","Name":"Yellow 15","NanColor":[0.25,0,0],"RGBPoints":[0,1,1,0.988235,0.002,1,1,0.988235,0.05,0.984314,0.988235,0.843137,0.1,0.988235,0.988235,0.741176,0.15,0.980392,0.968627,0.654902,0.2,0.980392,0.945098,0.576471,0.25,0.968627,0.905882,0.486275,0.3,0.968627,0.862745,0.388235,0.35,0.960784,0.803922,0.286275,0.4,0.94902,0.741176,0.219608,0.45,0.941176,0.678431,0.14902,0.5,0.929412,0.607843,0.094118,0.55,0.921569,0.545098,0.054902,0.6,0.909804,0.486275,0.035294,0.65,0.890196,0.411765,0.019608,0.7,0.8,0.305882,0,0.75,0.760784,0.239216,0,0.8,0.678431,0.180392,0.011765,0.85,0.6,0.121569,0.023529,0.9,0.501961,0.054902,0.031373,0.95,0.4,0.039216,0.058824,1,0.301961,0.047059,0.090196]},"Magma":{"ColorSpace":"Lab","Name":"Magma","NanColor":[0,1,0],"Source":"https://github.com/BIDS/colormap/blob/master/colormaps.py","License":"CC0","Creator":"Nathaniel J. Smith & Stefan van der Walt","RGBPoints":[0.0,0.001462,0.000466,0.013866,0.003922,0.002258,0.001295,0.018331,0.007843,0.003279,0.002305,0.023708,0.011765,0.004512,0.00349,0.029965,0.015686,0.00595,0.004843,0.03713,0.019608,0.007588,0.006356,0.044973,0.023529,0.009426,0.008022,0.052844,0.027451,0.011465,0.009828,0.06075,0.031373,0.013708,0.011771,0.068667,0.035294,0.016156,0.01384,0.076603,0.039216,0.018815,0.016026,0.084584,0.043137,0.021692,0.01832,0.09261,0.047059,0.024792,0.020715,0.100676,0.05098,0.028123,0.023201,0.108787,0.054902,0.031696,0.025765,0.116965,0.058824,0.03552,0.028397,0.125209,0.062745,0.039608,0.03109,0.133515,0.066667,0.04383,0.03383,0.141886,0.070588,0.048062,0.036607,0.150327,0.07451,0.05232,0.039407,0.158841,0.078431,0.056615,0.04216,0.167446,0.082353,0.060949,0.044794,0.176129,0.086275,0.06533,0.047318,0.184892,0.090196,0.069764,0.049726,0.193735,0.094118,0.074257,0.052017,0.20266,0.098039,0.078815,0.054184,0.211667,0.101961,0.083446,0.056225,0.220755,0.105882,0.088155,0.058133,0.229922,0.109804,0.092949,0.059904,0.239164,0.113725,0.097833,0.061531,0.248477,0.117647,0.102815,0.06301,0.257854,0.121569,0.107899,0.064335,0.267289,0.12549,0.113094,0.065492,0.276784,0.129412,0.118405,0.066479,0.286321,0.133333,0.123833,0.067295,0.295879,0.137255,0.12938,0.067935,0.305443,0.141176,0.135053,0.068391,0.315,0.145098,0.140858,0.068654,0.324538,0.14902,0.146785,0.068738,0.334011,0.152941,0.152839,0.068637,0.343404,0.156863,0.159018,0.068354,0.352688,0.160784,0.165308,0.067911,0.361816,0.164706,0.171713,0.067305,0.370771,0.168627,0.178212,0.066576,0.379497,0.172549,0.184801,0.065732,0.387973,0.176471,0.19146,0.064818,0.396152,0.180392,0.198177,0.063862,0.404009,0.184314,0.204935,0.062907,0.411514,0.188235,0.211718,0.061992,0.418647,0.192157,0.218512,0.061158,0.425392,0.196078,0.225302,0.060445,0.431742,0.2,0.232077,0.059889,0.437695,0.203922,0.238826,0.059517,0.443256,0.207843,0.245543,0.059352,0.448436,0.211765,0.25222,0.059415,0.453248,0.215686,0.258857,0.059706,0.45771,0.219608,0.265447,0.060237,0.46184,0.223529,0.271994,0.060994,0.46566,0.227451,0.278493,0.061978,0.46919,0.231373,0.284951,0.063168,0.472451,0.235294,0.291366,0.064553,0.475462,0.239216,0.29774,0.066117,0.478243,0.243137,0.304081,0.067835,0.480812,0.247059,0.310382,0.069702,0.483186,0.25098,0.316654,0.07169,0.48538,0.254902,0.322899,0.073782,0.487408,0.258824,0.329114,0.075972,0.489287,0.262745,0.335308,0.078236,0.491024,0.266667,0.341482,0.080564,0.492631,0.270588,0.347636,0.082946,0.494121,0.27451,0.353773,0.085373,0.495501,0.278431,0.359898,0.087831,0.496778,0.282353,0.366012,0.090314,0.49796,0.286275,0.372116,0.092816,0.499053,0.290196,0.378211,0.095332,0.500067,0.294118,0.384299,0.097855,0.501002,0.298039,0.390384,0.100379,0.501864,0.301961,0.396467,0.102902,0.502658,0.305882,0.402548,0.10542,0.503386,0.309804,0.408629,0.10793,0.504052,0.313725,0.414709,0.110431,0.504662,0.317647,0.420791,0.11292,0.505215,0.321569,0.426877,0.115395,0.505714,0.32549,0.432967,0.117855,0.50616,0.329412,0.439062,0.120298,0.506555,0.333333,0.445163,0.122724,0.506901,0.337255,0.451271,0.125132,0.507198,0.341176,0.457386,0.127522,0.507448,0.345098,0.463508,0.129893,0.507652,0.34902,0.46964,0.132245,0.507809,0.352941,0.47578,0.134577,0.507921,0.356863,0.481929,0.136891,0.507989,0.360784,0.488088,0.139186,0.508011,0.364706,0.494258,0.141462,0.507988,0.368627,0.500438,0.143719,0.50792,0.372549,0.506629,0.145958,0.507806,0.376471,0.512831,0.148179,0.507648,0.380392,0.519045,0.150383,0.507443,0.384314,0.52527,0.152569,0.507192,0.388235,0.531507,0.154739,0.506895,0.392157,0.537755,0.156894,0.506551,0.396078,0.544015,0.159033,0.506159,0.4,0.550287,0.161158,0.505719,0.403922,0.556571,0.163269,0.50523,0.407843,0.562866,0.165368,0.504692,0.411765,0.569172,0.167454,0.504105,0.415686,0.57549,0.16953,0.503466,0.419608,0.581819,0.171596,0.502777,0.423529,0.588158,0.173652,0.502035,0.427451,0.594508,0.175701,0.501241,0.431373,0.600868,0.177743,0.500394,0.435294,0.607238,0.179779,0.499492,0.439216,0.613617,0.181811,0.498536,0.443137,0.620005,0.18384,0.497524,0.447059,0.626401,0.185867,0.496456,0.45098,0.632805,0.187893,0.495332,0.454902,0.639216,0.189921,0.49415,0.458824,0.645633,0.191952,0.49291,0.462745,0.652056,0.193986,0.491611,0.466667,0.658483,0.196027,0.490253,0.470588,0.664915,0.198075,0.488836,0.47451,0.671349,0.200133,0.487358,0.478431,0.677786,0.202203,0.485819,0.482353,0.684224,0.204286,0.484219,0.486275,0.690661,0.206384,0.482558,0.490196,0.697098,0.208501,0.480835,0.494118,0.703532,0.210638,0.479049,0.498039,0.709962,0.212797,0.477201,0.501961,0.716387,0.214982,0.47529,0.505882,0.722805,0.217194,0.473316,0.509804,0.729216,0.219437,0.471279,0.513725,0.735616,0.221713,0.46918,0.517647,0.742004,0.224025,0.467018,0.521569,0.748378,0.226377,0.464794,0.52549,0.754737,0.228772,0.462509,0.529412,0.761077,0.231214,0.460162,0.533333,0.767398,0.233705,0.457755,0.537255,0.773695,0.236249,0.455289,0.541176,0.779968,0.238851,0.452765,0.545098,0.786212,0.241514,0.450184,0.54902,0.792427,0.244242,0.447543,0.552941,0.798608,0.24704,0.444848,0.556863,0.804752,0.249911,0.442102,0.560784,0.810855,0.252861,0.439305,0.564706,0.816914,0.255895,0.436461,0.568627,0.822926,0.259016,0.433573,0.572549,0.828886,0.262229,0.430644,0.576471,0.834791,0.26554,0.427671,0.580392,0.840636,0.268953,0.424666,0.584314,0.846416,0.272473,0.421631,0.588235,0.852126,0.276106,0.418573,0.592157,0.857763,0.279857,0.415496,0.596078,0.86332,0.283729,0.412403,0.6,0.868793,0.287728,0.409303,0.603922,0.874176,0.291859,0.406205,0.607843,0.879464,0.296125,0.403118,0.611765,0.884651,0.30053,0.400047,0.615686,0.889731,0.305079,0.397002,0.619608,0.8947,0.309773,0.393995,0.623529,0.899552,0.314616,0.391037,0.627451,0.904281,0.31961,0.388137,0.631373,0.908884,0.324755,0.385308,0.635294,0.913354,0.330052,0.382563,0.639216,0.917689,0.3355,0.379915,0.643137,0.921884,0.341098,0.377376,0.647059,0.925937,0.346844,0.374959,0.65098,0.929845,0.352734,0.372677,0.654902,0.933606,0.358764,0.370541,0.658824,0.937221,0.364929,0.368567,0.662745,0.940687,0.371224,0.366762,0.666667,0.944006,0.377643,0.365136,0.670588,0.94718,0.384178,0.363701,0.67451,0.95021,0.39082,0.362468,0.678431,0.953099,0.397563,0.361438,0.682353,0.955849,0.4044,0.360619,0.686275,0.958464,0.411324,0.360014,0.690196,0.960949,0.418323,0.35963,0.694118,0.96331,0.42539,0.359469,0.698039,0.965549,0.432519,0.359529,0.701961,0.967671,0.439703,0.35981,0.705882,0.96968,0.446936,0.360311,0.709804,0.971582,0.45421,0.36103,0.713725,0.973381,0.46152,0.361965,0.717647,0.975082,0.468861,0.363111,0.721569,0.97669,0.476226,0.364466,0.72549,0.97821,0.483612,0.366025,0.729412,0.979645,0.491014,0.367783,0.733333,0.981,0.498428,0.369734,0.737255,0.982279,0.505851,0.371874,0.741176,0.983485,0.51328,0.374198,0.745098,0.984622,0.520713,0.376698,0.74902,0.985693,0.528148,0.379371,0.752941,0.9867,0.535582,0.38221,0.756863,0.987646,0.543015,0.38521,0.760784,0.988533,0.550446,0.388365,0.764706,0.989363,0.557873,0.391671,0.768627,0.990138,0.565296,0.395122,0.772549,0.990871,0.572706,0.398714,0.776471,0.991558,0.580107,0.402441,0.780392,0.992196,0.587502,0.406299,0.784314,0.992785,0.594891,0.410283,0.788235,0.993326,0.602275,0.41439,0.792157,0.993834,0.609644,0.418613,0.796078,0.994309,0.616999,0.42295,0.8,0.994738,0.62435,0.427397,0.803922,0.995122,0.631696,0.431951,0.807843,0.99548,0.639027,0.436607,0.811765,0.99581,0.646344,0.441361,0.815686,0.996096,0.653659,0.446213,0.819608,0.996341,0.660969,0.45116,0.823529,0.99658,0.668256,0.456192,0.827451,0.996775,0.675541,0.461314,0.831373,0.996925,0.682828,0.466526,0.835294,0.997077,0.690088,0.471811,0.839216,0.997186,0.697349,0.477182,0.843137,0.997254,0.704611,0.482635,0.847059,0.997325,0.711848,0.488154,0.85098,0.997351,0.719089,0.493755,0.854902,0.997351,0.726324,0.499428,0.858824,0.997341,0.733545,0.505167,0.862745,0.997285,0.740772,0.510983,0.866667,0.997228,0.747981,0.516859,0.870588,0.997138,0.75519,0.522806,0.87451,0.997019,0.762398,0.528821,0.878431,0.996898,0.769591,0.534892,0.882353,0.996727,0.776795,0.541039,0.886275,0.996571,0.783977,0.547233,0.890196,0.996369,0.791167,0.553499,0.894118,0.996162,0.798348,0.55982,0.898039,0.995932,0.805527,0.566202,0.901961,0.99568,0.812706,0.572645,0.905882,0.995424,0.819875,0.57914,0.909804,0.995131,0.827052,0.585701,0.913725,0.994851,0.834213,0.592307,0.917647,0.994524,0.841387,0.598983,0.921569,0.994222,0.84854,0.605696,0.92549,0.993866,0.855711,0.612482,0.929412,0.993545,0.862859,0.619299,0.933333,0.99317,0.870024,0.626189,0.937255,0.992831,0.877168,0.633109,0.941176,0.99244,0.88433,0.640099,0.945098,0.992089,0.89147,0.647116,0.94902,0.991688,0.898627,0.654202,0.952941,0.991332,0.905763,0.661309,0.956863,0.99093,0.912915,0.668481,0.960784,0.99057,0.920049,0.675675,0.964706,0.990175,0.927196,0.682926,0.968627,0.989815,0.934329,0.690198,0.972549,0.989434,0.94147,0.697519,0.976471,0.989077,0.948604,0.704863,0.980392,0.988717,0.955742,0.712242,0.984314,0.988367,0.962878,0.719649,0.988235,0.988033,0.970012,0.727077,0.992157,0.987691,0.977154,0.734536,0.996078,0.987387,0.984288,0.742002,1.0,0.987053,0.991438,0.749504]},"Plasma":{"ColorSpace":"Lab","Name":"Plasma","NanColor":[0,1,0],"Source":"https://github.com/BIDS/colormap/blob/master/colormaps.py","License":"CC0","Creator":"Nathaniel J. Smith & Stefan van der Walt","RGBPoints":[0.0,0.050383,0.029803,0.527975,0.003922,0.063536,0.028426,0.533124,0.007843,0.075353,0.027206,0.538007,0.011765,0.086222,0.026125,0.542658,0.015686,0.096379,0.025165,0.547103,0.019608,0.10598,0.024309,0.551368,0.023529,0.115124,0.023556,0.555468,0.027451,0.123903,0.022878,0.559423,0.031373,0.132381,0.022258,0.56325,0.035294,0.140603,0.021687,0.566959,0.039216,0.148607,0.021154,0.570562,0.043137,0.156421,0.020651,0.574065,0.047059,0.16407,0.020171,0.577478,0.05098,0.171574,0.019706,0.580806,0.054902,0.17895,0.019252,0.584054,0.058824,0.186213,0.018803,0.587228,0.062745,0.193374,0.018354,0.59033,0.066667,0.200445,0.017902,0.593364,0.070588,0.207435,0.017442,0.596333,0.07451,0.21435,0.016973,0.599239,0.078431,0.221197,0.016497,0.602083,0.082353,0.227983,0.016007,0.604867,0.086275,0.234715,0.015502,0.607592,0.090196,0.241396,0.014979,0.610259,0.094118,0.248032,0.014439,0.612868,0.098039,0.254627,0.013882,0.615419,0.101961,0.261183,0.013308,0.617911,0.105882,0.267703,0.012716,0.620346,0.109804,0.274191,0.012109,0.622722,0.113725,0.280648,0.011488,0.625038,0.117647,0.287076,0.010855,0.627295,0.121569,0.293478,0.010213,0.62949,0.12549,0.299855,0.009561,0.631624,0.129412,0.30621,0.008902,0.633694,0.133333,0.312543,0.008239,0.6357,0.137255,0.318856,0.007576,0.63764,0.141176,0.32515,0.006915,0.639512,0.145098,0.331426,0.006261,0.641316,0.14902,0.337683,0.005618,0.643049,0.152941,0.343925,0.004991,0.64471,0.156863,0.35015,0.004382,0.646298,0.160784,0.356359,0.003798,0.64781,0.164706,0.362553,0.003243,0.649245,0.168627,0.368733,0.002724,0.650601,0.172549,0.374897,0.002245,0.651876,0.176471,0.381047,0.001814,0.653068,0.180392,0.387183,0.001434,0.654177,0.184314,0.393304,0.001114,0.655199,0.188235,0.399411,0.000859,0.656133,0.192157,0.405503,0.000678,0.656977,0.196078,0.41158,0.000577,0.65773,0.2,0.417642,0.000564,0.65839,0.203922,0.423689,0.000646,0.658956,0.207843,0.429719,0.000831,0.659425,0.211765,0.435734,0.001127,0.659797,0.215686,0.441732,0.00154,0.660069,0.219608,0.447714,0.00208,0.66024,0.223529,0.453677,0.002755,0.66031,0.227451,0.459623,0.003574,0.660277,0.231373,0.46555,0.004545,0.660139,0.235294,0.471457,0.005678,0.659897,0.239216,0.477344,0.00698,0.659549,0.243137,0.48321,0.00846,0.659095,0.247059,0.489055,0.010127,0.658534,0.25098,0.494877,0.01199,0.657865,0.254902,0.500678,0.014055,0.657088,0.258824,0.506454,0.016333,0.656202,0.262745,0.512206,0.018833,0.655209,0.266667,0.517933,0.021563,0.654109,0.270588,0.523633,0.024532,0.652901,0.27451,0.529306,0.027747,0.651586,0.278431,0.534952,0.031217,0.650165,0.282353,0.54057,0.03495,0.64864,0.286275,0.546157,0.038954,0.64701,0.290196,0.551715,0.043136,0.645277,0.294118,0.557243,0.047331,0.643443,0.298039,0.562738,0.051545,0.641509,0.301961,0.568201,0.055778,0.639477,0.305882,0.573632,0.060028,0.637349,0.309804,0.579029,0.064296,0.635126,0.313725,0.584391,0.068579,0.632812,0.317647,0.589719,0.072878,0.630408,0.321569,0.595011,0.07719,0.627917,0.32549,0.600266,0.081516,0.625342,0.329412,0.605485,0.085854,0.622686,0.333333,0.610667,0.090204,0.619951,0.337255,0.615812,0.094564,0.61714,0.341176,0.620919,0.098934,0.614257,0.345098,0.625987,0.103312,0.611305,0.34902,0.631017,0.107699,0.608287,0.352941,0.636008,0.112092,0.605205,0.356863,0.640959,0.116492,0.602065,0.360784,0.645872,0.120898,0.598867,0.364706,0.650746,0.125309,0.595617,0.368627,0.65558,0.129725,0.592317,0.372549,0.660374,0.134144,0.588971,0.376471,0.665129,0.138566,0.585582,0.380392,0.669845,0.142992,0.582154,0.384314,0.674522,0.147419,0.578688,0.388235,0.67916,0.151848,0.575189,0.392157,0.683758,0.156278,0.57166,0.396078,0.688318,0.160709,0.568103,0.4,0.69284,0.165141,0.564522,0.403922,0.697324,0.169573,0.560919,0.407843,0.701769,0.174005,0.557296,0.411765,0.706178,0.178437,0.553657,0.415686,0.710549,0.182868,0.550004,0.419608,0.714883,0.187299,0.546338,0.423529,0.719181,0.191729,0.542663,0.427451,0.723444,0.196158,0.538981,0.431373,0.72767,0.200586,0.535293,0.435294,0.731862,0.205013,0.531601,0.439216,0.736019,0.209439,0.527908,0.443137,0.740143,0.213864,0.524216,0.447059,0.744232,0.218288,0.520524,0.45098,0.748289,0.222711,0.516834,0.454902,0.752312,0.227133,0.513149,0.458824,0.756304,0.231555,0.509468,0.462745,0.760264,0.235976,0.505794,0.466667,0.764193,0.240396,0.502126,0.470588,0.76809,0.244817,0.498465,0.47451,0.771958,0.249237,0.494813,0.478431,0.775796,0.253658,0.491171,0.482353,0.779604,0.258078,0.487539,0.486275,0.783383,0.2625,0.483918,0.490196,0.787133,0.266922,0.480307,0.494118,0.790855,0.271345,0.476706,0.498039,0.794549,0.27577,0.473117,0.501961,0.798216,0.280197,0.469538,0.505882,0.801855,0.284626,0.465971,0.509804,0.805467,0.289057,0.462415,0.513725,0.809052,0.293491,0.45887,0.517647,0.812612,0.297928,0.455338,0.521569,0.816144,0.302368,0.451816,0.52549,0.819651,0.306812,0.448306,0.529412,0.823132,0.311261,0.444806,0.533333,0.826588,0.315714,0.441316,0.537255,0.830018,0.320172,0.437836,0.541176,0.833422,0.324635,0.434366,0.545098,0.836801,0.329105,0.430905,0.54902,0.840155,0.33358,0.427455,0.552941,0.843484,0.338062,0.424013,0.556863,0.846788,0.342551,0.420579,0.560784,0.850066,0.347048,0.417153,0.564706,0.853319,0.351553,0.413734,0.568627,0.856547,0.356066,0.410322,0.572549,0.85975,0.360588,0.406917,0.576471,0.862927,0.365119,0.403519,0.580392,0.866078,0.36966,0.400126,0.584314,0.869203,0.374212,0.396738,0.588235,0.872303,0.378774,0.393355,0.592157,0.875376,0.383347,0.389976,0.596078,0.878423,0.387932,0.3866,0.6,0.881443,0.392529,0.383229,0.603922,0.884436,0.397139,0.37986,0.607843,0.887402,0.401762,0.376494,0.611765,0.89034,0.406398,0.37313,0.615686,0.89325,0.411048,0.369768,0.619608,0.896131,0.415712,0.366407,0.623529,0.898984,0.420392,0.363047,0.627451,0.901807,0.425087,0.359688,0.631373,0.904601,0.429797,0.356329,0.635294,0.907365,0.434524,0.35297,0.639216,0.910098,0.439268,0.34961,0.643137,0.9128,0.444029,0.346251,0.647059,0.915471,0.448807,0.34289,0.65098,0.918109,0.453603,0.339529,0.654902,0.920714,0.458417,0.336166,0.658824,0.923287,0.463251,0.332801,0.662745,0.925825,0.468103,0.329435,0.666667,0.928329,0.472975,0.326067,0.670588,0.930798,0.477867,0.322697,0.67451,0.933232,0.48278,0.319325,0.678431,0.93563,0.487712,0.315952,0.682353,0.93799,0.492667,0.312575,0.686275,0.940313,0.497642,0.309197,0.690196,0.942598,0.502639,0.305816,0.694118,0.944844,0.507658,0.302433,0.698039,0.947051,0.512699,0.299049,0.701961,0.949217,0.517763,0.295662,0.705882,0.951344,0.52285,0.292275,0.709804,0.953428,0.52796,0.288883,0.713725,0.95547,0.533093,0.28549,0.717647,0.957469,0.53825,0.282096,0.721569,0.959424,0.543431,0.278701,0.72549,0.961336,0.548636,0.275305,0.729412,0.963203,0.553865,0.271909,0.733333,0.965024,0.559118,0.268513,0.737255,0.966798,0.564396,0.265118,0.741176,0.968526,0.5697,0.261721,0.745098,0.970205,0.575028,0.258325,0.74902,0.971835,0.580382,0.254931,0.752941,0.973416,0.585761,0.25154,0.756863,0.974947,0.591165,0.248151,0.760784,0.976428,0.596595,0.244767,0.764706,0.977856,0.602051,0.241387,0.768627,0.979233,0.607532,0.238013,0.772549,0.980556,0.613039,0.234646,0.776471,0.981826,0.618572,0.231287,0.780392,0.983041,0.624131,0.227937,0.784314,0.984199,0.629718,0.224595,0.788235,0.985301,0.63533,0.221265,0.792157,0.986345,0.640969,0.217948,0.796078,0.987332,0.646633,0.214648,0.8,0.98826,0.652325,0.211364,0.803922,0.989128,0.658043,0.2081,0.807843,0.989935,0.663787,0.204859,0.811765,0.990681,0.669558,0.201642,0.815686,0.991365,0.675355,0.198453,0.819608,0.991985,0.681179,0.195295,0.823529,0.992541,0.68703,0.19217,0.827451,0.993032,0.692907,0.189084,0.831373,0.993456,0.69881,0.186041,0.835294,0.993814,0.704741,0.183043,0.839216,0.994103,0.710698,0.180097,0.843137,0.994324,0.716681,0.177208,0.847059,0.994474,0.722691,0.174381,0.85098,0.994553,0.728728,0.171622,0.854902,0.994561,0.734791,0.168938,0.858824,0.994495,0.74088,0.166335,0.862745,0.994355,0.746995,0.163821,0.866667,0.994141,0.753137,0.161404,0.870588,0.993851,0.759304,0.159092,0.87451,0.993482,0.765499,0.156891,0.878431,0.993033,0.77172,0.154808,0.882353,0.992505,0.777967,0.152855,0.886275,0.991897,0.784239,0.151042,0.890196,0.991209,0.790537,0.149377,0.894118,0.990439,0.796859,0.14787,0.898039,0.989587,0.803205,0.146529,0.901961,0.988648,0.809579,0.145357,0.905882,0.987621,0.815978,0.144363,0.909804,0.986509,0.822401,0.143557,0.913725,0.985314,0.828846,0.142945,0.917647,0.984031,0.835315,0.142528,0.921569,0.982653,0.841812,0.142303,0.92549,0.98119,0.848329,0.142279,0.929412,0.979644,0.854866,0.142453,0.933333,0.977995,0.861432,0.142808,0.937255,0.976265,0.868016,0.143351,0.941176,0.974443,0.874622,0.144061,0.945098,0.97253,0.88125,0.144923,0.94902,0.970533,0.887896,0.145919,0.952941,0.968443,0.894564,0.147014,0.956863,0.966271,0.901249,0.14818,0.960784,0.964021,0.90795,0.14937,0.964706,0.961681,0.914672,0.15052,0.968627,0.959276,0.921407,0.151566,0.972549,0.956808,0.928152,0.152409,0.976471,0.954287,0.934908,0.152921,0.980392,0.951726,0.941671,0.152925,0.984314,0.949151,0.948435,0.152178,0.988235,0.946602,0.95519,0.150328,0.992157,0.944152,0.961916,0.146861,0.996078,0.941896,0.96859,0.140956,1.0,0.940015,0.975158,0.131326]},"Cividis":{"ColorSpace":"Lab","Name":"Cividis","Source":"https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0199239","License":"CC0","Creator":"Jamie R. Nuñez, Christopher R. Anderton, and Ryan S. Renslow","RGBPoints":[0,0.0,0.1262,0.3015,1,0.0,0.1292,0.3077,2,0.0,0.1321,0.3142,3,0.0,0.135,0.3205,4,0.0,0.1379,0.3269,5,0.0,0.1408,0.3334,6,0.0,0.1437,0.34,7,0.0,0.1465,0.3467,8,0.0,0.1492,0.3537,9,0.0,0.1519,0.3606,10,0.0,0.1546,0.3676,11,0.0,0.1574,0.3746,12,0.0,0.1601,0.3817,13,0.0,0.1629,0.3888,14,0.0,0.1657,0.396,15,0.0,0.1685,0.4031,16,0.0,0.1714,0.4102,17,0.0,0.1743,0.4172,18,0.0,0.1773,0.4241,19,0.0,0.1798,0.4307,20,0.0,0.1817,0.4347,21,0.0,0.1834,0.4363,22,0.0,0.1852,0.4368,23,0.0,0.1872,0.4368,24,0.0,0.1901,0.4365,25,0.0,0.193,0.4361,26,0.0,0.1958,0.4356,27,0.0,0.1987,0.4349,28,0.0,0.2015,0.4343,29,0.0,0.2044,0.4336,30,0.0,0.2073,0.4329,31,0.0055,0.2101,0.4322,32,0.0236,0.213,0.4314,33,0.0416,0.2158,0.4308,34,0.0576,0.2187,0.4301,35,0.071,0.2215,0.4293,36,0.0827,0.2244,0.4287,37,0.0932,0.2272,0.428,38,0.103,0.23,0.4274,39,0.112,0.2329,0.4268,40,0.1204,0.2357,0.4262,41,0.1283,0.2385,0.4256,42,0.1359,0.2414,0.4251,43,0.1431,0.2442,0.4245,44,0.15,0.247,0.4241,45,0.1566,0.2498,0.4236,46,0.163,0.2526,0.4232,47,0.1692,0.2555,0.4228,48,0.1752,0.2583,0.4224,49,0.1811,0.2611,0.422,50,0.1868,0.2639,0.4217,51,0.1923,0.2667,0.4214,52,0.1977,0.2695,0.4212,53,0.203,0.2723,0.4209,54,0.2082,0.2751,0.4207,55,0.2133,0.278,0.4205,56,0.2183,0.2808,0.4204,57,0.2232,0.2836,0.4203,58,0.2281,0.2864,0.4202,59,0.2328,0.2892,0.4201,60,0.2375,0.292,0.42,61,0.2421,0.2948,0.42,62,0.2466,0.2976,0.42,63,0.2511,0.3004,0.4201,64,0.2556,0.3032,0.4201,65,0.2599,0.306,0.4202,66,0.2643,0.3088,0.4203,67,0.2686,0.3116,0.4205,68,0.2728,0.3144,0.4206,69,0.277,0.3172,0.4208,70,0.2811,0.32,0.421,71,0.2853,0.3228,0.4212,72,0.2894,0.3256,0.4215,73,0.2934,0.3284,0.4218,74,0.2974,0.3312,0.4221,75,0.3014,0.334,0.4224,76,0.3054,0.3368,0.4227,77,0.3093,0.3396,0.4231,78,0.3132,0.3424,0.4236,79,0.317,0.3453,0.424,80,0.3209,0.3481,0.4244,81,0.3247,0.3509,0.4249,82,0.3285,0.3537,0.4254,83,0.3323,0.3565,0.4259,84,0.3361,0.3593,0.4264,85,0.3398,0.3622,0.427,86,0.3435,0.365,0.4276,87,0.3472,0.3678,0.4282,88,0.3509,0.3706,0.4288,89,0.3546,0.3734,0.4294,90,0.3582,0.3763,0.4302,91,0.3619,0.3791,0.4308,92,0.3655,0.3819,0.4316,93,0.3691,0.3848,0.4322,94,0.3727,0.3876,0.4331,95,0.3763,0.3904,0.4338,96,0.3798,0.3933,0.4346,97,0.3834,0.3961,0.4355,98,0.3869,0.399,0.4364,99,0.3905,0.4018,0.4372,100,0.394,0.4047,0.4381,101,0.3975,0.4075,0.439,102,0.401,0.4104,0.44,103,0.4045,0.4132,0.4409,104,0.408,0.4161,0.4419,105,0.4114,0.4189,0.443,106,0.4149,0.4218,0.444,107,0.4183,0.4247,0.445,108,0.4218,0.4275,0.4462,109,0.4252,0.4304,0.4473,110,0.4286,0.4333,0.4485,111,0.432,0.4362,0.4496,112,0.4354,0.439,0.4508,113,0.4388,0.4419,0.4521,114,0.4422,0.4448,0.4534,115,0.4456,0.4477,0.4547,116,0.4489,0.4506,0.4561,117,0.4523,0.4535,0.4575,118,0.4556,0.4564,0.4589,119,0.4589,0.4593,0.4604,120,0.4622,0.4622,0.462,121,0.4656,0.4651,0.4635,122,0.4689,0.468,0.465,123,0.4722,0.4709,0.4665,124,0.4756,0.4738,0.4679,125,0.479,0.4767,0.4691,126,0.4825,0.4797,0.4701,127,0.4861,0.4826,0.4707,128,0.4897,0.4856,0.4714,129,0.4934,0.4886,0.4719,130,0.4971,0.4915,0.4723,131,0.5008,0.4945,0.4727,132,0.5045,0.4975,0.473,133,0.5083,0.5005,0.4732,134,0.5121,0.5035,0.4734,135,0.5158,0.5065,0.4736,136,0.5196,0.5095,0.4737,137,0.5234,0.5125,0.4738,138,0.5272,0.5155,0.4739,139,0.531,0.5186,0.4739,140,0.5349,0.5216,0.4738,141,0.5387,0.5246,0.4739,142,0.5425,0.5277,0.4738,143,0.5464,0.5307,0.4736,144,0.5502,0.5338,0.4735,145,0.5541,0.5368,0.4733,146,0.5579,0.5399,0.4732,147,0.5618,0.543,0.4729,148,0.5657,0.5461,0.4727,149,0.5696,0.5491,0.4723,150,0.5735,0.5522,0.472,151,0.5774,0.5553,0.4717,152,0.5813,0.5584,0.4714,153,0.5852,0.5615,0.4709,154,0.5892,0.5646,0.4705,155,0.5931,0.5678,0.4701,156,0.597,0.5709,0.4696,157,0.601,0.574,0.4691,158,0.605,0.5772,0.4685,159,0.6089,0.5803,0.468,160,0.6129,0.5835,0.4673,161,0.6168,0.5866,0.4668,162,0.6208,0.5898,0.4662,163,0.6248,0.5929,0.4655,164,0.6288,0.5961,0.4649,165,0.6328,0.5993,0.4641,166,0.6368,0.6025,0.4632,167,0.6408,0.6057,0.4625,168,0.6449,0.6089,0.4617,169,0.6489,0.6121,0.4609,170,0.6529,0.6153,0.46,171,0.657,0.6185,0.4591,172,0.661,0.6217,0.4583,173,0.6651,0.625,0.4573,174,0.6691,0.6282,0.4562,175,0.6732,0.6315,0.4553,176,0.6773,0.6347,0.4543,177,0.6813,0.638,0.4532,178,0.6854,0.6412,0.4521,179,0.6895,0.6445,0.4511,180,0.6936,0.6478,0.4499,181,0.6977,0.6511,0.4487,182,0.7018,0.6544,0.4475,183,0.706,0.6577,0.4463,184,0.7101,0.661,0.445,185,0.7142,0.6643,0.4437,186,0.7184,0.6676,0.4424,187,0.7225,0.671,0.4409,188,0.7267,0.6743,0.4396,189,0.7308,0.6776,0.4382,190,0.735,0.681,0.4368,191,0.7392,0.6844,0.4352,192,0.7434,0.6877,0.4338,193,0.7476,0.6911,0.4322,194,0.7518,0.6945,0.4307,195,0.756,0.6979,0.429,196,0.7602,0.7013,0.4273,197,0.7644,0.7047,0.4258,198,0.7686,0.7081,0.4241,199,0.7729,0.7115,0.4223,200,0.7771,0.715,0.4205,201,0.7814,0.7184,0.4188,202,0.7856,0.7218,0.4168,203,0.7899,0.7253,0.415,204,0.7942,0.7288,0.4129,205,0.7985,0.7322,0.4111,206,0.8027,0.7357,0.409,207,0.807,0.7392,0.407,208,0.8114,0.7427,0.4049,209,0.8157,0.7462,0.4028,210,0.82,0.7497,0.4007,211,0.8243,0.7532,0.3984,212,0.8287,0.7568,0.3961,213,0.833,0.7603,0.3938,214,0.8374,0.7639,0.3915,215,0.8417,0.7674,0.3892,216,0.8461,0.771,0.3869,217,0.8505,0.7745,0.3843,218,0.8548,0.7781,0.3818,219,0.8592,0.7817,0.3793,220,0.8636,0.7853,0.3766,221,0.8681,0.7889,0.3739,222,0.8725,0.7926,0.3712,223,0.8769,0.7962,0.3684,224,0.8813,0.7998,0.3657,225,0.8858,0.8035,0.3627,226,0.8902,0.8071,0.3599,227,0.8947,0.8108,0.3569,228,0.8992,0.8145,0.3538,229,0.9037,0.8182,0.3507,230,0.9082,0.8219,0.3474,231,0.9127,0.8256,0.3442,232,0.9172,0.8293,0.3409,233,0.9217,0.833,0.3374,234,0.9262,0.8367,0.334,235,0.9308,0.8405,0.3306,236,0.9353,0.8442,0.3268,237,0.9399,0.848,0.3232,238,0.9444,0.8518,0.3195,239,0.949,0.8556,0.3155,240,0.9536,0.8593,0.3116,241,0.9582,0.8632,0.3076,242,0.9628,0.867,0.3034,243,0.9674,0.8708,0.299,244,0.9721,0.8746,0.2947,245,0.9767,0.8785,0.2901,246,0.9814,0.8823,0.2856,247,0.986,0.8862,0.2807,248,0.9907,0.8901,0.2759,249,0.9954,0.894,0.2708,250,1.0,0.8979,0.2655,251,1.0,0.9018,0.26,252,1.0,0.9057,0.2593,253,1.0,0.9094,0.2634,254,1.0,0.9131,0.268,255,1.0,0.9169,0.2731]},"KAAMS Step":{"Name":"KAAMS Step","ColorSpace":"Step","RGBPoints":[0.0,1.0,1.0,1.0,0.083,1.0,1.0,1.0,0.166,1.0,0.0,0.0,0.25,0.0,1.0,0.0,0.333,0.0,0.0,1.0,0.416,1.0,1.0,0.0,0.5,1.0,0.0,1.0,0.583,0.0,1.0,1.0,0.666,0.63,0.63,1.0,0.75,0.67,0.5,0.33,0.833,1.0,0.5,0.75,0.916,0.53,0.35,0.7,1.0,1.0,0.75,0.5]},"Spectrum Step":{"Name":"Spectrum Step","ColorSpace":"Step","NanColor":[0.6509803921568628,0.3372549019607843,0.1568627450980392],"RGBPoints":[0.0,0.0,0.0,0.0,0.143,0.0,0.0,0.0,0.286,0.8941176470588236,0.1019607843137255,0.1098039215686274,0.429,0.2156862745098039,0.4941176470588236,0.7215686274509804,0.571,0.3019607843137255,0.6862745098039216,0.2901960784313726,0.714,0.596078431372549,0.3058823529411765,0.6392156862745098,0.857,1.0,0.4980392156862745,0.0,1.0,0.6509803921568628,0.3372549019607843,0.1568627450980392]},"Warm Step":{"Name":"Warm Step","ColorSpace":"Step","NanColor":[1,0.8980392156862745,0.02352941176470588],"RGBPoints":[0.0,0.4745098039215686,0.09019607843137255,0.09019607843137255,0.166,0.4745098039215686,0.09019607843137255,0.09019607843137255,0.333,0.7098039215686275,0.00392156862745098,0.00392156862745098,0.5,0.9372549019607843,0.2784313725490196,0.09803921568627451,0.666,0.9764705882352941,0.5137254901960784,0.1411764705882353,0.833,1.0,0.7058823529411765,0.0,1.0,1.0,0.8980392156862745,0.02352941176470588]},"Cool Step":{"Name":"Cool Step","ColorSpace":"Step","NanColor":[0.3725490196078431,0.2,0.5019607843137255],"RGBPoints":[0.0,0.4588235294117647,0.6941176470588235,0.00392156862745098,0.143,0.4588235294117647,0.6941176470588235,0.00392156862745098,0.286,0.3450980392156863,0.5019607843137255,0.1607843137254902,0.429,0.3137254901960784,0.8431372549019608,0.7490196078431373,0.571,0.1098039215686274,0.5843137254901961,0.803921568627451,0.714,0.2313725490196079,0.407843137254902,0.6705882352941176,0.857,0.6039215686274509,0.407843137254902,1.0,1.0,0.3725490196078431,0.2,0.5019607843137255]},"Blues Step":{"Name":"Blues Step","ColorSpace":"Step","NanColor":[0.06274509803921569,0.1647058823529412,0.3215686274509804],"RGBPoints":[0.0,0.2313725490196079,0.407843137254902,0.6705882352941176,0.143,0.2313725490196079,0.407843137254902,0.6705882352941176,0.286,0.1098039215686274,0.5843137254901961,0.803921568627451,0.429,0.3058823529411765,0.8509803921568627,0.9176470588235294,0.571,0.4509803921568628,0.6039215686274509,0.8352941176470589,0.714,0.2588235294117647,0.2392156862745098,0.6627450980392157,0.857,0.3137254901960784,0.3294117647058823,0.5294117647058824,1.0,0.06274509803921569,0.1647058823529412,0.3215686274509804]},"Wild Flower Step":{"Name":"Wild Flower Step","ColorSpace":"Step","NanColor":[0.2392156862745098,0.06274509803921569,0.3215686274509804],"RGBPoints":[0.0,0.1098039215686274,0.5843137254901961,0.803921568627451,0.143,0.1098039215686274,0.5843137254901961,0.803921568627451,0.286,0.2313725490196079,0.407843137254902,0.6705882352941176,0.429,0.4,0.2431372549019608,0.7176470588235294,0.571,0.6352941176470588,0.3294117647058823,0.8117647058823529,0.714,0.8705882352941177,0.3803921568627451,0.807843137254902,0.857,0.8627450980392157,0.3803921568627451,0.5843137254901961,1.0,0.2392156862745098,0.06274509803921569,0.3215686274509804]},"Citrus Step":{"Name":"Citrus Step","ColorSpace":"Step","NanColor":[0.9764705882352941,0.5137254901960784,0.1411764705882353],"RGBPoints":[0.0,0.396078431372549,0.4862745098039216,0.2156862745098039,0.166,0.396078431372549,0.4862745098039216,0.2156862745098039,0.333,0.4588235294117647,0.6941176470588235,0.00392156862745098,0.5,0.6980392156862745,0.7294117647058823,0.1882352941176471,0.666,1.0,0.8980392156862745,0.02352941176470588,0.833,1.0,0.7058823529411765,0.0,1.0,0.9764705882352941,0.5137254901960784,0.1411764705882353]},"Traffic Lights Step":{"Name":"Traffic Lights Step","ColorSpace":"Step","NanColor":[0.803922,0,0.803922],"RGBPoints":[0.0,1.0,0.0,0.0,0.333,1.0,0.0,0.0,0.666,1.0,0.862745,0.0,1.0,0.0,0.695201,0.0]},"Traffic Lights For Deuteranopes Step":{"Name":"Traffic Lights For Deuteranopes Step","ColorSpace":"Step","NanColor":[0.803922,0,0.803922],"RGBPoints":[0.0,0.908659,0.604013,0.581857,0.333,0.908659,0.604013,0.581857,0.666,1.0,0.862745,0.0,1.0,0.0,0.695201,0.0]},"Traffic Lights For Deuteranopes 2 Step":{"Name":"Traffic Lights For Deuteranopes 2 Step","ColorSpace":"Step","NanColor":[0.803922,0,0.803922],"RGBPoints":[0.0,0.4196078431372549,0.0,0.07058823529411765,0.333,0.4196078431372549,0.0,0.07058823529411765,0.666,0.9019607843137255,0.9411764705882353,0.0196078431372549,1.0,0.01568627450980392,0.6196078431372549,0.00784313725490196]},"Fast":{"Name":"Fast","ColorSpace":"Lab","Creator":"Francesca Samsel, and Alan W. Scott","DefaultMap":true,"NanColor":[0.0,1.0,0.0],"RGBPoints":[0,0.05639999999999999,0.05639999999999999,0.47,0.17159223942480895,0.24300000000000013,0.4603500000000004,0.81,0.2984914818394138,0.3568143826543521,0.7450246485363142,0.954367702893722,0.4321287371255907,0.6882,0.93,0.9179099999999999,0.5,0.8994959551205902,0.944646394975174,0.7686567142818399,0.5882260353170073,0.957107977357604,0.8338185108985666,0.5089156299842102,0.7061412605695164,0.9275207599610714,0.6214389091739178,0.31535705838676426,0.8476395308725272,0.8,0.3520000000000001,0.15999999999999998,1,0.59,0.07670000000000013,0.11947499999999994]},"Fast (Reds)":{"Name":"Fast (Reds)","ColorSpace":"Lab","Creator":"Francesca Samsel, and Alan W. Scott","NanColor":[0.0,1.0,0.0],"RGBPoints":[0,0.8994959551205902,0.944646394975174,0.7686567142818399,0.176452070634014,0.957107977357604,0.8338185108985666,0.5089156299842102,0.412282521139032,0.9275207599610714,0.6214389091739178,0.31535705838676426,0.695279061745054,0.8,0.3520000000000001,0.15999999999999998,1,0.59,0.07670000000000013,0.11947499999999994]},"Fast (Blues)":{"Name":"Fast (Blues)","ColorSpace":"Lab","Creator":"Francesca Samsel, and Alan W. Scott","NanColor":[0.0,1.0,0.0],"RGBPoints":[0,0.05639999999999999,0.05639999999999999,0.47,0.343184478849616,0.24300000000000013,0.4603500000000004,0.81,0.596982963678826,0.3568143826543521,0.7450246485363142,0.954367702893722,0.86425747425118,0.6882,0.93,0.9179099999999999,1,0.8994959551205902,0.944646394975174,0.7686567142818399]},"ParaView 6.0.0 Default":{"Name":"ParaView 6.0.0 Default","ColorSpace":"Diverging","Creator":"ParaView 6.0.0","NanColor":[0.0,1.0,0.0],"RGBPoints":[0,0.05639999999999999,0.05639999999999999,0.47,0.17159223942480895,0.24300000000000013,0.4603500000000004,0.81,0.2984914818394138,0.3568143826543521,0.7450246485363142,0.954367702893722,0.4321287371255907,0.6882,0.93,0.9179099999999999,0.5,0.8994959551205902,0.944646394975174,0.7686567142818399,0.5882260353170073,0.957107977357604,0.8338185108985666,0.5089156299842102,0.7061412605695164,0.9275207599610714,0.6214389091739178,0.31535705838676426,0.8476395308725272,0.8,0.3520000000000001,0.15999999999999998,1,0.59,0.07670000000000013,0.11947499999999994]},"Inferno (matplotlib)":{"ColorSpace":"Diverging","Name":"Inferno (matplotlib)","NanColor":[0,1,0],"Source":"https://github.com/BIDS/colormap/blob/master/colormaps.py","License":"CC0","Creator":"Nathaniel J. Smith & Stefan van der Walt","DefaultMap":true,"RGBPoints":[0.0,0.001462,0.000466,0.013866,0.003922,0.002267,0.00127,0.01857,0.007843,0.003299,0.002249,0.024239,0.011765,0.004547,0.003392,0.030909,0.015686,0.006006,0.004692,0.038558,0.019608,0.007676,0.006136,0.046836,0.023529,0.009561,0.007713,0.055143,0.027451,0.011663,0.009417,0.06346,0.031373,0.013995,0.011225,0.071862,0.035294,0.016561,0.013136,0.080282,0.039216,0.019373,0.015133,0.088767,0.043137,0.022447,0.017199,0.097327,0.047059,0.025793,0.019331,0.10593,0.05098,0.029432,0.021503,0.114621,0.054902,0.033385,0.023702,0.123397,0.058824,0.037668,0.025921,0.132232,0.062745,0.042253,0.028139,0.141141,0.066667,0.046915,0.030324,0.150164,0.070588,0.051644,0.032474,0.159254,0.07451,0.056449,0.034569,0.168414,0.078431,0.06134,0.03659,0.177642,0.082353,0.066331,0.038504,0.186962,0.086275,0.071429,0.040294,0.196354,0.090196,0.076637,0.041905,0.205799,0.094118,0.081962,0.043328,0.215289,0.098039,0.087411,0.044556,0.224813,0.101961,0.09299,0.045583,0.234358,0.105882,0.098702,0.046402,0.243904,0.109804,0.104551,0.047008,0.25343,0.113725,0.110536,0.047399,0.262912,0.117647,0.116656,0.047574,0.272321,0.121569,0.122908,0.047536,0.281624,0.12549,0.129285,0.047293,0.290788,0.129412,0.135778,0.046856,0.299776,0.133333,0.142378,0.046242,0.308553,0.137255,0.149073,0.045468,0.317085,0.141176,0.15585,0.044559,0.325338,0.145098,0.162689,0.043554,0.333277,0.14902,0.169575,0.042489,0.340874,0.152941,0.176493,0.041402,0.348111,0.156863,0.183429,0.040329,0.354971,0.160784,0.190367,0.039309,0.361447,0.164706,0.197297,0.0384,0.367535,0.168627,0.204209,0.037632,0.373238,0.172549,0.211095,0.03703,0.378563,0.176471,0.217949,0.036615,0.383522,0.180392,0.224763,0.036405,0.388129,0.184314,0.231538,0.036405,0.3924,0.188235,0.238273,0.036621,0.396353,0.192157,0.244967,0.037055,0.400007,0.196078,0.25162,0.037705,0.403378,0.2,0.258234,0.038571,0.406485,0.203922,0.26481,0.039647,0.409345,0.207843,0.271347,0.040922,0.411976,0.211765,0.27785,0.042353,0.414392,0.215686,0.284321,0.043933,0.416608,0.219608,0.290763,0.045644,0.418637,0.223529,0.297178,0.04747,0.420491,0.227451,0.303568,0.049396,0.422182,0.231373,0.309935,0.051407,0.423721,0.235294,0.316282,0.05349,0.425116,0.239216,0.32261,0.055634,0.426377,0.243137,0.328921,0.057827,0.427511,0.247059,0.335217,0.06006,0.428524,0.25098,0.3415,0.062325,0.429425,0.254902,0.347771,0.064616,0.430217,0.258824,0.354032,0.066925,0.430906,0.262745,0.360284,0.069247,0.431497,0.266667,0.366529,0.071579,0.431994,0.270588,0.372768,0.073915,0.4324,0.27451,0.379001,0.076253,0.432719,0.278431,0.385228,0.078591,0.432955,0.282353,0.391453,0.080927,0.433109,0.286275,0.397674,0.083257,0.433183,0.290196,0.403894,0.08558,0.433179,0.294118,0.410113,0.087896,0.433098,0.298039,0.416331,0.090203,0.432943,0.301961,0.422549,0.092501,0.432714,0.305882,0.428768,0.09479,0.432412,0.309804,0.434987,0.097069,0.432039,0.313725,0.441207,0.099338,0.431594,0.317647,0.447428,0.101597,0.43108,0.321569,0.453651,0.103848,0.430498,0.32549,0.459875,0.106089,0.429846,0.329412,0.4661,0.108322,0.429125,0.333333,0.472328,0.110547,0.428334,0.337255,0.478558,0.112764,0.427475,0.341176,0.484789,0.114974,0.426548,0.345098,0.491022,0.117179,0.425552,0.34902,0.497257,0.119379,0.424488,0.352941,0.503493,0.121575,0.423356,0.356863,0.50973,0.123769,0.422156,0.360784,0.515967,0.12596,0.420887,0.364706,0.522206,0.12815,0.419549,0.368627,0.528444,0.130341,0.418142,0.372549,0.534683,0.132534,0.416667,0.376471,0.54092,0.134729,0.415123,0.380392,0.547157,0.136929,0.413511,0.384314,0.553392,0.139134,0.411829,0.388235,0.559624,0.141346,0.410078,0.392157,0.565854,0.143567,0.408258,0.396078,0.572081,0.145797,0.406369,0.4,0.578304,0.148039,0.404411,0.403922,0.584521,0.150294,0.402385,0.407843,0.590734,0.152563,0.40029,0.411765,0.59694,0.154848,0.398125,0.415686,0.603139,0.157151,0.395891,0.419608,0.60933,0.159474,0.393589,0.423529,0.615513,0.161817,0.391219,0.427451,0.621685,0.164184,0.388781,0.431373,0.627847,0.166575,0.386276,0.435294,0.633998,0.168992,0.383704,0.439216,0.640135,0.171438,0.381065,0.443137,0.64626,0.173914,0.378359,0.447059,0.652369,0.176421,0.375586,0.45098,0.658463,0.178962,0.372748,0.454902,0.66454,0.181539,0.369846,0.458824,0.670599,0.184153,0.366879,0.462745,0.676638,0.186807,0.363849,0.466667,0.682656,0.189501,0.360757,0.470588,0.688653,0.192239,0.357603,0.47451,0.694627,0.195021,0.354388,0.478431,0.700576,0.197851,0.351113,0.482353,0.7065,0.200728,0.347777,0.486275,0.712396,0.203656,0.344383,0.490196,0.718264,0.206636,0.340931,0.494118,0.724103,0.20967,0.337424,0.498039,0.729909,0.212759,0.333861,0.501961,0.735683,0.215906,0.330245,0.505882,0.741423,0.219112,0.326576,0.509804,0.747127,0.222378,0.322856,0.513725,0.752794,0.225706,0.319085,0.517647,0.758422,0.229097,0.315266,0.521569,0.76401,0.232554,0.311399,0.52549,0.769556,0.236077,0.307485,0.529412,0.775059,0.239667,0.303526,0.533333,0.780517,0.243327,0.299523,0.537255,0.785929,0.247056,0.295477,0.541176,0.791293,0.250856,0.29139,0.545098,0.796607,0.254728,0.287264,0.54902,0.801871,0.258674,0.283099,0.552941,0.807082,0.262692,0.278898,0.556863,0.812239,0.266786,0.274661,0.560784,0.817341,0.270954,0.27039,0.564706,0.822386,0.275197,0.266085,0.568627,0.827372,0.279517,0.26175,0.572549,0.832299,0.283913,0.257383,0.576471,0.837165,0.288385,0.252988,0.580392,0.841969,0.292933,0.248564,0.584314,0.846709,0.297559,0.244113,0.588235,0.851384,0.30226,0.239636,0.592157,0.855992,0.307038,0.235133,0.596078,0.860533,0.311892,0.230606,0.6,0.865006,0.316822,0.226055,0.603922,0.869409,0.321827,0.221482,0.607843,0.873741,0.326906,0.216886,0.611765,0.878001,0.33206,0.212268,0.615686,0.882188,0.337287,0.207628,0.619608,0.886302,0.342586,0.202968,0.623529,0.890341,0.347957,0.198286,0.627451,0.894305,0.353399,0.193584,0.631373,0.898192,0.358911,0.18886,0.635294,0.902003,0.364492,0.184116,0.639216,0.905735,0.37014,0.17935,0.643137,0.90939,0.375856,0.174563,0.647059,0.912966,0.381636,0.169755,0.65098,0.916462,0.387481,0.164924,0.654902,0.919879,0.393389,0.16007,0.658824,0.923215,0.399359,0.155193,0.662745,0.92647,0.405389,0.150292,0.666667,0.929644,0.411479,0.145367,0.670588,0.932737,0.417627,0.140417,0.67451,0.935747,0.423831,0.13544,0.678431,0.938675,0.430091,0.130438,0.682353,0.941521,0.436405,0.125409,0.686275,0.944285,0.442772,0.120354,0.690196,0.946965,0.449191,0.115272,0.694118,0.949562,0.45566,0.110164,0.698039,0.952075,0.462178,0.105031,0.701961,0.954506,0.468744,0.099874,0.705882,0.956852,0.475356,0.094695,0.709804,0.959114,0.482014,0.089499,0.713725,0.961293,0.488716,0.084289,0.717647,0.963387,0.495462,0.079073,0.721569,0.965397,0.502249,0.073859,0.72549,0.967322,0.509078,0.068659,0.729412,0.969163,0.515946,0.063488,0.733333,0.970919,0.522853,0.058367,0.737255,0.97259,0.529798,0.053324,0.741176,0.974176,0.53678,0.048392,0.745098,0.975677,0.543798,0.043618,0.74902,0.977092,0.55085,0.03905,0.752941,0.978422,0.557937,0.034931,0.756863,0.979666,0.565057,0.031409,0.760784,0.980824,0.572209,0.028508,0.764706,0.981895,0.579392,0.02625,0.768627,0.982881,0.586606,0.024661,0.772549,0.983779,0.593849,0.02377,0.776471,0.984591,0.601122,0.023606,0.780392,0.985315,0.608422,0.024202,0.784314,0.985952,0.61575,0.025592,0.788235,0.986502,0.623105,0.027814,0.792157,0.986964,0.630485,0.030908,0.796078,0.987337,0.63789,0.034916,0.8,0.987622,0.64532,0.039886,0.803922,0.987819,0.652773,0.045581,0.807843,0.987926,0.66025,0.05175,0.811765,0.987945,0.667748,0.058329,0.815686,0.987874,0.675267,0.065257,0.819608,0.987714,0.682807,0.072489,0.823529,0.987464,0.690366,0.07999,0.827451,0.987124,0.697944,0.087731,0.831373,0.986694,0.70554,0.095694,0.835294,0.986175,0.713153,0.103863,0.839216,0.985566,0.720782,0.112229,0.843137,0.984865,0.728427,0.120785,0.847059,0.984075,0.736087,0.129527,0.85098,0.983196,0.743758,0.138453,0.854902,0.982228,0.751442,0.147565,0.858824,0.981173,0.759135,0.156863,0.862745,0.980032,0.766837,0.166353,0.866667,0.978806,0.774545,0.176037,0.870588,0.977497,0.782258,0.185923,0.87451,0.976108,0.789974,0.196018,0.878431,0.974638,0.797692,0.206332,0.882353,0.973088,0.805409,0.216877,0.886275,0.971468,0.813122,0.227658,0.890196,0.969783,0.820825,0.238686,0.894118,0.968041,0.828515,0.249972,0.898039,0.966243,0.836191,0.261534,0.901961,0.964394,0.843848,0.273391,0.905882,0.962517,0.851476,0.285546,0.909804,0.960626,0.859069,0.29801,0.913725,0.95872,0.866624,0.31082,0.917647,0.956834,0.874129,0.323974,0.921569,0.954997,0.881569,0.337475,0.92549,0.953215,0.888942,0.351369,0.929412,0.951546,0.896226,0.365627,0.933333,0.950018,0.903409,0.380271,0.937255,0.948683,0.910473,0.395289,0.941176,0.947594,0.917399,0.410665,0.945098,0.946809,0.924168,0.426373,0.94902,0.946392,0.930761,0.442367,0.952941,0.946403,0.937159,0.458592,0.956863,0.946903,0.943348,0.47497,0.960784,0.947937,0.949318,0.491426,0.964706,0.949545,0.955063,0.50786,0.968627,0.95174,0.960587,0.524203,0.972549,0.954529,0.965896,0.540361,0.976471,0.957896,0.971003,0.556275,0.980392,0.961812,0.975924,0.571925,0.984314,0.966249,0.980678,0.587206,0.988235,0.971162,0.985282,0.602154,0.992157,0.976511,0.989753,0.61676,0.996078,0.982257,0.994109,0.631017,1.0,0.988362,0.998364,0.644924]},"Viridis (matplotlib)":{"ColorSpace":"Diverging","Name":"Viridis (matplotlib)","NanColor":[1,0,0],"Source":"https://github.com/BIDS/colormap/blob/master/colormaps.py","License":"CC0","Creator":"Eric Firing","DefaultMap":true,"RGBPoints":[0.0,0.267004,0.004874,0.329415,0.003922,0.26851,0.009605,0.335427,0.007843,0.269944,0.014625,0.341379,0.011765,0.271305,0.019942,0.347269,0.015686,0.272594,0.025563,0.353093,0.019608,0.273809,0.031497,0.358853,0.023529,0.274952,0.037752,0.364543,0.027451,0.276022,0.044167,0.370164,0.031373,0.277018,0.050344,0.375715,0.035294,0.277941,0.056324,0.381191,0.039216,0.278791,0.062145,0.386592,0.043137,0.279566,0.067836,0.391917,0.047059,0.280267,0.073417,0.397163,0.05098,0.280894,0.078907,0.402329,0.054902,0.281446,0.08432,0.407414,0.058824,0.281924,0.089666,0.412415,0.062745,0.282327,0.094955,0.417331,0.066667,0.282656,0.100196,0.42216,0.070588,0.28291,0.105393,0.426902,0.07451,0.283091,0.110553,0.431554,0.078431,0.283197,0.11568,0.436115,0.082353,0.283229,0.120777,0.440584,0.086275,0.283187,0.125848,0.44496,0.090196,0.283072,0.130895,0.449241,0.094118,0.282884,0.13592,0.453427,0.098039,0.282623,0.140926,0.457517,0.101961,0.28229,0.145912,0.46151,0.105882,0.281887,0.150881,0.465405,0.109804,0.281412,0.155834,0.469201,0.113725,0.280868,0.160771,0.472899,0.117647,0.280255,0.165693,0.476498,0.121569,0.279574,0.170599,0.479997,0.12549,0.278826,0.17549,0.483397,0.129412,0.278012,0.180367,0.486697,0.133333,0.277134,0.185228,0.489898,0.137255,0.276194,0.190074,0.493001,0.141176,0.275191,0.194905,0.496005,0.145098,0.274128,0.199721,0.498911,0.14902,0.273006,0.20452,0.501721,0.152941,0.271828,0.209303,0.504434,0.156863,0.270595,0.214069,0.507052,0.160784,0.269308,0.218818,0.509577,0.164706,0.267968,0.223549,0.512008,0.168627,0.26658,0.228262,0.514349,0.172549,0.265145,0.232956,0.516599,0.176471,0.263663,0.237631,0.518762,0.180392,0.262138,0.242286,0.520837,0.184314,0.260571,0.246922,0.522828,0.188235,0.258965,0.251537,0.524736,0.192157,0.257322,0.25613,0.526563,0.196078,0.255645,0.260703,0.528312,0.2,0.253935,0.265254,0.529983,0.203922,0.252194,0.269783,0.531579,0.207843,0.250425,0.27429,0.533103,0.211765,0.248629,0.278775,0.534556,0.215686,0.246811,0.283237,0.535941,0.219608,0.244972,0.287675,0.53726,0.223529,0.243113,0.292092,0.538516,0.227451,0.241237,0.296485,0.539709,0.231373,0.239346,0.300855,0.540844,0.235294,0.237441,0.305202,0.541921,0.239216,0.235526,0.309527,0.542944,0.243137,0.233603,0.313828,0.543914,0.247059,0.231674,0.318106,0.544834,0.25098,0.229739,0.322361,0.545706,0.254902,0.227802,0.326594,0.546532,0.258824,0.225863,0.330805,0.547314,0.262745,0.223925,0.334994,0.548053,0.266667,0.221989,0.339161,0.548752,0.270588,0.220057,0.343307,0.549413,0.27451,0.21813,0.347432,0.550038,0.278431,0.21621,0.351535,0.550627,0.282353,0.214298,0.355619,0.551184,0.286275,0.212395,0.359683,0.55171,0.290196,0.210503,0.363727,0.552206,0.294118,0.208623,0.367752,0.552675,0.298039,0.206756,0.371758,0.553117,0.301961,0.204903,0.375746,0.553533,0.305882,0.203063,0.379716,0.553925,0.309804,0.201239,0.38367,0.554294,0.313725,0.19943,0.387607,0.554642,0.317647,0.197636,0.391528,0.554969,0.321569,0.19586,0.395433,0.555276,0.32549,0.1941,0.399323,0.555565,0.329412,0.192357,0.403199,0.555836,0.333333,0.190631,0.407061,0.556089,0.337255,0.188923,0.41091,0.556326,0.341176,0.187231,0.414746,0.556547,0.345098,0.185556,0.41857,0.556753,0.34902,0.183898,0.422383,0.556944,0.352941,0.182256,0.426184,0.55712,0.356863,0.180629,0.429975,0.557282,0.360784,0.179019,0.433756,0.55743,0.364706,0.177423,0.437527,0.557565,0.368627,0.175841,0.44129,0.557685,0.372549,0.174274,0.445044,0.557792,0.376471,0.172719,0.448791,0.557885,0.380392,0.171176,0.45253,0.557965,0.384314,0.169646,0.456262,0.55803,0.388235,0.168126,0.459988,0.558082,0.392157,0.166617,0.463708,0.558119,0.396078,0.165117,0.467423,0.558141,0.4,0.163625,0.471133,0.558148,0.403922,0.162142,0.474838,0.55814,0.407843,0.160665,0.47854,0.558115,0.411765,0.159194,0.482237,0.558073,0.415686,0.157729,0.485932,0.558013,0.419608,0.15627,0.489624,0.557936,0.423529,0.154815,0.493313,0.55784,0.427451,0.153364,0.497,0.557724,0.431373,0.151918,0.500685,0.557587,0.435294,0.150476,0.504369,0.55743,0.439216,0.149039,0.508051,0.55725,0.443137,0.147607,0.511733,0.557049,0.447059,0.14618,0.515413,0.556823,0.45098,0.144759,0.519093,0.556572,0.454902,0.143343,0.522773,0.556295,0.458824,0.141935,0.526453,0.555991,0.462745,0.140536,0.530132,0.555659,0.466667,0.139147,0.533812,0.555298,0.470588,0.13777,0.537492,0.554906,0.47451,0.136408,0.541173,0.554483,0.478431,0.135066,0.544853,0.554029,0.482353,0.133743,0.548535,0.553541,0.486275,0.132444,0.552216,0.553018,0.490196,0.131172,0.555899,0.552459,0.494118,0.129933,0.559582,0.551864,0.498039,0.128729,0.563265,0.551229,0.501961,0.127568,0.566949,0.550556,0.505882,0.126453,0.570633,0.549841,0.509804,0.125394,0.574318,0.549086,0.513725,0.124395,0.578002,0.548287,0.517647,0.123463,0.581687,0.547445,0.521569,0.122606,0.585371,0.546557,0.52549,0.121831,0.589055,0.545623,0.529412,0.121148,0.592739,0.544641,0.533333,0.120565,0.596422,0.543611,0.537255,0.120092,0.600104,0.54253,0.541176,0.119738,0.603785,0.5414,0.545098,0.119512,0.607464,0.540218,0.54902,0.119423,0.611141,0.538982,0.552941,0.119483,0.614817,0.537692,0.556863,0.119699,0.61849,0.536347,0.560784,0.120081,0.622161,0.534946,0.564706,0.120638,0.625828,0.533488,0.568627,0.12138,0.629492,0.531973,0.572549,0.122312,0.633153,0.530398,0.576471,0.123444,0.636809,0.528763,0.580392,0.12478,0.640461,0.527068,0.584314,0.126326,0.644107,0.525311,0.588235,0.128087,0.647749,0.523491,0.592157,0.130067,0.651384,0.521608,0.596078,0.132268,0.655014,0.519661,0.6,0.134692,0.658636,0.517649,0.603922,0.137339,0.662252,0.515571,0.607843,0.14021,0.665859,0.513427,0.611765,0.143303,0.669459,0.511215,0.615686,0.146616,0.67305,0.508936,0.619608,0.150148,0.676631,0.506589,0.623529,0.153894,0.680203,0.504172,0.627451,0.157851,0.683765,0.501686,0.631373,0.162016,0.687316,0.499129,0.635294,0.166383,0.690856,0.496502,0.639216,0.170948,0.694384,0.493803,0.643137,0.175707,0.6979,0.491033,0.647059,0.180653,0.701402,0.488189,0.65098,0.185783,0.704891,0.485273,0.654902,0.19109,0.708366,0.482284,0.658824,0.196571,0.711827,0.479221,0.662745,0.202219,0.715272,0.476084,0.666667,0.20803,0.718701,0.472873,0.670588,0.214,0.722114,0.469588,0.67451,0.220124,0.725509,0.466226,0.678431,0.226397,0.728888,0.462789,0.682353,0.232815,0.732247,0.459277,0.686275,0.239374,0.735588,0.455688,0.690196,0.24607,0.73891,0.452024,0.694118,0.252899,0.742211,0.448284,0.698039,0.259857,0.745492,0.444467,0.701961,0.266941,0.748751,0.440573,0.705882,0.274149,0.751988,0.436601,0.709804,0.281477,0.755203,0.432552,0.713725,0.288921,0.758394,0.428426,0.717647,0.296479,0.761561,0.424223,0.721569,0.304148,0.764704,0.419943,0.72549,0.311925,0.767822,0.415586,0.729412,0.319809,0.770914,0.411152,0.733333,0.327796,0.77398,0.40664,0.737255,0.335885,0.777018,0.402049,0.741176,0.344074,0.780029,0.397381,0.745098,0.35236,0.783011,0.392636,0.74902,0.360741,0.785964,0.387814,0.752941,0.369214,0.788888,0.382914,0.756863,0.377779,0.791781,0.377939,0.760784,0.386433,0.794644,0.372886,0.764706,0.395174,0.797475,0.367757,0.768627,0.404001,0.800275,0.362552,0.772549,0.412913,0.803041,0.357269,0.776471,0.421908,0.805774,0.35191,0.780392,0.430983,0.808473,0.346476,0.784314,0.440137,0.811138,0.340967,0.788235,0.449368,0.813768,0.335384,0.792157,0.458674,0.816363,0.329727,0.796078,0.468053,0.818921,0.323998,0.8,0.477504,0.821444,0.318195,0.803922,0.487026,0.823929,0.312321,0.807843,0.496615,0.826376,0.306377,0.811765,0.506271,0.828786,0.300362,0.815686,0.515992,0.831158,0.294279,0.819608,0.525776,0.833491,0.288127,0.823529,0.535621,0.835785,0.281908,0.827451,0.545524,0.838039,0.275626,0.831373,0.555484,0.840254,0.269281,0.835294,0.565498,0.84243,0.262877,0.839216,0.575563,0.844566,0.256415,0.843137,0.585678,0.846661,0.249897,0.847059,0.595839,0.848717,0.243329,0.85098,0.606045,0.850733,0.236712,0.854902,0.616293,0.852709,0.230052,0.858824,0.626579,0.854645,0.223353,0.862745,0.636902,0.856542,0.21662,0.866667,0.647257,0.8584,0.209861,0.870588,0.657642,0.860219,0.203082,0.87451,0.668054,0.861999,0.196293,0.878431,0.678489,0.863742,0.189503,0.882353,0.688944,0.865448,0.182725,0.886275,0.699415,0.867117,0.175971,0.890196,0.709898,0.868751,0.169257,0.894118,0.720391,0.87035,0.162603,0.898039,0.730889,0.871916,0.156029,0.901961,0.741388,0.873449,0.149561,0.905882,0.751884,0.874951,0.143228,0.909804,0.762373,0.876424,0.137064,0.913725,0.772852,0.877868,0.131109,0.917647,0.783315,0.879285,0.125405,0.921569,0.79376,0.880678,0.120005,0.92549,0.804182,0.882046,0.114965,0.929412,0.814576,0.883393,0.110347,0.933333,0.82494,0.88472,0.106217,0.937255,0.83527,0.886029,0.102646,0.941176,0.845561,0.887322,0.099702,0.945098,0.85581,0.888601,0.097452,0.94902,0.866013,0.889868,0.095953,0.952941,0.876168,0.891125,0.09525,0.956863,0.886271,0.892374,0.095374,0.960784,0.89632,0.893616,0.096335,0.964706,0.906311,0.894855,0.098125,0.968627,0.916242,0.896091,0.100717,0.972549,0.926106,0.89733,0.104071,0.976471,0.935904,0.89857,0.108131,0.980392,0.945636,0.899815,0.112838,0.984314,0.9553,0.901065,0.118128,0.988235,0.964894,0.902323,0.123941,0.992157,0.974417,0.90359,0.130215,0.996078,0.983868,0.904867,0.136897,1.0,0.993248,0.906157,0.143936]},"Magma (matplotlib)":{"ColorSpace":"Diverging","Name":"Magma (matplotlib)","NanColor":[0,1,0],"Source":"https://github.com/BIDS/colormap/blob/master/colormaps.py","License":"CC0","Creator":"Nathaniel J. Smith & Stefan van der Walt","RGBPoints":[0.0,0.001462,0.000466,0.013866,0.003922,0.002258,0.001295,0.018331,0.007843,0.003279,0.002305,0.023708,0.011765,0.004512,0.00349,0.029965,0.015686,0.00595,0.004843,0.03713,0.019608,0.007588,0.006356,0.044973,0.023529,0.009426,0.008022,0.052844,0.027451,0.011465,0.009828,0.06075,0.031373,0.013708,0.011771,0.068667,0.035294,0.016156,0.01384,0.076603,0.039216,0.018815,0.016026,0.084584,0.043137,0.021692,0.01832,0.09261,0.047059,0.024792,0.020715,0.100676,0.05098,0.028123,0.023201,0.108787,0.054902,0.031696,0.025765,0.116965,0.058824,0.03552,0.028397,0.125209,0.062745,0.039608,0.03109,0.133515,0.066667,0.04383,0.03383,0.141886,0.070588,0.048062,0.036607,0.150327,0.07451,0.05232,0.039407,0.158841,0.078431,0.056615,0.04216,0.167446,0.082353,0.060949,0.044794,0.176129,0.086275,0.06533,0.047318,0.184892,0.090196,0.069764,0.049726,0.193735,0.094118,0.074257,0.052017,0.20266,0.098039,0.078815,0.054184,0.211667,0.101961,0.083446,0.056225,0.220755,0.105882,0.088155,0.058133,0.229922,0.109804,0.092949,0.059904,0.239164,0.113725,0.097833,0.061531,0.248477,0.117647,0.102815,0.06301,0.257854,0.121569,0.107899,0.064335,0.267289,0.12549,0.113094,0.065492,0.276784,0.129412,0.118405,0.066479,0.286321,0.133333,0.123833,0.067295,0.295879,0.137255,0.12938,0.067935,0.305443,0.141176,0.135053,0.068391,0.315,0.145098,0.140858,0.068654,0.324538,0.14902,0.146785,0.068738,0.334011,0.152941,0.152839,0.068637,0.343404,0.156863,0.159018,0.068354,0.352688,0.160784,0.165308,0.067911,0.361816,0.164706,0.171713,0.067305,0.370771,0.168627,0.178212,0.066576,0.379497,0.172549,0.184801,0.065732,0.387973,0.176471,0.19146,0.064818,0.396152,0.180392,0.198177,0.063862,0.404009,0.184314,0.204935,0.062907,0.411514,0.188235,0.211718,0.061992,0.418647,0.192157,0.218512,0.061158,0.425392,0.196078,0.225302,0.060445,0.431742,0.2,0.232077,0.059889,0.437695,0.203922,0.238826,0.059517,0.443256,0.207843,0.245543,0.059352,0.448436,0.211765,0.25222,0.059415,0.453248,0.215686,0.258857,0.059706,0.45771,0.219608,0.265447,0.060237,0.46184,0.223529,0.271994,0.060994,0.46566,0.227451,0.278493,0.061978,0.46919,0.231373,0.284951,0.063168,0.472451,0.235294,0.291366,0.064553,0.475462,0.239216,0.29774,0.066117,0.478243,0.243137,0.304081,0.067835,0.480812,0.247059,0.310382,0.069702,0.483186,0.25098,0.316654,0.07169,0.48538,0.254902,0.322899,0.073782,0.487408,0.258824,0.329114,0.075972,0.489287,0.262745,0.335308,0.078236,0.491024,0.266667,0.341482,0.080564,0.492631,0.270588,0.347636,0.082946,0.494121,0.27451,0.353773,0.085373,0.495501,0.278431,0.359898,0.087831,0.496778,0.282353,0.366012,0.090314,0.49796,0.286275,0.372116,0.092816,0.499053,0.290196,0.378211,0.095332,0.500067,0.294118,0.384299,0.097855,0.501002,0.298039,0.390384,0.100379,0.501864,0.301961,0.396467,0.102902,0.502658,0.305882,0.402548,0.10542,0.503386,0.309804,0.408629,0.10793,0.504052,0.313725,0.414709,0.110431,0.504662,0.317647,0.420791,0.11292,0.505215,0.321569,0.426877,0.115395,0.505714,0.32549,0.432967,0.117855,0.50616,0.329412,0.439062,0.120298,0.506555,0.333333,0.445163,0.122724,0.506901,0.337255,0.451271,0.125132,0.507198,0.341176,0.457386,0.127522,0.507448,0.345098,0.463508,0.129893,0.507652,0.34902,0.46964,0.132245,0.507809,0.352941,0.47578,0.134577,0.507921,0.356863,0.481929,0.136891,0.507989,0.360784,0.488088,0.139186,0.508011,0.364706,0.494258,0.141462,0.507988,0.368627,0.500438,0.143719,0.50792,0.372549,0.506629,0.145958,0.507806,0.376471,0.512831,0.148179,0.507648,0.380392,0.519045,0.150383,0.507443,0.384314,0.52527,0.152569,0.507192,0.388235,0.531507,0.154739,0.506895,0.392157,0.537755,0.156894,0.506551,0.396078,0.544015,0.159033,0.506159,0.4,0.550287,0.161158,0.505719,0.403922,0.556571,0.163269,0.50523,0.407843,0.562866,0.165368,0.504692,0.411765,0.569172,0.167454,0.504105,0.415686,0.57549,0.16953,0.503466,0.419608,0.581819,0.171596,0.502777,0.423529,0.588158,0.173652,0.502035,0.427451,0.594508,0.175701,0.501241,0.431373,0.600868,0.177743,0.500394,0.435294,0.607238,0.179779,0.499492,0.439216,0.613617,0.181811,0.498536,0.443137,0.620005,0.18384,0.497524,0.447059,0.626401,0.185867,0.496456,0.45098,0.632805,0.187893,0.495332,0.454902,0.639216,0.189921,0.49415,0.458824,0.645633,0.191952,0.49291,0.462745,0.652056,0.193986,0.491611,0.466667,0.658483,0.196027,0.490253,0.470588,0.664915,0.198075,0.488836,0.47451,0.671349,0.200133,0.487358,0.478431,0.677786,0.202203,0.485819,0.482353,0.684224,0.204286,0.484219,0.486275,0.690661,0.206384,0.482558,0.490196,0.697098,0.208501,0.480835,0.494118,0.703532,0.210638,0.479049,0.498039,0.709962,0.212797,0.477201,0.501961,0.716387,0.214982,0.47529,0.505882,0.722805,0.217194,0.473316,0.509804,0.729216,0.219437,0.471279,0.513725,0.735616,0.221713,0.46918,0.517647,0.742004,0.224025,0.467018,0.521569,0.748378,0.226377,0.464794,0.52549,0.754737,0.228772,0.462509,0.529412,0.761077,0.231214,0.460162,0.533333,0.767398,0.233705,0.457755,0.537255,0.773695,0.236249,0.455289,0.541176,0.779968,0.238851,0.452765,0.545098,0.786212,0.241514,0.450184,0.54902,0.792427,0.244242,0.447543,0.552941,0.798608,0.24704,0.444848,0.556863,0.804752,0.249911,0.442102,0.560784,0.810855,0.252861,0.439305,0.564706,0.816914,0.255895,0.436461,0.568627,0.822926,0.259016,0.433573,0.572549,0.828886,0.262229,0.430644,0.576471,0.834791,0.26554,0.427671,0.580392,0.840636,0.268953,0.424666,0.584314,0.846416,0.272473,0.421631,0.588235,0.852126,0.276106,0.418573,0.592157,0.857763,0.279857,0.415496,0.596078,0.86332,0.283729,0.412403,0.6,0.868793,0.287728,0.409303,0.603922,0.874176,0.291859,0.406205,0.607843,0.879464,0.296125,0.403118,0.611765,0.884651,0.30053,0.400047,0.615686,0.889731,0.305079,0.397002,0.619608,0.8947,0.309773,0.393995,0.623529,0.899552,0.314616,0.391037,0.627451,0.904281,0.31961,0.388137,0.631373,0.908884,0.324755,0.385308,0.635294,0.913354,0.330052,0.382563,0.639216,0.917689,0.3355,0.379915,0.643137,0.921884,0.341098,0.377376,0.647059,0.925937,0.346844,0.374959,0.65098,0.929845,0.352734,0.372677,0.654902,0.933606,0.358764,0.370541,0.658824,0.937221,0.364929,0.368567,0.662745,0.940687,0.371224,0.366762,0.666667,0.944006,0.377643,0.365136,0.670588,0.94718,0.384178,0.363701,0.67451,0.95021,0.39082,0.362468,0.678431,0.953099,0.397563,0.361438,0.682353,0.955849,0.4044,0.360619,0.686275,0.958464,0.411324,0.360014,0.690196,0.960949,0.418323,0.35963,0.694118,0.96331,0.42539,0.359469,0.698039,0.965549,0.432519,0.359529,0.701961,0.967671,0.439703,0.35981,0.705882,0.96968,0.446936,0.360311,0.709804,0.971582,0.45421,0.36103,0.713725,0.973381,0.46152,0.361965,0.717647,0.975082,0.468861,0.363111,0.721569,0.97669,0.476226,0.364466,0.72549,0.97821,0.483612,0.366025,0.729412,0.979645,0.491014,0.367783,0.733333,0.981,0.498428,0.369734,0.737255,0.982279,0.505851,0.371874,0.741176,0.983485,0.51328,0.374198,0.745098,0.984622,0.520713,0.376698,0.74902,0.985693,0.528148,0.379371,0.752941,0.9867,0.535582,0.38221,0.756863,0.987646,0.543015,0.38521,0.760784,0.988533,0.550446,0.388365,0.764706,0.989363,0.557873,0.391671,0.768627,0.990138,0.565296,0.395122,0.772549,0.990871,0.572706,0.398714,0.776471,0.991558,0.580107,0.402441,0.780392,0.992196,0.587502,0.406299,0.784314,0.992785,0.594891,0.410283,0.788235,0.993326,0.602275,0.41439,0.792157,0.993834,0.609644,0.418613,0.796078,0.994309,0.616999,0.42295,0.8,0.994738,0.62435,0.427397,0.803922,0.995122,0.631696,0.431951,0.807843,0.99548,0.639027,0.436607,0.811765,0.99581,0.646344,0.441361,0.815686,0.996096,0.653659,0.446213,0.819608,0.996341,0.660969,0.45116,0.823529,0.99658,0.668256,0.456192,0.827451,0.996775,0.675541,0.461314,0.831373,0.996925,0.682828,0.466526,0.835294,0.997077,0.690088,0.471811,0.839216,0.997186,0.697349,0.477182,0.843137,0.997254,0.704611,0.482635,0.847059,0.997325,0.711848,0.488154,0.85098,0.997351,0.719089,0.493755,0.854902,0.997351,0.726324,0.499428,0.858824,0.997341,0.733545,0.505167,0.862745,0.997285,0.740772,0.510983,0.866667,0.997228,0.747981,0.516859,0.870588,0.997138,0.75519,0.522806,0.87451,0.997019,0.762398,0.528821,0.878431,0.996898,0.769591,0.534892,0.882353,0.996727,0.776795,0.541039,0.886275,0.996571,0.783977,0.547233,0.890196,0.996369,0.791167,0.553499,0.894118,0.996162,0.798348,0.55982,0.898039,0.995932,0.805527,0.566202,0.901961,0.99568,0.812706,0.572645,0.905882,0.995424,0.819875,0.57914,0.909804,0.995131,0.827052,0.585701,0.913725,0.994851,0.834213,0.592307,0.917647,0.994524,0.841387,0.598983,0.921569,0.994222,0.84854,0.605696,0.92549,0.993866,0.855711,0.612482,0.929412,0.993545,0.862859,0.619299,0.933333,0.99317,0.870024,0.626189,0.937255,0.992831,0.877168,0.633109,0.941176,0.99244,0.88433,0.640099,0.945098,0.992089,0.89147,0.647116,0.94902,0.991688,0.898627,0.654202,0.952941,0.991332,0.905763,0.661309,0.956863,0.99093,0.912915,0.668481,0.960784,0.99057,0.920049,0.675675,0.964706,0.990175,0.927196,0.682926,0.968627,0.989815,0.934329,0.690198,0.972549,0.989434,0.94147,0.697519,0.976471,0.989077,0.948604,0.704863,0.980392,0.988717,0.955742,0.712242,0.984314,0.988367,0.962878,0.719649,0.988235,0.988033,0.970012,0.727077,0.992157,0.987691,0.977154,0.734536,0.996078,0.987387,0.984288,0.742002,1.0,0.987053,0.991438,0.749504]},"Plasma (matplotlib)":{"ColorSpace":"Diverging","Name":"Plasma (matplotlib)","NanColor":[0,1,0],"Source":"https://github.com/BIDS/colormap/blob/master/colormaps.py","License":"CC0","Creator":"Nathaniel J. Smith & Stefan van der Walt","RGBPoints":[0.0,0.050383,0.029803,0.527975,0.003922,0.063536,0.028426,0.533124,0.007843,0.075353,0.027206,0.538007,0.011765,0.086222,0.026125,0.542658,0.015686,0.096379,0.025165,0.547103,0.019608,0.10598,0.024309,0.551368,0.023529,0.115124,0.023556,0.555468,0.027451,0.123903,0.022878,0.559423,0.031373,0.132381,0.022258,0.56325,0.035294,0.140603,0.021687,0.566959,0.039216,0.148607,0.021154,0.570562,0.043137,0.156421,0.020651,0.574065,0.047059,0.16407,0.020171,0.577478,0.05098,0.171574,0.019706,0.580806,0.054902,0.17895,0.019252,0.584054,0.058824,0.186213,0.018803,0.587228,0.062745,0.193374,0.018354,0.59033,0.066667,0.200445,0.017902,0.593364,0.070588,0.207435,0.017442,0.596333,0.07451,0.21435,0.016973,0.599239,0.078431,0.221197,0.016497,0.602083,0.082353,0.227983,0.016007,0.604867,0.086275,0.234715,0.015502,0.607592,0.090196,0.241396,0.014979,0.610259,0.094118,0.248032,0.014439,0.612868,0.098039,0.254627,0.013882,0.615419,0.101961,0.261183,0.013308,0.617911,0.105882,0.267703,0.012716,0.620346,0.109804,0.274191,0.012109,0.622722,0.113725,0.280648,0.011488,0.625038,0.117647,0.287076,0.010855,0.627295,0.121569,0.293478,0.010213,0.62949,0.12549,0.299855,0.009561,0.631624,0.129412,0.30621,0.008902,0.633694,0.133333,0.312543,0.008239,0.6357,0.137255,0.318856,0.007576,0.63764,0.141176,0.32515,0.006915,0.639512,0.145098,0.331426,0.006261,0.641316,0.14902,0.337683,0.005618,0.643049,0.152941,0.343925,0.004991,0.64471,0.156863,0.35015,0.004382,0.646298,0.160784,0.356359,0.003798,0.64781,0.164706,0.362553,0.003243,0.649245,0.168627,0.368733,0.002724,0.650601,0.172549,0.374897,0.002245,0.651876,0.176471,0.381047,0.001814,0.653068,0.180392,0.387183,0.001434,0.654177,0.184314,0.393304,0.001114,0.655199,0.188235,0.399411,0.000859,0.656133,0.192157,0.405503,0.000678,0.656977,0.196078,0.41158,0.000577,0.65773,0.2,0.417642,0.000564,0.65839,0.203922,0.423689,0.000646,0.658956,0.207843,0.429719,0.000831,0.659425,0.211765,0.435734,0.001127,0.659797,0.215686,0.441732,0.00154,0.660069,0.219608,0.447714,0.00208,0.66024,0.223529,0.453677,0.002755,0.66031,0.227451,0.459623,0.003574,0.660277,0.231373,0.46555,0.004545,0.660139,0.235294,0.471457,0.005678,0.659897,0.239216,0.477344,0.00698,0.659549,0.243137,0.48321,0.00846,0.659095,0.247059,0.489055,0.010127,0.658534,0.25098,0.494877,0.01199,0.657865,0.254902,0.500678,0.014055,0.657088,0.258824,0.506454,0.016333,0.656202,0.262745,0.512206,0.018833,0.655209,0.266667,0.517933,0.021563,0.654109,0.270588,0.523633,0.024532,0.652901,0.27451,0.529306,0.027747,0.651586,0.278431,0.534952,0.031217,0.650165,0.282353,0.54057,0.03495,0.64864,0.286275,0.546157,0.038954,0.64701,0.290196,0.551715,0.043136,0.645277,0.294118,0.557243,0.047331,0.643443,0.298039,0.562738,0.051545,0.641509,0.301961,0.568201,0.055778,0.639477,0.305882,0.573632,0.060028,0.637349,0.309804,0.579029,0.064296,0.635126,0.313725,0.584391,0.068579,0.632812,0.317647,0.589719,0.072878,0.630408,0.321569,0.595011,0.07719,0.627917,0.32549,0.600266,0.081516,0.625342,0.329412,0.605485,0.085854,0.622686,0.333333,0.610667,0.090204,0.619951,0.337255,0.615812,0.094564,0.61714,0.341176,0.620919,0.098934,0.614257,0.345098,0.625987,0.103312,0.611305,0.34902,0.631017,0.107699,0.608287,0.352941,0.636008,0.112092,0.605205,0.356863,0.640959,0.116492,0.602065,0.360784,0.645872,0.120898,0.598867,0.364706,0.650746,0.125309,0.595617,0.368627,0.65558,0.129725,0.592317,0.372549,0.660374,0.134144,0.588971,0.376471,0.665129,0.138566,0.585582,0.380392,0.669845,0.142992,0.582154,0.384314,0.674522,0.147419,0.578688,0.388235,0.67916,0.151848,0.575189,0.392157,0.683758,0.156278,0.57166,0.396078,0.688318,0.160709,0.568103,0.4,0.69284,0.165141,0.564522,0.403922,0.697324,0.169573,0.560919,0.407843,0.701769,0.174005,0.557296,0.411765,0.706178,0.178437,0.553657,0.415686,0.710549,0.182868,0.550004,0.419608,0.714883,0.187299,0.546338,0.423529,0.719181,0.191729,0.542663,0.427451,0.723444,0.196158,0.538981,0.431373,0.72767,0.200586,0.535293,0.435294,0.731862,0.205013,0.531601,0.439216,0.736019,0.209439,0.527908,0.443137,0.740143,0.213864,0.524216,0.447059,0.744232,0.218288,0.520524,0.45098,0.748289,0.222711,0.516834,0.454902,0.752312,0.227133,0.513149,0.458824,0.756304,0.231555,0.509468,0.462745,0.760264,0.235976,0.505794,0.466667,0.764193,0.240396,0.502126,0.470588,0.76809,0.244817,0.498465,0.47451,0.771958,0.249237,0.494813,0.478431,0.775796,0.253658,0.491171,0.482353,0.779604,0.258078,0.487539,0.486275,0.783383,0.2625,0.483918,0.490196,0.787133,0.266922,0.480307,0.494118,0.790855,0.271345,0.476706,0.498039,0.794549,0.27577,0.473117,0.501961,0.798216,0.280197,0.469538,0.505882,0.801855,0.284626,0.465971,0.509804,0.805467,0.289057,0.462415,0.513725,0.809052,0.293491,0.45887,0.517647,0.812612,0.297928,0.455338,0.521569,0.816144,0.302368,0.451816,0.52549,0.819651,0.306812,0.448306,0.529412,0.823132,0.311261,0.444806,0.533333,0.826588,0.315714,0.441316,0.537255,0.830018,0.320172,0.437836,0.541176,0.833422,0.324635,0.434366,0.545098,0.836801,0.329105,0.430905,0.54902,0.840155,0.33358,0.427455,0.552941,0.843484,0.338062,0.424013,0.556863,0.846788,0.342551,0.420579,0.560784,0.850066,0.347048,0.417153,0.564706,0.853319,0.351553,0.413734,0.568627,0.856547,0.356066,0.410322,0.572549,0.85975,0.360588,0.406917,0.576471,0.862927,0.365119,0.403519,0.580392,0.866078,0.36966,0.400126,0.584314,0.869203,0.374212,0.396738,0.588235,0.872303,0.378774,0.393355,0.592157,0.875376,0.383347,0.389976,0.596078,0.878423,0.387932,0.3866,0.6,0.881443,0.392529,0.383229,0.603922,0.884436,0.397139,0.37986,0.607843,0.887402,0.401762,0.376494,0.611765,0.89034,0.406398,0.37313,0.615686,0.89325,0.411048,0.369768,0.619608,0.896131,0.415712,0.366407,0.623529,0.898984,0.420392,0.363047,0.627451,0.901807,0.425087,0.359688,0.631373,0.904601,0.429797,0.356329,0.635294,0.907365,0.434524,0.35297,0.639216,0.910098,0.439268,0.34961,0.643137,0.9128,0.444029,0.346251,0.647059,0.915471,0.448807,0.34289,0.65098,0.918109,0.453603,0.339529,0.654902,0.920714,0.458417,0.336166,0.658824,0.923287,0.463251,0.332801,0.662745,0.925825,0.468103,0.329435,0.666667,0.928329,0.472975,0.326067,0.670588,0.930798,0.477867,0.322697,0.67451,0.933232,0.48278,0.319325,0.678431,0.93563,0.487712,0.315952,0.682353,0.93799,0.492667,0.312575,0.686275,0.940313,0.497642,0.309197,0.690196,0.942598,0.502639,0.305816,0.694118,0.944844,0.507658,0.302433,0.698039,0.947051,0.512699,0.299049,0.701961,0.949217,0.517763,0.295662,0.705882,0.951344,0.52285,0.292275,0.709804,0.953428,0.52796,0.288883,0.713725,0.95547,0.533093,0.28549,0.717647,0.957469,0.53825,0.282096,0.721569,0.959424,0.543431,0.278701,0.72549,0.961336,0.548636,0.275305,0.729412,0.963203,0.553865,0.271909,0.733333,0.965024,0.559118,0.268513,0.737255,0.966798,0.564396,0.265118,0.741176,0.968526,0.5697,0.261721,0.745098,0.970205,0.575028,0.258325,0.74902,0.971835,0.580382,0.254931,0.752941,0.973416,0.585761,0.25154,0.756863,0.974947,0.591165,0.248151,0.760784,0.976428,0.596595,0.244767,0.764706,0.977856,0.602051,0.241387,0.768627,0.979233,0.607532,0.238013,0.772549,0.980556,0.613039,0.234646,0.776471,0.981826,0.618572,0.231287,0.780392,0.983041,0.624131,0.227937,0.784314,0.984199,0.629718,0.224595,0.788235,0.985301,0.63533,0.221265,0.792157,0.986345,0.640969,0.217948,0.796078,0.987332,0.646633,0.214648,0.8,0.98826,0.652325,0.211364,0.803922,0.989128,0.658043,0.2081,0.807843,0.989935,0.663787,0.204859,0.811765,0.990681,0.669558,0.201642,0.815686,0.991365,0.675355,0.198453,0.819608,0.991985,0.681179,0.195295,0.823529,0.992541,0.68703,0.19217,0.827451,0.993032,0.692907,0.189084,0.831373,0.993456,0.69881,0.186041,0.835294,0.993814,0.704741,0.183043,0.839216,0.994103,0.710698,0.180097,0.843137,0.994324,0.716681,0.177208,0.847059,0.994474,0.722691,0.174381,0.85098,0.994553,0.728728,0.171622,0.854902,0.994561,0.734791,0.168938,0.858824,0.994495,0.74088,0.166335,0.862745,0.994355,0.746995,0.163821,0.866667,0.994141,0.753137,0.161404,0.870588,0.993851,0.759304,0.159092,0.87451,0.993482,0.765499,0.156891,0.878431,0.993033,0.77172,0.154808,0.882353,0.992505,0.777967,0.152855,0.886275,0.991897,0.784239,0.151042,0.890196,0.991209,0.790537,0.149377,0.894118,0.990439,0.796859,0.14787,0.898039,0.989587,0.803205,0.146529,0.901961,0.988648,0.809579,0.145357,0.905882,0.987621,0.815978,0.144363,0.909804,0.986509,0.822401,0.143557,0.913725,0.985314,0.828846,0.142945,0.917647,0.984031,0.835315,0.142528,0.921569,0.982653,0.841812,0.142303,0.92549,0.98119,0.848329,0.142279,0.929412,0.979644,0.854866,0.142453,0.933333,0.977995,0.861432,0.142808,0.937255,0.976265,0.868016,0.143351,0.941176,0.974443,0.874622,0.144061,0.945098,0.97253,0.88125,0.144923,0.94902,0.970533,0.887896,0.145919,0.952941,0.968443,0.894564,0.147014,0.956863,0.966271,0.901249,0.14818,0.960784,0.964021,0.90795,0.14937,0.964706,0.961681,0.914672,0.15052,0.968627,0.959276,0.921407,0.151566,0.972549,0.956808,0.928152,0.152409,0.976471,0.954287,0.934908,0.152921,0.980392,0.951726,0.941671,0.152925,0.984314,0.949151,0.948435,0.152178,0.988235,0.946602,0.95519,0.150328,0.992157,0.944152,0.961916,0.146861,0.996078,0.941896,0.96859,0.140956,1.0,0.940015,0.975158,0.131326]},"Green Linear (9_17f)":{"Name":"Green Linear (9_17f)","ColorSpace":"Lab","RGBPoints":[0.0,1.0,0.988235,0.968627,0.02,1.0,0.952941,0.878431,0.05,0.968627,0.905882,0.776471,0.1,0.94902,0.898039,0.647059,0.15,0.901961,0.878431,0.556863,0.2,0.847059,0.858824,0.482353,0.25,0.690196,0.819608,0.435294,0.3,0.513725,0.768627,0.384314,0.35,0.337255,0.721569,0.337255,0.4,0.278431,0.658824,0.392157,0.45,0.231373,0.639216,0.435294,0.5,0.203922,0.6,0.486275,0.55,0.172549,0.568627,0.537255,0.6,0.141176,0.517647,0.54902,0.65,0.133333,0.458824,0.541176,0.7,0.12549,0.396078,0.529412,0.75,0.117647,0.321569,0.521569,0.8,0.121569,0.258824,0.509804,0.85,0.133333,0.227451,0.501961,0.9,0.145098,0.192157,0.490196,0.95,0.188235,0.164706,0.470588,1.0,0.258824,0.196078,0.439216],"Creator":"Francesca Samsel","Source":"https://www.paraview.org/paraview-docs/v5.9.1/python/_modules/paraview/_colorMaps.html"}}')
STREAM_COLOR_MAPS=tuple(dict.fromkeys(STREAM_COLOR_MAPS+tuple(PARAVIEW_COLOR_PRESETS)))

# The preset records and BSD notice above are bundled from Kitware/ParaView.
_STREAM_PRESET_CACHE = {}
_STREAM_PRESET_NAMES = {name.casefold(): name for name in PARAVIEW_COLOR_PRESETS}


def paraview_color_map(name):
    from matplotlib.colors import ListedColormap
    if name in _STREAM_PRESET_CACHE:return _STREAM_PRESET_CACHE[name]
    import vtk
    preset=PARAVIEW_COLOR_PRESETS[name]
    points=np.asarray(preset['RGBPoints'],dtype=float).reshape(-1,4)
    lo,hi=points[0,0],points[-1,0]
    if hi<=lo:raise ValueError('Invalid preset range: '+name)
    transfer=vtk.vtkColorTransferFunction()
    space=preset.get('ColorSpace','RGB')
    setter={'RGB':'SetColorSpaceToRGB','Lab':'SetColorSpaceToLab','CIELAB':'SetColorSpaceToLab',
            'Diverging':'SetColorSpaceToDiverging','HSV':'SetColorSpaceToHSV','Step':'SetColorSpaceToStep'}[space]
    getattr(transfer,setter)()
    for value,r,g,b in points:transfer.AddRGBPoint((value-lo)/(hi-lo),r,g,b)
    # Lab interpolation can leave the RGB gamut; renderers clamp those colors.
    table=np.clip(np.array([transfer.GetColor(float(t)) for t in np.linspace(0,1,1025)]),0,1)
    cmap=ListedColormap(table,name='ParaView: '+name)
    cmap.set_bad(preset.get('NanColor',(1,1,0)))
    cmap.set_under(table[0]);cmap.set_over(table[-1])
    _STREAM_PRESET_CACHE[name]=cmap
    return cmap


class StreamPresetPicker(tk.Toplevel):
    """Searchable offline preset gallery with actual transfer-function previews."""
    def __init__(self,parent,current,apply):
        super().__init__(parent)
        self.title('Choose color preset');self.transient(parent);self.geometry('620x530');self.minsize(420,300)
        self.apply_callback=apply;self.names=list(STREAM_COLOR_MAPS);self.photos={};self.current=current
        self.search=tk.StringVar();self.label=tk.StringVar()
        top=ttk.Frame(self,padding=8);top.pack(fill='x')
        ttk.Label(top,text='Search').pack(side='left',padx=(0,8))
        search=ttk.Entry(top,textvariable=self.search);search.pack(side='left',fill='x',expand=True)
        body=ttk.Frame(self,padding=(8,0,8,0));body.pack(fill='both',expand=True)
        ttk.Style(self).configure('MuMaxPreset.Treeview',rowheight=27)
        self.tree=ttk.Treeview(body,columns=('name',),show='tree headings',selectmode='browse',style='MuMaxPreset.Treeview')
        self.tree.heading('#0',text='Preview');self.tree.column('#0',width=150,minwidth=100,stretch=False)
        self.tree.heading('name',text='Preset');self.tree.column('name',width=380,minwidth=160)
        scroll=ttk.Scrollbar(body,orient='vertical',command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set);scroll.pack(side='right',fill='y');self.tree.pack(fill='both',expand=True)
        self.tree.bind('<<TreeviewSelect>>',self.show_selected)
        self.tree.bind('<Double-1>',lambda event:self.apply_selected())
        preview=ttk.Frame(self,padding=8);preview.pack(fill='x')
        ttk.Label(preview,textvariable=self.label).pack(anchor='center')
        self.preview=ttk.Label(preview);self.preview.pack(fill='x',pady=4)
        buttons=ttk.Frame(self,padding=8);buttons.pack(fill='x')
        ttk.Button(buttons,text='Apply',command=self.apply_selected).pack(side='right')
        ttk.Button(buttons,text='Cancel',command=self.destroy).pack(side='right',padx=6)
        self.bind('<Escape>',lambda event:self.destroy());self.bind('<Return>',lambda event:self.apply_selected())
        self.search.trace_add('write',lambda *args:self.populate());self.populate();search.focus_set()

    def preview_image(self,name,width,height):
        color=stream_color_map(name)(np.linspace(0,1,width))[:,:3]
        row=np.uint8(np.rint(color*255))[None,:,:]
        return Image.fromarray(np.repeat(row,height,axis=0))

    def populate(self):
        selected=self.selected_name() or self.current
        self.tree.delete(*self.tree.get_children())
        query=self.search.get().strip().casefold();matches=[name for name in self.names if query in name.casefold()]
        for name in matches:
            if name not in self.photos:self.photos[name]=ImageTk.PhotoImage(self.preview_image(name,140,19),master=self)
            self.tree.insert('', 'end',iid=name,image=self.photos[name],values=(name,))
        if matches:
            name=selected if selected in matches else matches[0];self.tree.selection_set(name);self.tree.see(name)
        self.show_selected()

    def selected_name(self):
        selection=self.tree.selection();return selection[0] if selection else None

    def show_selected(self,event=None):
        name=self.selected_name();self.label.set(name or 'No matching presets')
        if name:
            self.preview_photo=ImageTk.PhotoImage(self.preview_image(name,560,30),master=self)
            self.preview.configure(image=self.preview_photo)
        else:self.preview.configure(image='')

    def apply_selected(self):
        name=self.selected_name()
        if name:self.apply_callback(name);self.destroy()



def show_shared_color_presets(parent,owner,variable,changed):
    picker=getattr(owner,'color_preset_picker',None)
    if picker is not None and picker.winfo_exists():
        picker.lift();picker.focus_set();return picker
    def apply(name):
        variable.set(name)
        changed()
    return StreamPresetPicker(parent,variable.get(),apply)


def stream_number(value,decimals=True):
    return format(float(value),'.3f' if decimals else '.0f')


def stream_color_map(value):
    from matplotlib import colormaps
    from matplotlib.colors import to_rgba
    value=value.strip()
    if value in PARAVIEW_COLOR_PRESETS:return paraview_color_map(value)
    if ',' not in value:
        try:return colormaps[value]
        except KeyError:
            name=_STREAM_PRESET_NAMES.get(value.casefold())
            if name:return paraview_color_map(name)
            raise ValueError('Choose a color preset or enter three comma-separated colors.')
    names=[name.strip() for name in value.split(',')]
    if len(names)!=3:raise ValueError('Enter three comma-separated colors.')
    colors=[]
    for name in names:
        if name.lower() in MUMAX_RGB:colors.append(np.array(MUMAX_RGB[name.lower()])/255.)
        else:
            try:colors.append(to_rgba(name))
            except ValueError:raise ValueError('Unknown color: '+name)
    return LinearSegmentedColormap.from_list('m_components',colors,N=257)


def stream_scalar_rgb(values,colors,low,high):
    cmap=stream_color_map(colors)
    normalized=np.clip((values[::-1].astype(float)-low)/(high-low),0,1)
    return Image.fromarray(np.uint8(np.rint(cmap(normalized)[:,:,:3]*255)))


def configure_mumax_vtk_cpu():
    """Select the compiled CPU backend once, before viewer filter workers exist."""
    global _MUMAX_VTK_CPU
    if _MUMAX_VTK_CPU is not None:return _MUMAX_VTK_CPU
    require_tab4_backend()
    import vtk
    if vtk.vtkSMPTools.GetBackend()=='Sequential':vtk.vtkSMPTools.SetBackend('STDThread')
    vtk.vtkSMPTools.Initialize(max(1,min(8,(os.cpu_count() or 2)-1)))
    _MUMAX_VTK_CPU=(vtk.vtkSMPTools.GetBackend(),vtk.vtkSMPTools.GetEstimatedNumberOfThreads())
    return _MUMAX_VTK_CPU


def refine_stream_volume(vectors,mesh,factor,roi):
    """Refine only the ROI plus interpolation halo, exactly matching full-volume refinement."""
    if roi is None:return refine_vector_grid(vectors,mesh,factor)
    cropped_mesh=dict(mesh);bounds={}
    for axis,count in zip('xy',(vectors.shape[2],vectors.shape[1])):
        if mesh[axis]['unit']!=roi['unit']:raise ValueError('ROI and field use different coordinate units.')
        low,high=mesh[axis]['edges'];step=(high-low)/count
        centers=low+(np.arange(count*factor)+.5)*step/factor
        first,last=np.searchsorted(centers,roi[axis],side='left')
        first,last=max(0,int(first)),min(count*factor,int(last))
        if last<=first:raise ValueError('The selected ROI contains no samples.')
        begin=max(0,int(math.floor((first+.5)/factor-.5)))
        end=min(count,int(math.ceil((last-.5)/factor-.5))+1)
        bounds[axis]=(begin,end)
        cropped_mesh[axis]=dict(mesh[axis],centers=mesh[axis]['centers'][begin:end],edges=(low+begin*step,low+end*step))
    patch=vectors[:,slice(*bounds['y']),slice(*bounds['x']),:]
    refined,refined_mesh=refine_vector_grid(patch,cropped_mesh,factor)
    return crop_stream_roi(refined,refined_mesh,roi)


def magnetization_vtk_grid(vectors,mesh):
    pv = require_tab4_backend()
    nz,ny,nx = vectors.shape[:3]
    if min(nx,ny,nz)<2:
        raise ValueError('3D analysis requires two samples per axis. Increase upscaling for a one-layer file.')
    spacing = [(mesh[a]['edges'][1]-mesh[a]['edges'][0])/n for a,n in zip('xyz',(nx,ny,nz))]
    grid = pv.ImageData(dimensions=(nx,ny,nz),spacing=spacing,origin=[mesh[a]['centers'][0] for a in 'xyz'])
    values = np.ascontiguousarray(vectors.reshape(-1,3),dtype=np.float32)
    grid.point_data['m'] = values
    for index,key in enumerate(('mx','my','mz')):grid.point_data[key] = values[:,index]
    magnitude = np.linalg.norm(values,axis=1)
    grid.point_data['mag'] = magnitude
    grid.set_active_vectors('m')
    material = grid
    if np.any(magnitude<=1e-8):
        grid.point_data['material'] = (magnitude>1e-8).astype(np.uint8)
        material = grid.threshold(.5,scalars='material',preference='point',all_scalars=True)
    return grid,material,spacing


def streamline_seed_points(bounds,count,seed_plane='Volume',seed_type='Uniform grid',spacing=None,
                           center=None,radius=None,line_start=None,line_end=None,z_layers=None,section=None,grid_center=None):
    """Seed counts are exact; automatic grids follow physical aspect, including thin films."""
    bounds = np.asarray(bounds,dtype=float).reshape(3,2)
    spans = bounds[:,1]-bounds[:,0]
    normal_axis = None
    position = None
    if section and section['mode']=='Plane':
        normal_axis = {'XY':2,'YZ':0,'XZ':1}[section['view']]
        position = section['position']
    elif seed_plane!='Volume':
        normal_axis = 'XYZ'.index(seed_plane)
        position = float(np.mean(bounds[normal_axis]))
    active = [a for a in range(3) if a!=normal_axis and spans[a]>1e-12]
    if not active:return np.empty((0,3),float)
    mid = np.mean(bounds,axis=1)
    if grid_center is not None and normal_axis is None:
        mid=np.clip(np.asarray(grid_center,float),bounds[:,0],bounds[:,1])
    if normal_axis is not None:mid[normal_axis] = position
    if seed_type=='Line':
        axis = max(active,key=lambda a:spans[a])
        start = mid.copy() if line_start is None else np.array(line_start,float)
        end = mid.copy() if line_end is None else np.array(line_end,float)
        if line_start is None:start[axis] = bounds[axis,0]+spans[axis]*.05
        if line_end is None:end[axis] = bounds[axis,1]-spans[axis]*.05
        if normal_axis is not None:start[normal_axis] = end[normal_axis] = position
        if np.linalg.norm(end-start)<1e-12:raise ValueError('Seed line endpoints must differ.')
        points = np.linspace(start,end,count)
    elif seed_type=='Point cloud':
        origin = mid.copy() if center is None else np.array(center,float)
        if normal_axis is not None:origin[normal_axis] = position
        size = min(spans[active])*.25 if radius is None else radius
        rng = np.random.default_rng(0)
        directions = rng.normal(size=(count,len(active)))
        directions /= np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-20)
        offsets = directions*(rng.random(count)**(1/len(active))*size)[:,None]
        points = np.tile(origin,(count,1));points[:,active] += offsets
    else:
        lengths = spans*.9
        counts = np.ones(3,dtype=int)
        if spacing is not None:
            for a in active:counts[a] = max(1,int(np.floor(lengths[a]/spacing))+1)
        else:
            fixed_z = z_layers if normal_axis is None and z_layers is not None else None
            free = [a for a in active if not (a==2 and fixed_z is not None)]
            if fixed_z is not None:counts[2] = fixed_z
            target = max(1,count/int(np.prod(counts)))
            if free:
                low,high = 1e-12,max(lengths[free])*2+1
                for _ in range(80):
                    pitch = (low+high)/2
                    product = np.prod([max(1,int(np.ceil(lengths[a]/pitch))) for a in free])
                    if product>=target:low = pitch
                    else:high = pitch
                for a in free:counts[a] = max(1,int(np.ceil(lengths[a]/max(low,1e-12))))
        total = int(np.prod(counts))
        if total>100_000:raise ValueError('The requested spacing produces more than 100000 seeds. Increase spacing or reduce Z layers.')
        axes = []
        for a in range(3):
            if a==normal_axis:axes.append(np.array([position]))
            elif counts[a]==1:axes.append(np.array([mid[a]]))
            elif spacing is not None:
                axes.append(mid[a]+(np.arange(counts[a])-(counts[a]-1)/2)*spacing)
            else:axes.append(np.linspace(bounds[a,0]+spans[a]*.05,bounds[a,1]-spans[a]*.05,counts[a]))
        points = np.column_stack([p.ravel() for p in np.meshgrid(*axes,indexing='ij')])
        if spacing is None and len(points)>count:
            points = points[np.linspace(0,len(points)-1,count,dtype=int)]
    if normal_axis is not None:points[:,normal_axis] = position
    valid = np.all((points>=bounds[:,0]-1e-8)&(points<=bounds[:,1]+1e-8),axis=1)
    return np.ascontiguousarray(points[valid])


def material_seed_source(domain,points,count,refill=False):
    """Drop vacuum seeds; automatic grid counts are filled from valid material cells."""
    pv=require_tab4_backend()
    if not len(points) or not domain.n_cells:return pv.PolyData()
    source=pv.PolyData(points)
    sampled=source.sample(domain)
    points=points[np.asarray(sampled['vtkValidPointMask'],bool)]
    if refill and len(points)<count:
        centers=np.asarray(domain.cell_centers(vertex=False).points)
        if len(centers):
            needed=count-len(points)
            # Select throughout the valid cells rather than clustering near the first cells.
            indexes=np.linspace(0,len(centers)-1,min(len(centers),max(needed*2,count)),dtype=int)
            extra=centers[indexes]
            if len(points):
                combined=np.vstack((points,extra));_,indices=np.unique(combined,axis=0,return_index=True)
                points=combined[np.sort(indices)][:count]
            else:points=extra[:count]
            if len(points)<count:
                # Distinct convex combinations of vertices avoid duplicated seeds
                # when the requested count exceeds the number of material cells.
                rng=np.random.default_rng(1);extra=[]
                for index in range(count-len(points)):
                    vertices=np.asarray(domain.get_cell(index%domain.n_cells).points)
                    weight=rng.random(len(vertices));weight/=weight.sum()
                    extra.append(weight@vertices)
                candidate=pv.PolyData(np.asarray(extra))
                valid=np.asarray(candidate.sample(domain)['vtkValidPointMask'],bool)
                points=np.vstack((points,candidate.points[valid]))
    return pv.PolyData(np.ascontiguousarray(points))


def vtk_material_center(dataset):
    # Implicit ImageData coordinates are uniform and need no explicit point array.
    if hasattr(dataset,'center_of_mass'):return dataset.center_of_mass()
    return np.asarray(dataset.center)


def clip_section_xy(dataset,xy_bounds,normal_axis=None):
    """Clip tangential coordinates without reducing a plane to a one-cell VTK grid."""
    output=dataset
    for axis in 'xy':
        index='xyz'.index(axis)
        if index==normal_axis or not output.n_cells:continue
        low,high=xy_bounds[axis];bounds=np.asarray(output.bounds).reshape(3,2)
        normal=np.eye(3)[index];origin=np.mean(bounds,axis=1)
        if low>bounds[index,0]+1e-9:
            origin[index]=low;output=output.clip(normal=normal,origin=origin,invert=False)
        if high<bounds[index,1]-1e-9 and output.n_cells:
            origin[index]=high;output=output.clip(normal=normal,origin=origin,invert=True)
    return output


def build_stream_iso_scene(vectors,mesh,component,levels,seed_count,seed_plane,streams,iso,
                           options=None,section=None,prepared=None):
    pv = require_tab4_backend();options = options or {}
    grid,material,spacing = prepared if prepared is not None else magnetization_vtk_grid(vectors,mesh)
    bounds = np.asarray(grid.bounds).reshape(3,2).copy()
    slice_mesh = pv.PolyData();normal = None;section_info = None
    domain = material
    if section:
        section_info = dict(section)
        axis = {'XY':2,'YZ':0,'XZ':1}[section['view']]
        normal = np.eye(3)[axis]
        if section.get('xy_bounds'):
            material=clip_section_xy(material,section['xy_bounds'],normal_axis=axis)
            for coordinate in 'xy':
                index='xyz'.index(coordinate)
                if index!=axis:
                    lo,hi=section['xy_bounds'][coordinate]
                    bounds[index]=[max(bounds[index,0],lo),min(bounds[index,1],hi)]
            domain=material
        position = section.get('position')
        auto_center=vtk_material_center(material) if material.n_points else np.mean(bounds,axis=1)
        position = float(auto_center[axis]) if position is None else float(np.clip(position,*bounds[axis]))
        section_info['position'] = position
        origin = np.mean(bounds,axis=1);origin[axis] = position
        if section['mode']=='Slab':
            half = section['thickness']/2
            bounds[axis] = [max(bounds[axis,0],position-half),min(bounds[axis,1],position+half)]
            if material.n_cells:
                lower = origin.copy();lower[axis] = bounds[axis,0]
                upper = origin.copy();upper[axis] = bounds[axis,1]
                domain = material.clip(normal=normal,origin=lower,invert=False)
                domain = domain.clip(normal=normal,origin=upper,invert=True)
            section_info['bounds'] = bounds.ravel().tolist()
        if material.n_cells:
            slice_mesh = material.slice(normal=normal,origin=origin,generate_triangles=True)
        if section['mode']=='Plane' and section.get('stream_domain','On slice')=='On slice':
            domain = slice_mesh.copy()
            if domain.n_points:
                data = np.asarray(domain['m'])
                domain.point_data['slice_vectors'] = data-(data@normal)[:,None]*normal
        contour_domain = slice_mesh if section['mode']=='Plane' else domain
    else:contour_domain = material
    contour_component = options.get('contour_component') or component
    surface = pv.PolyData()
    if iso and contour_domain.n_cells:
        if contour_domain is grid:
            # Flying Edges needs attribute interpolation explicitly enabled when
            # contouring one component and coloring by another.
            import vtk
            contour=vtk.vtkFlyingEdges3D();contour.SetInputData(grid)
            contour.SetInputArrayToProcess(0,0,0,0,contour_component)
            contour.SetInterpolateAttributes(True);contour.SetComputeNormals(False)
            contour.SetNumberOfContours(len(levels))
            for index,value in enumerate(levels):contour.SetValue(index,value)
            contour.Update();surface=pv.wrap(contour.GetOutput())
        else:surface=contour_domain.contour(isosurfaces=levels,scalars=contour_component)
    seeds = pv.PolyData();lines = pv.PolyData()
    if streams and domain.n_cells:
        seed_domain=slice_mesh if section and section['mode']=='Plane' else domain
        seed_bounds=bounds.copy()
        if not section or section['mode']=='Slab':seed_bounds=np.asarray(domain.bounds).reshape(3,2)
        points = streamline_seed_points(seed_bounds,seed_count,seed_plane,options.get('seed_type','Uniform grid'),
                    spacing=options.get('seed_spacing'),center=options.get('seed_center'),radius=options.get('seed_radius'),
                    line_start=options.get('line_start'),line_end=options.get('line_end'),z_layers=options.get('z_layers'),section=section_info,
                    grid_center=vtk_material_center(domain) if domain.n_points else None)
        refill=options.get('seed_type','Uniform grid')=='Uniform grid' and not options.get('seed_spacing') and (seed_plane=='Volume' or bool(section and section['mode']=='Plane'))
        seeds=material_seed_source(seed_domain,points,seed_count,refill=refill)
        projected = bool(section and section['mode']=='Plane' and section.get('stream_domain','On slice')=='On slice')
        vector_name = 'slice_vectors' if projected else 'm'
        if seeds.n_points and np.any(np.linalg.norm(np.asarray(domain[vector_name]),axis=1)>options.get('terminal_speed',1e-6)):
            step_unit=options.get('step_unit','cl')
            initial_step=options.get('initial_step',.1);minimum_step=options.get('min_step',.001);maximum_step=options.get('max_step',.3)
            if options.get('auto_step',True):
                active=[a for a in range(3) if not projected or a!=int(np.argmax(normal))]
                cell_step=min(spacing[a] for a in active)
                if section and section['mode']=='Slab':cell_step=min(cell_step,section['thickness'])
                step_unit='l';initial_step=cell_step*.3;minimum_step=cell_step*.01;maximum_step=cell_step*.8
            lines = domain.streamlines_from_source(seeds,vectors=vector_name,
                integrator_type=options.get('integrator',45),integration_direction=options.get('direction','both'),
                surface_streamlines=projected,step_unit=step_unit,
                initial_step_length=initial_step,min_step_length=minimum_step,
                max_step_length=maximum_step,max_steps=options.get('max_steps',2000),
                max_length=options.get('max_length') or float(np.linalg.norm(bounds[:,1]-bounds[:,0])*3),
                terminal_speed=options.get('terminal_speed',1e-6),max_error=options.get('max_error',1e-6),
                compute_vorticity=False,interpolator_type='cell')
    return dict(grid=grid,iso=surface,streams=lines,seeds=seeds,spacing=spacing,slice=slice_mesh,
                section=section_info,display_bounds=tuple(bounds.ravel()),
                scalar_range=(float(np.min(grid[component])),float(np.max(grid[component]))),
                contour_range=(float(np.min(grid[contour_component])),float(np.max(grid[contour_component]))))


def streamline_arrow_glyphs(lines, tube_radius, max_arrows=8000, options=None):
    """Resample by physical arc length and orient arrows with the vector field.

    Backward integration reverses point order, so line tangents alone would
    show the wrong direction. Interpolated m (or slice_vectors) is authoritative.
    A global deterministic sample cap distributes glyphs across all traces.
    """
    pv=require_tab4_backend()
    if not lines.n_lines:return pv.PolyData()
    points=np.asarray(lines.points);connectivity=np.asarray(lines.lines)
    diagonal=float(np.linalg.norm(np.ptp(points,axis=0)))
    options=options or {}
    spacing=options.get('spacing') or max(diagonal/12,tube_radius*12,1e-6)
    length=options.get('length') or max(spacing*.36,tube_radius*4.8)
    radius=options.get('radius') or max(spacing*.144,tube_radius*2.5)
    resolution=options.get('resolution',24);max_arrows=options.get('limit',max_arrows)
    cells=[];offset=0;total=0
    while offset<len(connectivity):
        count=int(connectivity[offset]);ids=connectivity[offset+1:offset+1+count];offset+=count+1
        if count<2:continue
        distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points[ids],axis=0),axis=1))]
        if distance[-1]<=1e-10:continue
        samples=max(1,int(math.ceil(distance[-1]/spacing)))
        cells.append((ids,distance,samples,total));total+=samples
    if not total:return pv.PolyData()
    selected=np.linspace(0,total-1,min(total,max_arrows),dtype=np.int64)
    origins=[];vectors=[];colors={a:[] for a in ('mx','my','mz','mag')}
    vector_name='slice_vectors' if 'slice_vectors' in lines.point_data else 'm'
    field=np.asarray(lines[vector_name])
    for ids,distance,count,first in cells:
        begin,end=np.searchsorted(selected,[first,first+count])
        if begin==end:continue
        # Keep the requested nm spacing exact; balance leftover length at both ends.
        target=(distance[-1]-(count-1)*spacing)/2+(selected[begin:end]-first)*spacing
        hi=np.searchsorted(distance,target,side='right').clip(1,len(ids)-1);lo=hi-1
        t=(target-distance[lo])/np.maximum(distance[hi]-distance[lo],1e-20)
        a,b=ids[lo],ids[hi]
        origins.append(points[a]*(1-t[:,None])+points[b]*t[:,None])
        vectors.append(field[a]*(1-t[:,None])+field[b]*t[:,None])
        for name in colors:
            values=np.asarray(lines[name]);colors[name].append(values[a]*(1-t)+values[b]*t)
    origins=np.concatenate(origins);vectors=np.concatenate(vectors)
    magnitude=np.linalg.norm(vectors,axis=1);valid=np.isfinite(magnitude)&(magnitude>1e-8)
    if not np.any(valid):return pv.PolyData()
    data=pv.PolyData(origins[valid]);data['arrow_vectors']=vectors[valid]/magnitude[valid,None]
    data['arrow_length']=np.full(valid.sum(),length)
    for name,values in colors.items():data[name]=np.concatenate(values)[valid]
    source=pv.Cone(center=(-.5,0,0),direction=(1,0,0),height=1.,
        radius=radius/length,resolution=resolution,capping=True)
    glyphs=data.glyph(orient='arrow_vectors',scale='arrow_length',factor=1.,geom=source)
    glyphs.field_data['arrow_count']=np.array([valid.sum()])
    glyphs.field_data['requested_arrow_count']=np.array([total])
    glyphs.field_data['head_length_nm']=np.array([length]);glyphs.field_data['head_radius_nm']=np.array([radius])
    glyphs.field_data['head_spacing_nm']=np.array([spacing])
    return glyphs




class StreamIsoViewBase(ttk.Frame):
    def __init__(self,gui,parent):
        super().__init__(parent,padding=8)
        self.gui = gui;self.closed = False
        self.files = [];self.vectors = self.mesh = None;self.time_ns = None
        self.native_cache = None;self.imported = None;self.result = None;self.plotter = None
        self.vtk_cache = Tab4VTKCache()
        self.native_stamp = None;self.native_vtk_path = None;self.empty_message = None
        self.worker_busy = False;self.generation = 0;self.update_timer = self.resize_timer = None
        self.scene_valid = False;self.photo = None;self.drag = None
        self.roi_physical = None
        self.use_tab3_roi = tk.BooleanVar(value=True)
        self.roi_label = tk.StringVar(value='ROI: full volume')
        self.folder = tk.StringVar();self.frame = tk.DoubleVar(value=1)
        self.frame_label = tk.StringVar(value='No frames')
        self.factor = tk.StringVar(value='1x');self.component = tk.StringVar(value='mz')
        self.levels = tk.StringVar(value='0');self.seeds = tk.StringVar(value='125')
        self.seed_plane = tk.StringVar(value='Volume');self.tube_radius = tk.StringVar(value='0.15')
        self.opacity = tk.DoubleVar(value=.55)
        self.show_streams = tk.BooleanVar(value=True);self.show_iso = tk.BooleanVar(value=True)
        self.show_scales = tk.BooleanVar(value=True);self.show_bar = tk.BooleanVar(value=True)
        self.show_time = tk.BooleanVar(value=True);self.colors = tk.StringVar(value='blue,white,red')
        self.status = tk.StringVar(value='Load a vector-field folder or use Tab 3 data. Left-drag pans; middle-drag rotates; wheel zooms.')
        self.columnconfigure(0,weight=1);self.rowconfigure(0,weight=1)
        self.divider = ttk.Panedwindow(self,orient='horizontal');self.divider.grid(row=0,column=0,sticky='nsew')
        view = ttk.Frame(self.divider)
        self.canvas = tk.Canvas(view,background='white',highlightthickness=0);self.canvas.pack(fill='both',expand=True)
        panel = ttk.Frame(self.divider,width=300);panel.rowconfigure(0,weight=1);panel.columnconfigure(0,weight=1)
        sidebar = tk.Canvas(panel,width=285,highlightthickness=0)
        sidebar.grid(row=0,column=0,sticky='nsew')
        scroll = ttk.Scrollbar(panel,orient='vertical',command=sidebar.yview);scroll.grid(row=0,column=1,sticky='ns')
        sidebar.configure(yscrollcommand=scroll.set)
        self.controls = ttk.Frame(sidebar,padding=6)
        item = sidebar.create_window((0,0),window=self.controls,anchor='nw',width=285)
        sidebar.bind('<Configure>',lambda e:sidebar.itemconfigure(item,width=max(180,e.width)))
        self.controls.bind('<Configure>',lambda e:sidebar.configure(scrollregion=sidebar.bbox('all')))
        self.divider.add(view,weight=4);self.divider.add(panel,weight=1)
        footer = ttk.Label(self,textvariable=self.status,wraplength=950,padding=(5,3));footer.grid(row=1,column=0,sticky='ew')
        self.bind('<Configure>',lambda e:footer.configure(wraplength=max(200,e.width-20)),add='+')
        group = self.group('Frames')
        ttk.Entry(group,textvariable=self.folder).pack(fill='x')
        row = ttk.Frame(group);row.pack(fill='x',pady=3)
        ttk.Button(row,text='Browse',command=self.browse).pack(side='left',padx=2)
        row = ttk.Frame(group);row.pack(fill='x',pady=3)
        row.columnconfigure((0,1,2),weight=1)
        ttk.Button(row,text='Load',command=self.load_folder,width=6).grid(row=0,column=0,sticky='ew',padx=1)
        self.vtk_convert_busy = False;self.vtk_convert_stop = threading.Event()
        self.vtk_convert_button = ttk.Button(row,text='VTK Convert',command=self.convert_vtk,width=11)
        self.vtk_convert_button.grid(row=0,column=1,sticky='ew',padx=1)
        ttk.Button(row,text='VTK output',command=self.open_vtk_cache,width=10).grid(row=0,column=2,sticky='ew',padx=1)
        ttk.Button(group,text='Use Tab 3 data',command=self.use_tab3).pack(fill='x',pady=2)
        self.frame_slider = ttk.Scale(group,from_=1,to=1,variable=self.frame,command=lambda v:self.schedule())
        self.frame_slider.pack(fill='x');ttk.Label(group,textvariable=self.frame_label,wraplength=260).pack(anchor='w')
        self.dropdown(group,'Upscaling',self.factor,['1x','2x','4x','8x','10x'])
        ttk.Checkbutton(group,text='Use Tab 3 ROI',variable=self.use_tab3_roi,command=self.schedule).pack(anchor='w',pady=2)
        ttk.Label(group,textvariable=self.roi_label,wraplength=260).pack(anchor='w',pady=2)
        group = self.group('Component')
        row = ttk.Frame(group);row.pack(fill='x')
        for name,icon in zip(('mx','my','mz'),gui.component_icons):
            ttk.Radiobutton(row,image=icon,variable=self.component,value=name,command=self.schedule).pack(side='left',padx=3)
        ttk.Radiobutton(row,text='|m|',variable=self.component,value='mag',command=self.schedule).pack(side='left',padx=3)
        self.dropdown(group,'Colors',self.colors,COLOR_PRESETS)
        group = self.group('Streamlines')
        ttk.Checkbutton(group,text='Show',variable=self.show_streams,command=self.schedule).pack(anchor='w')
        self.entry(group,'Seeds',self.seeds)
        self.dropdown(group,'Seed region',self.seed_plane,['Volume','X','Y','Z'])
        self.entry(group,'Tube radius (nm)',self.tube_radius)
        group = self.group('Isosurfaces')
        ttk.Checkbutton(group,text='Show',variable=self.show_iso,command=self.schedule).pack(anchor='w')
        self.entry(group,'Iso values (comma separated)',self.levels)
        ttk.Label(group,text='Opacity').pack(anchor='w')
        ttk.Scale(group,from_=0,to=1,variable=self.opacity,command=lambda v:self.schedule()).pack(fill='x')
        group = self.group('Display')
        for text,var in (('XYZ scales',self.show_scales),('Color bar',self.show_bar),('Time (ns)',self.show_time)):
            ttk.Checkbutton(group,text=text,variable=var,command=self.schedule).pack(anchor='w')
        ttk.Button(group,text='Reset',command=self.reset_camera).pack(fill='x',pady=2)
        ttk.Button(group,text='Export PNG',command=self.export_png).pack(fill='x',pady=2)
        self.canvas.bind('<Configure>',self.resize)
        self.canvas.bind('<ButtonPress-1>',lambda e:self.start_drag(e,'pan'))
        self.canvas.bind('<ButtonPress-2>',lambda e:self.start_drag(e,'rotate'))
        self.canvas.bind('<B1-Motion>',self.move_drag);self.canvas.bind('<B2-Motion>',self.move_drag)
        self.canvas.bind('<ButtonRelease-1>',lambda e:setattr(self,'drag',None));self.canvas.bind('<ButtonRelease-2>',lambda e:setattr(self,'drag',None))
        self.canvas.bind('<MouseWheel>',self.zoom)
        self.master.bind('<<NotebookTabChanged>>',self.tab_selected,add='+')
        self.show_message('Load a vector-field folder or use Tab 3 data.')

    def convert_vtk(self):
        if self.closed: return
        if self.vtk_convert_busy:
            self.vtk_convert_stop.set()
            self.vtk_convert_button.configure(state='disabled')
            self.status.set('Stopping VTK conversion…')
            return
        try:
            folder_text = self.folder.get().strip()
            if folder_text:
                folder = Path(folder_text).expanduser()
                if not folder.is_dir(): raise ValueError('Choose a valid field folder.')
                sources = field_files(folder)
            else:
                sources = list(self.files)
            if not sources: raise ValueError('Load an OVF, OOMMF, or VTK field folder first.')
            require_tab4_backend()
        except Exception as exc:
            self.status.set('VTK Convert: '+str(exc)); return
        self.vtk_convert_busy = True; self.vtk_convert_stop.clear()
        self.vtk_convert_button.configure(text='Stop VTK')
        self.status.set(f'VTK Convert: 0/{len(sources)}')
        def worker():
            completed = 0; errors = []; stopped = False
            try:
                for index, source in enumerate(sources, 1):
                    check_cancel(self.vtk_convert_stop)
                    try:
                        self.vtk_cache.convert_and_read(source, self.vtk_convert_stop)
                        completed += 1
                        message = f'VTK Convert: {index}/{len(sources)} — {source.name}'
                    except ConversionCancelled:
                        raise
                    except Exception as exc:
                        errors.append(f'{source.name}: {exc}')
                        message = f'VTK Convert: {index}/{len(sources)} — skipped {source.name}'
                    if not self.closed:
                        self.gui.events.put((self.vtk_conversion_progress, (message,)))
            except ConversionCancelled:
                stopped = True
            except Exception as exc:
                errors.append(str(exc))
            if not self.closed:
                self.gui.events.put((self.vtk_conversion_complete, (completed, len(sources), errors, stopped)))
        threading.Thread(target=worker, daemon=True).start()

    def vtk_conversion_progress(self, message):
        if not self.closed and self.vtk_convert_busy:
            self.status.set(message)

    def vtk_conversion_complete(self, completed, total, errors, stopped):
        if self.closed: return
        self.vtk_convert_busy = False
        self.vtk_convert_button.configure(text='VTK Convert', state='normal')
        message = f'VTK conversion {"stopped" if stopped else "complete"}: {completed}/{total} files'
        if errors:
            message += f'; {len(errors)} skipped. '+errors[0]
            directory = self.vtk_cache.directory
            if directory is not None and directory.is_dir():
                (directory/'conversion_errors.txt').write_text('\n\n'.join(errors), encoding='utf-8')
        self.status.set(message+'. Use VTK output to open the cache folder.')


    def open_vtk_cache(self):
        if self.closed:return
        try:
            folder = self.vtk_cache.ensure_directory()
            open_folder(folder)
            self.status.set('VTK output: '+str(folder))
        except (OSError, ConversionCancelled) as exc:
            self.status.set('Could not open the VTK output folder: '+str(exc))

    def show_message(self,message):
        self.empty_message = message
        self.canvas.delete('all')
        width,height = max(200,self.canvas.winfo_width()),max(180,self.canvas.winfo_height())
        self.canvas.create_text(width/2,height/2,text=message,width=max(180,width-80),fill='#333333',font=('Segoe UI',11),justify='center')

    def tab_selected(self,event=None):
        if not self.closed and self.master.select()==str(self):self.sync_roi(True)

    def sync_roi(self,schedule=False):
        roi = physical_tab3_roi(self.gui.cross_tab) if self.use_tab3_roi.get() else None
        changed = roi!=self.roi_physical
        self.roi_physical = roi
        if roi is None:self.roi_label.set('ROI: full volume' if self.use_tab3_roi.get() else 'ROI: independent full volume')
        else:self.roi_label.set(f'ROI: X {roi["x"][0]:.3f}…{roi["x"][1]:.3f}, Y {roi["y"][0]:.3f}…{roi["y"][1]:.3f} {roi["unit"]}')
        if changed and schedule and self.files:self.schedule()

    def group(self,text):
        group = ttk.LabelFrame(self.controls,text=text,padding=6);group.pack(fill='x',pady=4);return group

    def entry(self,parent,text,variable):
        ttk.Label(parent,text=text).pack(anchor='w')
        entry = ttk.Entry(parent,textvariable=variable);entry.pack(fill='x',pady=2)
        entry.bind('<Return>',lambda e:self.schedule());entry.bind('<FocusOut>',lambda e:self.schedule())

    def dropdown(self,parent,text,variable,values):
        ttk.Label(parent,text=text).pack(anchor='w')
        box = ttk.Combobox(parent,textvariable=variable,values=values,state='readonly');box.pack(fill='x',pady=2)
        box.bind('<<ComboboxSelected>>',lambda e:self.schedule())

    def browse(self):
        folder = filedialog.askdirectory(title='OVF, OOMMF, or VTK field folder')
        if folder:self.folder.set(folder)

    def load_folder(self):
        text = self.folder.get().strip()
        folder = Path(text).expanduser()
        if not text or not folder.is_dir():self.status.set('Choose a valid field folder.');return
        files = field_files(folder)
        if not files:self.status.set('No OVF, OOMMF, or VTK vector fields found in this folder.');return
        self.files = files;self.imported = self.native_cache = None
        self.native_stamp = self.native_vtk_path = None
        self.frame.set(1);self.frame_slider.configure(to=len(files));self.schedule()

    def use_tab3(self):
        source = self.gui.cross_tab
        if source.vectors is None or source.loading:self.status.set('Load a readable frame in Tab 3 first.');return
        result = source.native_result[2] if source.native_result else (source.vectors,source.mesh,source.time_ns,None)
        path = source.files[source.loaded_index]
        self.files = list(source.files);self.folder.set(str(path.parent))
        self.imported = (path,result);self.native_cache = self.imported
        info = path.stat();self.native_stamp = (info.st_size,info.st_mtime_ns);self.native_vtk_path = None
        self.factor.set(source.figure_upscale.get());self.frame.set(source.loaded_index+1)
        self.frame_slider.configure(to=len(self.files));self.schedule()

    def schedule(self):
        if self.closed or self.gui.closed:return
        self.generation += 1
        if self.update_timer is not None:self.after_cancel(self.update_timer)
        self.update_timer = self.after(150,self.start_build)

    def start_build(self):
        self.update_timer = None
        if self.closed or self.worker_busy or not self.files:return
        try:
            index = min(len(self.files)-1,max(0,round(self.frame.get())-1));source = self.files[index]
            factor = int(self.factor.get().replace('x',''))
            levels = [float(v.strip()) for v in self.levels.get().split(',') if v.strip()]
            if self.show_iso.get() and (not levels or not all(math.isfinite(v) for v in levels)):
                raise ValueError('Enter finite iso values, such as -0.5, 0, 0.5.')
            seeds = int(self.seeds.get());radius = float(self.tube_radius.get())
            if not 1<=seeds<=2000 or not math.isfinite(radius) or radius<=0:raise ValueError('Use 1–2000 seeds and a positive tube radius.')
            settings = dict(component=self.component.get(),levels=levels,seed_count=seeds,seed_plane=self.seed_plane.get(),
                            streams=self.show_streams.get(),iso=self.show_iso.get())
            style = dict(colors=self.colors.get(),opacity=float(self.opacity.get()),scales=self.show_scales.get(),
                         bar=self.show_bar.get(),time=self.show_time.get(),radius=radius,component=self.component.get())
        except Exception as exc:self.status.set(str(exc));return
        self.sync_roi(False)
        roi = self.roi_physical
        style['roi'] = roi
        token = self.generation;cache = self.native_cache
        cache_stamp = self.native_stamp;cached_vtk = self.native_vtk_path
        self.worker_busy = True;self.frame_label.set(f'{index+1}/{len(self.files)}: {source.name}')
        self.status.set('Computing streamlines and isosurfaces…')
        if not self.scene_valid:self.show_message('Preparing the 3D view…')
        def worker():
            try:
                require_tab4_backend()
                info = source.stat();stamp = (info.st_size,info.st_mtime_ns)
                if cache is not None and cache[0]==source and cache_stamp==stamp:
                    native,vtk_path = cache[1],cached_vtk
                else:native,vtk_path = self.vtk_cache.read_source(source)
                style['source_stamp'] = stamp;style['vtk_path'] = str(vtk_path) if vtk_path is not None else None
                vectors,mesh = refine_vector_grid(native[0],native[1],factor)
                vectors,mesh = crop_stream_roi(vectors,mesh,roi)
                result = build_stream_iso_scene(vectors,mesh,**settings)
                payload = (source,native,vectors,mesh,result,style,None)
            except Exception as exc:payload = (source,None,None,None,None,None,str(exc))
            if not self.closed and not self.gui.closed:self.gui.events.put((self.built,(token,payload)))
        threading.Thread(target=worker,daemon=True).start()

    def built(self,token,payload):
        self.worker_busy = False
        if self.closed:return
        if token!=self.generation:
            if self.update_timer is not None:self.after_cancel(self.update_timer);self.update_timer = None
            self.start_build();return
        source,native,vectors,mesh,result,style,error = payload
        if error:
            self.scene_valid = False;self.show_message(error);self.status.set(f'{source.name}: {error}');return
        self.native_cache = (source,native);self.vectors,self.mesh,self.time_ns = vectors,mesh,native[2]
        self.native_stamp = style['source_stamp'];self.native_vtk_path = style['vtk_path']
        self.result,self.style = result,style
        try:self.display_scene()
        except Exception as exc:self.scene_valid = False;self.show_message('3D view failed: '+str(exc));self.status.set('3D view failed: '+str(exc));return
        lo,hi = result['scalar_range']
        message = f'{source.name} | {result["streams"].n_lines} streamlines | {result["iso"].n_cells} surface cells | Values {lo:.3f}…{hi:.3f}'
        if field_source_note(source):message += ' | '+field_source_note(source)
        if self.native_vtk_path:message += ' | VTK cache: '+self.native_vtk_path
        if not result['streams'].n_points and not result['iso'].n_points:message += ' | No geometry at these iso values/seeds; adjust the controls or ROI.'
        self.status.set(message)

    def display_scene(self):
        import pyvista as pv
        width,height = max(200,self.canvas.winfo_width()),max(180,self.canvas.winfo_height())
        camera = None
        if self.plotter is None:self.plotter = pv.Plotter(off_screen=True,window_size=(width,height))
        elif self.scene_valid:camera = self.plotter.camera_position
        plotter = self.plotter;plotter.clear();plotter.set_background('white')
        names = [x.strip().lower() for x in self.style['colors'].split(',')]
        if len(names)!=3 or any(x not in MUMAX_RGB for x in names):raise ValueError('Choose three valid colors.')
        cmap = LinearSegmentedColormap.from_list('m_components',[np.array(MUMAX_RGB[x])/255 for x in names])
        component = self.style['component'];clim = (0.,max(1.,self.result['scalar_range'][1])) if component=='mag' else (-1.,1.)
        bar_used = False
        for kind in ('iso','streams'):
            data = self.result[kind]
            if not data.n_points:continue
            if kind=='streams':data = data.tube(radius=self.style['radius'],n_sides=12)
            plotter.add_mesh(data,scalars=component,cmap=cmap,clim=clim,opacity=self.style['opacity'] if kind=='iso' else 1.,
                smooth_shading=True,ambient=.35,diffuse=.75,specular=.25,specular_power=25,
                show_scalar_bar=self.style['bar'] and not bar_used,
                scalar_bar_args=dict(title='',vertical=True,position_x=.88,position_y=.2,width=.07,height=.6,
                                     n_labels=3,fmt='%.3f',color='black',label_font_size=10))
            bar_used = True
        plotter.add_mesh(self.result['grid'].outline(),color='#999999',line_width=1)
        if self.style['scales']:
            unit = self.mesh['x']['unit']
            plotter.show_bounds(bounds=self.result['grid'].bounds,xtitle=f'X ({unit})',ytitle=f'Y ({unit})',ztitle=f'Z ({unit})',
                color='black',font_size=10,n_xlabels=4,n_ylabels=4,n_zlabels=4,fmt='%.3f',grid='back',location='outer')
        if camera is None:plotter.view_isometric();plotter.reset_camera()
        else:plotter.camera_position = camera
        plotter.enable_anti_aliasing('fxaa')
        self.empty_message = None;self.scene_valid = True;self.paint()

    def overlay(self,image,scale=1):
        draw = ImageDraw.Draw(image)
        label = '|m|' if self.style['component']=='mag' else self.style['component']
        draw_scientific_text(draw,(10*scale,8*scale),label,annotation_font(14*scale),'black')
        if self.style['time'] and self.time_ns is not None:
            draw.text((10*scale,30*scale),f't = {self.time_ns:.3f} ns',font=annotation_font(11*scale),fill='black')
        return image

    def paint(self):
        if not self.scene_valid or self.plotter is None or self.closed:return
        width,height = max(200,self.canvas.winfo_width()),max(180,self.canvas.winfo_height())
        self.plotter.window_size = (width,height)
        self.plotter.render()
        self.image = self.overlay(Image.fromarray(np.asarray(self.plotter.screenshot(return_img=True))))
        self.photo = ImageTk.PhotoImage(self.image,master=self.gui.root)
        self.canvas.delete('all');self.canvas.create_image(0,0,image=self.photo,anchor='nw')

    def resize(self,event):
        if self.resize_timer is not None:self.after_cancel(self.resize_timer)
        self.resize_timer = self.after(50,self.resize_done)

    def resize_done(self):
        self.resize_timer = None
        if self.scene_valid:self.paint()
        elif self.empty_message:self.show_message(self.empty_message)

    def reset_camera(self):
        if self.plotter is not None and self.scene_valid:self.plotter.view_isometric();self.plotter.reset_camera();self.paint()

    def start_drag(self,event,mode):self.drag = event.x,event.y,mode

    def move_drag(self,event):
        if self.drag is None or not self.scene_valid:return
        x,y,mode = self.drag;dx,dy = event.x-x,event.y-y
        camera = self.plotter.camera
        if mode=='rotate':camera.Azimuth(-dx*.4);camera.Elevation(dy*.4);camera.OrthogonalizeViewUp()
        else:
            position,focal = np.array(camera.position),np.array(camera.focal_point)
            normal = (focal-position)/np.linalg.norm(focal-position)
            up = np.array(camera.up);right = np.cross(normal,up);right /= np.linalg.norm(right)
            up = np.cross(right,normal)
            span = 2*np.linalg.norm(focal-position)*math.tan(math.radians(camera.view_angle)/2)
            shift = (-dx*right+dy*up)*span/max(1,self.canvas.winfo_height())
            camera.position = position+shift;camera.focal_point = focal+shift
        self.drag = event.x,event.y,mode;self.paint()

    def cursor_world(self,x,y):
        renderer = self.plotter.renderer
        focal = self.plotter.camera.focal_point
        renderer.SetWorldPoint(*focal,1.);renderer.WorldToDisplay();depth = renderer.GetDisplayPoint()[2]
        renderer.SetDisplayPoint(x,self.canvas.winfo_height()-y,depth);renderer.DisplayToWorld()
        value = np.array(renderer.GetWorldPoint());return value[:3]/value[3]

    def zoom(self,event):
        if not self.scene_valid:return
        before = self.cursor_world(event.x,event.y)
        self.plotter.camera.Zoom(1.2 if event.delta>0 else 1/1.2)
        after = self.cursor_world(event.x,event.y);shift = before-after
        self.plotter.camera.position = np.array(self.plotter.camera.position)+shift
        self.plotter.camera.focal_point = np.array(self.plotter.camera.focal_point)+shift
        self.paint();return 'break'

    def export_png(self):
        if not self.scene_valid:self.status.set('Load a readable volume first.');return
        path = filedialog.asksaveasfilename(title='Export streamlines / isosurfaces',defaultextension='.png',filetypes=[('PNG','*.png')])
        if path:self.save_png(path)

    def save_png(self,path):
        image = Image.fromarray(np.asarray(self.plotter.screenshot(return_img=True,scale=3)))
        self.overlay(image,3).save(path,dpi=(300,300));self.status.set(f'Saved {path} (300 PPI)')

    def close(self):
        self.closed = True;self.generation += 1
        self.vtk_convert_stop.set()
        self.vtk_cache.close()
        for timer in (self.update_timer,self.resize_timer):
            if timer is not None:self.after_cancel(timer)
        if self.plotter is not None:self.plotter.close();self.plotter = None
        self.result = self.native_cache = self.imported = None
        self.vectors = self.mesh = None
        import gc
        gc.collect()




class StreamVolumeControlsBase(StreamIsoViewBase):
    """Stream Tracer, Contour and display properties for Tab 4."""
    def __init__(self,gui,parent):
        super().__init__(gui,parent)
        try:configure_mumax_vtk_cpu()
        except Tab4DependencyError as exc:self.show_message(str(exc));self.status.set(str(exc))
        self.auto_apply = tk.BooleanVar(value=True);self.advanced = tk.BooleanVar(value=False)
        self.analysis_cache = None;self.geometry_cache = None;self.pending = False
        self.actor_cache={};self.display_cache_key=None;self.paint_timer=None
        self.seeds.set('1000')
        self.seed_type = tk.StringVar(value='Uniform grid');self.seed_spacing = tk.StringVar()
        self.seed_center = tk.StringVar();self.seed_radius = tk.StringVar()
        self.line_start = tk.StringVar();self.line_end = tk.StringVar();self.z_layers = tk.StringVar()
        self.direction = tk.StringVar(value='Both');self.integrator = tk.StringVar(value='Runge-Kutta 4-5')
        self.max_length = tk.StringVar();self.initial_step = tk.StringVar(value='0.3')
        self.auto_step = tk.BooleanVar(value=True)
        self.initial_step.set('0.1');self.min_step = tk.StringVar(value='0.001');self.max_step = tk.StringVar(value='0.3')
        self.step_unit = tk.StringVar(value='Cell length');self.max_steps = tk.StringVar(value='2000')
        self.terminal_speed = tk.StringVar(value='0.000001');self.max_error = tk.StringVar(value='0.000001')
        self.stream_style = tk.StringVar(value='Tubes');self.line_width = tk.StringVar(value='2')
        self.show_stream_arrows=tk.BooleanVar(value=False);self.stream_arrow_cache=(None,None)
        self.head_spacing=tk.StringVar();self.head_length=tk.StringVar();self.head_radius=tk.StringVar()
        self.head_resolution=tk.StringVar(value='24');self.head_limit=tk.StringVar(value='8000')
        self.head_color=tk.StringVar(value='By component')
        self.tube_sides = tk.StringVar(value='24');self.stream_opacity = tk.DoubleVar(value=1.)
        self.contour_component = tk.StringVar(value='color');self.contour_style = tk.StringVar(value='Surface')
        self.contour_color = tk.StringVar(value='By component');self.smooth_normals = tk.BooleanVar(value=True)
        self.level_min = tk.StringVar(value='-0.9');self.level_max = tk.StringVar(value='0.9');self.level_count = tk.StringVar(value='7')
        self.contour_range_label = tk.StringVar(value='Range: load a frame')
        self.show_seeds = tk.BooleanVar(value=False);self.show_outline = tk.BooleanVar(value=True)
        self.bar_size=tk.StringVar(value='Small');self.bar_position=tk.StringVar(value='Middle');self.text_size=tk.StringVar(value='10')
        self.show_decimals=tk.BooleanVar(value=True)
        self.ambient_light=tk.DoubleVar(value=.35)
        self.color_range_mode = tk.StringVar(value='Fixed');self.color_low = tk.StringVar(value='-1');self.color_high = tk.StringVar(value='1')
        groups = {str(g.cget('text')):g for g in self.controls.winfo_children() if isinstance(g,ttk.LabelFrame)}
        self.groups = groups
        for child in groups['Component'].winfo_children():
            if isinstance(child,ttk.Label) and str(child.cget('text'))=='Colors':child.configure(text='Color map')
            if isinstance(child,ttk.Combobox) and str(child.cget('textvariable'))==str(self.colors):
                child.configure(values=STREAM_COLOR_MAPS,state='normal')
                child.bind('<Return>',lambda e:self.schedule());child.bind('<FocusOut>',lambda e:self.schedule())
                ttk.Button(groups['Component'],text='Choose preset…',command=self.choose_color_preset).pack(fill='x',after=child,pady=3)
        self.volume_view=tk.StringVar(value='3D')
        view_group=self.group('View');view_group.pack_configure(before=groups['Streamlines'])
        box=ttk.Combobox(view_group,textvariable=self.volume_view,values=['3D','XY','YZ','XZ'],state='readonly')
        box.pack(fill='x');box.bind('<<ComboboxSelected>>',self.volume_view_changed)
        frames = groups['Frames']
        row = ttk.Frame(frames);row.pack(fill='x',pady=3)
        ttk.Button(row,text='Apply',command=self.apply).pack(side='left',fill='x',expand=True)
        ttk.Checkbutton(row,text='Auto apply',variable=self.auto_apply,command=self.auto_changed).pack(side='left',padx=4)
        group = groups['Streamlines'];group.configure(text='Stream Tracer')
        for child in group.winfo_children():child.destroy()
        ttk.Checkbutton(group,text='Show streamlines',variable=self.show_streams,command=self.schedule).pack(anchor='w')
        self.spin(group,'Seed count',self.seeds,1,100000)
        self.dropdown(group,'Seed source',self.seed_type,['Uniform grid','Point cloud','Line'])
        self.dropdown(group,'Seed region',self.seed_plane,['Volume','X','Y','Z'])
        self.entry(group,'Seed spacing (nm, blank = count)',self.seed_spacing)
        self.dropdown(group,'Integration direction',self.direction,['Both','Forward','Backward'])
        self.dropdown(group,'Integrator',self.integrator,['Runge-Kutta 4-5','Runge-Kutta 4','Runge-Kutta 2'])
        self.entry(group,'Maximum length (nm, blank = auto)',self.max_length)
        self.dropdown(group,'Representation',self.stream_style,['Tubes','Lines'])
        self.entry(group,'Tube radius (nm)',self.tube_radius)
        self.spin(group,'Line width (px)',self.line_width,1,20)
        self.dropdown(group,'Tube sides',self.tube_sides,['8','12','24','36','48'])
        self.head_checkbox=ttk.Checkbutton(group,text='Arrow heads',variable=self.show_stream_arrows,command=self.schedule)
        self.head_checkbox.pack(anchor='w')
        self.head_controls=ttk.Frame(group,padding=(10,0,0,3))
        for label,var in (('Head spacing (nm, blank = auto)',self.head_spacing),('Head length (nm, blank = auto)',self.head_length),
                          ('Head radius (nm, blank = auto)',self.head_radius)):self.entry(self.head_controls,label,var)
        self.dropdown(self.head_controls,'Head resolution',self.head_resolution,['8','12','16','24','36','48','64'])
        self.spin(self.head_controls,'Maximum heads',self.head_limit,1,50000)
        self.dropdown(self.head_controls,'Head color',self.head_color,['By component','Black','White'])
        self.show_stream_arrows.trace_add('write',self.reveal_head_controls)
        ttk.Checkbutton(group,text='Show seed points',variable=self.show_seeds,command=self.schedule).pack(anchor='w')
        ttk.Checkbutton(group,text='Advanced integration / seeds',variable=self.advanced,command=self.toggle_advanced).pack(anchor='w',pady=3)
        self.advanced_panel = ttk.Frame(group)
        for label,var in (('Point center X, Y, Z (nm; blank = center)',self.seed_center),('Point radius (nm; blank = auto)',self.seed_radius),
                          ('Line start X, Y, Z (nm; blank = auto)',self.line_start),('Line end X, Y, Z (nm; blank = auto)',self.line_end),
                          ('Z seed layers (blank = auto)',self.z_layers)):
            self.entry(self.advanced_panel,label,var)
        self.dropdown(self.advanced_panel,'Step unit',self.step_unit,['Cell length','nm'])
        ttk.Checkbutton(self.advanced_panel,text='Automatic step size',variable=self.auto_step,command=self.schedule).pack(anchor='w')
        for label,var in (('Initial step',self.initial_step),('Minimum step',self.min_step),('Maximum step',self.max_step),
                          ('Maximum steps',self.max_steps),('Terminal magnitude',self.terminal_speed),('Maximum error',self.max_error)):
            self.entry(self.advanced_panel,label,var)
        ttk.Label(group,text='Streamline opacity').pack(anchor='w')
        ttk.Scale(group,from_=0,to=1,variable=self.stream_opacity,command=lambda v:self.schedule()).pack(fill='x')
        group = groups['Isosurfaces'];group.configure(text='Contour / Isosurfaces')
        for child in group.winfo_children():child.destroy()
        ttk.Checkbutton(group,text='Show contours / isosurfaces',variable=self.show_iso,command=self.schedule).pack(anchor='w')
        ttk.Label(group,text='Contour by').pack(anchor='w')
        row = ttk.Frame(group);row.pack(fill='x')
        ttk.Radiobutton(row,text='Color',value='color',variable=self.contour_component,command=self.schedule).pack(side='left')
        for name,icon in zip(('mx','my','mz'),gui.component_icons):
            ttk.Radiobutton(row,image=icon,value=name,variable=self.contour_component,command=self.schedule).pack(side='left',padx=2)
        ttk.Radiobutton(row,text='|m|',value='mag',variable=self.contour_component,command=self.schedule).pack(side='left')
        self.entry(group,'Values (comma separated)',self.levels)
        ttk.Label(group,textvariable=self.contour_range_label).pack(anchor='w')
        row = ttk.Frame(group);row.pack(fill='x',pady=2)
        for label,var in (('From',self.level_min),('To',self.level_max),('Count',self.level_count)):
            frame = ttk.Frame(row);frame.pack(side='left',fill='x',expand=True)
            ttk.Label(frame,text=label).pack(anchor='w');ttk.Entry(frame,textvariable=var,width=6).pack(fill='x',padx=1)
        ttk.Button(group,text='Generate values',command=self.generate_levels).pack(fill='x',pady=2)
        self.dropdown(group,'Representation',self.contour_style,['Surface','Wireframe','Surface with edges'])
        self.dropdown(group,'Contour color',self.contour_color,['By component','Black','White'])
        ttk.Checkbutton(group,text='Smooth normals',variable=self.smooth_normals,command=self.schedule).pack(anchor='w')
        ttk.Label(group,text='Contour opacity').pack(anchor='w')
        ttk.Scale(group,from_=0,to=1,variable=self.opacity,command=lambda v:self.schedule()).pack(fill='x')
        group = groups['Display']
        ttk.Checkbutton(group,text='Decimals',variable=self.show_decimals,command=self.schedule).pack(anchor='w')
        ttk.Checkbutton(group,text='Volume outline',variable=self.show_outline,command=self.schedule).pack(anchor='w')
        self.dropdown(group,'Color bar size',self.bar_size,['Small','Medium','Bigger'])
        self.dropdown(group,'Color bar position',self.bar_position,['Top','Middle','Bottom'])
        self.spin(group,'Text size',self.text_size,6,40)
        self.dropdown(group,'Color range',self.color_range_mode,['Fixed','Data range'])
        row = ttk.Frame(group);row.pack(fill='x')
        for label,var in (('Min',self.color_low),('Max',self.color_high)):
            frame = ttk.Frame(row);frame.pack(side='left',fill='x',expand=True)
            ttk.Label(frame,text=label).pack(anchor='w');entry=ttk.Entry(frame,textvariable=var,width=8);entry.pack(fill='x')
            entry.bind('<Return>',lambda e:self.schedule());entry.bind('<FocusOut>',lambda e:self.schedule())

        self.layer_fill_actors={}
        self.layer_fill_control=LayerFillControl(self.controls,self.schedule,self.status)
        self.layer_fill_control.pack(fill='x',pady=5,before=self.groups['Component'])

        group=self.groups['Display'];lighting=ttk.Frame(group)
        lighting.pack(fill='x',pady=3,before=group.pack_slaves()[0])
        ttk.Label(lighting,text='Ambient light').pack(anchor='w')
        row=ttk.Frame(lighting);row.pack(fill='x')
        ttk.Scale(row,from_=0,to=1,variable=self.ambient_light,command=lambda value:self.schedule()).pack(side='left',fill='x',expand=True)
        ambient_spin=ttk.Spinbox(row,from_=0,to=1,increment=.05,textvariable=self.ambient_light,width=6,command=self.schedule)
        ambient_spin.pack(side='right',padx=3)
        ambient_spin.bind('<Return>',lambda event:self.schedule());ambient_spin.bind('<FocusOut>',lambda event:self.schedule())

        add_light_controls(self,lighting,self.schedule)
        add_reflection_controls(self,lighting,self.schedule)

    def spin(self,parent,label,var,first,last):
        ttk.Label(parent,text=label).pack(anchor='w')
        widget=ttk.Spinbox(parent,from_=first,to=last,textvariable=var,command=self.schedule)
        widget.pack(fill='x',pady=2)
        widget.bind('<Return>',lambda e:self.schedule());widget.bind('<FocusOut>',lambda e:self.schedule())

    def reveal_head_controls(self,*args):
        if self.show_stream_arrows.get():self.head_controls.pack(fill='x',after=self.head_checkbox)
        else:self.head_controls.pack_forget()

    def choose_color_preset(self):
        picker=getattr(self,'preset_picker',None)
        if picker is not None and picker.winfo_exists():picker.lift();picker.focus_set();return
        def apply(name):self.colors.set(name);self.schedule()
        self.preset_picker=StreamPresetPicker(self.gui.root,self.colors.get(),apply)

    def toggle_advanced(self):
        if self.advanced.get():self.advanced_panel.pack(fill='x',pady=3)
        else:self.advanced_panel.pack_forget()

    def auto_changed(self):
        if self.auto_apply.get() and self.pending:self.apply()

    def schedule(self):
        if hasattr(self,'auto_apply') and not self.auto_apply.get():
            self.pending=True;self.status.set('Settings changed. Click Apply.');return
        super().schedule()

    def apply(self):
        self.pending=False;super().schedule()

    def generate_levels(self):
        try:
            lo,hi = float(self.level_min.get()),float(self.level_max.get());count=int(self.level_count.get())
            if not all(math.isfinite(v) for v in (lo,hi)) or lo>hi or not 1<=count<=100:raise ValueError()
        except ValueError:self.status.set('Use finite contour bounds, From ≤ To, and 1–100 values.');return
        self.levels.set(', '.join(f'{v:.6g}' for v in np.linspace(lo,hi,count)))
        self.schedule()

    def section_settings(self):return None

    def prepare_section_request(self,settings,style,native,factor,roi):
        pass

    def filter_settings(self):
        def positive(var,label,optional=False,integer=False,zero=False):
            text=var.get().strip()
            if optional and not text:return None
            try:value=int(text) if integer else float(text)
            except ValueError:raise ValueError(label+' must be a number.')
            if not math.isfinite(value) or value<0 or (value==0 and not zero):raise ValueError(label+' must be '+('nonnegative.' if zero else 'positive.'))
            return value
        def point(var,label):
            if not var.get().strip():return None
            try:values=[float(v) for v in re.split(r'[,;\s]+',var.get().strip())]
            except ValueError:raise ValueError(label+' needs X, Y, Z coordinates.')
            if len(values)!=3 or not all(math.isfinite(v) for v in values):raise ValueError(label+' needs three finite coordinates.')
            return values
        levels=[float(v.strip()) for v in self.levels.get().split(',') if v.strip()]
        if self.show_iso.get() and (not levels or len(levels)>100 or not all(math.isfinite(v) for v in levels)):
            raise ValueError('Enter 1–100 finite contour values.')
        count=positive(self.seeds,'Seed count',integer=True)
        if count>100000:raise ValueError('Seed count must be 1–100000.')
        options=dict(seed_type=self.seed_type.get(),seed_spacing=positive(self.seed_spacing,'Seed spacing',True),
            seed_center=point(self.seed_center,'Point center'),seed_radius=positive(self.seed_radius,'Point radius',True),
            line_start=point(self.line_start,'Line start'),line_end=point(self.line_end,'Line end'),
            z_layers=positive(self.z_layers,'Z layers',True,integer=True),direction=self.direction.get().lower(),
            auto_step=self.auto_step.get(),
            integrator={'Runge-Kutta 4-5':45,'Runge-Kutta 4':4,'Runge-Kutta 2':2}[self.integrator.get()],
            max_length=positive(self.max_length,'Maximum length',True),initial_step=positive(self.initial_step,'Initial step'),
            min_step=positive(self.min_step,'Minimum step'),max_step=positive(self.max_step,'Maximum step'),
            step_unit='cl' if self.step_unit.get()=='Cell length' else 'l',max_steps=positive(self.max_steps,'Maximum steps',integer=True),
            terminal_speed=positive(self.terminal_speed,'Terminal magnitude',zero=True),max_error=positive(self.max_error,'Maximum error'),
            contour_component=self.component.get() if self.contour_component.get()=='color' else self.contour_component.get())
        if not self.auto_step.get() and not options['min_step']<=options['initial_step']<=options['max_step']:
            raise ValueError('Integration steps must satisfy minimum ≤ initial ≤ maximum.')
        radius=positive(self.tube_radius,'Tube radius');line_width=positive(self.line_width,'Line width')
        text_size=positive(self.text_size,'Text size',integer=True)
        if not 6<=text_size<=40:raise ValueError('Text size must be 6–40.')
        lo,hi=float(self.color_low.get()),float(self.color_high.get())
        if self.color_range_mode.get()=='Fixed' and (not math.isfinite(lo) or not math.isfinite(hi) or lo>=hi):
            raise ValueError('Color minimum must be less than maximum.')
        stream_color_map(self.colors.get())
        arrow_style=dict(show=False,spacing=None,length=None,radius=None,resolution=24,limit=8000,color='By component')
        if self.show_stream_arrows.get():
            arrow_style.update(show=True,spacing=positive(self.head_spacing,'Head spacing',True),
                length=positive(self.head_length,'Head length',True),radius=positive(self.head_radius,'Head radius',True),
                resolution=positive(self.head_resolution,'Head resolution',integer=True),limit=positive(self.head_limit,'Maximum heads',integer=True),color=self.head_color.get())
            if arrow_style['resolution'] not in (8,12,16,24,36,48,64):raise ValueError('Choose a listed head resolution.')
            if not 1<=arrow_style['limit']<=50000:raise ValueError('Maximum heads must be 1–50000.')
        settings=dict(component=self.component.get(),levels=levels,seed_count=count,seed_plane=self.seed_plane.get(),
                      streams=self.show_streams.get(),iso=self.show_iso.get(),options=options,section=self.section_settings())
        section_visual={}
        if settings['section']:
            section_visual={name:settings['section'].pop(name) for name in ('show_slice','slice_opacity')}
        style=dict(colors=self.colors.get(),opacity=float(self.opacity.get()),scales=self.show_scales.get(),bar=self.show_bar.get(),
            time=self.show_time.get(),radius=radius,component=self.component.get(),line_width=line_width,tube_sides=int(self.tube_sides.get()),
            stream_style=self.stream_style.get(),stream_opacity=float(self.stream_opacity.get()),contour_style=self.contour_style.get(),
            contour_color=self.contour_color.get(),smooth_normals=self.smooth_normals.get(),seeds=self.show_seeds.get(),
            outline=self.show_outline.get(),range_mode=self.color_range_mode.get(),clim=(lo,hi),section_visual=section_visual,
            bar_size=self.bar_size.get(),bar_position=self.bar_position.get(),text_size=text_size,decimals=self.show_decimals.get(),arrows=arrow_style)
        style['layer_fill']=self.layer_fill_control.snapshot()
        ambient=float(self.ambient_light.get())
        if not math.isfinite(ambient) or not 0<=ambient<=1:raise ValueError('Ambient light must be 0–1.')
        style['ambient_light']=ambient
        style['light']=light_control_settings(self)
        style['reflection']=reflection_control_settings(self)
        return settings,style

    def start_build(self):
        self.update_timer=None
        if self.closed or self.worker_busy or not self.files:return
        try:
            index=min(len(self.files)-1,max(0,round(self.frame.get())-1));source=self.files[index]
            factor=int(self.factor.get().replace('x',''));settings,style=self.filter_settings()
            self.sync_roi(False);roi=self.roi_physical;style['roi']=roi
        except Exception as exc:self.status.set(str(exc));return
        token=self.generation;cache=self.native_cache;cache_stamp=self.native_stamp;cached_vtk=self.native_vtk_path
        analysis_cache=self.analysis_cache;geometry_cache=self.geometry_cache
        self.worker_busy=True;self.pending=False
        self.frame_label.set(f'{index+1}/{len(self.files)}: {source.name}')
        self.status.set('Computing…')
        if not self.scene_valid:self.show_message('Preparing the view…')
        def worker():
            try:
                from time import perf_counter
                started=perf_counter()
                require_tab4_backend();info=source.stat();stamp=(info.st_size,info.st_mtime_ns)
                if cache is not None and cache[0]==source and cache_stamp==stamp:native,vtk_path=cache[1],cached_vtk
                else:native,vtk_path=self.vtk_cache.read_source(source)
                style['source_stamp']=stamp;style['vtk_path']=str(vtk_path) if vtk_path is not None else None
                self.prepare_section_request(settings,style,native,factor,roi)
                key=(str(source.resolve()),stamp,factor,json.dumps(roi,sort_keys=True))
                if analysis_cache is not None and analysis_cache[0]==key:
                    vectors,mesh,prepared=analysis_cache[1:]
                else:
                    vectors,mesh=refine_stream_volume(native[0],native[1],factor,roi)
                    prepared=magnetization_vtk_grid(vectors,mesh)
                style['analysis_cache']=(key,vectors,mesh,prepared)
                if settings['section']:
                    axis={'XY':2,'YZ':0,'XZ':1}[settings['section']['view']]
                    bounds=np.asarray(prepared[0].bounds).reshape(3,2)
                    position=settings['section']['position']
                    auto_center=vtk_material_center(prepared[1]) if prepared[1].n_points else np.mean(bounds,axis=1)
                    position=auto_center[axis] if position is None else np.clip(position,*bounds[axis])
                    settings['section']['position']=round(float(position),9)
                geometry_key=(key,json.dumps(settings,sort_keys=True))
                if geometry_cache is not None and geometry_cache[0]==geometry_key:result=geometry_cache[1]
                else:result=build_stream_iso_scene(vectors,mesh,**settings,prepared=prepared)
                style['geometry_key']=geometry_key
                style['compute_seconds']=perf_counter()-started
                if hasattr(self,'build_overview_data'):
                    style['overview_data']=self.build_overview_data(source,stamp,native,factor,style)
                payload=(source,native,vectors,mesh,result,style,None)
            except Exception as exc:payload=(source,None,None,None,None,None,str(exc))
            if not self.closed and not self.gui.closed:self.gui.events.put((self.built,(token,payload)))
        threading.Thread(target=worker,daemon=True).start()

    def built(self,token,payload):
        accepted=token==self.generation and not self.closed
        super().built(token,payload)
        if not accepted or payload[-1] or not self.scene_valid:return
        self.layer_fill_control.sync_layers(layer_fill_source_z(self))
        self.analysis_cache=self.style.pop('analysis_cache')
        self.geometry_cache=(self.style.pop('geometry_key'),self.result)
        lo,hi=self.result['contour_range'];self.contour_range_label.set('Range: '+stream_number(lo,self.style['decimals'])+' … '+stream_number(hi,self.style['decimals']))
        lines=self.result['streams'];surface=self.result['iso'];seeds=self.result['seeds']
        source=payload[0]
        message=f'{source.name} | {seeds.n_points} seeds | {lines.n_lines} streamlines | {surface.n_cells} contour cells'
        if self.style['arrows']['show'] and self.stream_arrow_cache[1] is not None:
            glyphs=self.stream_arrow_cache[1]
            if glyphs.n_points:
                count=int(glyphs.field_data['arrow_count'][0]);requested=int(glyphs.field_data['requested_arrow_count'][0])
                message+=f' | {count} arrow heads'+(f' (capped from {requested})' if requested>count else '')
        section=self.result.get('section')
        if section:message+=f' | {section["view"]} {section["mode"].lower()} at '+stream_number(section['position'],self.style['decimals'])+' '+self.mesh['x']['unit']
        if self.show_iso.get() and not surface.n_points:message+=' | No contour at the selected values.'
        if self.show_streams.get() and not lines.n_points:message+=' | No streamlines: check seeds, vector direction, or terminal magnitude.'
        if field_source_note(source):message += ' | '+field_source_note(source)
        if self.native_vtk_path:message+=' | VTK cache: '+self.native_vtk_path
        message+=f' | Compute {self.style["compute_seconds"]:.3f} s'
        self.status.set(message)

    def camera_default(self):
        view=self.volume_view.get()
        if view=='3D':self.plotter.view_isometric();self.plotter.disable_parallel_projection()
        else:
            {'XY':self.plotter.view_xy,'YZ':self.plotter.view_yz,'XZ':self.plotter.view_xz}[view]()
            self.plotter.enable_parallel_projection()
        self.plotter.reset_camera()

    def volume_view_changed(self,event=None):
        if self.scene_valid:StreamVolumeControlsBase.camera_default(self);self.paint()

    def display_scene(self):
        pv=require_tab4_backend();width,height=max(200,self.canvas.winfo_width()),max(180,self.canvas.winfo_height())
        camera=None
        if self.plotter is None:
            self.plotter=pv.Plotter(off_screen=True,window_size=(width,height),shape=(1,2),col_weights=[.88,.12],border=False)
            self.plotter.subplot(0,0)
        elif self.scene_valid:camera=self.plotter.camera_position
        plotter=self.plotter
        cmap=stream_color_map(self.style['colors'])
        component=self.style['component']
        clim=self.result['scalar_range'] if self.style['range_mode']=='Data range' else self.style['clim']
        if clim[0]==clim[1]:clim=(clim[0]-.5,clim[1]+.5)
        if component=='mag' and self.style['range_mode']=='Fixed' and clim==(-1.,1.):clim=(0.,max(1.,self.result['scalar_range'][1]))
        section=self.result.get('section');bar_used=False
        section_visual=self.style.get('section_visual',{})
        structural={name:self.style[name] for name in ('component','radius','tube_sides','stream_style','contour_style',
            'contour_color','smooth_normals','seeds','outline','scales','bar','bar_size','bar_position','text_size','decimals')}
        structural['show_slice']=section_visual.get('show_slice',True)
        structural['arrows']=self.style['arrows']
        fill_state=layer_fill_effective_state(self.style.get('layer_fill',{}))
        structural['merge_layer']=bool(fill_state.get('merge'))
        structural['layer_fill']=[dict({k:v for k,v in r.items() if k!='transparency'},visible=r['transparency']<100)
            for r in fill_state.get('layers',[])] if fill_state.get('enabled') else []
        if any(r['mode']=='Component color map' for r in fill_state.get('layers',[])):
            structural['fill_map']=(self.style['colors'],tuple(clim))
        display_key=(id(self.result),json.dumps(structural,sort_keys=True))
        if self.scene_valid and self.display_cache_key==display_key:
            apply_stream_light_controls(self)
            for kind,(actor,colored) in self.actor_cache.items():
                opacity={'slice':section_visual.get('slice_opacity',.35),'iso':self.style['opacity'],'streams':self.style['stream_opacity'],'arrows':self.style['stream_opacity']}[kind]
                actor.prop.opacity=opacity;actor.prop.line_width=self.style['line_width']
                if actor.prop.lighting:
                    actor.prop.ambient=self.style.get('ambient_light',.35);actor.prop.diffuse=1-self.style.get('ambient_light',.35)
                if colored:
                    actor.mapper.lookup_table.n_values=257
                    actor.mapper.lookup_table.cmap=cmap
                    actor.mapper.scalar_range=clim;actor.mapper.lookup_table.scalar_range=clim
            transparency={r['index']:r['transparency'] for r in fill_state.get('layers',[])}
            fill_records={r['index']:r for r in fill_state.get('layers',[])}
            for layer,actor in getattr(self,'layer_fill_actors',{}).items():
                actor.prop.opacity=1-transparency.get(layer,100)/100
                if layer in fill_records:apply_layer_finish(actor,fill_records[layer],self.style.get('ambient_light',.35))
            if getattr(self,'reorient',False):self.camera_default();self.reorient=False
            apply_stream_reflection(self)
            self.paint();return
        plotter.clear();plotter.set_background('white',all_renderers=True);self.actor_cache={};self.layer_fill_actors={}
        plotter.subplot(0,0);bar_mapper=None
        apply_stream_light_controls(self)
        def add(data,**kwargs):
            nonlocal bar_used,bar_mapper
            if not data.n_points:return None
            colored=kwargs.pop('colored',True)
            if kwargs.get('lighting',True):
                kwargs['ambient']=self.style.get('ambient_light',.35);kwargs['diffuse']=1-self.style.get('ambient_light',.35)
            common=dict(show_scalar_bar=False)
            if colored:common.update(scalars=component,cmap=cmap,clim=clim,n_colors=257)
            actor=plotter.add_mesh(data,**common,**kwargs)
            if colored:
                bar_used=True
                if bar_mapper is None:bar_mapper=actor.mapper
            return actor,colored
        if section and section_visual.get('show_slice',True):
            actor=add(self.result['slice'],opacity=section_visual.get('slice_opacity',.35),lighting=False)
            if actor is not None:self.actor_cache['slice']=actor
        surface=self.result['iso']
        if surface.n_points:
            plane=bool(section and section['mode']=='Plane')
            color=self.style['contour_color']
            extra={} if color=='By component' else dict(color=color.lower(),colored=False)
            representation=self.style['contour_style']
            self.actor_cache['iso']=add(surface,opacity=self.style['opacity'],smooth_shading=self.style['smooth_normals'],
                style='wireframe' if representation=='Wireframe' else 'surface',show_edges=representation=='Surface with edges',
                line_width=max(1,self.style['line_width']),ambient=.35,diffuse=.75,specular=.25,specular_power=25,
                lighting=not plane,**extra)
        lines=self.result['streams']
        if lines.n_points:
            if self.style['stream_style']=='Tubes':
                tube_key=(id(lines),self.style['radius'],self.style['tube_sides'])
                if getattr(self,'tube_cache',(None,None))[0]!=tube_key:
                    self.tube_cache=(tube_key,lines.tube(radius=self.style['radius'],n_sides=self.style['tube_sides']))
                data=self.tube_cache[1]
            else:data=lines
            self.actor_cache['streams']=add(data,opacity=self.style['stream_opacity'],line_width=self.style['line_width'],smooth_shading=True,
                ambient=.35,diffuse=.75,specular=.25,specular_power=25)
            if self.style['arrows']['show']:
                head_geometry={k:v for k,v in self.style['arrows'].items() if k not in ('show','color')}
                arrow_key=(id(lines),json.dumps(head_geometry,sort_keys=True),self.style['radius'])
                if self.stream_arrow_cache[0]!=arrow_key:
                    self.stream_arrow_cache=(arrow_key,streamline_arrow_glyphs(lines,self.style['radius'],options=head_geometry))
                glyphs=self.stream_arrow_cache[1]
                if glyphs.n_points:
                    extra={} if self.style['arrows']['color']=='By component' else dict(color=self.style['arrows']['color'].lower(),colored=False)
                    self.actor_cache['arrows']=add(glyphs,opacity=self.style['stream_opacity'],smooth_shading=True,
                        ambient=.2,diffuse=.7,specular=.2,specular_power=25,**extra)
        fill_planes=stream_layer_fill_planes(self,component,clim)
        for plane in fill_planes:
            actor=plotter.add_mesh(vtk_layer_fill_mesh(plane),scalars='layer_rgba',rgba=True,preference='cell',
                opacity=plane['opacity'],show_scalar_bar=False,**layer_finish_material(plane['record'],self.style.get('ambient_light',.35)))
            apply_layer_finish(actor,plane['record'],self.style.get('ambient_light',.35))
            self.layer_fill_actors[plane['index']]=actor
        if fill_planes:plotter.enable_depth_peeling(number_of_peels=16,occlusion_ratio=0.)
        if self.style['seeds'] and self.result['seeds'].n_points:
            plotter.add_mesh(self.result['seeds'],color='#222222',point_size=5,render_points_as_spheres=True)
        if self.style['outline']:plotter.add_mesh(self.result['grid'].outline(),color='#999999',line_width=1)
        have_bar=self.style['bar'] and bar_mapper is not None
        plotter.renderers[0].SetViewport(0.,0.,.88 if have_bar else 1.,1.)
        plotter.renderers[1].SetDraw(bool(have_bar))
        self.update_header_viewport()
        if have_bar:
            plotter.subplot(0,1)
            height,bar_width={'Small':(.4,.16),'Medium':(.55,.22),'Bigger':(.7,.28)}[self.style['bar_size']]
            bottom={'Top':.95-height,'Middle':(1-height)/2,'Bottom':.05}[self.style['bar_position']]
            plotter.add_scalar_bar(mapper=bar_mapper,title='',vertical=True,position_x=.12,position_y=bottom,
                width=bar_width,height=height,n_labels=3,fmt='%.3f' if self.style['decimals'] else '%.0f',color='black',
                label_font_size=self.style['text_size'],title_font_size=self.style['text_size'],
                use_opacity=False)
            plotter.subplot(0,0)
        bounds=self.result['display_bounds']
        axes=dict()
        if section and section['mode']=='Plane' and not self.style['outline'] and self.result['slice'].n_points:
            bounds=self.result['slice'].bounds;axes['show_'+{'XY':'z','YZ':'x','XZ':'y'}[section['view']]+'axis']=False
        if self.style['scales']:
            unit=self.mesh['x']['unit']
            plotter.show_bounds(bounds=bounds,xtitle=f'X ({unit})',ytitle=f'Y ({unit})',ztitle=f'Z ({unit})',
                color='black',font_size=self.style['text_size'],n_xlabels=4,n_ylabels=4,n_zlabels=4,fmt='%.3f' if self.style['decimals'] else '%.0f',grid='back',location='outer',**axes)
        if camera is None or getattr(self,'reorient',False):self.camera_default();self.reorient=False
        else:plotter.camera_position=camera
        plotter.enable_anti_aliasing('fxaa')
        self.display_cache_key=display_key
        apply_stream_reflection(self)
        self.empty_message=None;self.scene_valid=True;self.paint()

    def reset_camera(self):
        if self.plotter is not None and self.scene_valid:self.camera_default();self.paint()

    def move_drag(self,event):
        if self.drag is None or not self.scene_valid:return
        x,y,mode=self.drag;dx,dy=event.x-x,event.y-y;camera=self.plotter.camera
        if mode=='rotate':camera.Azimuth(-dx*.4);camera.Elevation(dy*.4);camera.OrthogonalizeViewUp()
        else:
            position,focal=np.array(camera.position),np.array(camera.focal_point)
            normal=(focal-position)/np.linalg.norm(focal-position)
            up=np.array(camera.up);right=np.cross(normal,up);right/=np.linalg.norm(right);up=np.cross(right,normal)
            span=2*camera.parallel_scale if camera.parallel_projection else 2*np.linalg.norm(focal-position)*math.tan(math.radians(camera.view_angle)/2)
            shift=(-dx*right+dy*up)*span/max(1,self.canvas.winfo_height())
            camera.position=position+shift;camera.focal_point=focal+shift
        self.drag=event.x,event.y,mode
        if self.paint_timer is None:self.paint_timer=self.after(16,self.paint)

    def paint(self):
        if getattr(self,'paint_timer',None) is not None:self.after_cancel(self.paint_timer);self.paint_timer=None
        if self.plotter is not None:
            self.plotter.window_size=(max(200,self.canvas.winfo_width()),max(180,self.canvas.winfo_height()))
            self.update_header_viewport()
        super().paint()

    def export_png(self):
        if not self.scene_valid:self.status.set('Load a readable volume first.');return
        if self.worker_busy or self.update_timer is not None:self.status.set('Wait for the current frame to finish loading.');return
        parent=filedialog.askdirectory(parent=self.gui.root,title='Choose output folder for Tab 4 PNG')
        if not parent:return
        folder=None
        try:
            index=min(len(self.files)-1,max(0,round(self.frame.get())-1));source=self.files[index]
            folder=auto_export_folder(parent,4,source,'Images')
            view='Analysis_'+(self.slice_view.get() if self.cross_section_enabled.get() else self.volume_view.get())
            filename=auto_export_filename(source,view,self.style['component'],int(self.factor.get().replace('x','')),'png',index)
            self.save_png(folder/filename)
            self.status.set('Saved 300 PPI PNG in '+str(folder));open_folder(folder)
        except Exception as exc:
            discard_empty_export_folder(folder);self.status.set('Image export failed: '+str(exc))

    def save_png(self,path):
        self.paint();super().save_png(path)

    def figure_header_lines(self):
        label='|m|' if self.style['component']=='mag' else self.style['component']
        lines=[label]
        if self.style['time'] and self.time_ns is not None:
            lines.append('t = '+stream_number(self.time_ns,self.style['decimals'])+' ns')
        section=self.result.get('section') if self.result else None
        if section:
            axis={'XY':'Z','YZ':'X','XZ':'Y'}[section['view']]
            label=f'{section["view"]} | {axis} = '+stream_number(section['position'],self.style['decimals'])+' '+self.mesh['x']['unit']
            if section['mode']=='Slab':label+=' | thickness '+stream_number(section['thickness'],self.style['decimals'])
            lines.append(label)
        return lines

    def update_header_viewport(self):
        height=self.plotter.window_size[1]
        top=max(.2,1-figure_header_height(self.figure_header_lines(),self.style['text_size'])/height)
        have_bar=bool(self.plotter.renderers[1].GetDraw())
        self.plotter.renderers[0].SetViewport(0.,0.,.88 if have_bar else 1.,top)
        self.plotter.renderers[1].SetViewport(.88,0.,1.,top)

    def overlay(self,image,scale=1):
        image=draw_figure_header(image,self.figure_header_lines(),self.style['text_size'],scale)
        if self.plotter is None or not self.style['bar'] or not len(self.plotter.scalar_bars):return image
        renderer=self.plotter.renderers[1]
        if not renderer.GetDraw():return image
        bar=next(iter(self.plotter.scalar_bars.values()))
        rect=[0,0,0,0];bar.GetScalarBarRect(rect,renderer)
        if rect[2]<=0 or rect[3]<=0:return image
        width,height=self.plotter.window_size;viewport=renderer.GetViewport()
        sx,sy=image.width/width,image.height/height
        center_x=(viewport[0]*width+rect[0]+rect[2]/2)*sx
        top_y=image.height-(viewport[1]*height+rect[1]+rect[3])*sy
        size=self.style['text_size'];font=annotation_font(round(size*scale))
        component='|m|' if self.style['component']=='mag' else self.style['component']
        title=component_title_image(component,font)
        # Reserve a full line for VTK endpoint text, including high-DPI exports.
        tick_half_height=font.size
        gap=max(round(8*scale),round(size*.8*scale))
        x=round(center_x-title.width/2)
        y=max(round(2*scale),round(top_y-tick_half_height-gap-title.height))
        image.paste(title,(x,y),title)
        return image

    def close(self):
        if self.paint_timer is not None:self.after_cancel(self.paint_timer);self.paint_timer=None
        picker=getattr(self,'preset_picker',None)
        if picker is not None and picker.winfo_exists():picker.destroy()
        self.actor_cache={};self.layer_fill_actors={};self.display_cache_key=None
        self.analysis_cache=self.geometry_cache=None;self.tube_cache=(None,None)
        self.stream_arrow_cache=(None,None)
        super().close()


from types import SimpleNamespace


def stream_refined_mesh(native_mesh,shape,factor):
    result={}
    for axis,count in zip('xyz',shape[:3][::-1]):
        info=native_mesh[axis];low,high=info['edges'];n=count*factor
        result[axis]=dict(info,centers=low+(np.arange(n)+.5)*(high-low)/n)
    return result


def stream_default_bounds(native,factor,available_roi=None):
    vectors,mesh=native[:2];full=stream_refined_mesh(mesh,vectors.shape,factor)
    cells={}
    for axis in 'xy':
        centers=full[axis]['centers'];first,last=1,len(centers)
        if available_roi:
            first,last=np.searchsorted(centers,available_roi[axis],side='left')
            first,last=max(1,int(first)+1),min(len(centers),int(last))
            if last<first:raise ValueError('The ROI contains no cells.')
        cells[axis]=[first,last]
    weight=np.count_nonzero(np.linalg.norm(vectors,axis=-1)>1e-8,axis=(1,2))
    z=float(np.average(mesh['z']['centers'],weights=weight)) if weight.sum() else float(np.mean(mesh['z']['centers']))
    cells['z']=min(len(full['z']['centers']),max(1,int(np.argmin(abs(full['z']['centers']-z)))+1))
    return full,cells


def stream_resolve_bounds(native,factor,roi,request):
    full,defaults=stream_default_bounds(native,factor,roi)
    cells=request.get('cells')
    old_factor=request.get('factor',factor) or factor
    if cells is None:cells=defaults
    else:
        cells=json.loads(json.dumps(cells));ratio=factor/old_factor
        for axis in 'xy':
            start,end=cells[axis]
            if ratio!=1:
                if start==end:start=end=round((start-.5)*ratio+.5)
                else:start,end=math.floor((start-1)*ratio)+1,math.ceil(end*ratio)
            low,high=defaults[axis]
            start=min(high,max(low,round(start)));end=min(high,max(start,round(end)))
            cells[axis]=[start,end]
        cells['z']=min(len(full['z']['centers']),max(1,round((cells['z']-.5)*ratio+.5)))
    return full,cells


def stream_section_from_bounds(section,full,cells):
    output=dict(section);output.pop('bounds_request',None)
    xy={}
    for axis in 'xy':
        first,last=cells[axis];low,high=full[axis]['edges'];step=(high-low)/len(full[axis]['centers'])
        xy[axis]=[float(low+(first-1)*step),float(low+last*step)]
    normal={'XY':'z','YZ':'x','XZ':'y'}[section['view']]
    if normal=='z':
        output.update(mode='Plane',position=float(full['z']['centers'][cells['z']-1]))
    else:
        first,last=cells[normal]
        if first==last:output.update(mode='Plane',position=float(full[normal]['centers'][first-1]))
        else:output.update(mode='Slab',position=sum(xy[normal])/2,thickness=xy[normal][1]-xy[normal][0])
    output['xy_bounds']=xy
    return output


def stream_refined_xy_plane(native,factor,z_cell):
    """Refine XY before blending Z, matching the full-volume interpolation order."""
    vectors,mesh=native[:2]
    total=vectors.shape[0]*factor;index=min(total-1,max(0,int(z_cell)-1))
    position=float(np.clip((index+.5)/factor-.5,0,vectors.shape[0]-1))
    lower=int(math.floor(position));upper=min(lower+1,vectors.shape[0]-1)
    data=np.asarray(vectors[[lower,upper]],dtype=np.float32)
    for axis in (2,1):
        count=data.shape[axis];positions=np.clip((np.arange(count*factor)+.5)/factor-.5,0,count-1)
        low=np.floor(positions).astype(int);high=np.minimum(low+1,count-1)
        shape=list(data.shape);shape[axis]*=factor;refined=np.empty(shape,np.float32)
        for first in range(0,len(positions),8):
            last=min(len(positions),first+8);weight_shape=[1]*4;weight_shape[axis]=last-first
            weight=(positions[first:last]-low[first:last]).astype(np.float32).reshape(weight_shape)
            a=np.take(data,low[first:last],axis=axis);b=np.take(data,high[first:last],axis=axis)
            target=[slice(None)]*4;target[axis]=slice(first,last);refined[tuple(target)]=a+(b-a)*weight
        data=refined
    plane=data[0]+(data[1]-data[0])*np.float32(position-lower)
    occupied=np.any(vectors[index//factor]!=0,axis=-1)
    occupied=np.repeat(np.repeat(occupied,factor,axis=0),factor,axis=1)
    plane[~occupied]=0
    return plane,stream_refined_mesh(mesh,vectors.shape,factor)


class StreamAnalysisControlsMixin:
    def initialize_position_controls(self):
        self.cross_section_enabled=tk.BooleanVar(value=False)
        self.use_bounds=tk.BooleanVar(value=True)
        self.positions={a:tk.DoubleVar(value=1) for a in 'xyz'}
        self.band_ends={a:tk.DoubleVar(value=1) for a in 'xy'}
        self.position_labels={a:tk.StringVar() for a in 'xyz'}
        self.end_labels={a:tk.StringVar() for a in 'xy'}
        self.position_sliders={};self.end_sliders={}
        self.bounds_initialized=False;self.bounds_factor=1;self.full_mesh=None
        self.updating_positions=False;self.local_roi=None;self.roi_enabled=tk.BooleanVar(value=False)
        self.roi_selecting=False;self.roi_drag=None;self.reference_drag=None;self.overview_pan=None
        self.overview_plane=None;self.overview_cache=None;self.overview_scalar_cache=None
        self.overview_zoom=1.;self.overview_center=(.5,.5);self.overview_rect=None
        self.active_view='Analysis';self.range_start=tk.StringVar(value='1');self.range_end=tk.StringVar(value='1')
        self.play_fps=tk.StringVar(value='24');self.playing=False;self.play_timer=None
        self.mp4_busy=False;self.mp4_stop=threading.Event();self.mp4_lock=threading.Lock()
        self.mp4_job=None;self.mp4_thread=None

    def build_position_roi_movie_controls(self,old_group):
        # Cross-section controls share Tab 4's data and rendering pipeline.
        children=old_group.winfo_children()
        for child in children[:2]:child.destroy()
        old_group.configure(text='Slice display / manual position')
        group=self.group('Cross-section position');group.pack_configure(before=old_group)
        ttk.Label(group,text='View').pack(anchor='w')
        box=ttk.Combobox(group,textvariable=self.slice_view,values=['XY','YZ','XZ'],state='readonly')
        box.pack(fill='x');box.bind('<<ComboboxSelected>>',self.slice_view_changed)
        ttk.Checkbutton(group,text='Use X/Y bounds',variable=self.use_bounds,command=self.schedule).pack(anchor='w',pady=2)
        for axis in 'xy':
            for is_end in (False,True):
                label=axis.upper()+('2 end' if is_end else '1 start')
                variable=self.band_ends[axis] if is_end else self.positions[axis]
                value=self.end_labels[axis] if is_end else self.position_labels[axis]
                ttk.Label(group,text=label).pack(anchor='w')
                scale=ttk.Scale(group,from_=1,to=1,variable=variable,command=lambda v,a=axis,e=is_end:self.band_changed(a,e))
                scale.pack(fill='x');ttk.Label(group,textvariable=value).pack(anchor='w')
                (self.end_sliders if is_end else self.position_sliders)[axis]=scale
        ttk.Label(group,text='Z (XY layer)').pack(anchor='w')
        self.position_sliders['z']=ttk.Scale(group,from_=1,to=1,variable=self.positions['z'],command=lambda v:self.z_changed())
        self.position_sliders['z'].pack(fill='x');ttk.Label(group,textvariable=self.position_labels['z']).pack(anchor='w')
        position_group=group
        group=self.group('Region of interest');group.pack_configure(before=old_group)
        row=ttk.Frame(group);row.pack(fill='x',pady=3)
        ttk.Button(row,text='2D overview',command=lambda:self.open_section_view('Overview')).pack(side='left',fill='x',expand=True)
        ttk.Button(row,text='Analysis view',command=lambda:self.open_section_view('Analysis')).pack(side='left',fill='x',expand=True)
        ttk.Checkbutton(group,text='Enable ROI',variable=self.roi_enabled,command=self.toggle_local_roi).pack(anchor='w')
        ttk.Button(group,text='Select / change ROI',command=self.select_local_roi).pack(fill='x',pady=2)
        ttk.Button(group,text='Clear ROI',command=self.clear_local_roi).pack(fill='x',pady=2)
        parent=self.canvas.master
        self.overview_canvas=tk.Canvas(parent,background='white',highlightthickness=0)
        self.overview_canvas.bind('<Configure>',lambda e:self.paint_overview())
        self.overview_canvas.bind('<ButtonPress-1>',self.overview_press)
        self.overview_canvas.bind('<B1-Motion>',self.overview_move)
        self.overview_canvas.bind('<ButtonRelease-1>',self.overview_release)
        self.overview_canvas.bind('<MouseWheel>',self.overview_scroll)
        for button in (2,3):
            self.overview_canvas.bind(f'<ButtonPress-{button}>',self.overview_pan_start)
            self.overview_canvas.bind(f'<B{button}-Motion>',self.overview_pan_move)
            self.overview_canvas.bind(f'<ButtonRelease-{button}>',lambda e:setattr(self,'overview_pan',None))
        group=self.group('Playback / MP4');group.pack_configure(before=self.groups['Component'])
        row=ttk.Frame(group);row.pack(fill='x')
        self.play_button=ttk.Button(row,text='Play',command=self.toggle_play);self.play_button.pack(side='left',fill='x',expand=True)
        ttk.Label(row,text='FPS').pack(side='left',padx=4)
        ttk.Spinbox(row,from_=1,to=120,textvariable=self.play_fps,width=6).pack(side='left')
        row=ttk.Frame(group);row.pack(fill='x',pady=3)
        for label,var in (('Start frame',self.range_start),('End frame',self.range_end)):
            frame=ttk.Frame(row);frame.pack(side='left',fill='x',expand=True)
            ttk.Label(frame,text=label).pack(anchor='w');ttk.Spinbox(frame,from_=1,to=1000000,textvariable=var,width=8).pack(fill='x')
        row=ttk.Frame(group);row.pack(fill='x')
        ttk.Button(row,text='Export MP4',command=self.export_mp4).pack(side='left',fill='x',expand=True)
        self.mp4_stop_button=ttk.Button(row,text='Stop',command=self.stop_mp4,state='disabled')
        self.mp4_stop_button.pack(side='left',padx=2)
        self.section_groups=(position_group,old_group)
        ttk.Checkbutton(self.groups['Frames'],text='Cross-section analysis',variable=self.cross_section_enabled,command=self.cross_section_changed).pack(anchor='w',pady=3)
        for widget in self.section_groups:widget.pack_forget()

    def reveal_section_controls(self):
        for widget in self.section_groups:widget.pack(fill='x',pady=4,before=self.groups['Component'])

    def cross_section_changed(self):
        if self.cross_section_enabled.get():self.reveal_section_controls()
        else:
            for widget in self.section_groups:widget.pack_forget()
        self.reorient=True;self.schedule();self.paint_overview()

    def reset_section_bounds(self):
        self.pause_play();self.bounds_initialized=False;self.full_mesh=None
        self.overview_plane=self.overview_cache=self.overview_scalar_cache=None
        self.local_roi=None;self.roi_enabled.set(False);self.roi_selecting=False;self.roi_drag=None
        self.overview_zoom=1.;self.overview_center=(.5,.5)

    def bounds_snapshot(self):
        cells=None
        if self.bounds_initialized:
            cells={a:[round(self.positions[a].get()),round(self.band_ends[a].get())] for a in 'xy'}
            cells['z']=round(self.positions['z'].get())
        return dict(cells=cells,factor=self.bounds_factor)

    def filter_settings(self):
        settings,style=super().filter_settings()
        style['position_request']=self.bounds_snapshot()
        return settings,style

    def prepare_section_request(self,settings,style,native,factor,roi):
        style['layer_source_z']=native[1]['z']
        style['layer_native']=native[:2]
        full,cells=stream_resolve_bounds(native,factor,roi,style['position_request'])
        section=settings['section']
        if section and section.get('bounds_request') is not None:
            section=stream_section_from_bounds(section,full,cells);settings['section']=section
        style['resolved_cells']=cells;style['full_mesh']=full;style['bounds_factor']=factor
        style['overview_z']=cells['z']

    def build_overview_data(self,source,stamp,native,factor,style):
        key=(str(source),stamp,factor,style['overview_z'])
        if self.overview_cache is not None and self.overview_cache[0]==key:return self.overview_cache
        plane,mesh=stream_refined_xy_plane(native,factor,style['overview_z'])
        return key,plane,mesh

    def accept_overview_data(self):
        self.full_mesh=self.style.pop('full_mesh')
        cells=self.style.pop('resolved_cells');self.bounds_factor=self.style.pop('bounds_factor')
        self.overview_cache=self.style.pop('overview_data');self.overview_plane=self.overview_cache[1]
        self.updating_positions=True
        try:
            for axis in 'xy':
                first,last=cells[axis];count=len(self.full_mesh[axis]['centers'])
                self.positions[axis].set(first);self.band_ends[axis].set(last)
                self.position_sliders[axis].configure(from_=1,to=count);self.end_sliders[axis].configure(from_=1,to=count)
            self.positions['z'].set(cells['z']);self.position_sliders['z'].configure(from_=1,to=len(self.full_mesh['z']['centers']))
            self.bounds_initialized=True
        finally:self.updating_positions=False
        self.update_position_labels();self.paint_overview()

    def update_position_labels(self):
        if self.full_mesh is None:return
        for axis in 'xyz':
            info=self.full_mesh[axis];count=len(info['centers']);index=min(count,max(1,round(self.positions[axis].get())))
            self.position_labels[axis].set(stream_number(info['centers'][index-1],self.show_decimals.get())+f' {info["unit"]} | cell {index}/{count}')
            if axis in 'xy':
                index=min(count,max(1,round(self.band_ends[axis].get())))
                self.end_labels[axis].set(stream_number(info['centers'][index-1],self.show_decimals.get())+f' {info["unit"]} | cell {index}/{count}')

    def band_changed(self,axis,is_end):
        if self.updating_positions or self.mp4_busy:return
        start,end=round(self.positions[axis].get()),round(self.band_ends[axis].get())
        if start>end:(self.positions[axis] if is_end else self.band_ends[axis]).set(end if is_end else start)
        self.cross_section_enabled.set(True);self.reveal_section_controls()
        self.use_bounds.set(True);self.update_position_labels();self.paint_overview();self.schedule()

    def z_changed(self):
        if self.updating_positions or self.mp4_busy:return
        self.cross_section_enabled.set(True);self.reveal_section_controls()
        self.use_bounds.set(True);self.update_position_labels();self.schedule()

    def open_section_view(self,view):
        self.active_view=view
        self.canvas.pack_forget();self.overview_canvas.pack_forget()
        (self.overview_canvas if view=='Overview' else self.canvas).pack(fill='both',expand=True)
        self.update_idletasks()
        if view=='Overview':self.paint_overview()
        else:self.paint()

    def paint(self):
        super().paint()
        if getattr(self,'active_view',None)=='Overview':self.paint_overview()

    def overview_world(self,x,y,clamp=False):
        if self.overview_rect is None or self.full_mesh is None:return None
        left,top,right,bottom=self.overview_rect
        if not clamp and not (left<=x<=right and top<=y<=bottom):return None
        fx=np.clip((x-left)/(right-left),0,1);fy=np.clip(1-(y-top)/(bottom-top),0,1)
        return np.array([self.full_mesh['x']['edges'][0]+fx*np.ptp(self.full_mesh['x']['edges']),
                         self.full_mesh['y']['edges'][0]+fy*np.ptp(self.full_mesh['y']['edges'])])

    def overview_screen(self,x,y):
        left,top,right,bottom=self.overview_rect
        ex=self.full_mesh['x']['edges'];ey=self.full_mesh['y']['edges']
        return left+(x-ex[0])/np.ptp(ex)*(right-left),bottom-(y-ey[0])/np.ptp(ey)*(bottom-top)

    def paint_overview(self):
        if not hasattr(self,'overview_canvas') or self.closed:return
        width,height=self.overview_canvas.winfo_width(),self.overview_canvas.winfo_height()
        if width<2 or height<2 or self.overview_plane is None:return
        image=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(image)
        size=int(self.text_size.get());font=annotation_font(size)
        low,high=float(self.color_low.get()),float(self.color_high.get())
        plane=self.overview_plane;component=self.component.get()
        values=np.linalg.norm(plane,axis=-1) if component=='mag' else plane[:,:,{'mx':0,'my':1,'mz':2}[component]]
        if self.color_range_mode.get()=='Data range':low,high=float(values.min()),float(values.max())
        if component=='mag' and (low,high)==(-1.,1.):low,high=0.,max(1.,float(values.max()))
        if low==high:high=low+1
        key=(id(plane),component,self.colors.get(),low,high)
        if self.overview_scalar_cache is None or self.overview_scalar_cache[0]!=key:
            self.overview_scalar_cache=(key,stream_scalar_rgb(values,self.colors.get(),low,high))
        scalar=self.overview_scalar_cache[1]
        zpos=self.full_mesh['z']['centers'][min(len(self.full_mesh['z']['centers']),max(1,round(self.positions['z'].get())))-1]
        scalar=layer_fill_2d(scalar,plane,zpos,layer_fill_source_z(self),self.style.get('layer_fill',{}),component,self.colors.get(),(low,high),native=layer_fill_native_data(self))
        ex,ey=self.full_mesh['x']['edges'],self.full_mesh['y']['edges']
        zcell=min(len(self.full_mesh['z']['centers']),max(1,round(self.positions['z'].get())))
        header=[component if component!='mag' else '|m|']
        if self.show_time.get() and self.time_ns is not None:header.append('t = '+stream_number(self.time_ns,self.show_decimals.get())+' ns')
        header.append('XY overview | Z = '+stream_number(self.full_mesh['z']['centers'][zcell-1],self.show_decimals.get())+' '+self.full_mesh['z']['unit'])
        margins=(55,figure_header_height(header,size)+8,25,48);area_w=max(1,width-margins[0]-margins[2]);area_h=max(1,height-margins[1]-margins[3])
        physical_scale=min(area_w/np.ptp(ex),area_h/np.ptp(ey))*self.overview_zoom
        w,h=np.ptp(ex)*physical_scale,np.ptp(ey)*physical_scale
        cx=margins[0]+area_w/2;cy=margins[1]+area_h/2
        left,top=cx-self.overview_center[0]*w,cy-self.overview_center[1]*h
        self.overview_rect=(left,top,left+w,top+h)
        clip=(margins[0],margins[1],width-margins[2],height-margins[3])
        x0,y0=max(clip[0],math.ceil(left)),max(clip[1],math.ceil(top))
        x1,y1=min(clip[2],math.floor(left+w)),min(clip[3],math.floor(top+h))
        if x0<x1 and y0<y1:
            extent=((x0-left)*scalar.width/w,(y0-top)*scalar.height/h,(x1-left)*scalar.width/w,(y1-top)*scalar.height/h)
            image.paste(scalar.transform((x1-x0,y1-y0),Image.Transform.EXTENT,extent,resample=Image.Resampling.BILINEAR),(x0,y0))
            for axis in 'xy':
                count=len(self.full_mesh[axis]['centers'])
                for end,var in ((False,self.positions[axis]),(True,self.band_ends[axis])):
                    cell=min(count,max(1,round(var.get())));coordinate=self.full_mesh[axis]['centers'][cell-1]
                    px,py=self.overview_screen(coordinate if axis=='x' else ex[0],coordinate if axis=='y' else ey[0])
                    color=(('#ffe600','#ff9000') if axis=='x' else ('#00e6ff','#ff00cc'))[int(end)]
                    if axis=='x' and x0<=px<x1:
                        draw.line((px,y0,px,y1),fill='black',width=3);draw.line((px,y0,px,y1),fill=color,width=1)
                        draw.text((min(x1-25,px+3),y0+2),axis.upper()+str(int(end)+1),font=font,fill='black',stroke_width=1,stroke_fill='white')
                    if axis=='y' and y0<=py<y1:
                        draw.line((x0,py,x1,py),fill='black',width=3);draw.line((x0,py,x1,py),fill=color,width=1)
                        draw.text((x0+2,min(y1-size-3,py+2)),axis.upper()+str(int(end)+1),font=font,fill='black',stroke_width=1,stroke_fill='white')
            for roi,color in ((self.shared_roi if self.use_tab3_roi.get() else None,'#777777'),(self.local_roi if self.roi_enabled.get() else None,'#7b00ff')):
                if roi:
                    a,b=self.overview_screen(roi['x'][0],roi['y'][1]);c,d=self.overview_screen(roi['x'][1],roi['y'][0])
                    rectangle=(max(x0,a),max(y0,b),min(x1,c),min(y1,d))
                    if rectangle[0]<=rectangle[2] and rectangle[1]<=rectangle[3]:draw.rectangle(rectangle,outline=color,width=2)
            if self.show_scales.get():
                wx0,wy0=self.overview_world(x0,y1,True);wx1,wy1=self.overview_world(x1,y0,True)
                for i in range(4):
                    f=i/3;px=x0+(x1-x0)*f;py=y1-(y1-y0)*f
                    draw.text((px-15,y1+4),stream_number(wx0+(wx1-wx0)*f,self.show_decimals.get()),font=font,fill='black')
                    draw.text((max(0,x0-52),py-size/2),stream_number(wy0+(wy1-wy0)*f,self.show_decimals.get()),font=font,fill='black')
                draw.text(((x0+x1)/2,y1+size+8),'X ('+self.full_mesh['x']['unit']+')',font=font,fill='black')
                draw.text((2,max(40,(y0+y1)/2)),'Y ('+self.full_mesh['y']['unit']+')',font=font,fill='black')
        draw_figure_header(image,header,size)
        self.overview_image=image;self.overview_photo=ImageTk.PhotoImage(image,master=self.gui.root)
        self.overview_canvas.delete('all');self.overview_canvas.create_image(0,0,image=self.overview_photo,anchor='nw')

    def toggle_local_roi(self):
        if self.roi_enabled.get() and self.local_roi is None:self.select_local_roi()
        else:self.schedule();self.paint_overview()

    def select_local_roi(self):
        if self.overview_plane is None:self.status.set('Load a readable frame before selecting an ROI.');return
        self.roi_enabled.set(True);self.roi_selecting=True;self.roi_drag=None
        self.open_section_view('Overview');self.overview_canvas.configure(cursor='crosshair')
        self.status.set('Drag a rectangle on the XY overview to select the ROI.')

    def clear_local_roi(self):
        self.local_roi=None;self.roi_enabled.set(False);self.roi_selecting=False;self.roi_drag=None
        self.overview_canvas.configure(cursor='');self.paint_overview();self.schedule()

    def overview_press(self,event):
        if self.mp4_busy:return
        point=self.overview_world(event.x,event.y)
        if point is None:return
        if self.roi_selecting:self.roi_drag=(point,point);return
        choices=[]
        for axis in 'xy':
            for end,var in ((False,self.positions[axis]),(True,self.band_ends[axis])):
                cell=min(len(self.full_mesh[axis]['centers']),max(1,round(var.get())))
                coordinate=self.full_mesh[axis]['centers'][cell-1]
                px,py=self.overview_screen(coordinate if axis=='x' else self.full_mesh['x']['edges'][0],coordinate if axis=='y' else self.full_mesh['y']['edges'][0])
                choices.append((abs(event.x-px) if axis=='x' else abs(event.y-py),axis,end))
        distance,axis,end=min(choices)
        if distance<=14:self.reference_drag=(axis,end);self.overview_move(event)
        else:self.overview_pan_start(event)

    def overview_move(self,event):
        point=self.overview_world(event.x,event.y,True)
        if point is None:return
        if self.roi_drag is not None:
            self.roi_drag=(self.roi_drag[0],point);self.paint_overview()
            a,b=self.overview_screen(*self.roi_drag[0]);c,d=self.overview_screen(*point)
            self.overview_canvas.create_rectangle(a,b,c,d,outline='#7b00ff',width=2,dash=(4,2));return
        if self.reference_drag is not None:
            axis,end=self.reference_drag;index='xy'.index(axis)
            cell=int(np.argmin(abs(self.full_mesh[axis]['centers']-point[index])))+1
            (self.band_ends[axis] if end else self.positions[axis]).set(cell);self.band_changed(axis,end);return
        if self.overview_pan is not None:self.overview_pan_move(event)

    def overview_release(self,event):
        if self.roi_drag is not None:
            self.overview_move(event);start,end=self.roi_drag;roi={'unit':self.full_mesh['x']['unit']}
            for index,axis in enumerate('xy'):
                low,high=sorted((start[index],end[index]));edge=self.full_mesh[axis]['edges'];count=len(self.full_mesh[axis]['centers']);step=np.ptp(edge)/count
                first=max(0,min(count-1,int(math.floor((low-edge[0])/step))))
                last=min(count,max(first+1,int(math.ceil((high-edge[0])/step))))
                if last-first<2:
                    self.status.set('Select an ROI at least two cells wide on X and Y.');self.roi_drag=None;return
                roi[axis]=[float(edge[0]+first*step),float(edge[0]+last*step)]
            base=self.shared_roi if self.use_tab3_roi.get() else None
            if base and any(min(base[a][1],roi[a][1])<=max(base[a][0],roi[a][0]) for a in 'xy'):
                self.status.set('The ROI must overlap the selected Tab 3 ROI.');self.roi_drag=None;return
            self.local_roi=roi;self.roi_selecting=False;self.roi_drag=None
            self.overview_canvas.configure(cursor='');self.paint_overview();self.schedule()
        self.reference_drag=None;self.overview_pan=None

    def overview_scroll(self,event):
        if self.overview_rect is None:return
        left,top,right,bottom=self.overview_rect;w,h=right-left,bottom-top
        fraction=((event.x-left)/w,(event.y-top)/h)
        before=self.overview_zoom;self.overview_zoom=min(50.,max(.2,before*(1.2 if event.delta>0 else 1/1.2)))
        ratio=self.overview_zoom/before
        self.overview_center=(self.overview_center[0]+(fraction[0]-self.overview_center[0])*(1-1/ratio),
                              self.overview_center[1]+(fraction[1]-self.overview_center[1])*(1-1/ratio))
        self.paint_overview();return 'break'

    def overview_pan_start(self,event):self.overview_pan=(event.x,event.y,self.overview_center)

    def overview_pan_move(self,event):
        if self.overview_pan is None or self.overview_rect is None:return
        x,y,center=self.overview_pan;left,top,right,bottom=self.overview_rect
        self.overview_center=(center[0]-(event.x-x)/(right-left),center[1]-(event.y-y)/(bottom-top))
        self.paint_overview()

    def update_movie_range(self):
        self.range_start.set('1');self.range_end.set(str(max(1,len(self.files))))

    def frame_range(self):
        try:first,last=int(self.range_start.get()),int(self.range_end.get())
        except ValueError:raise ValueError('Start and end frames must be integers.')
        if not self.files or not 1<=first<=last<=len(self.files):raise ValueError(f'Use a frame range within 1–{len(self.files)}.')
        return first-1,last-1

    def fps_value(self):
        try:value=float(self.play_fps.get())
        except ValueError:raise ValueError('FPS must be a number.')
        if not math.isfinite(value) or not 0<value<=120:raise ValueError('FPS must be greater than zero and at most 120.')
        return value

    def toggle_play(self):
        if self.playing:self.pause_play();return
        if self.mp4_busy:return
        try:self.frame_range();self.fps_value()
        except ValueError as exc:self.status.set(str(exc));return
        self.playing=True;self.play_button.configure(text='Pause');self.play_tick()

    def pause_play(self):
        self.playing=False
        if self.play_timer is not None:self.after_cancel(self.play_timer);self.play_timer=None
        if hasattr(self,'play_button'):self.play_button.configure(text='Play')

    def play_tick(self):
        self.play_timer=None
        if not self.playing or self.closed:return
        try:first,last=self.frame_range();fps=self.fps_value()
        except ValueError as exc:self.pause_play();self.status.set(str(exc));return
        if not self.worker_busy and self.update_timer is None:
            current=round(self.frame.get())-1;index=first if current<first or current>=last else current+1
            self.frame.set(index+1);self.apply()
        self.play_timer=self.after(max(1,round(1000/fps)),self.play_tick)

    def set_movie_controls(self,busy):
        CrossSectionTab.set_movie_controls(self,busy)

    def export_mp4(self):
        if self.mp4_busy:return
        try:
            first,last=self.frame_range();fps=self.fps_value()
            if self.worker_busy or self.update_timer is not None:raise ValueError('Wait for the current frame to finish loading.')
            if not self.scene_valid:raise ValueError('Load a readable frame first.')
        except ValueError as exc:self.status.set(str(exc));return
        parent=filedialog.askdirectory(parent=self.gui.root,title='Choose output folder for Tab 4 MP4')
        if not parent:return
        folder=None
        try:
            source=self.files[first]
            folder=auto_export_folder(parent,4,source,'MP4')
            view='Analysis_'+(self.slice_view.get() if self.cross_section_enabled.get() else self.volume_view.get())
            filename=auto_export_filename(source,view,self.component.get(),int(self.factor.get().replace('x','')),'mp4',first,last,fps)
            self.start_mp4_export(folder/filename,auto_folder=folder)
        except Exception as exc:
            discard_empty_export_folder(folder);self.status.set('MP4 export failed: '+str(exc))

    def start_mp4_export(self,path,auto_folder=None):
        import copy
        import vtk
        if self.mp4_busy:raise ValueError('A Tab 4 MP4 export is already running.')
        if self.worker_busy or self.update_timer is not None:raise ValueError('Wait for the current frame to finish loading.')
        if not self.scene_valid or self.native_cache is None:raise ValueError('Load a readable frame first.')
        first,last=self.frame_range();fps=self.fps_value();settings,style=self.filter_settings()
        self.sync_roi(False);roi=copy.deepcopy(self.roi_physical);style['roi']=roi
        target=Path(path).resolve()
        if not target.parent.is_dir():raise ValueError('Choose an existing output folder.')
        self.pause_play();self.gui.root.update_idletasks()
        widget=self.overview_canvas if self.active_view=='Overview' else self.canvas
        width=max(2,widget.winfo_width()//2*2);height=max(2,widget.winfo_height()//2*2)
        partial=target.with_name(target.stem+'.partial-'+str(time.time_ns())+'.mp4')
        camera=vtk.vtkCamera();camera.DeepCopy(self.plotter.camera)
        clone=copy.copy(self)
        for key,value in vars(self).items():
            if isinstance(value,tk.Variable):setattr(clone,key,FrozenViewValue(value.get()))
        clone.positions={a:FrozenViewValue(v.get()) for a,v in self.positions.items()}
        clone.band_ends={a:FrozenViewValue(v.get()) for a,v in self.band_ends.items()}
        clone.gui=SimpleNamespace(root=self.gui.root,closed=False)
        clone.canvas=SimpleNamespace(winfo_width=lambda:width,winfo_height=lambda:height)
        clone.plotter=None;clone.scene_valid=False;clone.actor_cache={};clone.layer_fill_actors={};clone.display_cache_key=None
        clone.tube_cache=(None,None);clone.paint_timer=None;clone.closed=False;clone.reorient=True
        clone.stream_arrow_cache=(None,None)
        clone.active_view='Analysis';clone.overview_cache=None;clone.overview_plane=None
        def paint_export():
            clone.plotter.window_size=(width,height);clone.update_header_viewport();clone.plotter.render()
            clone.image=clone.overlay(Image.fromarray(np.asarray(clone.plotter.screenshot(return_img=True))))
        def camera_export():
            clone.plotter.camera.DeepCopy(camera);clone.plotter.reset_camera_clipping_range()
        clone.paint=paint_export;clone.camera_default=camera_export
        self.mp4_stop.clear();self.mp4_busy=True
        cache=Tab4VTKCache()
        job=dict(clone=clone,camera=camera,first=first,last=last,fps=fps,size=(width,height),files=list(self.files),
                 factor=int(self.factor.get().replace('x','')),settings=copy.deepcopy(settings),style=copy.deepcopy(style),
                 roi=roi,path=target,partial=partial,cache=cache,encoded=0,skipped=[],error=None,state='running',auto_folder=auto_folder)
        self.mp4_job=job;self.set_movie_controls(True)
        self.status.set(f'Exporting Tab 4 frames {first+1}–{last+1} at {fps:g} FPS…')
        def worker():
            writer=None
            try:
                writer=cv2.VideoWriter(str(partial),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
                if not writer.isOpened():raise ValueError('Could not open the MP4 encoder.')
                for index in range(first,last+1):
                    if self.mp4_stop.is_set() or self.gui.closed:break
                    source=job['files'][index]
                    try:
                        native,_=cache.read_source(source)
                        frame_settings=copy.deepcopy(job['settings']);frame_style=copy.deepcopy(job['style'])
                        clone.prepare_section_request(frame_settings,frame_style,native,job['factor'],job['roi'])
                        vectors,mesh=refine_stream_volume(native[0],native[1],job['factor'],job['roi'])
                        result=build_stream_iso_scene(vectors,mesh,**frame_settings)
                    except ConversionCancelled:break
                    except Exception as exc:job['skipped'].append((source.name,str(exc)));continue
                    if self.mp4_stop.is_set() or self.gui.closed:break
                    ack=threading.Event();reply={}
                    self.gui.events.put((self.render_movie_frame,(job,index,(vectors,mesh,native[2],result,frame_style),ack,reply)))
                    while not ack.wait(.05):
                        if self.mp4_stop.is_set() or self.gui.closed:break
                    if self.mp4_stop.is_set() or self.gui.closed:break
                    if 'error' in reply:raise ValueError(reply['error'])
                    writer.write(reply['frame']);job['encoded']+=1
                writer.release();writer=None
                with self.mp4_lock:
                    if self.mp4_stop.is_set() or self.gui.closed:job['state']='cancelled'
                    elif not job['encoded']:raise ValueError('No readable frames in the selected range.')
                    else:os.replace(partial,target);job['state']='saved'
            except Exception as exc:job['error']=str(exc);job['state']='failed'
            finally:
                if writer is not None:writer.release()
                if job['state']!='saved':partial.unlink(missing_ok=True)
                cache.close()
                if not self.gui.closed:self.gui.events.put((self.finish_mp4,(job,)))
        self.mp4_thread=threading.Thread(target=worker,daemon=True);self.mp4_thread.start()

    def render_movie_frame(self,job,index,payload,ack,reply):
        try:
            if self.gui.closed or self.mp4_stop.is_set() or self.mp4_job is not job:return
            clone=job['clone'];clone.vectors,clone.mesh,clone.time_ns,clone.result,clone.style=payload
            clone.reorient=True;clone.display_scene()
            frame=np.asarray(clone.image)
            reply['frame']=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            self.status.set(f'Exporting frame {index+1}/{len(job["files"])} | Range {job["first"]+1}–{job["last"]+1} | {job["fps"]:g} FPS')
        except Exception as exc:reply['error']=str(exc)
        finally:ack.set()

    def stop_mp4(self):
        if not self.mp4_busy:return
        with self.mp4_lock:
            if self.mp4_job['state']!='running':return
            self.mp4_stop.set();self.mp4_job['cache'].close()
        self.status.set('Stopping Tab 4 MP4 export…')

    def finish_mp4(self,job):
        if self.mp4_job is not job:return
        clone=job['clone'];clone.actor_cache={};clone.stream_arrow_cache=(None,None)
        if clone.plotter is not None:clone.plotter.close();clone.plotter=None
        clone.result=clone.vectors=clone.mesh=None
        self.mp4_busy=False;self.mp4_job=None;self.set_movie_controls(False)
        if job['state']=='saved':self.status.set(f'Saved {job["path"]} | {job["encoded"]} frames | Skipped {len(job["skipped"])} unreadable')
        elif job['state']=='cancelled':self.status.set('MP4 export stopped. Partial video removed.')
        else:self.status.set('MP4 export failed: '+str(job['error']))
        if job.get('auto_folder'):
            if job['state']=='saved':open_folder(job['auto_folder'])
            else:discard_empty_export_folder(job['auto_folder'])

    def close_position_movie_controls(self):
        self.pause_play();self.mp4_stop.set()
        if self.mp4_job:
            self.mp4_job['cache'].close();clone=self.mp4_job['clone'];clone.actor_cache={}
            if clone.plotter is not None:clone.plotter.close();clone.plotter=None
        self.overview_plane=self.overview_cache=self.overview_scalar_cache=None


class StreamIsoTab(StreamAnalysisControlsMixin,StreamVolumeControlsBase):
    def __init__(self,gui,parent):
        self.shared_roi=None;self.slice_control_key=None;self.reorient=True
        self.slice_view=tk.StringVar(value='XY');self.slice_mode=tk.StringVar(value='Plane')
        self.slice_position=tk.StringVar();self.slice_slider_value=tk.DoubleVar(value=0)
        self.slice_exact_position=None;self.slice_display_text=''
        self.slice_thickness=tk.StringVar(value='2');self.show_slice=tk.BooleanVar(value=True)
        self.slice_opacity=tk.DoubleVar(value=.35);self.stream_domain=tk.StringVar(value='On slice')
        self.camera_mode=tk.StringVar(value='Slice');self.slice_bounds_label=tk.StringVar(value='Load a frame')
        self.slice_positions={}
        self.initialize_position_controls()
        super().__init__(gui,parent)
        self.camera_trace=self.camera_mode.trace_add('write',self.camera_mode_changed)
        group=self.group('Slice / Clip');group.pack_configure(before=self.groups['Component'])
        ttk.Label(group,text='Plane').pack(anchor='w')
        box=ttk.Combobox(group,textvariable=self.slice_view,values=['XY','YZ','XZ'],state='readonly')
        box.pack(fill='x');box.bind('<<ComboboxSelected>>',self.slice_view_changed)
        self.dropdown(group,'Section',self.slice_mode,['Plane','Slab'])
        ttk.Label(group,text='Position (nm)').pack(anchor='w')
        self.slice_slider=ttk.Scale(group,from_=0,to=1,variable=self.slice_slider_value,command=self.slice_moved)
        self.slice_slider.pack(fill='x')
        entry=ttk.Spinbox(group,textvariable=self.slice_position,from_=0,to=1,increment=.1,command=self.position_entered)
        entry.pack(fill='x');entry.bind('<Return>',self.position_entered);entry.bind('<FocusOut>',self.position_entered)
        self.slice_entry=entry
        ttk.Label(group,textvariable=self.slice_bounds_label).pack(anchor='w')
        self.entry(group,'Slab thickness (nm)',self.slice_thickness)
        self.dropdown(group,'Streamline domain',self.stream_domain,['On slice','Through volume'])
        ttk.Checkbutton(group,text='Show colored slice',variable=self.show_slice,command=self.schedule).pack(anchor='w')
        ttk.Label(group,text='Slice opacity').pack(anchor='w')
        ttk.Scale(group,from_=0,to=1,variable=self.slice_opacity,command=lambda v:self.schedule()).pack(fill='x')
        self.dropdown(group,'Camera',self.camera_mode,['Slice','3D'])
        ttk.Button(group,text='Align to slice',command=self.align_slice).pack(fill='x',pady=2)
        self.build_position_roi_movie_controls(group)
        self.show_message('Load a vector-field folder or use Tab 3 data.')

    def use_tab3(self):
        super().use_tab3()
        self.reset_section_bounds();self.update_movie_range()

    def load_folder(self):
        self.shared_roi=None;self.slice_position.set('');self.slice_control_key=None;self.reorient=True
        super().load_folder()
        self.reset_section_bounds();self.update_movie_range()

    def tab_selected(self,event=None):
        if self.closed or self.master.select()!=str(self):return
        self.sync_roi(True)

    def sync_roi(self,schedule=False):
        roi=physical_tab3_roi(self.gui.cross_tab) if self.use_tab3_roi.get() else None
        self.shared_roi=roi
        if self.roi_enabled.get() and self.local_roi is not None:
            if roi is None:roi=json.loads(json.dumps(self.local_roi))
            else:
                roi=dict(roi)
                for axis in 'xy':
                    roi[axis]=[max(roi[axis][0],self.local_roi[axis][0]),min(roi[axis][1],self.local_roi[axis][1])]
        changed=roi!=self.roi_physical;self.roi_physical=roi
        if roi is None:self.roi_label.set('Region: full volume')
        else:
            fmt=lambda v:stream_number(v,self.show_decimals.get())
            self.roi_label.set(f'Region: X {fmt(roi["x"][0])}…{fmt(roi["x"][1])}, Y {fmt(roi["y"][0])}…{fmt(roi["y"][1])} {roi["unit"]}')
        if changed and schedule and self.files:self.schedule()

    def slice_view_changed(self,event=None):
        self.slice_position.set('');self.slice_control_key=None;self.reorient=True;self.schedule()

    def slice_moved(self,value):
        self.use_bounds.set(False)
        self.slice_position.set(f'{float(value):.3f}');self.schedule()

    def position_entered(self,event=None):
        self.use_bounds.set(False)
        try:
            position=float(self.slice_position.get())
            if not math.isfinite(position):raise ValueError()
            self.slice_slider_value.set(position)
        except ValueError:self.status.set('Enter a finite slice position.');return
        self.schedule()

    def section_settings(self):
        if not self.cross_section_enabled.get():return None
        text=self.slice_position.get().strip();position=float(text) if text else None
        if text and text==self.slice_display_text:position=self.slice_exact_position
        thickness=float(self.slice_thickness.get())
        if position is not None and not math.isfinite(position):raise ValueError('Slice position must be finite.')
        if not math.isfinite(thickness) or thickness<=0:raise ValueError('Slab thickness must be positive.')
        section=dict(view=self.slice_view.get(),mode=self.slice_mode.get(),position=position,thickness=thickness,
                     stream_domain=self.stream_domain.get(),show_slice=self.show_slice.get(),slice_opacity=float(self.slice_opacity.get()))
        if self.use_bounds.get():
            section['bounds_request']=self.bounds_snapshot()
        return section

    def built(self,token,payload):
        super().built(token,payload)
        if token!=self.generation or payload[-1] or not self.scene_valid:return
        axis={'XY':'z','YZ':'x','XZ':'y'}[self.slice_view.get()]
        centers=self.mesh[axis]['centers'];lo,hi=float(centers[0]),float(centers[-1])
        self.slice_slider.configure(from_=lo,to=hi)
        step=(self.mesh[axis]['edges'][1]-self.mesh[axis]['edges'][0])/len(centers)
        self.slice_entry.configure(from_=lo,to=hi,increment=step)
        section=self.result.get('section')
        value=section['position'] if section else float(np.mean(centers))
        self.slice_exact_position=value;self.slice_display_text=stream_number(value,self.show_decimals.get())
        self.slice_slider_value.set(value);self.slice_position.set(self.slice_display_text)
        self.slice_bounds_label.set(f'{axis.upper()}: '+stream_number(lo,self.show_decimals.get())+' … '+stream_number(hi,self.show_decimals.get())+' '+self.mesh[axis]['unit'])
        self.accept_overview_data()

    def camera_default(self):
        if not self.cross_section_enabled.get():return super().camera_default()
        if self.camera_mode.get()=='3D':self.plotter.view_isometric();self.plotter.disable_parallel_projection()
        else:
            {'XY':self.plotter.view_xy,'YZ':self.plotter.view_yz,'XZ':self.plotter.view_xz}[self.slice_view.get()]()
            self.plotter.enable_parallel_projection()
        self.plotter.reset_camera()

    def align_slice(self):
        self.camera_mode.set('Slice');self.reset_camera()

    def camera_mode_changed(self,*args):
        self.reorient=True
        if self.scene_valid:self.camera_default();self.paint();self.reorient=False

    def close(self):
        self.close_position_movie_controls()
        self.camera_mode.trace_remove('write',self.camera_trace)
        super().close()



class ConverterGUI:
    def __init__(self, root):
        self.root = root
        root.title("MuMax3 Image Tools")
        root.geometry("960x840")
        root.minsize(820, 640)
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self.ovf_tab = ttk.Frame(notebook, padding=14)
        self.mp4_tab = ttk.Frame(notebook, padding=14)
        notebook.add(self.ovf_tab, text="Field to Image")
        notebook.add(self.mp4_tab, text="Image / Field to MP4")
        self.frame_range_enabled = tk.BooleanVar(value=False)
        self.frame_start = tk.StringVar(value="1")
        self.frame_end = tk.StringVar(value="1")
        self.colorbar = tk.BooleanVar(value=False)
        self.colorbar_size = tk.StringVar(value="Small")
        self.colorbar_position = tk.StringVar(value="Middle")
        self.colorbar_decimals = tk.BooleanVar(value=False)
        self.legend_enabled = tk.BooleanVar(value=False)
        self.display_time = tk.BooleanVar(value=False)
        self.legend_text = ""
        self.legend_location = tk.StringVar(value="Top center")
        self.time_location = tk.StringVar(value="Top left")
        self.annotation_size = tk.StringVar(value="8")
        self.line_spacing = tk.StringVar(value="1")
        self.text_alignment = tk.StringVar(value="Align left")
        self.annotation_font_path = tk.StringVar(value="")
        self.color_min = tk.StringVar(value="-1")
        self.color_max = tk.StringVar(value="1")
        self.legend_window = None
        self.last_outputs = {"ovf": [], "video": []}
        self.cancel_events = {"ovf": threading.Event(), "video": threading.Event()}
        self.preview_windows = []
        self.closed = False
        self.build_ovf_tab()
        self.build_mp4_tab()
        self.cross_tab = CrossSectionTab(self,notebook)
        notebook.add(self.cross_tab,text="Cross section analysis")
        self.stream_tab = StreamIsoTab(self,notebook)
        notebook.add(self.stream_tab,text="Streamlines / Isosurfaces")
        self.events = queue.Queue()
        self.running = {"ovf": False, "video": False}
        self.pipeline_pending = False
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self.poll_events)

    def close(self):
        self.closed = True
        self.stream_tab.close()
        self.cross_tab.close()
        for event in self.cancel_events.values():
            event.set()
        for viewer in self.preview_windows:
            if not viewer.closed:
                viewer.close()
        self.root.destroy()

    def poll_events(self):
        if self.closed:
            return
        for _ in range(100):
            try:
                callback, args = self.events.get_nowait()
            except queue.Empty:
                break
            try:
                callback(*args)
            except Exception:
                # A preview callback failure must not stop future worker results.
                self.root.report_callback_exception(*sys.exc_info())
            if self.closed:
                return
        self.root.after(50, self.poll_events)

    def finish_job(self, kind, callback, *args):
        self.running[kind] = False
        (self.ovf_button if kind == "ovf" else self.video_button).configure(state="normal")
        if kind == "ovf":
            self.ovf_video_button.configure(state="normal")
            if self.pipeline_pending:
                self.pipeline_pending = False
                self.video_button.configure(state="normal")
        callback(*args)
        self.refresh_stop_buttons()

    def refresh_stop_buttons(self):
        active = any(self.running.values())
        requested = any(self.cancel_events[k].is_set() for k in self.running if self.running[k])
        for button in (self.ovf_stop_button, self.video_stop_button):
            button.configure(state="normal" if active and not requested else "disabled")

    def stop_conversions(self):
        for kind in ("ovf", "video"):
            if self.running[kind]:
                self.cancel_events[kind].set()
                (self.ovf_status if kind == "ovf" else self.video_status).set("Stopping conversion…")
        if self.pipeline_pending:
            self.cancel_events["video"].set()
        self.refresh_stop_buttons()

    def conversion_stopped(self, kind, output_dir=None):
        status = self.ovf_status if kind == "ovf" else self.video_status
        status.set("Stopped. Completed output files were kept.")
        if kind == "video":
            self.ovf_status.set("Stopped. Completed images and MP4 files were kept.")
        if output_dir is not None:
            folder = Path(output_dir)
            kept = sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS | {".mp4"}], key=natural_key)
            if kept:
                self.last_outputs[kind] = kept

    def preview_output(self, kind):
        # Reuse a live preview rather than opening hidden/overlapping windows.
        for existing in self.preview_windows:
            if isinstance(existing, InputPreview) and existing.kind == kind and not existing.closed:
                try:
                    if existing.window.winfo_exists():
                        existing.window.deiconify()
                        existing.window.lift()
                        existing.window.focus_set()
                        existing.refresh()
                        return existing
                except tk.TclError:
                    existing.closed = True
        viewer = InputPreview(self, kind)
        viewer.window.deiconify()
        viewer.window.lift()
        viewer.window.focus_set()
        self.preview_windows = [v for v in self.preview_windows if not v.closed]
        self.preview_windows.append(viewer)
        return viewer

    def snapshot_frame_range(self):
        if not self.frame_range_enabled.get():
            return None
        start, end = int(self.frame_start.get()), int(self.frame_end.get())
        if not 1 <= start <= end:
            raise ValueError("Frame range must have 1 <= start <= end (inclusive).")
        return start, end

    def frame_range_controls(self, parent, row):
        controls = ttk.Frame(parent)
        controls.grid(row=row,column=0,columnspan=4,sticky="ew",pady=4)
        ttk.Checkbutton(controls, text="Use frame range", variable=self.frame_range_enabled).pack(side="left",padx=4)
        ttk.Label(controls, text="Start").pack(side="left",padx=4)
        ttk.Spinbox(controls, from_=1, to=1000000000, textvariable=self.frame_start, width=8).pack(side="left")
        ttk.Label(controls, text="End").pack(side="left",padx=4)
        ttk.Spinbox(controls, from_=1, to=1000000000, textvariable=self.frame_end, width=8).pack(side="left")
        ttk.Label(controls, text="1-based, inclusive, in natural filename order; unchecked = all frames").pack(side="left",padx=8)

    def annotation_controls(self, parent, row):
        container = ttk.Frame(parent)
        container.grid(row=row,column=0,columnspan=4,sticky="ew",pady=8)
        controls = ttk.Frame(container)
        controls.pack(fill="x")
        positioning = ttk.Frame(container)
        positioning.pack(fill="x",pady=(4,0))
        ttk.Label(positioning, text="Color-bar position (right side):").pack(side="left",padx=4)
        ttk.Combobox(positioning, textvariable=self.colorbar_position, values=["Top", "Bottom", "Middle"],
                     state="readonly", width=10).pack(side="left",padx=4)
        ttk.Label(positioning, text="Anchored to original OVF area, excluding legend/time margins").pack(side="left",padx=8)
        ttk.Checkbutton(controls, text="Color bar scale (right)", variable=self.colorbar).pack(side="left",padx=4)
        ttk.Combobox(controls, textvariable=self.colorbar_size, values=["Small", "Medium", "Bigger"],
                     state="readonly", width=8).pack(side="left",padx=4)
        ttk.Checkbutton(controls, text="Decimals (3)", variable=self.colorbar_decimals).pack(side="left",padx=4)
        ttk.Checkbutton(controls, text="Legend", variable=self.legend_enabled).pack(side="left",padx=4)
        ttk.Checkbutton(controls, text="Display time (ns)", variable=self.display_time).pack(side="left",padx=4)
        ttk.Button(controls, text="Edit legend / display…", command=self.edit_legend).pack(side="left",padx=8)

    def edit_legend(self):
        if self.legend_window is not None and self.legend_window.winfo_exists():
            self.legend_window.lift()
            return
        window = self.legend_window = tk.Toplevel(self.root)
        window.title("Legend and display settings")
        window.geometry("720x640")
        window.columnconfigure(1, weight=1)
        window.rowconfigure(1, weight=1)
        ttk.Checkbutton(window, text="Display Legend", variable=self.legend_enabled).grid(row=0,column=0,sticky="w",padx=10,pady=8)
        ttk.Label(window, text="Type or paste symbols, letters, numbers, and multiline text:").grid(row=0,column=1,sticky="w",padx=8)
        self.legend_editor = tk.Text(window, wrap="word", undo=True, height=8, font=("Segoe UI", 12))
        self.legend_editor.grid(row=1,column=0,columnspan=3,sticky="nsew",padx=10,pady=4)
        self.legend_editor.insert("1.0", self.legend_text)
        menu = tk.Menu(self.legend_editor, tearoff=False)
        for label, event in (("Cut", "<<Cut>>"), ("Copy", "<<Copy>>"), ("Paste", "<<Paste>>"), ("Select all", "<<SelectAll>>")):
            menu.add_command(label=label, command=lambda e=event: self.legend_editor.event_generate(e))
        self.legend_editor.bind("<Button-3>", lambda event: menu.tk_popup(event.x_root, event.y_root))
        locations = ["Top center", "Top left", "Top right", "Bottom center", "Bottom left", "Bottom right", "Left", "Right"]
        for row, label, variable in ((2, "Legend location", self.legend_location), (3, "Time location", self.time_location)):
            ttk.Label(window, text=label).grid(row=row,column=0,sticky="w",padx=10,pady=6)
            ttk.Combobox(window, textvariable=variable, values=locations, state="readonly").grid(row=row,column=1,sticky="w",padx=8)
        ttk.Label(window, text="Text size (at 1x)").grid(row=4,column=0,sticky="w",padx=10,pady=6)
        ttk.Spinbox(window, from_=6, to=72, textvariable=self.annotation_size, width=8).grid(row=4,column=1,sticky="w",padx=8)
        ttk.Label(window, text="Color-bar min / max").grid(row=5,column=0,sticky="w",padx=10,pady=6)
        limits = ttk.Frame(window)
        limits.grid(row=5,column=1,sticky="w",padx=8)
        ttk.Entry(limits, textvariable=self.color_min, width=10).pack(side="left")
        ttk.Entry(limits, textvariable=self.color_max, width=10).pack(side="left",padx=8)
        ttk.Checkbutton(window, text="Color-bar decimals (3)", variable=self.colorbar_decimals).grid(row=5,column=2,padx=8)
        ttk.Label(window, text="Font (optional .ttf/.ttc/.otf)").grid(row=6,column=0,sticky="w",padx=10,pady=6)
        ttk.Entry(window, textvariable=self.annotation_font_path).grid(row=6,column=1,sticky="ew",padx=8)
        def browse_font():
            selected = filedialog.askopenfilename(parent=window, title="Select a font for your symbols", filetypes=[("Fonts", "*.ttf *.ttc *.otf"), ("All files", "*.*")])
            if selected:
                self.annotation_font_path.set(selected)
        ttk.Button(window, text="Browse…", command=browse_font).grid(row=6,column=2,padx=8)
        ttk.Label(window, text="Text occupies a margin around the image. The color bar uses these same limits for OVF rendering.\nTime comes from OVF metadata, not playback FPS. Choose a font that contains your characters.",
                  wraplength=640).grid(row=7,column=0,columnspan=3,sticky="w",padx=10,pady=8)
        def save():
            self.legend_text = self.legend_editor.get("1.0", "end-1c")
            window.destroy()
            self.legend_window = None
        ttk.Label(window,text="Line spacing").grid(row=8,column=0,sticky="w",padx=10,pady=6)
        ttk.Combobox(window,textvariable=self.line_spacing,values=["1","1.15","1.5","2"],state="readonly",width=12).grid(row=8,column=1,sticky="w",padx=8)
        ttk.Label(window,text="Text alignment").grid(row=9,column=0,sticky="w",padx=10,pady=6)
        ttk.Combobox(window,textvariable=self.text_alignment,values=["Align left","Center","Align right","Justified"],state="readonly").grid(row=9,column=1,sticky="w",padx=8)
        ttk.Button(window, text="Save and close", command=save).grid(row=10,column=0,columnspan=3,pady=10)
        window.protocol("WM_DELETE_WINDOW", save)

    def snapshot_annotations(self):
        if self.legend_window is not None and self.legend_window.winfo_exists():
            self.legend_text = self.legend_editor.get("1.0", "end-1c")
        low, high = float(self.color_min.get()), float(self.color_max.get())
        size = int(self.annotation_size.get())
        if not math.isfinite(low) or not math.isfinite(high) or low >= high or not 6 <= size <= 72:
            raise ValueError("Use finite color-bar min < max and a text size from 6 to 72.")
        return dict(colorbar=self.colorbar.get(), colorbar_size=self.colorbar_size.get(), colorbar_position=self.colorbar_position.get(), colorbar_decimals=self.colorbar_decimals.get(), legend=self.legend_enabled.get(), display_time=self.display_time.get(),
                    legend_text=self.legend_text, legend_location=self.legend_location.get(), time_location=self.time_location.get(),
                    font_size=size, font_path=self.annotation_font_path.get(), vmin=low, vmax=high,
                    line_spacing=float(self.line_spacing.get()), text_alignment=self.text_alignment.get())

    def folder_row(self, parent, row, label, var, choose_command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8)
        ttk.Button(parent, text="Browse…", command=choose_command).grid(row=row, column=2, sticky="ew")
        parent.columnconfigure(1, weight=1)

    def choose_tab1_color_preset(self):
        def changed():
            for viewer in self.preview_windows:
                if isinstance(viewer,InputPreview) and not viewer.closed:
                    viewer.refresh()
        self.color_preset_picker = show_shared_color_presets(self.root,self,self.colors,changed)

    def build_ovf_tab(self):
        p = self.ovf_tab
        self.ovf_input, self.ovf_output = tk.StringVar(), tk.StringVar()
        self.component = tk.StringVar(value="mz")
        self.out_format = tk.StringVar(value="PNG")
        self.colors = tk.StringVar(value="blue,white,red")
        self.ppi = tk.StringVar(value="300")
        self.upscale = tk.StringVar(value="1x")
        self.arrows = tk.BooleanVar(value=False)
        self.arrow_density = tk.StringVar(value="80")
        self.arrow_length = tk.StringVar(value="2.5")
        self.arrow_head = tk.StringVar(value="1")
        self.arrow_mode = tk.StringVar(value="Magnitude")
        self.ovf_status = tk.StringVar(value="Choose an OVF, OOMMF, or VTK field folder.")
        self.ovf_progress = tk.DoubleVar(value=0)
        self.folder_row(p, 0, "Input field folder:", self.ovf_input, self.choose_ovf_input)
        self.folder_row(p, 1, "Output parent folder:", self.ovf_output, self.choose_ovf_output)

        settings = ttk.LabelFrame(p, text="Image settings", padding=10)
        settings.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(12,8))
        for c in range(4): settings.columnconfigure(c, weight=1)
        ttk.Label(settings, text="m component").grid(row=0,column=0,sticky="w",padx=4,pady=4)
        component_picker = ttk.Frame(settings)
        component_picker.grid(row=0,column=1,sticky="ew",padx=4)
        self.component_icons = []
        for name in ("mx","my","mz"):
            icon = ImageTk.PhotoImage(component_title_image(name,annotation_font(15)),master=self.root)
            self.component_icons.append(icon)
            ttk.Radiobutton(component_picker,image=icon,variable=self.component,value=name).pack(side="left",padx=3)
        ttk.Radiobutton(component_picker,text="ALL",variable=self.component,value="ALL").pack(side="left",padx=3)
        ttk.Label(settings, text="Format").grid(row=0,column=2,sticky="w",padx=4)
        ttk.Combobox(settings, textvariable=self.out_format, values=["JPG","PNG","TIFF"], state="readonly", width=10).grid(row=0,column=3,sticky="ew",padx=4)
        ttk.Label(settings, text="Color map").grid(row=1,column=0,sticky="w",padx=4,pady=4)
        color_controls = ttk.Frame(settings)
        color_controls.grid(row=1,column=1,sticky="ew",padx=4,pady=4)
        self.color_box = ttk.Combobox(color_controls, textvariable=self.colors, values=STREAM_COLOR_MAPS, state="normal", width=28)
        self.color_box.pack(fill='x')
        self.color_preset_button = ttk.Button(color_controls,text='Choose preset…',command=self.choose_tab1_color_preset)
        self.color_preset_button.pack(fill='x',pady=(3,0))
        ttk.Label(settings, text="PPI metadata").grid(row=1,column=2,sticky="w",padx=4)
        ttk.Combobox(settings, textvariable=self.ppi, values=["150","300","600"], state="readonly", width=10).grid(row=1,column=3,sticky="ew",padx=4)
        ttk.Label(settings, text="Upscaling").grid(row=2,column=0,sticky="w",padx=4,pady=4)
        ttk.Combobox(settings, textvariable=self.upscale, values=["1x","2x","4x","8x","10x"], state="readonly", width=10).grid(row=2,column=1,sticky="ew",padx=4)
        ttk.Checkbutton(settings, text="Overlay solid magnetization arrows (mₓ/mᵧ)", variable=self.arrows,
                        command=self.toggle_arrow_options).grid(row=2,column=2,columnspan=2,sticky="w",padx=4)
        self.arrow_frame = ttk.Frame(settings)
        self.arrow_frame.grid(row=3,column=0,columnspan=4,sticky="ew",pady=(8,0))
        ttk.Label(self.arrow_frame,text="Arrow density across x").grid(row=0,column=0,sticky="w",padx=4)
        ttk.Spinbox(self.arrow_frame,from_=4,to=100,textvariable=self.arrow_density,width=8).grid(row=0,column=1,sticky="w",padx=4)
        ttk.Label(self.arrow_frame,text="Length scale").grid(row=0,column=2,sticky="w",padx=4)
        ttk.Spinbox(self.arrow_frame,from_=0.2,to=3.0,increment=0.1,textvariable=self.arrow_length,width=8).grid(row=0,column=3,sticky="w",padx=4)
        ttk.Label(self.arrow_frame,text="Arrowhead size (1x px)").grid(row=0,column=4,sticky="w",padx=4)
        ttk.Spinbox(self.arrow_frame,from_=1,to=40,textvariable=self.arrow_head,width=8).grid(row=0,column=5,sticky="w",padx=4)
        ttk.Label(self.arrow_frame, text="Arrow scaling").grid(row=1,column=0,sticky="w",padx=4,pady=(8,0))
        self.arrow_mode_box = ttk.Combobox(self.arrow_frame, textvariable=self.arrow_mode,
                                         values=["Magnitude", "Normalized"], state="readonly", width=12)
        self.arrow_mode_box.grid(row=1,column=1,columnspan=2,sticky="w",padx=4,pady=(8,0))
        ttk.Label(self.arrow_frame, text="Magnitude: length, head and shaft width scale with √(mₓ² + mᵧ²).\nNormalized: equal sizes; direction is retained. Zero vectors are hidden.",
                  wraplength=740).grid(row=2,column=0,columnspan=6,sticky="w",padx=4,pady=(8,0))
        ttk.Label(p, text="ALL exports three component images per OVF. PPI is embedded as image DPI metadata; upscaling increases pixel dimensions.",
                  wraplength=820).grid(row=3,column=0,columnspan=3,sticky="w",pady=5)
        ttk.Progressbar(p,variable=self.ovf_progress,maximum=100).grid(row=4,column=0,columnspan=3,sticky="ew",pady=8)
        ttk.Label(p,textvariable=self.ovf_status,wraplength=820).grid(row=5,column=0,columnspan=3,sticky="w",pady=4)
        self.ovf_button = ttk.Button(p,text="Convert to Images",command=self.start_ovf)
        self.ovf_button.grid(row=6,column=0,columnspan=3,pady=12,ipadx=18,ipady=5)
        self.ovf_video_button = ttk.Button(p, text="Convert fields to MP4", command=lambda: self.start_ovf(make_video=True))
        self.ovf_video_button.grid(row=7,column=0,columnspan=3,pady=4,ipadx=18,ipady=5)
        ttk.Label(p, text="OVF to MP4 uses FPS and resolution from the Image / Field to MP4 tab. ALL creates a separate movie for each component.",
                  wraplength=800).grid(row=8,column=0,columnspan=3,sticky="w",pady=4)
        actions = ttk.Frame(p)
        actions.grid(row=9,column=0,columnspan=3,pady=6)
        ttk.Button(actions, text="Preview before conversion…", command=lambda: self.preview_output("ovf")).pack(side="left",padx=6)
        self.ovf_stop_button = ttk.Button(actions, text="Stop conversion", command=self.stop_conversions, state="disabled")
        self.ovf_stop_button.pack(side="left",padx=6)
        self.annotation_controls(settings, 4)
        self.frame_range_controls(settings, 5)
        self.toggle_arrow_options()

    def toggle_arrow_options(self):
        state = "normal" if self.arrows.get() else "disabled"
        for child in self.arrow_frame.winfo_children():
            try: child.configure(state=state)
            except tk.TclError: pass
        self.arrow_mode_box.configure(state="readonly" if self.arrows.get() else "disabled")

    def build_mp4_tab(self):
        p = self.mp4_tab
        self.video_input, self.video_output = tk.StringVar(), tk.StringVar()
        self.fps = tk.StringVar(value="10")
        self.video_component = tk.StringVar(value="mz")
        self.resolution = tk.StringVar(value="720p")
        self.video_status = tk.StringVar(value="Choose folders and video settings.")
        self.video_progress = tk.DoubleVar(value=0)
        self.folder_row(p,0,"Input image / field folder:",self.video_input,self.choose_video_input)
        self.folder_row(p,1,"Output parent folder:",self.video_output,self.choose_video_output)
        ttk.Label(p,text="Frames per second (FPS):").grid(row=2,column=0,sticky="w",pady=8)
        ttk.Entry(p,textvariable=self.fps,width=12).grid(row=2,column=1,sticky="w",padx=8)
        ttk.Label(p,text="Output resolution:").grid(row=3,column=0,sticky="w",pady=8)
        ttk.Combobox(p,textvariable=self.resolution,values=list(RESOLUTIONS.keys()),state="readonly",width=12).grid(row=3,column=1,sticky="w",padx=8)
        field_row = ttk.Frame(p)
        field_row.grid(row=4,column=0,columnspan=3,sticky="w",pady=8)
        ttk.Label(field_row,text="Field component:").pack(side='left')
        ttk.Combobox(field_row,textvariable=self.video_component,values=('mx','my','mz'),state='readonly',width=7).pack(side='left',padx=8)
        ttk.Label(field_row,text="Field frames use Tab 1 color, arrow, and upscaling settings.").pack(side='left')
        ttk.Progressbar(p,variable=self.video_progress,maximum=100).grid(row=5,column=0,columnspan=3,sticky="ew",pady=8)
        ttk.Label(p,textvariable=self.video_status,wraplength=820).grid(row=6,column=0,columnspan=3,sticky="w",pady=4)
        self.video_button = ttk.Button(p,text="Convert to MP4",command=self.start_video)
        self.video_button.grid(row=7,column=0,columnspan=3,pady=12,ipadx=18,ipady=5)
        self.annotation_controls(p, 8)
        self.frame_range_controls(p, 10)
        actions = ttk.Frame(p)
        actions.grid(row=9,column=0,columnspan=3,pady=6)
        ttk.Button(actions, text="Preview before conversion…", command=lambda: self.preview_output("video")).pack(side="left",padx=6)
        self.video_stop_button = ttk.Button(actions, text="Stop conversion", command=self.stop_conversions, state="disabled")
        self.video_stop_button.pack(side="left",padx=6)

    def choose_ovf_input(self):
        path=filedialog.askdirectory(title="Select OVF, OOMMF, or VTK field folder")
        if path:
            self.ovf_input.set(path)
            if not self.ovf_output.get(): self.ovf_output.set(str(Path(path).parent))
    def choose_ovf_output(self):
        path=filedialog.askdirectory(title="Select output parent folder")
        if path: self.ovf_output.set(path)
    def choose_video_input(self):
        path=filedialog.askdirectory(title="Select folder containing images")
        if path:
            self.video_input.set(path)
            if not self.video_output.get(): self.video_output.set(str(Path(path).parent))
    def choose_video_output(self):
        path=filedialog.askdirectory(title="Select output parent folder")
        if path: self.video_output.set(path)

    def start_ovf(self, make_video=False):
        if self.running["ovf"]:
            return
        video_settings = None
        if make_video:
            if self.running["video"]:
                messagebox.showerror("Video busy", "Wait for the current video conversion to finish.")
                return
            try:
                fps = float(self.fps.get())
                resolution = self.resolution.get()
                if not math.isfinite(fps) or not 0.1 <= fps <= 240 or resolution not in RESOLUTIONS:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid video settings", "Choose FPS from 0.1 to 240 and a resolution in the MP4 tab.")
                return
            video_settings = (fps, resolution)
        if not self.ovf_input.get().strip() or not self.ovf_output.get().strip():
            messagebox.showerror("Missing folders","Select both input and output folders."); return
        try:
            frame_range = self.snapshot_frame_range()
            annotations = self.snapshot_annotations()
            density=int(self.arrow_density.get()); length=float(self.arrow_length.get()); head=float(self.arrow_head.get())
            if density<1 or not all(math.isfinite(v) and v > 0 for v in (length, head)): raise ValueError
            ppi=int(self.ppi.get())
            upscale=int(self.upscale.get().replace("x",""))
            colors=self.colors.get().strip()
            stream_color_map(colors)
        except Exception as e:
            messagebox.showerror("Invalid settings",str(e) or "Check arrow settings, PPI, upscaling, and color map."); return
        args = (self.ovf_input.get(), self.ovf_output.get(), self.component.get(),
                self.out_format.get(), colors, ppi, upscale, self.arrows.get(), density, length, head)
        arrow_mode = self.arrow_mode.get()
        self.cancel_events["ovf"].clear()
        if make_video:
            self.cancel_events["video"].clear()
        self.running["ovf"] = True
        self.refresh_stop_buttons()
        self.ovf_button.configure(state="disabled")
        self.ovf_video_button.configure(state="disabled")
        self.pipeline_pending = make_video
        if make_video:
            self.video_button.configure(state="disabled")
        self.ovf_progress.set(0); self.ovf_status.set("Starting field conversion…")
        def prog(i,n,name):
            self.events.put((self.update_progress, (self.ovf_progress,self.ovf_status,i,n,name)))
        def worker():
            try:
                outdir,count=convert_ovf_folder(*args, progress=prog, arrow_mode=arrow_mode, annotations=annotations, cancel_event=self.cancel_events["ovf"], frame_range=frame_range)
                self.events.put((self.finish_job, ("ovf", self.ovf_done, outdir, count, video_settings)))
            except ConversionCancelled as exc:
                self.events.put((self.finish_job, ("ovf", self.conversion_stopped, "ovf", exc.output_dir)))
            except Exception as e:
                self.events.put((self.finish_job, ("ovf", self.fail, self.ovf_status, "OVF conversion failed", str(e))))
        threading.Thread(target=worker,daemon=True).start()

    def start_video(self):
        if self.pipeline_pending or self.running["video"]:
            return
        try:
            frame_range = self.snapshot_frame_range()
            annotations = self.snapshot_annotations()
            colors = self.colors.get()
            sources = video_source_files(self.video_input.get()) if Path(self.video_input.get()).is_dir() else []
            field_settings = field_render_settings(self) if sources and sources[0].suffix.lower() in FIELD_EXTS else None
            fps=float(self.fps.get())
            if not 0.1<=fps<=240: raise ValueError
        except ValueError:
            messagebox.showerror("Invalid FPS","Enter an FPS value from 0.1 to 240."); return
        if not self.video_input.get().strip() or not self.video_output.get().strip():
            messagebox.showerror("Missing folders","Select both input and output folders."); return
        self.cancel_events["video"].clear()
        args = (self.video_input.get(), self.video_output.get(), fps, self.resolution.get())
        self.running["video"] = True
        self.refresh_stop_buttons()
        self.video_button.configure(state="disabled")
        self.video_progress.set(0); self.video_status.set("Starting video conversion…")
        def prog(i,n,name):
            self.events.put((self.update_progress, (self.video_progress,self.video_status,i,n,name)))
        def worker():
            try:
                outdir,outfile,count=convert_images_to_video(*args, progress=prog, annotations=annotations, colors=colors, cancel_event=self.cancel_events["video"], frame_range=frame_range, field_settings=field_settings)
                self.events.put((self.finish_job, ("video", self.video_done, outdir, outfile, count)))
            except ConversionCancelled as exc:
                self.events.put((self.finish_job, ("video", self.conversion_stopped, "video", exc.output_dir)))
            except Exception as e:
                self.events.put((self.finish_job, ("video", self.fail, self.video_status, "MP4 conversion failed", str(e))))
        threading.Thread(target=worker,daemon=True).start()

    def update_progress(self,var,status,i,n,name):
        var.set(100*i/n)
        status.set(f"{i}/{n}: {name}")
    def ovf_done(self,outdir,count,video_settings=None):
        report = outdir / "skipped_ovf_files.txt"
        skip_note = "\n\nUnreadable OVF files were skipped. See skipped_ovf_files.txt in the output folder." if report.exists() else ""
        self.ovf_progress.set(100)
        self.ovf_status.set(f"Done: created {count} image(s)." + (" Unreadable files skipped; see report." if report.exists() else ""))
        if count:
            self.last_outputs["ovf"] = sorted([p for p in outdir.iterdir() if p.suffix.lower() in IMAGE_EXTS], key=natural_key)
            self.video_input.set(str(outdir))
            self.video_output.set(str(outdir.parent))
        if video_settings is not None and (self.cancel_events["ovf"].is_set() or self.cancel_events["video"].is_set()):
            self.conversion_stopped("ovf", outdir)
            return
        if video_settings is not None and count:
            self.start_ovf_videos(outdir, video_settings)
            return
        messagebox.showinfo("OVF conversion complete",f"Created {count} image(s):\n{outdir}{skip_note}\n\nOpening output folder.")
        open_folder(outdir)

    def start_ovf_videos(self, image_dir, settings):
        # Keep component sequences separate so ALL never interleaves mx/my/mz frames.
        fps, resolution = settings
        groups = {comp: sorted([p for p in image_dir.iterdir()
                  if p.suffix.lower() in IMAGE_EXTS and p.stem.endswith("_" + comp)], key=natural_key)
                  for comp in ("mx", "my", "mz")}
        groups = {comp: paths for comp, paths in groups.items() if paths}
        self.running["video"] = True
        self.refresh_stop_buttons()
        self.video_button.configure(state="disabled")
        self.video_progress.set(0)
        self.video_status.set("Creating MP4 from exported OVF images…")
        self.ovf_status.set(f"Created {sum(map(len, groups.values()))} images. Creating MP4…")
        def worker():
            try:
                movies = []
                total = sum(map(len, groups.values()))
                done = 0
                for comp, paths in groups.items():
                    # Reuse the video converter without duplicating exported images on disk.
                    def progress(i, n, name, offset=done, tag=comp):
                        self.events.put((self.update_progress, (self.video_progress, self.video_status,
                                         offset+i, total, f"{tag}: {name}")))
                    _, movie, _ = convert_images_to_video(image_dir, image_dir, fps, resolution,
                                                         progress=progress, image_paths=paths, output_label=comp, cancel_event=self.cancel_events["video"])
                    movies.append(movie)
                    done += len(paths)
                check_cancel(self.cancel_events["video"], image_dir)
                self.events.put((self.finish_job, ("video", self.ovf_videos_done, image_dir, movies)))
            except ConversionCancelled:
                self.events.put((self.finish_job, ("video", self.conversion_stopped, "video", image_dir)))
            except Exception as exc:
                self.events.put((self.finish_job, ("video", self.pipeline_failed, str(exc))))
        threading.Thread(target=worker, daemon=True).start()

    def pipeline_failed(self, error):
        self.ovf_status.set("Image export complete; MP4 creation failed.")
        self.fail(self.video_status, "OVF to MP4 failed", error)

    def ovf_videos_done(self, image_dir, movies):
        self.last_outputs["video"] = list(movies)
        self.last_outputs["ovf"] = list(movies) + sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS], key=natural_key)
        self.video_progress.set(100)
        summary = f"Done: created {len(movies)} MP4 file(s)."
        self.video_status.set(summary)
        self.ovf_status.set(summary + (" Unreadable files skipped; see report." if (image_dir / "skipped_ovf_files.txt").exists() else ""))
        detail = "\n".join(str(p) for p in movies)
        skip_note = "\n\nUnreadable files were skipped; see skipped_ovf_files.txt." if (image_dir / "skipped_ovf_files.txt").exists() else ""
        messagebox.showinfo("OVF to MP4 complete", f"{summary}\n{detail}{skip_note}")
        open_folder(image_dir)
    def video_done(self,outdir,outfile,count):
        self.last_outputs["video"] = [outfile]
        self.video_progress.set(100); self.video_status.set(f"Done: converted {count} image(s).")
        messagebox.showinfo("Video conversion complete",f"Video:\n{outfile}\n\nFrames: {count}\n\nOpening output folder.")
        open_folder(outdir)
    def fail(self,status,title,error):
        status.set("Conversion failed.")
        messagebox.showerror(title,error)
