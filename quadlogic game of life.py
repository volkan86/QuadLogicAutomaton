import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

# -------------------------- Constants -----------------------------
GRID_SIZE = (40, 40)
CMAP = ListedColormap(['#2ecc71', '#e74c3c', '#f1c40f', '#7f8c8d'])
INIT_RATIO = [0.25]*4
MAX_FRAMES = 1000
HISTORY_WINDOW = 100  # Son 100 adımı göster

class QLASimulation:
    def __init__(self):
        self.state, self.prev_state = self.initialize_grid()
        self.alpha = 0.5
        self.theta = 0.75
        self.history = {'T': [], 'F': [], 'V': [], 'I': []}
        self.setup_ui()
        self.ani = None

    def initialize_grid(self):
        h, w = GRID_SIZE
        state = np.zeros((h, w, 4), dtype=np.float32)
        choices = np.random.choice(4, size=(h, w))
        for i in range(4):
            state[..., i] = (choices == i).astype(np.float32)
        return state, np.zeros((h, w))

    def neighborhood_average(self, layer):
        return (np.roll(layer,1,0) + np.roll(layer,-1,0) + 
               np.roll(layer,1,1) + np.roll(layer,-1,1)) /4.0

    def update_state(self):
        neighbor_avg = np.stack([self.neighborhood_average(self.state[...,k]) for k in range(4)], -1)
        mixed = self.alpha * self.state + (1 - self.alpha) * neighbor_avg
        
        max_val = mixed.max(axis=-1, keepdims=True)
        collapse_mask = (mixed >= self.theta) & (mixed == max_val)
        mixed[collapse_mask] = 1.0
        mixed[~collapse_mask] = 0.0
        
        sum_vals = mixed.sum(axis=-1, keepdims=True)
        return np.divide(mixed, sum_vals, where=sum_vals!=0)

    def setup_ui(self):
        plt.close('all')
        self.fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        gs = gridspec.GridSpec(4, 2, figure=self.fig)
        
        # Eksenler
        self.ax_grid = self.fig.add_subplot(gs[0, 0])
        self.ax_plot = self.fig.add_subplot(gs[0, 1])
        self.ax_phase = self.fig.add_subplot(gs[1, :])
        self.ax_alpha = self.fig.add_subplot(gs[2, 0])
        self.ax_theta = self.fig.add_subplot(gs[2, 1])
        self.ax_reset = self.fig.add_subplot(gs[3, :])
        
        # Görsel elementler
        self.grid_img = self.ax_grid.imshow(np.argmax(self.state, -1), cmap=CMAP, vmin=0, vmax=3)
        self.ax_grid.set_title("QLA State Grid")
        
        self.lines = {
            k: self.ax_plot.plot([], [], color=c, label=k, linewidth=2)[0]
            for k,c in zip(['T','F','V','I'], CMAP.colors)
        }
        self.ax_plot.set_title("State Proportions (0-1 Scale)")
        self.ax_plot.set_ylim(0, 1.0)
        self.ax_plot.set_xlim(0, HISTORY_WINDOW)
        self.ax_plot.legend()
        self.ax_plot.grid(True, linestyle=':')
        
        # Kontroller
        self.alpha_slider = Slider(self.ax_alpha, 'Alpha', 0, 1, valinit=self.alpha)
        self.theta_slider = Slider(self.ax_theta, 'Theta', 0, 1, valinit=self.theta)
        self.reset_button = Button(self.ax_reset, 'Reset Simulation', hovercolor='0.85')
        
        # Event handlers
        self.alpha_slider.on_changed(lambda val: setattr(self, 'alpha', val))
        self.theta_slider.on_changed(lambda val: setattr(self, 'theta', val))
        self.reset_button.on_clicked(self.reset_simulation)

    def reset_simulation(self, event=None):
        self.state, self.prev_state = self.initialize_grid()
        self.history = {'T': [], 'F': [], 'V': [], 'I': []}
        self.grid_img.set_data(np.argmax(self.state, -1))
        for line in self.lines.values():
            line.set_data([], [])
        self.ax_plot.set_xlim(0, HISTORY_WINDOW)
        self.fig.canvas.draw_idle()

    def start_animation(self):
        def update(frame):
            self.state = self.update_state()
            self.grid_img.set_data(np.argmax(self.state, -1))
            
            # Verileri normalize et ve kırp
            counts = np.clip(self.state.mean(axis=(0,1)), 0.0, 1.0)
            
            for i, (k, line) in enumerate(self.lines.items()):
                self.history[k].append(counts[i])
                
                # Son HISTORY_WINDOW adımı tut
                if len(self.history[k]) > HISTORY_WINDOW:
                    self.history[k] = self.history[k][-HISTORY_WINDOW:]
                
                line.set_data(
                    np.arange(len(self.history[k])),
                    self.history[k]
                )
            
            # Eksen limitlerini güncelle
            current_length = len(self.history['T'])
            x_start = max(0, current_length - HISTORY_WINDOW)
            x_end = max(HISTORY_WINDOW, current_length)
            self.ax_plot.set_xlim(x_start, x_end)
            
            return (self.grid_img, *self.lines.values())
        
        self.ani = FuncAnimation(
            self.fig, 
            update, 
            interval=50, 
            blit=True, 
            cache_frame_data=False,
            save_count=MAX_FRAMES
        )
        plt.show()

# Kullanım
sim = QLASimulation()
sim.start_animation() AttributeError                            Traceback (most recent call last)
Cell In[12], line 2
      1 import numpy as np
----> 2 import matplotlib.pyplot as plt
      3 from matplotlib.colors import ListedColormap
      4 from matplotlib.widgets import Slider, Button

File ~\anaconda3\Lib\site-packages\matplotlib\__init__.py:161
    157 from packaging.version import parse as parse_version
    159 # cbook must import matplotlib only within function
    160 # definitions, so it is safe to import from it here.
--> 161 from . import _api, _version, cbook, _docstring, rcsetup
    162 from matplotlib.cbook import sanitize_sequence
    163 from matplotlib._api import MatplotlibDeprecationWarning

File ~\anaconda3\Lib\site-packages\matplotlib\rcsetup.py:28
     26 from matplotlib.cbook import ls_mapper
     27 from matplotlib.colors import Colormap, is_color_like
---> 28 from matplotlib._fontconfig_pattern import parse_fontconfig_pattern
     29 from matplotlib._enums import JoinStyle, CapStyle
     31 # Don't let the original cycler collide with our validating cycler

File ~\anaconda3\Lib\site-packages\matplotlib\colors.py:57
     55 import matplotlib as mpl
     56 import numpy as np
---> 57 from matplotlib import _api, _cm, cbook, scale, _image
     58 from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS
     61 class _ColorMapping(dict):

File ~\anaconda3\Lib\site-packages\matplotlib\scale.py:764
    755         docs.extend([
    756             f"    {name!r}",
    757             "",
    758             textwrap.indent(docstring, " " * 8),
    759             ""
    760         ])
    761     return "\n".join(docs)
--> 764 _docstring.interpd.register(
    765     scale_type='{%s}' % ', '.join([repr(x) for x in get_scale_names()]),
    766     scale_docs=_get_scale_docs().rstrip(),
    767     )

AttributeError: '_ArtistPropertiesSubstitution' object has no attribute 'register'





