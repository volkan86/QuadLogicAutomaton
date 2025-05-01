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
        self.frame_count = 0  # Frame sayacı
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
        neighbor_avg = np.stack([self.neighborhood_average(self.state[...,k]) 
                               for k in range(4)], -1)
        mixed = self.alpha * self.state + (1 - self.alpha) * neighbor_avg
        
        max_val = mixed.max(axis=-1, keepdims=True)
        collapse_mask = (mixed >= self.theta) & (mixed == max_val)
        mixed[collapse_mask] = 1.0
        mixed[~collapse_mask] = 0.0
        
        sum_vals = mixed.sum(axis=-1, keepdims=True)
        return np.divide(mixed, sum_vals, where=sum_vals!=0)

    def setup_ui(self):
        plt.close('all')
        self.fig = plt.figure(figsize=(10, 10), constrained_layout=True)
        gs = gridspec.GridSpec(
            nrows=6, ncols=2,
            figure=self.fig,
            height_ratios=[1, 1, 1, 1, 1, 0.5],
            hspace=0.3
        )
        
        # Eksenler
        self.ax_grid = self.fig.add_subplot(gs[0, 0])
        self.ax_plot = self.fig.add_subplot(gs[0, 1])
        self.ax_table = self.fig.add_subplot(gs[1:3, :])
        self.ax_alpha = self.fig.add_subplot(gs[3, 0])
        self.ax_theta = self.fig.add_subplot(gs[3, 1])
        self.ax_reset = self.fig.add_subplot(gs[4, :])
        
        # Grid görselleştirmesi
        self.grid_img = self.ax_grid.imshow(np.argmax(self.state, -1), cmap=CMAP, vmin=0, vmax=3)
        self.ax_grid.set_title("QLA State Grid")
        
        # Proportion grafikleri
        self.lines = {
            k: self.ax_plot.plot([], [], color=c, label=k, linewidth=2)[0]
            for k,c in zip(['T','F','V','I'], CMAP.colors)
        }
        self.ax_plot.set_title("State Proportions (0-1 Scale)")
        self.ax_plot.set_ylim(0, 1.0)
        self.ax_plot.set_xlim(0, HISTORY_WINDOW)
        self.ax_plot.legend()
        self.ax_plot.grid(True, linestyle=':')
        
        # Dinamik tablo
        self.table = self.ax_table.table(
            cellText=[
                ['Frame', '0'],
                ['Alpha', f"{self.alpha:.2f}"],
                ['Theta', f"{self.theta:.2f}"],
                ['T', '0.25 (400)'],
                ['F', '0.25 (400)'],
                ['V', '0.25 (400)'],
                ['I', '0.25 (400)']
            ],
            colWidths=[0.25, 0.25],
            loc='center',
            cellLoc='center'
        )
        self.table.auto_set_font_size(False)
        self.table.set_fontsize(9)
        self.table.scale(1.0, 1.0)
        self.ax_table.axis('off')
        self.ax_table.set_title("Simulation Metrics", pad=20)

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
        self.frame_count = 0
        self.grid_img.set_data(np.argmax(self.state, -1))
        
        # Grafikleri sıfırla
        for line in self.lines.values():
            line.set_data([], [])
        self.ax_plot.set_xlim(0, HISTORY_WINDOW)
        
        # Tabloyu sıfırla
        total_cells = GRID_SIZE[0] * GRID_SIZE[1]
        for i, key in enumerate(['T', 'F', 'V', 'I']):
            prop = INIT_RATIO[i]
            count = int(round(prop * total_cells))
            self.table._cells[(i + 3, 1)]._text.set_text(f"{prop:.2f} ({count})")
        self.table._cells[(0, 1)]._text.set_text("0")
        self.table._cells[(1, 1)]._text.set_text(f"{self.alpha:.2f}")
        self.table._cells[(2, 1)]._text.set_text(f"{self.theta:.2f}")
        
        self.fig.canvas.draw_idle()

    def start_animation(self):
        def update(frame):
            self.state = self.update_state()
            self.grid_img.set_data(np.argmax(self.state, -1))
            
            # Durum oranlarını hesapla
            counts = np.clip(self.state.mean(axis=(0, 1)), 0.0, 1.0)
            total_cells = GRID_SIZE[0] * GRID_SIZE[1]
            
            # Verileri grafiklere ekle
            for i, (k, line) in enumerate(self.lines.items()):
                self.history[k].append(counts[i])
                if len(self.history[k]) > HISTORY_WINDOW:
                    self.history[k] = self.history[k][-HISTORY_WINDOW:]
                line.set_data(
                    np.arange(len(self.history[k])),
                    self.history[k]
                )
            
            # Frame sayacını güncelle
            self.frame_count += 1
            
            # Tabloyu güncelle
            self.table._cells[(0, 1)]._text.set_text(str(self.frame_count))
            self.table._cells[(1, 1)]._text.set_text(f"{self.alpha:.2f}")
            self.table._cells[(2, 1)]._text.set_text(f"{self.theta:.2f}")

            for row, val in enumerate(counts):
                count = int(round(val * total_cells))
                self.table._cells[(row + 3, 1)]._text.set_text(f"{val:.2f} ({count})")

            # X ekseni limitlerini güncelle
            current_length = len(self.history['T'])
            x_start = max(0, current_length - HISTORY_WINDOW)
            x_end = max(HISTORY_WINDOW, current_length)
            self.ax_plot.set_xlim(x_start, x_end)
            
            return (self.grid_img, *self.lines.values())

        self.ani = FuncAnimation(
            self.fig, 
            update, 
            interval=50, 
            blit=False,  # Tablo animasyonu için blit devre dışı
            cache_frame_data=False,
            save_count=MAX_FRAMES
        )
        plt.show()

# Kullanım
sim = QLASimulation()
sim.start_animation()
