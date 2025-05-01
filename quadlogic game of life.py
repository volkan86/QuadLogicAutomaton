#!/usr/bin/env python
# coding: utf-8

# In[5]:


"""
Quad-Logic Automaton (QLA) – Interactive Simulation
---------------------------------------------------
4-durumlu hücresel otomat:
    T (Truth) | F (Falsity) | V (Volatility) | I (Indeterminacy)

Kullanım
--------
Terminal  : $ python qla_interactive.py
Jupyter   :
    %matplotlib widget      # veya %matplotlib notebook
    import qla_interactive as qla
    ani = qla.run()         # referansı saklayın

Kontroller
----------
• Alpha  (atalet)  : 0–1      → komşu etkisi
• Theta  (eşik)    : 0–1      → “çökme” kesinlik eşiği
• Reset butonu     : yeni rastgele başlangıç
"""
get_ipython().run_line_magic('matplotlib', 'widget')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

# -------------------------- Sabitler -----------------------------
H, W = 80, 80                                     # Izgara boyutu
INIT_RATIO = dict(T=.25, F=.25, V=.25, I=.25)     # Başlangıç dağılımı

ALPHA_INIT = 0.5   # Atalet
THETA_INIT = 0.75  # Çökme eşiği

CMAP = ListedColormap(['#2ecc71',  # T – yeşil
                       '#e74c3c',  # F – kırmızı
                       '#f1c40f',  # V – sarı
                       '#7f8c8d']) # I – gri


# ----------------------- Yardımcı Fonksiyonlar -------------------
def init_state():
    """H×W×4 boyutlu birim-sıcak başlangıç matrisi üret."""
    state = np.zeros((H, W, 4), dtype=np.float32)
    choices = np.random.choice(4, size=(H, W),
                               p=[INIT_RATIO[k] for k in ('T', 'F', 'V', 'I')])
    for i in range(4):
        state[..., i] = (choices == i).astype(np.float32)
    return state


def neighbor_mean(layer):
    """Moore 8-komşu ortalaması (periodik sınır)."""
    return (np.roll(layer, 1, 0) + np.roll(layer, -1, 0) +
            np.roll(layer, 1, 1) + np.roll(layer, -1, 1) +
            np.roll(np.roll(layer, 1, 0), 1, 1) +
            np.roll(np.roll(layer, 1, 0), -1, 1) +
            np.roll(np.roll(layer, -1, 0), 1, 1) +
            np.roll(np.roll(layer, -1, 0), -1, 1)) / 8.0


def step(state, alpha, theta):
    """Tek hücresel otomat adımı."""
    neigh_avg = np.stack([neighbor_mean(state[..., k]) for k in range(4)], axis=-1)
    mixed = alpha * state + (1 - alpha) * neigh_avg

    # Keskin çökme
    max_comp = mixed.max(axis=-1, keepdims=True)
    collapsed = (mixed >= theta) & (mixed == max_comp)
    if collapsed.any():
        mixed[collapsed] = 1.0
        mixed[~collapsed] = 0.0

    # Normalizasyon
    s = mixed.sum(axis=-1, keepdims=True)
    s[s == 0] = 1.0
    return mixed / s


def make_figure():
    """Ana figürü ve eksenleri oluştur."""
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(2, 2, height_ratios=[20, 1], width_ratios=[1, 1])
    ax_grid = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])
    ax_alpha = fig.add_subplot(gs[1, 0])
    ax_theta = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(hspace=0.3, wspace=0.25, bottom=0.15)
    return fig, ax_grid, ax_plot, ax_alpha, ax_theta


# ------------------------ Ana Çalıştırıcı ------------------------
def run():
    """Etkileşimli pencereyi başlat; `FuncAnimation` nesnesini döndür."""
    state = init_state()
    history = {'T': [], 'F': [], 'V': [], 'I': []}
    time_axis = []

    fig, ax_g, ax_p, ax_a, ax_t = make_figure()

    # Izgara görseli
    im = ax_g.imshow(np.argmax(state, axis=-1), cmap=CMAP, vmin=0, vmax=3)
    ax_g.set_title('Quad-Logic Grid')
    ax_g.axis('off')

    # Zaman serisi çizgileri
    line_T, = ax_p.plot([], [], label='T')
    line_F, = ax_p.plot([], [], label='F')
    line_V, = ax_p.plot([], [], label='V')
    line_I, = ax_p.plot([], [], label='I')
    ax_p.set_ylim(0, 1)
    ax_p.set_xlim(0, 200)
    ax_p.set_xlabel('Step')
    ax_p.set_ylabel('Ratio')
    ax_p.legend(loc='upper right', fontsize=8)
    ax_p.grid(ls=':')

    # Slaytlar
    s_alpha = Slider(ax_a, 'alpha', 0.0, 1.0, valinit=ALPHA_INIT, valstep=0.01)
    s_theta = Slider(ax_t, 'theta', 0.0, 1.0, valinit=THETA_INIT, valstep=0.01)

    # Reset tuşu
    reset_ax = fig.add_axes([0.45, 0.02, 0.1, 0.04])
    button = Button(reset_ax, 'Reset', hovercolor='0.975')

    # ---------------------- İç Fonksiyonlar ----------------------
    def animate(frame):
        nonlocal state
        alpha = s_alpha.val
        theta = s_theta.val
        state = step(state, alpha, theta)
        im.set_data(np.argmax(state, axis=-1))

        counts = state.mean(axis=(0, 1))
        for key, idx in zip(history.keys(), range(4)):
            history[key].append(counts[idx])
        time_axis.append(frame)

        line_T.set_data(time_axis, history['T'])
        line_F.set_data(time_axis, history['F'])
        line_V.set_data(time_axis, history['V'])
        line_I.set_data(time_axis, history['I'])

        if frame > ax_p.get_xlim()[1]:
            ax_p.set_xlim(0, frame + 50)
        return im, line_T, line_F, line_V, line_I

    def reset(event):
        nonlocal state, history, time_axis
        state = init_state()
        history = {k: [] for k in history}
        time_axis = []

    button.on_clicked(reset)

    # Animasyon nesnesi referansta tutuluyor
    ani = FuncAnimation(fig, animate,
                        interval=100, cache_frame_data=False)

    plt.show()
    return ani   # Jupyter’da referans olarak saklayın


# --------------------------- Main -------------------------------
if __name__ == '__main__':
    run()


# In[ ]:




