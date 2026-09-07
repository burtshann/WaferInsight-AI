import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from waferinsight.core import clusters


def wafer_figure(a):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='#0b1220')
    ax.set_facecolor('#0b1220')
    ax.imshow(a, cmap=ListedColormap(['#0b1220', '#25b7a1', '#ff635e']), vmin=0, vmax=2, interpolation='nearest')
    for c in clusters(a):
        ax.add_patch(Rectangle((c['x_min']-.5,c['y_min']-.5), c['x_max']-c['x_min']+1,
                              c['y_max']-c['y_min']+1, fill=False, edgecolor='#ffc857', linewidth=1))
    ax.set_title('DIE MAP  /  PASS · FAIL · CLUSTER', color='white', pad=18)
    ax.axis('off')
    fig.tight_layout()
    return fig
