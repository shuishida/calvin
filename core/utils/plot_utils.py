import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import patches

from core.domains.gridworld.actions import GridActionSet

ALMOST_NEG_INF = -10000 * 0.99


def preprocess_path(path, lower_origin=False):
    if path is None: return None
    return path if lower_origin else [(y, x) for x, y in path]


def preprocess_grid(grid, lower_origin=False):
    if isinstance(grid, torch.Tensor): grid = grid.cpu().data.numpy()
    if grid.ndim == 3:
        n_channels = grid.shape[0]
        if n_channels == 3:
            if lower_origin:
                grid = grid.transpose((2, 1, 0))
        elif n_channels == 1:
            grid = grid[0]
        else:
            weights = 2 ** (np.arange(n_channels) / n_channels).reshape((-1, 1, 1))
            grid = (grid * weights).sum(axis=0)
    if grid.ndim == 2 and lower_origin:
        grid = grid.T
        # if grid.size <= 400:
        #     for (i, j), val in np.ndenumerate(grid):
        #         ax.text(j, i, f"{val:.2f}", c='m', ha='center', va='center')
    return grid


def visualise(grids, opt_path=None, pred_path=None, curr_pose=None, fig=None, axes=None, titles=None, show=True, auto_close=True,
              save_path=None, highlight=None, lower_origin=True, legend_loc='lower center', bbox_to_anchor=None, show_legend=True):
    if axes is None:
        n = len(grids)
        h = 2 if n >= 4 and n % 2 == 0 else 1
        w = n // h
        fig, axes = plt.subplots(h, w)
        if n == 1: axes = np.array([axes])
        else: axes = axes.flat

    if not lower_origin:
        opt_path, pred_path = preprocess_path(opt_path), preprocess_path(pred_path)
        if curr_pose is not None: curr_pose = (curr_pose[-1], curr_pose[-2])

    assert len(grids) == len(axes), "number of images do not equal number of axes"
    if isinstance(curr_pose, np.ndarray): curr_pose = tuple(curr_pose)
    elif isinstance(curr_pose, torch.Tensor): curr_pose = tuple(curr_pose.cpu().data.numpy())
    if highlight is not None:
        if isinstance(highlight, torch.Tensor): highlight = highlight.cpu().data.numpy()
    for i_grid, grid in enumerate(grids):
        ax = axes[i_grid]
        grid = preprocess_grid(grid, lower_origin)
        ax.axis('off')
        im = ax.imshow(grid, origin='lower' if lower_origin else 'upper')
        if grid.ndim == 2:
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, ax=ax, orientation='vertical')

        opt_path = list(opt_path) if opt_path is not None else None
        pred_path = list(pred_path) if pred_path is not None else None

        if opt_path:
            ax.plot(*tuple(zip(*opt_path)), c='tab:pink', label='Optimal Path', linewidth=3.0)
        if pred_path:
            ax.plot(*tuple(zip(*pred_path)), c='tab:red', label='Predicted Path', linewidth=3.0)
        if curr_pose:
            if len(curr_pose) == 2:
                ax.plot(*curr_pose, '-o', c='tab:blue', label='Current Pose')
            else:
                (dx, dy), x, y = curr_pose
                ax.arrow(x, y, dx * 0.3, dy * 0.3, width=0.2)
        if opt_path or pred_path:
            path = opt_path or pred_path
            ax.plot(*path[0], '-o', c='tab:red', label='Start', )
            ax.plot(*path[-1], '-o', c='tab:orange', label='Goal')
        if titles is not None:
            ax.set_title(titles[i_grid])
        if highlight is not None:
            for x, y in zip(*np.where(highlight)):
                ax.add_patch(patches.Rectangle((x - 0.5, y - 0.5), 1, 1, color='r', fill=False))

    if show_legend:
        if bbox_to_anchor is None:
            legend = plt.legend()
        else:
            legend = plt.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor,
                                bbox_transform=plt.gcf().transFigure, ncol=2, borderaxespad=0.)
        for label in legend.get_texts():
            label.set_fontsize('x-small')   # The legend text size
        for label in legend.get_lines():
            label.set_linewidth(0.5)        # The legend line width

    plt.tight_layout()

    if show:
        plt.draw()
        plt.waitforbuttonpress(0)

    if save_path:
        if not isinstance(save_path, list):
            save_path = [save_path]
        for i_save_path in save_path:
            fig.savefig(i_save_path)

    if auto_close:
        for ax in axes:
            [p.remove() for p in reversed(ax.patches)]
        plt.close(fig)

    return fig, axes


def show_dir_values(dir_values, ax, fig=None, dirs=None, show_colorbar=True):
    """
    convert direction values of shape (9, map_x, map_y) to a grid of (map_x * 3, map_y * 3)
    :param dir_values:
    :return:
    """
    if isinstance(dir_values, torch.Tensor): dir_values = dir_values.cpu().data.numpy()
    dir_values = np.copy(dir_values)
    n_actions, size_x, size_y = dir_values.shape
    if not dirs:
        dirs = GridActionSet()
    if dir_values.min() <= ALMOST_NEG_INF:
        dir_values[dir_values <= ALMOST_NEG_INF] = -np.inf
    grid = np.zeros((size_x * 3, size_y * 3))
    for i, direction in enumerate(dirs):
        dx, dy = direction
        grid[(1+dx)::3, (1+dy)::3] = dir_values[i]

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, size_x * 3, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, size_y * 3, 3), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    im = ax.imshow(grid.T, origin='lower')
    ax.grid(which='minor', color='c', linestyle='-', linewidth=1)

    if show_colorbar:
        fig.colorbar(im, ax=ax, orientation='vertical')

    return ax