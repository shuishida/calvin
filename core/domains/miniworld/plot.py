import matplotlib.pyplot as plt
from einops import rearrange
import numpy as np

from core.utils.plot_utils import show_dir_values, visualise


def miniworld_plot(meta, *, is_expert=None, curr_rgb=None, features_exist=None, curr_top_view=None, q=None, q0=None, r=None, p=None, r_d=None, d=None,
                  _poses=None, vis=None, _returns=None,
                  action_set=None, dirs=None, show=True, save_path=None, step=None, curr_poses=None, _actions=None, **kwargs):
    print(kwargs.keys(), curr_poses, _actions)

    fig, axes = plt.subplots(3, 3)

    pred_path = [meta.state_index_to_grid_index(_p) for _p in _poses]
    dir, x, y = curr_poses
    curr_pose = (dirs[int(dir)], int(x), int(y))

    axes = axes.flat
    for i, (ax, action) in enumerate(zip(axes[:3], action_set[1:3] + action_set[-1:])):
        show_dir_values(q0[i], ax=ax, fig=fig, dirs=dirs)
        ax.set_title(f"Q0 {action}")

    # for i, (ax, action) in enumerate(zip(axes[:5], action_set)):
    #     show_dir_values(r[i], ax=ax, fig=fig, dirs=dirs)
    #     ax.set_title(f"reward {action}")

    for i, (ax, action) in enumerate(zip(axes[3:6], action_set[1:3] + action_set[-1:])):
        show_dir_values(q[i], ax=ax, fig=fig, dirs=dirs)
        ax.set_title(f"Pred action {action}")

    ax = axes[6]
    # show_dir_values(v[0], ax, fig=fig, dirs=dirs)
    # ax.set_title("pred values")
    direction = []
    xs = []
    ys = []
    "a () oi oo h w x y"
    for i, (dx, dy) in enumerate(dirs):
        direction.append(i)
        xs.append(dx + 1)
        ys.append(dy + 1)

    res = p[2, 0, direction, direction, xs, ys]

    curr_trans = p[2, 0, int(curr_poses[0]), :, :, :, int(curr_poses[1]), int(curr_poses[2])]

    show_dir_values(res, ax, fig=fig, dirs=dirs)
    ax.set_title("forward transitions")

    # visualise(grids=[feature_map], show=show, save_path=save_path, pred_path=pred_path[:step+1], curr_pose=curr_pose,
    #           fig=fig, axes=axes[11:], titles=['agent view'])

    ax = axes[6]
    ax.imshow(rearrange(curr_rgb, "c h w -> h w c"))
    ax.set_title("Agent view")
    ax.axis('off')

    ax = axes[7]
    ax.imshow(curr_top_view)
    ax.set_title("Top view (unavailable to agent)")
    ax.axis('off')

    visualise(grids=[features_exist],
              pred_path=pred_path,
              curr_pose=curr_pose, axes=axes[8:], fig=fig,
              titles=['Point cloud location'], save_path=save_path, show=show,
              lower_origin=False, legend_loc="center", bbox_to_anchor=(0.85, 0.3, 0.2, 0.4))


def mini_rollout_plot(meta, *, curr_rgb=None, feature_map=None, best_action_maps=None, v=None, q=None, q0=None, r=None, p=None, r_d=None, d=None, _poses=None, vis=None,
                      action_set=None, show=True, curr_top_view=None, save_path=None, step=None, curr_poses=None, _actions=None, **kwargs):
    print(kwargs.keys(), curr_poses, _actions)

    fig, axes = plt.subplots(2, 3, figsize=(24, 18))

    pred_path = [meta.state_index_to_grid_index(p) for p in _poses]

    H, W = meta.map_res

    action_dir = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

    axes = axes.flat
    ax = axes[0]
    ax.imshow(curr_top_view)
    ax.set_title("Top view")

    ax = axes[1]
    im = ax.imshow(rearrange(q[[3, 2, 1, 4, 8, 0, 5, 6, 7]], "(h w) x y -> (x h) (y w)", h=3))
    # plt.plot(*(data['curr_poses'][idx, [1, 0]] * 3 + 1), "*b")
    # plt.plot(*(data['target'][idx, [1, 0]] * 3 + 1), "sr")
    ax.set_xticks(np.arange(-0.5, H * 3, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, W * 3, 3), minor=True)
    ax.grid(which='minor', color='c', linestyle='-', linewidth=1)
    ax.set_title("pred action")
    fig.colorbar(im, ax=ax, orientation='vertical')

    ax.plot(*tuple(zip(*[(3*y + 1, 3*x + 1) for x, y in pred_path])), c='tab:red', label='Predicted Path', linewidth=3.0)

    ax = axes[2]
    im = ax.imshow(rearrange(r[[3, 2, 1, 4, 8, 0, 5, 6, 7]], "(h w) x y -> (x h) (y w)", h=3))
    # plt.plot(*(data['curr_poses'][idx, [1, 0]] * 3 + 1), "*b")
    # plt.plot(*(data['target'][idx, [1, 0]] * 3 + 1), "sr")
    ax.set_xticks(np.arange(-0.5, H * 3, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, W * 3, 3), minor=True)
    ax.grid(which='minor', color='c', linestyle='-', linewidth=1)
    ax.set_title("pred rewards")
    fig.colorbar(im, ax=ax, orientation='vertical')

    ax = axes[3]
    im = ax.imshow(rearrange(d[[3, 2, 1, 4, 8, 0, 5, 6, 7]], "(h w) x y -> (x h) (y w)", h=3))
    # plt.plot(*(data['curr_poses'][idx, [1, 0]] * 3 + 1), "*b")
    # plt.plot(*(data['target'][idx, [1, 0]] * 3 + 1), "sr")
    ax.set_xticks(np.arange(-0.5, H * 3, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, W * 3, 3), minor=True)
    ax.grid(which='minor', color='c', linestyle='-', linewidth=1)
    ax.set_title("pred dones")
    fig.colorbar(im, ax=ax, orientation='vertical')

    ax = axes[4]
    ax.imshow(rearrange(curr_rgb, "c h w -> h w c"))
    ax.set_title("Agent view")
    ax.axis('off')

    visualise(grids=[v], show=show, save_path=save_path, pred_path=pred_path[:step+1], curr_pose=curr_poses,
              fig=fig, axes=axes[5:], titles=['Agent view', 'Pred values', 'Pred rewards 0', 'Pred rewards 1',
                                              'Pred dones 0', 'vis'])


def mini_rollout_calvin_plot(meta, *, curr_rgb=None, aa=None, best_action_maps=None, v=None, q=None, q0=None, r_sa=None, p=None, r_d=None, d=None, _poses=None, vis=None,
                      action_set=None, show=True, curr_top_view=None, save_path=None, step=None, curr_poses=None, _actions=None, **kwargs):
    print(kwargs.keys(), curr_poses, _actions)

    fig, axes = plt.subplots(2, 3, figsize=(24, 18))

    pred_path = [meta.state_index_to_grid_index(p) for p in _poses]

    H, W = meta.map_res

    axes = axes.flat
    ax = axes[0]
    ax.imshow(curr_top_view)
    ax.set_title("Top view")

    ax = axes[1]
    im = ax.imshow(rearrange(q[[3, 2, 1, 4, 8, 0, 5, 6, 7]], "(h w) x y -> (x h) (y w)", h=3))
    # plt.plot(*(data['curr_poses'][idx, [1, 0]] * 3 + 1), "*b")
    # plt.plot(*(data['target'][idx, [1, 0]] * 3 + 1), "sr")
    ax.set_xticks(np.arange(-0.5, H * 3, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, W * 3, 3), minor=True)
    ax.grid(which='minor', color='c', linestyle='-', linewidth=1)
    ax.set_title("pred action")
    fig.colorbar(im, ax=ax, orientation='vertical')

    ax.plot(*tuple(zip(*[(3*y + 1, 3*x + 1) for x, y in pred_path])), c='tab:red', label='Predicted Path', linewidth=3.0)

    ax = axes[2]
    im = ax.imshow(rearrange(r_sa[[3, 2, 1, 4, 8, 0, 5, 6, 7]], "(h w) x y -> (x h) (y w)", h=3))
    # plt.plot(*(data['curr_poses'][idx, [1, 0]] * 3 + 1), "*b")
    # plt.plot(*(data['target'][idx, [1, 0]] * 3 + 1), "sr")
    ax.set_xticks(np.arange(-0.5, H * 3, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, W * 3, 3), minor=True)
    ax.grid(which='minor', color='c', linestyle='-', linewidth=1)
    ax.set_title("pred rewards")
    fig.colorbar(im, ax=ax, orientation='vertical')

    ax = axes[3]
    im = ax.imshow(rearrange(aa[[3, 2, 1, 4, 8, 0, 5, 6, 7]], "(h w) x y -> (x h) (y w)", h=3))
    # plt.plot(*(data['curr_poses'][idx, [1, 0]] * 3 + 1), "*b")
    # plt.plot(*(data['target'][idx, [1, 0]] * 3 + 1), "sr")
    ax.set_xticks(np.arange(-0.5, H * 3, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, W * 3, 3), minor=True)
    ax.grid(which='minor', color='c', linestyle='-', linewidth=1)
    ax.set_title("action availability")
    fig.colorbar(im, ax=ax, orientation='vertical')

    ax = axes[4]
    ax.imshow(rearrange(curr_rgb, "c h w -> h w c"))
    ax.set_title("Agent view")
    ax.axis('off')

    visualise(grids=[v], show=show, save_path=save_path, pred_path=pred_path[:step+1], curr_pose=curr_poses,
              fig=fig, axes=axes[5:], titles=['Agent view', 'Pred values', 'Pred rewards 0', 'Pred rewards 1',
                                              'Pred dones 0', 'vis'])


def ego_mini_rollout_plot(meta, *, curr_rgb=None, feature_counts=None, rgb_map=None, v=None, q=None, q0=None, r=None, p=None, r_d=None, d=None, _poses=None, vis=None,
                      action_set=None, show=True, curr_top_view=None, save_path=None, step=None, curr_poses=None, _actions=None, **kwargs):
    print(kwargs.keys(), curr_poses, _actions)

    fig, axes = plt.subplots(2, 2, figsize=(24, 18))

    pred_path = [meta.state_index_to_grid_index(p) for p in _poses]

    H, W = meta.map_res

    axes = axes.flat
    ax = axes[0]
    ax.imshow(curr_top_view)
    ax.set_title("Top view")

    ax = axes[1]
    ax.imshow(rearrange(curr_rgb, "c h w -> h w c"))
    ax.set_title("Agent view")
    ax.axis('off')

    rgb_map = rearrange(rgb_map, "c v h w -> v h w c")

    ax = axes[2]
    ax.imshow(rgb_map[0])
    ax.set_title("rgb map 0")
    ax.axis('off')

    ax = axes[3]
    ax.imshow(rgb_map[1])
    ax.set_title("rgb map 1")
    ax.axis('off')
    #
    # visualise(grids=[rgb_map[0], rgb_map[1]], show=show, save_path=save_path, pred_path=pred_path[:step+1], curr_pose=curr_poses,
    #           fig=fig, axes=axes[2:], titles=['rgb map 0', 'rgb map 1'])
