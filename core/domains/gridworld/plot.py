import matplotlib.pyplot as plt

from core.utils.plot_utils import show_dir_values, visualise


def grid_plot(meta, *, feature_map=None, traj_actions=None, vis=None, values=None, best_action_maps=None, counts=None,
              opt_path=None, curr_pose=None, action_set=None, show=True, save_path=None, **kwargs):
    print(kwargs.keys())
    fig, axes = plt.subplots(2, 3)

    axes = axes.flat
    ax = axes[0]
    show_dir_values(best_action_maps, ax, fig=fig, dirs=action_set)
    ax.set_title("Best action")
    ax = axes[1]
    show_dir_values(traj_actions, ax, fig=fig, dirs=action_set)
    ax.set_title("Traj action")
    visualise(grids=[feature_map, values, vis, counts], opt_path=opt_path, curr_pose=curr_pose, show=show, save_path=save_path,
              fig=fig, axes=axes[2:], titles=['Agent view', 'Values', 'Visibility', 'Node counts'])


def grid_rollout_plot(meta, *, feature_map=None, best_action_maps=None, v=None, q=None, q0=None, r=None, p=None, r_d=None, d=None, _poses=None, vis=None,
                      action_set=None, show=True, save_path=None, step=None, curr_poses=None, _actions=None, **kwargs):
    print(kwargs.keys(), curr_poses, _actions)

    fig, axes = plt.subplots(2, 3, figsize=(24, 18))

    pred_path = [meta.state_index_to_grid_index(p) for p in _poses]

    axes = axes.flat
    ax = axes[0]
    show_dir_values(best_action_maps, ax, fig=fig, dirs=action_set)
    ax.set_title("Best action")
    ax = axes[1]
    show_dir_values(q, ax, fig=fig, dirs=action_set)
    ax.set_title("pred action")

    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

    ax = axes[2]
    show_dir_values(r, ax, fig=fig, dirs=dirs)
    ax.set_title("rewards")
    ax = axes[3]
    show_dir_values(d, ax, fig=fig, dirs=action_set)
    ax.set_title("dones")
    visualise(grids=[feature_map, v], show=show, save_path=save_path, pred_path=pred_path[:step+1], curr_pose=curr_poses,
              fig=fig, axes=axes[4:], titles=['Agent view', 'Pred values'])


def ego_grid_plot(meta, *, is_expert=None, feature_map=None, best_action_maps=None, v=None, q=None, q0=None, r=None, p=None, r_d=None, d=None,
                  _poses=None, vis=None, _returns=None,
                  action_set=None, dirs=None, show=True, save_path=None, step=None, curr_poses=None, _actions=None, **kwargs):
    print(kwargs.keys(), curr_poses, _actions)

    fig, axes = plt.subplots(3, 4)

    pred_path = [meta.state_index_to_grid_index(_p) for _p in _poses]
    dir, x, y = curr_poses
    curr_pose = (dirs[int(dir)], int(x), int(y))

    axes = axes.flat
    # for i, (ax, action) in enumerate(zip(axes[:5], action_set)):
    #     show_dir_values(q0[i], ax=ax, fig=fig, dirs=dirs)
    #     ax.set_title(f"Q0 {action}")

    for i, (ax, action) in enumerate(zip(axes[:5], action_set)):
        show_dir_values(r[i], ax=ax, fig=fig, dirs=dirs)
        ax.set_title(f"reward {action}")

    for i, (ax, action) in enumerate(zip(axes[5:10], action_set)):
        show_dir_values(q[i], ax=ax, fig=fig, dirs=dirs)
        ax.set_title(f"Pred action {action}")

    ax = axes[10]
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

    res = p[0, 0, direction, direction, xs, ys]

    curr_trans = p[0, 0, int(curr_poses[0]), :, :, :, int(curr_poses[1]), int(curr_poses[2])]

    show_dir_values(res, ax, fig=fig, dirs=dirs)
    ax.set_title("forward transitions")

    visualise(grids=[feature_map], show=show, save_path=save_path, pred_path=pred_path[:step+1], curr_pose=curr_pose,
              fig=fig, axes=axes[11:], titles=['agent view'])
