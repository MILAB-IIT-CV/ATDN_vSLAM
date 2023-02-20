import argparse
import datetime

import matplotlib.pyplot as plt

import evo.main_ape as main_ape
import evo.main_traj as main_traj
from evo.tools import file_interface
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.core.metrics import PoseRelation
from evo.tools.settings import SETTINGS

# temporarily override some package settings
SETTINGS.plot_figsize = [6, 6]
SETTINGS.plot_split = True
SETTINGS.plot_usetex = False


if __name__ == "__main__":
    # arguments
    # --est:            estimated poses
    # --GT:             ground truth poses
    # --model_name:     current model name
    # --plots:          what plotting mode will be used

    parser = argparse.ArgumentParser(description="Odometry results plotter")

    parser.add_argument("est_path",
                        metavar="est",
                        type=str,
                        nargs=1,
                        help="path to estimation poses to be used (.txt)",
                        )

    parser.add_argument("GT_path",
                        metavar="GT",
                        type=str,
                        nargs=1,
                        help="path to ground truth poses to be used (.txt)",
                        )

    parser.add_argument("--model_name",
                        type=str,
                        default="",
                        help="name of the current model",
                        nargs="?",
                        )

    parser.add_argument("--plots",
                        type=str,
                        default="[xz]",
                        choices=["xz", "xy", "yz", "xyz"],
                        help="list plots - which axes will be used",
                        nargs="*",
                        )

    args = parser.parse_args()

    # parse plot modes
    plot_modes = []
    for plot_mode in args.plots:
        if plot_mode == "xz":
            plot_modes.append(PlotMode.xz)
            continue
        if plot_mode == "xy":
            plot_modes.append(PlotMode.xy)
            continue
        if plot_mode == "yz":
            plot_modes.append(PlotMode.yz)
            continue
        if plot_mode == "xyz":
            plot_modes.append(PlotMode.xyz)
            continue


    traj_ref = file_interface.read_kitti_poses_file(args.GT_path[0])
    traj_est = file_interface.read_kitti_poses_file(args.est_path[0])

    est_name = args.model_name

    count = 0
    results = []
    for plot_mode, plot_mode_string in zip(plot_modes, args.plots):
        result = main_ape.ape(traj_ref=traj_ref,
                              traj_est=traj_est,
                              est_name=est_name,
                              align=True,
                              correct_scale=True,
                              pose_relation=PoseRelation.translation_part,
                              )

        count += 1
        results.append(result)

        fig = plt.figure()
        ax = plot.prepare_axis(fig, plot_mode)
        plot.traj(ax, plot_mode, traj_ref, style="--", alpha=0.5)
        plot.traj_colormap(ax,
                           result.trajectories[est_name],
                           result.np_arrays["error_array"],
                           plot_mode,
                           min_map=result.stats["min"],
                           max_map=result.stats["max"],
                           )

        plt.savefig(f"{est_name}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}_{plot_mode_string}.png")

        plot_collection = plot.PlotCollection("evo_traj - trajectory plot")
        fig_xyz, axarr_xyz = plt.subplots(3, sharex="col", figsize=tuple(SETTINGS.plot_figsize))
        plot.traj_xyz(axarr_xyz,
                      traj_ref,
                      style=SETTINGS.plot_reference_linestyle,
                      color=SETTINGS.plot_reference_color,
                      label="ground truth",
                      alpha=SETTINGS.plot_reference_alpha,
                      start_timestamp=None,
                      )
        traj_est.align_origin(traj_ref=traj_ref)
        plot.traj_xyz(axarr_xyz,
                      traj_est,
                      style=SETTINGS.plot_trajectory_linestyle,
                      color="blue",
                      label=est_name,
                      alpha=SETTINGS.plot_trajectory_alpha,
                      start_timestamp=None,
                      )
        plot_collection.add_figure("xyz_view", fig_xyz)

        plot_collection.export(f"{est_name}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}_traj.png",
                               confirm_overwrite=False,
                               )
