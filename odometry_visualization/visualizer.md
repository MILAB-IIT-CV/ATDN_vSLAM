# Visualize VO results
For creating odometry results, first install the `evo` package from pip using

`pip install evo`

Save the output poses as a txt file (in our example `ATDN_predicition.txt`).

## Built-in method
Next, you can export the built-in figures using:

`evo_traj kitti ATDN_prediction.txt --ref=GT.txt --plot_mode=xz --save_plot`

Note, that `GT.txt` is the reference trajectory.

## Alternative method
Alternatively, you can use our own visualizer python script with:

`python visualizer.py ATDN_prediction.txt GT.txt --model_name ATDN_v1 --plots xz`

The optional argument `--plots` take several plot types (such as xz xyz). 