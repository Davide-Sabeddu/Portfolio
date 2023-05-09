# README: ultrasonic
*version: 20-01-2021*

The orca-lab folder project/ultrasonic contains the main function and main loop for searching the optimal set of transducer bowl center coordinates with respect to the max peak pressure, Isppa, and thermal rise maps.

## ultrasonic.m

The function runs a k-wave acoustic and thermal simulation in CPP interactive mode and water+skull setup. This to avoid extra walltime once it is performed into the ultrasonic_loop.m. As inputs it takes delta_y and delta_z to add to the geometric focus y and z coordinates, while x coordinate is computed as transducer_x_coordinate = geometric_focus_x_coordinate + focal length.
It returns the max peak pressure map, Isppa map, and thermal map.

## ultrasonic_loop.m

The loop allows to use the function ultrasonic.m in multiple simulations, saving the outputs in different matrices.

The code contains also a mask, because sometimes the output values range is too wide to be visualized well on a plot. Then, it is possible to mask the mean and standard deviation maps from a certain value, in order to have a figure of the values of interest and/or regions of interest.

In the terminal on the directory where the file is located, the ultrasonic_loop.m runs with the use of the command 'matlab_sub ultrasonic_loop.m'. The time and memory limits can be added to the command (https://dccn-hpc-wiki.readthedocs.io/en/latest/docs/cluster_howto/exercise_matlab/exercise.html) or chosen by default.

The output data is saved in the same folder where the ultrasonic_loop.m is located. The output data is ultrasonic_output.mat and it can be loaded on the workspace in Matlab.