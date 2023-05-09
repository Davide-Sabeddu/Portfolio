% The ultrasonic loop allows to run multiple ultrasonic simulations (both
% acoustic and thermal), to find the best location in cartesian coordinates where to place the
% transducer, relatively to max peak pressure map, Isppa map, and thermal rise map

% The loop can be initiate in the terminal at the path where it is saved
% this file, with the command matlab_sub ultrasonic_loop.m. The output file
% is saved in the same folder of the loop.

% Set working directory
workDir = fullfile('/project','2420104.01','personal','Davide','code');

% Add path to k-Wave toolbox
addpath (fullfile(workDir,'kwave'));
addpath (fullfile(workDir,'kwave','binaries'));
% Add function for the loop
addpath (fullfile(workDir,'orca-lab','project','RecreaTUS'));
% Add CT scan from which the simulations are created
addpath (fullfile('/project','2420104.01','personal','Davide','data'));

N = 9; % Number of simulations
s = 1; % Step size (delta_y, delta_x) [mm]
% The different locations of the transducer bowl center respect to the
% geometric focus coordinates and x = x + focal length
positions = [0, 0; s, 0; s, s; 0, s; -s, s; -s, 0; -s, -s; 0, -s; s, -s]; 

% Output data must be in the same size of the computational grid in the
% function we are using (in this loop ultrasonic.m)
X = 128;
Y = 128;
Z = 128;
% Define the output data structures
Max = zeros(X,Y,Z,N);
Isppa = zeros(X,Y,Z,N);
Thermal = zeros(X,Y,Z,N);

% Loop
% (in interactive mode, for 9 simulations (128x128x128 grid) and 0.1 sec of sonification (time step 0.001 sec),
% the walltime is ~5 min for each simulation)
for i=1:N
    [Max(:,:,:,i), Isppa(:,:,:,i), Thermal(:,:,:,i)] = ultrasonic(positions(i,1), positions(i,2));
end

% Extract the mean of the max peak pressure, Isppa, and themral rise map
Mean_Max = mean(Max,4);
Mean_Isppa = mean(Isppa,4);
Mean_Thermal = mean(Thermal,4);

% Extract the standard deviation of the max peak pressure, Isppa, and themral rise map
Std_Isppa = std(Isppa,0,4);
Std_Max = std(Max,0,4);
Std_Thermal = std(Thermal,0,4);

% Save the mean and standard deviation outputs
save('ultrasonic_output.mat','Mean_Isppa','Mean_Max','Mean_Thermal','Std_Isppa','Std_Max','Std_Thermal');

%% Check the output data

% after loading the ultrasonic_output.at into the workspace, we can check
% the values. These are just some parts which might be useful in reading the data

% Mask: 
% sometimes the values range is too wide to be visualized well on a plot.
% This masks the mean and std maps from a certain value, in order to have a figure
% of the values of interest

for i=1:X
    for j=1:Y
        for k=1:Z
            if Mean_Isppa(i,j,k)>=4
                Mean_Isppa(i,j,k) = 0;
            end
        end
    end
end

% Plot:
% Any output can be visualized in different planes. To change the
% coordinates order in the matrices we can permute them.

% In this example [x y z]->[x z y] 
Mean_Isppa = permute(Mean_Isppa,[1 3 2]);

% Define and plot the figure
f1 = figure;
imagesc(1, 1, Mean_Isppa(:,:,43));
h = colorbar;
xlabel(h, '[W/cm^2]');
ylabel('x-position [mm]');
xlabel('z-position [mm]');
axis image;
title('Isppa - Mean for 9 simulations - Cartesian coordinates');
% coordinates of the geometric focus (check the ultrasonic.m function)
text(59,76,43,'x');
% set colormap and enlarge figure window
colormap(jet(256));
scaleFig(1.5, 1);

% Save the figure in a specific path
saveas(f1,['~/Desktop/Mean_Isppa_9_output.jpg']);

% Close the figure
close(f1);