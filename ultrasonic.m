%% Acoustic and thermal simulation - CPP Interactive mode

function [Max_map, Isppa_map, Thermal_map] = ultrasonic(delta_y, delta_z)

% ultrasonic.m : acoustic and thermal simulation function to optimize the transducer position
%(y and z coordinates) keeping fixed the x coordinate.
%
% DESCRIPTION:
%     The function runs an acoustic and thermal 3D simulation using k-wave packages
%     (kspaceFirstOrder3DC, kWaveDiffusion) with CPP-interactive-mode-enhancement.
%     It returns the maximum peak pressure map, the Isppa map, and the thermal rise map.
%     The loaded image is a CT scan. That guides the grid size and the medium characterization.
%     The transducer is a geometrical bowl. The simulation time steps are computed based on
%     the the ppw and the cfl for the acoustic simulation, and they are instead given for
%     the thermal simulation which is set with sonification duration and pulse duration parameters.
%
% USAGE:
%     [Max_map, Isppa_map, Thermal_map] = ultrasonic(delta_y, delta_z)
%
% INPUTS:
%     delta_y = y step size to add or subtract from the geometric focus coordinate
%     delta_z = z step size to add or subtract from the geometric focus coordinate
%
% OUTPUTS:
%     Max_map          - Maximum peak pressure map sensored during the
%                        acoustic simulation [Pa]. It keeps the size of the
%                        computational grid (Nx x Ny x Nz).
%     Isppa_map        - Isppa map [W/cm^2] computed after the acoustic simulation
%                        has ended, with the use of the maximum peak
%                        pressure map. It keeps the size of the
%                        computational grid (Nx x Ny x Nz).
%     Thermal_map      - Thermal rise map [deg Celsius] computed after the thermal
%                        simulation has ended, with the use of the
%                        temperature map and the initial temperature value,
%                        which is subtracted from the temperature map (and taken in absolute value).

%% Parameters definition

% Chose a defined ultrasound frequency (250 KHz, 500 KHz, 750 KHz, 1 MHz)
f = 250;

% Transducer radius of curvature [mm]
radius = 63;

% Transducer diameter [mm]
diameter = 45;  %(make sure the diameter is an odd number of grid points, but considering it after the resolution reset,
                  % i.e if diameter before reset is 63 and after is
                  % 63/0.5=1fsleyes26, even tough at first it was an odd number, the final result is not)

% Focal distance [mm] (generally focal distance is the radius of curvature)
FD = 43;

% Sonification duration [e-1 sec] for the thermal simulation
SD = 1;

% Perfectly matched layer size (in 3D advisable 10 units)
%pml_size = 10;
pml_size = [10, 10, 10];

% Other initial values
f_Hz = f*1e3;    % Hz
f_MHz = f*1e-3;  % in MHz in case we use water absorption function
T_0 = 37.5;      % initial temperature for water k-wave functions and thermal rise map

Isppa0 = 18; % initial Isppa at the source [W/cm^2]
Isppa0_m2 = Isppa0 * 1e4; % Conversion [W/m^2]

% Source amplitude [Pa]
amp = ((2*waterDensity(T_0)*waterSoundSpeed(T_0)*Isppa0_m2)^0.5);

% Skull sound speed and density (From Simulation_Medium_Parameters.pdf)
Skullsoundspeed = 3360; %[m/s]
Skulldensity = 1908; %[Kg/m^3]

% Load CT scan (be sure the CT resolution is the same resolution of the grid)
input_ct = niftiread('/project/2420104.01/personal/Davide/data/CT/CT_Human_sampled_1mmres_crop.nii.gz');
brain_model = input_ct>200;                   % skull segmentation using thershold

% Methods to fill the holes
brain_model = imfill(brain_model,'holes');    % [1] First method
cubes = strel('cuboid',[2 2 2]);              % [2] Second method: create cuboids which will fill the holes in the skull image*
model = imclose(brain_model, cubes);          % [2] fill holes with the cuboids

%* the cuboids can also change in size (i.e. [2 4 6]) and then assume different shape other than cubes

% Skull selection
mx = 90; my = 100; mz = 110;        % the extremes coordinates of the model (found in software such as FSLeyes)
model = brain_model(1:mx,1:my,1:mz); % decrease useless space

% % CHECK: show CT (Warning: in Linux environment it may cost excessive time to open the volume viewer)
% volumeViewer(model); pause(2);

% % Create grid (Advisable to use power of 2 grid size for each dimension)
% If PML Inside is false (check input_args for kspaceFirstOrder) subtract
% from the power of 2 number the pml_size for each coordinate
% (i.e 128 - 10 = 118). In this way, the computational grid is still a power of 2
% size when pml is added.

Grid = zeros(118, 118, 118);

% Insert the model in the grid in the location of interest
ex = 10; ey = 1; ez = 1;                      % insert the coordinates where to set the (0,0,0) origin of the model in the grid
Grid(ex:mx+ex-1,ey:my+ey-1,ez:mz+ez-1) = model; % put the model in the new grid in the location of interest
model = Grid;                % for computational expenses the grid may be restricted and the model is saved again in the new grid

% define the grid parameters
[Nx, Ny, Nz] = size(model);     % number of grid points in the X/Y/Z direction

% grid resolution
dx = 1e-3;                    % grid point spacing in the X direction [m]
dy = 1e-3;                    % grid point spacing in the Y direction [m]
dz = 1e-3;                    % grid point spacing in the Z direction [m]

% create the 3D - computational grid
kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);

% =========================================================================
% ACOUSTIC SIMULATION
% =========================================================================
%% Set Medium

%Medium.alpha_coeff based on the ultrasound frequency (From Simulation_Medium_Parameters.pdf)
%[coeff. water, coeff. cortical skull, coeff. brain]

%Medium.alpha_coeff based on the ultrasound frequency
if f==250
   Alpha_coeff = [0.006 13.64 1.12]*8.686*10^-2/0.25;
elseif f==500
   Alpha_coeff = [0.013 27.28 2.76]*8.686*10^-2/0.5;
elseif f==750
   Alpha_coeff = [0.019 40.91 4.68]*8.686*10^-2/0.75;
elseif f==1000
   Alpha_coeff = [0.025 54.55 6.80]*8.686*10^-2;
end

% speed [m/s]
medium.sound_speed = waterSoundSpeed(T_0) * ones(Nx, Ny, Nz);    % water
medium.sound_speed(model==1)=Skullsoundspeed;                    % skull

% density [Kg/m3]
medium.density = waterDensity(T_0) * ones(Nx, Ny, Nz);           % water
medium.density(model==1)=Skulldensity;                           % skull

% take the max sound speed for calculate ppw
Max_sound_speed = max(mean(medium.sound_speed,'all'), mean(medium.sound_speed(model==1),'all'));

% alpha coefficient
medium.alpha_coeff = Alpha_coeff(1) * ones(Nx, Ny, Nz);          % water
medium.alpha_coeff(model==1) = Alpha_coeff(2);                   % skull

% medium.alpha_mode = 'no_dispersion'; %default parameter in case alpha_power = 1
medium.alpha_power = 1.5; %(From Simulation_Medium_Parameters.pdf)

%% Set Transducer and Target Area

radius   = (radius/dx)*1e-3;    % reset radius value  based on the grid resolution
diameter = (diameter/dx)*1e-3;  % reset diameter value based on the grid resolution

% Focal distance is the curvature radius
FD = (FD/dx)*1e-3;              % reset focal distance value based on the grid resolution

% define the target area coordinates M1 area
Gx = 59;
Gy = 43;
Gz = 76;

% define the transducer bowl center coordinates from the geometric focus coordinates.
% The use of round() avoids the possibility to not having integer grid coordinates,
% which is justifiable if the grid resolution is 1mm, otherwise, for example when the
% resolution is 0.5 mm, it would be ideal to be able to approximate each coordinate to the next integer
% or to the next integer +- 0.5.

Fx = round(Gx + FD);
Fy = round(Gy + delta_y);
Fz = round(Gz + delta_z);

% define a focused ultrasound transducer (Bowl)
source.p_mask = makeBowl([Nx, Ny, Nz], [Fx, Fy, Fz], radius, diameter, [Gx, Gy, Gz]);

%% Set Time Steps and Source Signal

% calculate the time step using an integer number of points per period
ppw = Max_sound_speed/ (f_Hz * dx);     % points per wavelength (optimal higher than 2)
cfl = 0.3;                              % cfl number
ppp = ceil(ppw / cfl);                  % points per period
T   = 1 / f_Hz;                         % period [s]
dt  = T / ppp;                          % time step [s]

% calculate the number of time steps to reach steady state
t_end = sqrt( kgrid.x_size.^2 + kgrid.y_size.^2 + kgrid.z_size.^2) / Max_sound_speed;
Nt = round(t_end / dt);
dt_stability_limit = checkStability(kgrid, medium);
dt = dt_stability_limit;

% create the time array
kgrid.setTime(Nt, dt);

% phase difference
phase = 0;

% define the input signal
source.p = createCWSignals(kgrid.t_array, f_Hz, amp, phase);

%% Set Sensor

% set the sensor mask to cover the entire grid
sensor.mask = ones(Nx, Ny, Nz);
sensor.record = {'p', 'p_max_all'}; % record variables of interest

% record the last 3 cycles in steady state
num_periods = 3;
T_points = round(num_periods * T / kgrid.dt);
sensor.record_start_index = Nt - T_points + 1;

input_args = {'PMLSize', pml_size, 'PMLInside', false, 'SystemCall', 'module load gcc'};

%% Run kspaceFirstOrder3DC - CPP Interactive mode

sensor_data = kspaceFirstOrder3DC(kgrid, medium, source, sensor, input_args{:});

%% Find Maximum Final Pressure (Acoustic Focus), its coordinates, and save Max Peak Pressure and Isppa map

% s = size(sensor_data.p_max_all); % size of the pressure map
% [v,ii] = max(reshape(sensor_data.p_max_all,[],s(3))); % reshape the pressure matrix in 1D array and save the maximum values and array indices in each gridpoint
% [i1,j1] = ind2sub(s(1:2),ii); % translate the indices in cartesian coordinates of the spatial grid
% out = [v;i1;j1;1:s(3)]'; % create out as a information matrix which contains maximum values and coordinates while the for loop is searching for the biggest value
%
% % Find max pressure and save max pressure and its coordinates
% MAX = [0,0];
% for i = 1:size(out,1)
%     if out(i,1)> MAX(1)
%         MAX(1) = out(i,1);
%         MAX(2) = i;
%     end
% end
%
% % Save the acoustic focus coordinates in Focus
% Focus = out(MAX(2),2:4);
%
% % Save the maximum pressure in MAX
% MAX = MAX(1);
%
% % Save acoustic focus coordinates in [Ax, Ay, Az]
% Ax = Focus(1); Ay = Focus(2); Az = Focus(3);

% Save Max Peak Pressure map and define Isppa map sizes
Max_map = sensor_data.p_max_all;
Isppa_map = zeros(Nx, Ny, Nz);

% Loop to compute the Isppa map from the Max peak pressure map
for i=1:Nx
    for j=1:Ny
        for k=1:Nz
            if medium.density(i,j,k)==waterDensity(T_0)
                Isppa_map(i,j,k) = ((Max_map(i,j,k))^2/(2*waterDensity(T_0)*waterSoundSpeed(T_0))).*1e-4;
            elseif medium.density(i,j,k)==Skulldensity
                Isppa_map(i,j,k) = ((Max_map(i,j,k))^2/(2*Skulldensity*Skullsoundspeed)).*1e-4;
            end
        end
    end
end

% =========================================================================
% THERMAL SIMULATION
% =========================================================================
%% Volume rate of heat deposition

% convert the absorption coefficient to nepers/m
alpha_np = db2neper(medium.alpha_coeff, medium.alpha_power) * ...
    (2 * pi * f_Hz).^medium.alpha_power;

% extract the pressure amplitude at each position
p = extractAmpPhase(sensor_data.p, 1/kgrid.dt, f_Hz);

% reshape the data, and calculate the volume rate of heat deposition
p = reshape(p, Nx, Ny, Nz);

% volume rate of heat deposition (how much heat is deposited in a unit
% volume for unit of time)
Q = alpha_np .* p.^2 ./ (medium.density .* medium.sound_speed);

%% clear the input structures

clear medium source sensor;

%% set the background temperature and heating term

source.Q = Q;

source.T0  = T_0; % * ones(Nx, Ny, Nz) + Cold_Water*24; %Celsius

%% define the properties of the propagation medium

% speed [m/s] (From Simulation_Medium_Parameters.pdf)
medium.sound_speed = waterSoundSpeed(T_0) * ones(Nx, Ny, Nz);       % water
medium.sound_speed(model==1)=Skullsoundspeed;                       % skull

% density [Kg/m3] (From Simulation_Medium_Parameters.pdf)
medium.density = waterDensity(T_0) * ones(Nx, Ny, Nz);              % water
medium.density(model==1)=Skulldensity;                              % skull

% thermal conductivity [W/(m.K)] (From Simulation_Medium_Parameters.pdf)
medium.thermal_conductivity = 0.6 * ones(Nx, Ny, Nz);               % water
medium.thermal_conductivity(model==1)=0.32;                         % skull

% specific heat [J/(kg.K)] (From Simulation_Medium_Parameters.pdf)
medium.specific_heat = 4178 * ones(Nx, Ny, Nz);                     % water
medium.specific_heat(model==1) = 1313;                              % skull

%% create kWaveDiffusion object

kdiff = kWaveDiffusion(kgrid, medium, source, [], 'PlotSim', false);

% set source on time and off time
time = 0;
dt = 0.001;  % set time step size [s];

% Loop simulating the pulse/burst time profile of the sonification
while time<SD

    % on_time + off_time = Pulse_Duration
    on_time  = 0.03;  % [s] 
    off_time = 0.07;  % [s]

    % take time steps
    kdiff.Q = Q;
    kdiff.takeTimeStep(round(on_time / dt), dt);

    % turn off heat source and take time steps
    kdiff.Q = 0;
    kdiff.takeTimeStep(round(off_time / dt), dt);

    time = time + 1;

end

%% Thermal map

T = kdiff.T;
%T_Max = T(Ax, Ay, Az);
Thermal_map = T - T_0;

end
