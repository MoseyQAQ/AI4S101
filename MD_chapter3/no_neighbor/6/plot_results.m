clear; close all
load energy.txt;
timeStep = 5/1000; % ps
sampleInterval = 100;
timeInterval = timeStep * sampleInterval;
numData = size(energy, 1);
time = (1 : numData) * timeInterval;
totalEnergy = sum(energy, 2);
relativeEnergy = totalEnergy / mean(abs(totalEnergy));

figure;

subplot(2,2,1)
plot(time, energy(:,1), '-', 'linewidth', 2);hold on;
xlabel('Time (ps)', 'fontsize', 15);
ylabel('Kinetic Energy (eV)', 'fontsize', 15);
title('(a)', 'fontsize', 15);
set(gca, 'fontsize', 15);

subplot(2,2,2)
plot(time, energy(:,2), '-', 'linewidth', 2); hold on;
xlabel('Time (ps)', 'fontsize', 15);
ylabel('Potential Energy (eV)', 'fontsize', 15);
title('(b)', 'fontsize', 15);
set(gca, 'fontsize', 15);

subplot(2,2,3)
plot(time, totalEnergy, '-', 'linewidth', 2); hold on;
xlabel('Time (ps)', 'fontsize', 15);
ylabel('Total Energy (eV)', 'fontsize', 15);
title('(c)', 'fontsize', 15);
set(gca, 'fontsize', 15);

subplot(2,2,4)
plot(time(2:end), relativeEnergy(2:end),'-', 'linewidth', 2); hold on;
xlabel('Time (ps)', 'fontsize', 15);
ylabel('Relative Energy', 'fontsize', 15);
title('(d)', 'fontsize', 15);
set(gca, 'fontsize', 15);

load energy_ref.txt

figure;
plot(energy-energy_ref);
