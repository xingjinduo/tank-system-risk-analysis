clear, clc, close all;
tic;
%% Preparation
% Process equation x[k] = sys(k, x[k-1], u[k]);
nx = 4;  % number of states
sys = @(k, xkm1, uk) [xkm1(1) + uk(1); xkm1(2) + uk(2);...
    xkm1(3) + uk(3); xkm1(4) + uk(4);]; % (returns column vector)

% Observation equation y[k] = obs(k, x[k], v[k]);
ny = 1;                                           % number of observations
obs = @(k, xk, vk) xk(1).*exp(xk(2).*k) + xk(3).*exp(xk(4).*k) + vk;   % (returns column vector)

% PDF of process noise and noise generator function
nu = 4;                                           % size of the vector of process noise
%sigma_u = 1e-1*[1e-5, 1e-6, 1e-6, 1e-5]';
sigma_u = 1e-5*[100, 1, 1, 100]';
% sigma_u = [0, 0, 0, 0]';
p_sys_noise   = @(u) normpdf(u, 0, sigma_u);
gen_sys_noise = @(u) normrnd(0, sigma_u);         % sample from p_sys_noise (returns column vector)

% PDF of observation noise and noise generator function
nv = 1;                                           % size of the vector of observation noise
%sigma_v = 5e-3;
sigma_v = 1e-2;
p_obs_noise   = @(v) normpdf(v, 0, sigma_v);
gen_obs_noise = @(v) normrnd(0, sigma_v);         % sample from p_obs_noise (returns column vector)

% Initial PDF
% p_x0 = @(x) normpdf(x, 0,sqrt(10));             % initial pdf
% gen_x0 = @(x) normrnd([.887, -8.86e-4, -2.32e-4, .0458]', sigma_u);               % sample from p_x0 (returns column vector)
gen_x0 = @(x) [random('unif',.88,.89), random('unif',-9e-4,-8e-4),...
    random('unif',-3e-4,-2e-4), random('unif',.03,.05),];

% Transition prior PDF p(x[k] | x[k-1])
% (under the suposition of additive process noise)
% p_xk_given_xkm1 = @(k, xk, xkm1) p_sys_noise(xk - sys(k, xkm1, 0));

% Observation likelihood PDF p(y[k] | x[k])
% (under the suposition of additive process noise)
p_yk_given_xk = @(k, yk, xk) p_obs_noise(yk - obs(k, xk, 0));

% Number of time steps
T = 120;

% Separate memory space
x = zeros(nx,T);  y = zeros(ny,T);
u = zeros(nu,T);  v = zeros(nv,T);
yReal = y;
% Simulate a system trajetory
xh0 = [.887, -8.86e-4, -2.32e-4, .0458]';                                  % initial state, true value
u(:,1) = gen_sys_noise();                               % initial process noise
v(:,1) = gen_obs_noise(sigma_v);          % initial observation noise
x(:,1) = xh0;
y(:,1) = obs(1, xh0, v(:,1));
for k = 2:T
   % here we are basically sampling from p_xk_given_xkm1 and from p_yk_given_xk
   u(:,k) = gen_sys_noise();              % simulate process noise
   v(:,k) = gen_obs_noise();              % simulate observation noise
   x(:,k) = sys(k, x(:,k-1), u(:,k));     % simulate state
   y(:,k) = obs(k, x(:,k),   v(:,k));     % simulate observation
end

% State estimation
% Separate memory
xh = zeros(nx, T); xh(:,1) = xh0;
yh = zeros(ny, T); yh(:,1) = obs(1, xh0, 0);

pf.k               = 1;                   % initial iteration number
pf.Ns              = 1e4;                 % number of particles
pf.w               = zeros(pf.Ns, T);     % weights
pf.particles       = zeros(nx, pf.Ns, T); % particles
pf.gen_x0          = gen_x0;              % function for sampling from initial pdf p_x0
pf.p_yk_given_xk   = p_yk_given_xk;       % function of the observation likelihood PDF p(y[k] | x[k])
pf.gen_sys_noise   = gen_sys_noise;       % function for generating system noise
%pf.p_x0 = p_x0;                          % initial prior PDF p(x[0])
%pf.p_xk_given_ xkm1 = p_xk_given_xkm1;   % transition prior PDF p(x[k] | x[k-1])

% Estimate state
for k = 2:T
   fprintf('Iteration = %d/%d\n',k,T);
   % state estimation
   pf.k = k;
   %[xh(:,k), pf] = particle_filter(sys, y(:,k), pf, 'multinomial_resampling');
   [xh(:,k), pf] = particle_filter(sys, y(:,k), pf, 'systematic_resampling');   
 
   % filtered observation
   yh(:,k) = obs(k, xh(:,k), 0);
end

for k = 1:T
   yReal(:,k) = obs(k, xh0, 0);     % simulate observation
end

 %plot of the observation vs filtered observation by the particle filter

figure
%plot(1:T,y,'b', 1:T, yh,'r', 1:T, yReal, 'k');
plot(1:T,y,'b', 1:T, yh,'r');
hold on
plot([0 120],[0.8 0.8]);
%legend('Observation','Filtered observation','Real-state','Threshold');
legend('Observation','Filtered observation','Threshold');

xlabel('Time');
ylabel('Capacity');
axis([0 120 0 1])
