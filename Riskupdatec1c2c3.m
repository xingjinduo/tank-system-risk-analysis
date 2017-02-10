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
%T = 140;
T=120;


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

%{
figure
plot(1:T,y,'b', 1:T, yh,'r', 1:T, yReal, 'k');
legend('observation','filtered observation','real-state');
title('Observation vs filtered observation by the particle filter','FontSize',14);
 %}

%% RUL prediction

%y_th = 0.7172;
y_th=0.8;

% for i = 1:T
%     if yh(i) < y_th
%         TrueTTF = i-1;
%         break
%     end
% end
% Initialization
sample_particles = pf.particles; % Particles
sample_w = pf.w; % Weights of each particle at each t
Ns = pf.Ns; % Number of particles
time_obs = 10:10:120;
n_obs = length(time_obs);

vect_n_obs = linspace(1,n_obs,n_obs);
vect_n_obs = vect_n_obs';
vect_R = zeros(12,1);

RUL = T*ones(Ns,n_obs); % Time to failure
%options = optimoptions('Display','off'); options = optimoptions(@fsolve,'Display','off');
options = optimoptions(@fsolve,'Display','off');
% Prediction
for t = 1:n_obs 
    count=0;
    fprintf('t = %d / %d\n',t,n_obs)
    sample_para = sample_particles(:,:,time_obs(t)); % Estimated xs by particles at each t
%     u_1 = normrnd(0,sigma_u(1),1,num_u);
%     u_2 = normrnd(0,sigma_u(2),1,num_u);
%     u_3 = normrnd(0,sigma_u(3),1,num_u);
%     u_4 = normrnd(0,sigma_u(4),1,num_u);
%     v = normrnd(0,sigma_v,1,num_u);
    for i = 1:Ns
        xkm = sample_para(:,i);
        obj = @(tt) obs(tt,xkm,0)-y_th;
        TTF = fsolve(obj,T,options);
        if TTF < time_obs(t)
            RUL(i,time_obs(t)) = 0;
        else
            RUL(i,time_obs(t)) = TTF - time_obs(t);
            count=count+1;
        end
    end
    vect_R(t,1)=count/Ns;
end
%plot(time_obs,vect_R,'o');
 
%Plot pdf of RUL
% figure
% hold on;
% xi = 1:T;
% yi = linspace(0,180,1000);
% [xx,yy] = meshgrid(xi,yi);
% den = zeros(size(xx));
% xhmode = zeros(size(xh));
% for i = xi
%    % for each time step perform a kernel density estimation
%    den(:,i) = ksdensity(RUL(:,i), yi,'kernel','epanechnikov','weights',sample_w(:,i));
%    [~, idx] = max(den(:,i));
% 
%    % estimate the mode of the density
%    xhmode(i) = yi(idx);
%    if mod(i,10) == 1
%        plot3(repmat(xi(i),length(yi),1), yi', den(:,i));
%    end
% end
% view(3);
% box on;
% title('Evolution of the RUL density','FontSize',14)

%Plot RUL VS t
obj = @(tt) obs(tt,xh0,0)-y_th;
TrueTTF = fsolve(obj,T);
% for t = 1:T

%figure
%plot(time_obs,sum(RUL(:,time_obs).*sample_w(:,time_obs)),'.-');
%hold on
%t = 0:T;
%plot(time_obs,TrueTTF-time_obs,'-r');
%f_3=[0 0 0 0 0 0 0.1747 0.8995 1 1 1 1 ];
f_3=ones(12,1)-vect_R;
f_3=f_3';
f_1=0.025;
f_4=0.015;
f_2=0.150;
f_5=0.045;
p_1=1-f_1;
p_2=1-f_2;
p_3=vect_R;
p_3=p_3';
p_4=1-f_4;
p_5=1-f_5;
c_1=p_1.*p_2;
c_2=p_1.*f_2.*p_3.*p_4;
c_3=p_1.*f_2.*p_3.*f_4.*p_5;
c_4=p_1.*f_2.*p_3.*f_4.*f_5;
c_5=p_1.*f_2.*f_3.*p_5;
c_6=p_1.*f_2.*f_3.*f_5;
c_7=f_1.*p_3.*p_4;
c_8=f_1.*p_3.*f_4.*p_5;
c_9=f_1.*p_3.*f_4.*f_5;
c_10=f_1.*f_3.*p_5;
c_11=f_1.*f_3.*f_5;
s1=c_1+c_2+c_7;
s2=c_3+c_5+c_8+c_10;
s3=c_4+c_6+c_9+c_11;
%subplot(1,3,1);
figure;
plot(time_obs,s1,'o');
%subplot(1,3,2);
figure;
plot(time_obs,s2,'*');
%subplot(1,3,3);
figure;
plot(time_obs,s3,'+')