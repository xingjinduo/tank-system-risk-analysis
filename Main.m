% This is to generate the degradation data, conduct PF and make RUL
% prediction using PF.
clear; clc; close all;
%% Prepare data for generating the degradation trajectory
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
sigma_u = 1e-5*[10, 1, 1, 10]';
p_sys_noise   = @(u) normpdf(u, 0, sigma_u);
gen_sys_noise = @(u) normrnd(0, sigma_u);         % sample from p_sys_noise (returns column vector)
% PDF of observation noise and noise generator function
nv = 1;                                           % size of the vector of observation noise
sigma_v = 5e-3;
p_obs_noise   = @(v) normpdf(v, 0, sigma_v);
gen_obs_noise = @(v) normrnd(0, sigma_v);         % sample from p_obs_noise (returns column vector)
% Initial value of the state variables
gen_x0 = @(x) [random('unif',.88,.89), random('unif',-9e-4,-8e-4),...
    random('unif',-3e-4,-2e-4), random('unif',.03,.05),];
% Observation likelihood PDF p(y[k] | x[k])
% (under the suposition of additive process noise)
p_yk_given_xk = @(k, yk, xk) p_obs_noise(yk - obs(k, xk, 0));
% Number of time steps
T = 150;
% Separate memory space
x = zeros(nx,T);  y = zeros(ny,T);
u = zeros(nu,T);  v = zeros(nv,T);
y_true = y;
yReal = y;
%% Simulate a system trajetory
xh0 = [.887, -8.86e-4, -2.32e-4, .0458]';                                  % initial state, true value
u(:,1) = gen_sys_noise();                               % initial process noise
v(:,1) = gen_obs_noise(sigma_v);          % initial observation noise
x(:,1) = xh0;
y(:,1) = obs(1, xh0, v(:,1));
y_true(:,1) = obs(1, xh0, 0);
for k = 2:T
   % here we are basically sampling from p_xk_given_xkm1 and from p_yk_given_xk
   u(:,k) = gen_sys_noise();              % simulate process noise
   v(:,k) = gen_obs_noise();              % simulate observation noise
   x(:,k) = sys(k, x(:,k-1), u(:,k));     % simulate state
   y(:,k) = obs(k, x(:,k),   v(:,k));     % simulate observation
   y_true(:,k) = obs(k, x(:,k), 0);
end
% Draw the generated data
figure
y_th = 0.7;
plot(1:T,y,'-k',...
    1:T,y_true,'-b',...
    1:T,y_th*ones(T,1),'k:');
%% PF filtering
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
% Estimate state
yh5 = yh; % Initial values
yh95 = yh; % Initial values
for k = 2:T
   fprintf('Iteration = %d/%d\n',k,T);
   % state estimation
   pf.k = k;
   %[xh(:,k), pf] = particle_filter(sys, y(:,k), pf, 'multinomial_resampling');
   [xh(:,k),pf,yh(:,k),yh5(:,k),yh95(:,k)] = particle_filter(sys, y(:,k), pf, 'systematic_resampling');
end
% Plot observed values V.S. filtered values
hold on;
plot(2:T,yh(2:T),'r-',...
    2:T,yh5(2:T),'--r',2:T,yh95(2:T),'--r');
legend('Observation','True value','Failure threshold','Filtered by PF','95% Confidence interval')
%% RUL prediction
y_th = 0.7; % Failure threshold
% Find true TTF
for i = 1:T
    if y_true(i) < y_th
        TTF_true = i;
        break;
    end
end
time = [30:10:TTF_true-21,TTF_true-20:TTF_true+5]; % Time instants that used to make prediciton
n_time = length(time);
sample_particles = pf.particles; % Particles
sample_w = pf.w; % Weights of each particle at each t
Ns = pf.Ns; % Number of particles
RUL = zeros(Ns,n_time); % Time to failure
MaxIter = 500;
% Prediction
for i = 1:n_time 
    fprintf('i = %d / %d\n',i,n_time)
    t = time(i);
    sample_para = sample_particles(:,:,t); % Estimated xs by particles at each t
    for j = 1:Ns
        xkm = sample_para(:,j);
        if obs(t,xkm,0) < y_th % if the current time has already failed
            RUL(j,i) = 0;
            continue;
        else 
            % Search for the TTF
            k = t+1;
            while 1
                xk_pred = sys(k,xkm,gen_sys_noise());
                yk_pred = obs(k,xk_pred,0);
                if yk_pred < y_th
                    RUL(j,i) = k-t;
                    break;
                else
                    xkm = xk_pred;
                    k = k+1;
                end
                if k == MaxIter
                    RUL(j,i) = MaxIter;
                    break;
                end
            end
        end
    end
end
% Determine the credibility interval
RUL_percentile = zeros(2,n_time);
alpha = .05; % Confidence level
for i = 1:n_time
    [RUL_sort, I] = sort(RUL(1:end,i));
    temp_w = sample_w(:,time(i));
    w_k_sort = temp_w(I);
    RUL_k_cdf = cumsum(w_k_sort);
    index_L = find(RUL_k_cdf>alpha,1);
    index_U = find(RUL_k_cdf>1-alpha,1);
    RUL_percentile(1,i) = RUL_sort(index_L);
    RUL_percentile(2,i) = RUL_sort(index_U);
end
% Plot RUL VS t
figure
plot(time,sum(RUL.*sample_w(:,time)),'sb-')
hold on
plot(30:TTF_true,TTF_true-[30:TTF_true],'-ok')
plot(time,RUL_percentile(1,1:end),'--r',...
    time,RUL_percentile(2,1:end),'--r')
legend('Estimated RUL','True RUL','95% belief interval');
xlabel('t')
ylabel('RUL')
%% Risk updating
t_update = [TTF_true-5:TTF_true+5];
n_t_update = length(t_update);
% Update reliability of component 3 at each t_update
R_3_update = zeros(n_t_update,1);
particles = pf.particles;
weight = pf.w;
index = 0;
for i = t_update
    fprintf('%d/%d\n',i,T);
    x_cur = particles(:,:,i);
    weight_cur = weight(:,i);
    Count_S = 0;
    for j = 1:Ns
        y_cur = obs(i,x_cur(:,j),0);
        if y_cur > y_th
            Count_S = Count_S + weight_cur(j);
        end
    end
    index = index + 1;
    R_3_update(index) = Count_S;
end
% Calculate the risk indexes
R_1 = 1-0.025;
R_2 = 1-0.150;
R_4 = 1-0.015;
R_5 = 1-0.045;
p_c_update = zeros(3,n_t_update);
for i = 1:n_t_update
    R_3 = R_3_update(i);
    R = [R_1,R_2,R_3,R_4,R_5];
    p_c_update(:,i) = CalPConsequence(R);
    figure
    labels = {['P_{C_1}:' num2str(p_c_update(1,i),3)],...
        ['P_{C_2}:' num2str(p_c_update(2,i),'%.2e')],...
        ['P_{C_3}:' num2str(p_c_update(3,i),'%.2e')]};
    pie(p_c_update(:,i),[1,0,0],labels);
    title(['t = ' num2str(t_update(i))]);
end
%% Risk prediction
p_c_1_th = .90; % Definition of threshold event
TCE_true = TTF_true; % True TCE
TCE = zeros(1,n_time); % Time to Critical Event, defined by p_c_1 < p_c_1_th
% Prediction
for i = 1:n_time 
    fprintf('i = %d / %d\n',i,n_time)
    t = time(i);
    temp_RUL = RUL(:,i); % Predicted RUL at t
    temp_weight = weight(:,t);
    [temp_RUL,I_RUL] = sort(temp_RUL);
    temp_t = temp_RUL + t;
    temp_weight = temp_weight(I_RUL);
    temp_R = 1-cumsum(temp_weight);
    for j = t:TCE_true+50
        if j < temp_t(1)
            R_3 = 1;
        else
            if j >= temp_t(end)
                R_3 = 0;
            else
                temp_index = find(temp_t>j,1,'first');
                R_3 = temp_R(temp_index-1);
            end
        end                          
        R = [R_1,R_2,R_3,R_4,R_5];
        p_c = CalPConsequence(R);
        p_c_1_pred = p_c(1);
        if p_c_1_pred < p_c_1_th
            TCE(i) = j-t;
            break;
        end
        if j == TCE_true+10
            TCE(i) = j-t;
        end
    end
end
% Plot RUL VS t
figure
plot(time,TCE,'sb-')
hold on
plot(30:TTF_true,TTF_true-[30:TTF_true],'-ok')
legend('Estimated RTCE','True RTCE');
xlabel('t')
ylabel('RTCE')
save('Result')