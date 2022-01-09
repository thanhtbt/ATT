%% DEMO: Adaptive TT for streaming 4-order tensors 
clear;clc; close all

run_path;

n_exp        = 5;
tt_dim       = [10 15 20 500];
tt_rank      = [2 3 5];
fac_noise    = 1e-3;
fac_time_varying = 1e-4;
epsilon      = fac_time_varying*ones(1,tt_dim(end));
epsilon(300) = 1; % Aim to create a significant change at t = 300

PER0  = zeros(1,tt_dim(end));

for ii = 1:n_exp
    fprintf(' run %d/%d \n',ii,n_exp);
    
    %% Data Generation
    [Xtrue,tt_core] = tt_generate_tensor(tt_dim,tt_rank,epsilon);
    X = Xtrue + fac_noise*randn(tt_dim);
  
    
    %% Algorithm
    OPTS_PER.Xtrue = Xtrue;
    OPTS_PER.lambda = 0.7;
    
    t_start = tic;
    [PER,~,~] = TT_FOA(X,tt_rank,OPTS_PER);
    toc(t_start);
    PER0 = PER0 + PER;
              
end

PER0 = PER0/n_exp;
 
%% Plot

makerSize = 11;
numbMarkers = 500;
LineWidth = 2;

color   = get(groot,'DefaultAxesColorOrder');
red_o   = [1,0,0];
blue_o  = [0, 0, 1];
mag_o   = [1 0 1];
gree_o  = [0, 0.5, 0];
black_o = [0.25, 0.25, 0.25];

blue_n  = color(1,:);
oran_n  = color(2,:);
yell_n  = color(3,:);
viol_n  = color(4,:);
gree_n  = color(5,:);
lblu_n  = color(6,:);
brow_n  = color(7,:);
lbrow_n = [0.5350    0.580    0.2840];

% %% SEP
T = tt_dim(end);
fig = figure;
hold on;
k = 2;


d1 = semilogy(1:k:T,PER0(1:k:end),...
    'linestyle','-','color',red_o,'LineWidth',LineWidth);
d11 = plot(1:100:T,PER0(1:100:end),...
    'marker','p','markersize',makerSize,...
    'linestyle','none','color',red_o,'LineWidth',LineWidth);
d12 = semilogy(1:1,PER0(1:1),...
    'marker','p','markersize',makerSize,...
    'linestyle','-','color',red_o,'LineWidth',LineWidth);

lgd = legend([d12],'\texttt{TT-FOA}');
lgd.FontSize = 18;
set(lgd, 'Interpreter', 'latex', 'Color', [0.95, 0.95, 0.95]);
%
xlabel('Time Index','interpreter','latex','FontSize',13,'FontName','Times New Roman');
ylabel('RE$(\mathcal{X}_{tr},\mathcal{X}_{es})$','interpreter','latex','FontSize',13,'FontName','Times New Roman');

set(fig, 'units', 'inches', 'position', [0.5 0.5 7.5 6.5]);
h=gca;
set(h,'FontSize',16,'XGrid','on','YGrid','on','GridLineStyle',':','MinorGridLineStyle',':','FontName','Times New Roman');
set(h,'FontSize', 24);
% axis([0 T 1*1e-4 1e1]);
grid on;
set(h, 'YScale', 'log','box','on')


