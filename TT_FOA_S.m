function [PER,tt_core_es,Xre] = TT_FOA_Stochastic(X,tt_rank,OPTS_PER)

% Adaptive Algorithm for Tensor Train Decomposition
% Author     : LE Trung-Thanh
% Affiliation: University of Orleans, France
% Contact    : trung-thanh.le@univ-orleans.fr // letrungthanhtbt@gmail.com
% Date       : 4/2/2019

%% 
tt_dim = size(X);
N = length(tt_dim);
I = tt_dim(1);
J = tt_dim(2);
K = tt_dim(3);
T = tt_dim(N);


Xtrue = OPTS_PER.Xtrue;
if isfield(OPTS_PER,'lambda'), % Forgetting factor
    lambda = OPTS_PER.lambda;
else lambda = 0.7;
end
rho = 0.01;


Xre   = zeros(tt_dim);
%% Initialization
G1 = randn(I,tt_rank(1));
G2 = randn(tt_rank(1),J,tt_rank(2));
G3 = randn(tt_rank(2),K,tt_rank(3));
G4 = [];
tt_core_es = cell(N,1);

for ii = 1 : T
    %% g4
    X_t       = X(:,:,:,ii);
    G12_t     = tt_product_tensors(G1,G2);
    G134      = tt_product_tensors(G12_t,G3);
    H         = ten2mat(tensor(G134),4)';
    g4        = H \ X_t.data(:);
    G4        = [G4,g4];
    
    X_re = H * g4;
    X_re = reshape(X_re,[I J K]);
    X_re = tensor(X_re);
    ER   = X_t - X_re;
    
    %% G1
    ER_Unfolding_1 = ten2mat(ER,1);
    
    G23       = tt_product_tensors(G2,G3);
    G234      = ttm(tensor(G23),g4',4);
    G234      = G234(:,:,:,1);
    G234_mat  = ten2mat(tensor(G234),1);
    
    HG  = G234_mat * G234_mat' + rho*eye(tt_rank(1));
    G_inv = HG \ G234_mat;

    G1  = G1 + ER_Unfolding_1 * G_inv';
    
    %% G2
    ER_Unfolding_2 = ten2mat(ER,2);
    
    G34  = ttm(tensor(G3),g4',3); G34  = G34(:,:,1);
    CB   = kron(G34.data, G1');
    HCB  = CB * CB' + rho*eye(tt_rank(1)*tt_rank(2));
    CB_inv = HCB \ CB;

    G2_Unfolding_2 = ten2mat(tensor(G2),2);
    G2_Unfolding_2 = G2_Unfolding_2 + ER_Unfolding_2 * CB_inv';
    G2             = mat2ten(G2_Unfolding_2,[tt_rank(1), J, tt_rank(2)],[2]);
    
    %% G3
    ER_Unfolding_3   = ten2mat(ER,3);
    
    G12_Unfolding_3 = ten2mat(tensor(G12_t),3);
    CA       = kron(g4,G12_Unfolding_3);
    HCA      = CA * CA' + rho*eye(tt_rank(2)*tt_rank(3));
    CA_inv   = HCA \ CA;

    G3_Unfolding_2 = ten2mat(tensor(G3),2);
    G3_Unfolding_2 = G3_Unfolding_2 + ER_Unfolding_3 * CA_inv';
    G3             = mat2ten(G3_Unfolding_2,[tt_rank(2), K, tt_rank(3)],[2]);
        
    %% Save
    tt_core_es{1,1} = G1;
    tt_core_es{2,1} = G2;
    tt_core_es{3,1} = G3;
    tt_core_es{4,1} = g4';
    
    %% Performance estimation

    X_t_true  = Xtrue(:,:,:,ii);
    X_re      = tt_recover_tensor(tt_core_es);
    X_re      = X_re(:,:,:,1);
    ER2       = X_t_true - X_re;
    PER(1,ii) = norm(ER2) / norm(X_t_true);
    Xre(:,:,:,ii) = X_re;
 
    
end
end
