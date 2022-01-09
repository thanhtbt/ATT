function [PER,tt_core_es,Xre] = TT_FOA(X,tt_rank,OPTS_PER)

% Adaptive Algorithm for Tensor Train Decomposition
% Author     : LE Trung-Thanh
% Affiliation: University of Orleans, France
% Contact    : trung-thanh.le@univ-orleans.fr // letrungthanhtbt@gmail.com

%%
tt_dim = size(X);
N = length(tt_dim);
T = tt_dim(N);

Xre = zeros(tt_dim);
PER = zeros(1,T);


if isfield(OPTS_PER,'lambda'), % Forgetting factor
    lambda = OPTS_PER.lambda;
else lambda = 0.7;
end



Xtrue = OPTS_PER.Xtrue;
%% Initialization
G{1} = randn(tt_dim(1),tt_rank(1)); 
G{2} = randn(tt_rank(1),tt_dim(2),tt_rank(2)); 
G{3} = randn(tt_rank(2),tt_dim(3),tt_rank(3)); 
G{4} = [];

tt_core_es = cell(N,1);

S{1} = 100*eye(tt_rank(1));
S{2} = 100*eye(tt_rank(1)*tt_rank(2));
S{3} = 100*eye(tt_rank(2)*tt_rank(3));

for ii = 1 : T
    %% g4
    X_t       = X(:,:,:,ii);
    
    %% The last TT-Core G4
    G4_buffer = tt_product_tensors(tt_product_tensors(G{1},G{2}),G{3});
    H_t       = ten2mat(tensor(G4_buffer),4)';
    x_t       = X_t.data(:); 
   
    g4        = H_t  \ x_t;
    G{4}      = [G{4},g4];
    X_t_re    = H_t * g4;
    X_t_re    = tensor(reshape(X_t_re,[tt_dim(1:end-1)]));
    Delta_X_t = X_t - X_t_re;
    
    %% G1
    ER_Unfolding_1 = ten2mat(Delta_X_t,1);
    G_buffer{1}    = ttm(tensor(tt_product_tensors(G{2},G{3})),g4',4);
    W{1}           = ten2mat(G_buffer{1},1);
    S{1}           = lambda*S{1} + W{1} * W{1}';
    V{1}           = S{1} \ W{1};
    G{1}           = G{1} + ER_Unfolding_1 * V{1}';
    
    %% G2
    ER_Unfolding_2 = ten2mat(Delta_X_t,2);
    G_buffer{2}    = ttm(tensor(G{3}),g4',3); 
    W{2}           = kron(G_buffer{2}.data, G{1}');
    S{2}           = lambda*S{2} + W{2}* W{2}';
    V{2}           = S{2} \ W{2};
    G2_Unfolding_2 = ten2mat(tensor(G{2}),2) + ER_Unfolding_2 *  V{2}';
    G{2}           = mat2ten(G2_Unfolding_2,[tt_rank(1), tt_dim(2), tt_rank(2)],[2]);
    
    %% G3
    ER_Unfolding_3  = ten2mat(Delta_X_t,3);
    G_buffer{3}     = ten2mat(tensor(tt_product_tensors(G{1},G{2})),3);
    W{3}            = kron(g4,G_buffer{3});
    S{3}            = W{3} * W{3}' + lambda*S{3};
    V{3}            = S{3} \ W{3};
    G3_Unfolding_2  = ten2mat(tensor(G{3}),2) + ER_Unfolding_3 * V{3}';
    G{3}            = mat2ten(G3_Unfolding_2,[tt_rank(2), tt_dim(3), tt_rank(3)],[2]);
    
    
    %% Save
    tt_core_es{1,1} = G{1};
    tt_core_es{2,1} = G{2};
    tt_core_es{3,1} = G{3};
    tt_core_es{4,1} = g4';
    
    %% Performance estimation
    
    X_t_true  = Xtrue(:,:,:,ii);
    X_re      = tt_recover_tensor(tt_core_es);
    X_re      = X_re(:,:,:,1);
    PER(1,ii) = norm(X_t_true - X_re) / norm(X_t_true);
    Xre(:,:,:,ii) = X_re;
    
    
end
end
