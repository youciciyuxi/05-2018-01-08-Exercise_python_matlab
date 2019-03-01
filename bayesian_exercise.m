%%
%Bayesian code
%Author by Yu

%% 设置的理论曲线
%data produce 
rseed	= 1;
rand('state',rseed);
randn('state',rseed);

Xn = 0:0.1:10;
Tn = sin(Xn);
plot(Xn, Tn, 'g', 'LineWidth', 2);
hold on;

noiseToSignal = 0.1;            % 引入的噪声方差
basisWidth = 0.05;              % 核函数的带宽
pSparse = 0.9;

N = max(Xn) - min(Xn);
X = Xn./N;
C = X; 
Basis = exp(-distSquared(X,C)/(basisWidth^2));         % 由核函数得到的基函数
[N M] = size(Basis);
w = randn(M, 1)*100/(M*(1 - pSparse));
sparse = rand(M, 1)<pSparse;
w(sparse) = 0;

Y = Basis * w;
noise = std(Y) * noiseToSignal;
Yn = Y + noise;                                   % 作为已知的函数目标值
plot(Xn, Yn, 'r:', 'LineWidth', 2);

%% 设置添加噪声后的曲线
% Sparse bayes inference section
init_alpha_max = 1000;                   % 设置阿尔法的初始值，越大越好，说明越稀疏
init_alpha_min = -1000;
beta = 1/noiseToSignal^2;

Yntemp = Yn;
proj = Basis' * Yntemp;
[foo used] = max(abs(proj));
Phi = Basis(:, used);                     % 相关向量

% 关于阿尔法的初始化
si = diag(Phi' * Phi) * beta;
qi = (Phi' * Yn) * beta;
alpha = si.^2./(qi.^2 - si);
alpha(alpha<0) = init_alpha_max;
beta = 1/noiseToSignal^2;

% 计算基函数和相关向量的乘积
Basis_Phi = Basis' * Phi;
Basis_Yn = Basis' * Yn;

%% 计算后验概率 Sigma,Mu
U = chol(Phi' * Phi * beta + diag(alpha));
Ui = inv(U);
Sigma = Ui * Ui';
Mu = (Sigma * (Phi' * Yn)) * beta;

gradient = 1e-6;
step_max = 25;
step_min = 1/(2^8);

A = diag(alpha);
Basis_Mu = Basis * Mu;
regulariser = (alpha' * (Mu.^2))/2;
newError = regulariser;

errorlog = zeros(step_max, 1);
for iteration = 1:step_max
    errorlog(iteration) = newError;
    error = Yn-y;
    g = Basis'*error - Alpha.*Mu;
    Basis_B = Basis .* (Basis * ones(1,M));
    H = (Basis_B' * Basis + A);               % H表示Hessian矩阵计算
    [U_new, pdErr] = chol(H);
    DeltaMu = U\(U' \ g);
    step = 1;
    while step > step_min
        Mu_new = Mu + step*DeltaMu;
        Basis_Mu = Basis * Mu_new;
        regulariser = (alpha' * (Mu_new.^2))/2;
        newError = regulariser;
        if newError >= errorlog
            step = step/2;
        else
            Mu = Mu_new;
            step = 0;
        end
    end
end
Ui_new = inv(U_new)
Sigma_new = Ui_new * Ui_new';

%% 计算和稀疏贝叶斯相关的状态量
% 计算对数形式边缘最大似然估计
logdet = sum(log(diag(U_new)))
logML = -(Mu.^2)'*alpha/2 + sum(log(alpha))/2 - logdet;
diagc= sum(Ui_new.^2,2);
Gamma = 1-alpha.*diagc;

% 计算Q和S的值
betaBasis_Phi = beta*Basis_Phi;
s_in = beta - sum((betaBasis_Phi*Ui_new).^2,2);
q_in = beta*(Basis_Yn - Basis_Phi*Mu);
% 当Q和S不在基函数内时
s_out(used) = (alpha .* s_in(used))./(alpha - s_in(used));
q_out(used) = (alpha .* q_in(used))./(alpha - s_in(used));
% 这样就可以计算用于判别阿尔法表达式分母的正负了
theta = q_out.*q_out - s_out;

addcount = 0;
deletecount = 0;
updatacount = 0;
% 阿尔法为正且有限，重新估计；为正无穷，添加核函数到基函数再重新更新；为负且无限，则删除并置为无限
% 重新估计
ZeroFactor = 1e-12;
usedtheta = theta(used);
iftheta = usedtheta > ZeroFactor;
index = used(iftheta);
newalpha = s_out(index).^2./theta(index);
delta = (1./newalpha - 1./alpha(index));    % 临时向量
% 删除
iftheta = ~iftheta;
index = used(iftheta);
delete = ~




































