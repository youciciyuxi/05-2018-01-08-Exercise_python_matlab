%%
%Bayesian code
%Author by Yu

%% ���õ���������
%data produce 
rseed	= 1;
rand('state',rseed);
randn('state',rseed);

Xn = 0:0.1:10;
Tn = sin(Xn);
plot(Xn, Tn, 'g', 'LineWidth', 2);
hold on;

noiseToSignal = 0.1;            % �������������
basisWidth = 0.05;              % �˺����Ĵ���
pSparse = 0.9;

N = max(Xn) - min(Xn);
X = Xn./N;
C = X; 
Basis = exp(-distSquared(X,C)/(basisWidth^2));         % �ɺ˺����õ��Ļ�����
[N M] = size(Basis);
w = randn(M, 1)*100/(M*(1 - pSparse));
sparse = rand(M, 1)<pSparse;
w(sparse) = 0;

Y = Basis * w;
noise = std(Y) * noiseToSignal;
Yn = Y + noise;                                   % ��Ϊ��֪�ĺ���Ŀ��ֵ
plot(Xn, Yn, 'r:', 'LineWidth', 2);

%% ������������������
% Sparse bayes inference section
init_alpha_max = 1000;                   % ���ð������ĳ�ʼֵ��Խ��Խ�ã�˵��Խϡ��
init_alpha_min = -1000;
beta = 1/noiseToSignal^2;

Yntemp = Yn;
proj = Basis' * Yntemp;
[foo used] = max(abs(proj));
Phi = Basis(:, used);                     % �������

% ���ڰ������ĳ�ʼ��
si = diag(Phi' * Phi) * beta;
qi = (Phi' * Yn) * beta;
alpha = si.^2./(qi.^2 - si);
alpha(alpha<0) = init_alpha_max;
beta = 1/noiseToSignal^2;

% �������������������ĳ˻�
Basis_Phi = Basis' * Phi;
Basis_Yn = Basis' * Yn;

%% ���������� Sigma,Mu
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
    H = (Basis_B' * Basis + A);               % H��ʾHessian�������
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

%% �����ϡ�豴Ҷ˹��ص�״̬��
% ���������ʽ��Ե�����Ȼ����
logdet = sum(log(diag(U_new)))
logML = -(Mu.^2)'*alpha/2 + sum(log(alpha))/2 - logdet;
diagc= sum(Ui_new.^2,2);
Gamma = 1-alpha.*diagc;

% ����Q��S��ֵ
betaBasis_Phi = beta*Basis_Phi;
s_in = beta - sum((betaBasis_Phi*Ui_new).^2,2);
q_in = beta*(Basis_Yn - Basis_Phi*Mu);
% ��Q��S���ڻ�������ʱ
s_out(used) = (alpha .* s_in(used))./(alpha - s_in(used));
q_out(used) = (alpha .* q_in(used))./(alpha - s_in(used));
% �����Ϳ��Լ��������б𰢶������ʽ��ĸ��������
theta = q_out.*q_out - s_out;

addcount = 0;
deletecount = 0;
updatacount = 0;
% ������Ϊ�������ޣ����¹��ƣ�Ϊ�������Ӻ˺����������������¸��£�Ϊ�������ޣ���ɾ������Ϊ����
% ���¹���
ZeroFactor = 1e-12;
usedtheta = theta(used);
iftheta = usedtheta > ZeroFactor;
index = used(iftheta);
newalpha = s_out(index).^2./theta(index);
delta = (1./newalpha - 1./alpha(index));    % ��ʱ����
% ɾ��
iftheta = ~iftheta;
index = used(iftheta);
delete = ~




































