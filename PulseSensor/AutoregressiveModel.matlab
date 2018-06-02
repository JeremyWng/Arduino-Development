% AR自回归模型

function [A,Sgm2] = AR(x,P)

%  x   - 输入数据，列向量
%  P   - AR 阶数,标量
%  A   - AR 系数[a1 a2 ...aP]，行向量;
% Sgm2 - 方差
A  = zeros(1,P);  % 初始化参数矩阵，行向量
Rx = Rxx(x,P);    % 自相关估计，size = P+1

% p = 1 时候的Yule-Walker方程解,注意，MATLAB下标从1开始
A(1) = - Rx(1+1)/Rx(0+1);
Sgm2  = Rx(0+1)*(1 - A(1)^2);

% p = 1->P 时候的递推,注意，MATLAB下标从1开始
for p = 1:P-1
    k = 1:p;
    K = -(Rx((p+1)+1) + A(k)*Rx((p+1-k)+1))/Sgm2;
    Sgm2 = Sgm2 * (1 - K*K);
    
    A(k) = A(k) + K * A(p+1-k);
    A(p+1) = K;
end


% 所用自相关函数

function Rx = Rxx(x,P)
% x      - 自变量
% P      - 自相关数目
% Rxx[m] = sum(x[n]*x[n+m])
%         (n = 0->inf,m = 0->P)

N  = length(x);

sizex = size(x);
if sizex(1)==1
    % x 为行向量那么Rx也为行向量
    Rx = zeros(1,P+1); 
    
    for m = 0:P
        Rx(m+1) = x(m+1:N)*x(1:N-m)';
    end
    
elseif sizex(2)==1
    % x 为列向量那么Rx也为列向量
    Rx = zeros(P+1,1); 
    for m = 0:P
        Rx(m+1) = x(m+1:N)'*x(1:N-m);
    end
    
end