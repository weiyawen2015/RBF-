clear all 
clc
close all 
%% 读取数据
[X_test X yd_test yd X_my yd_my] = read_cat();
train_data = double(X')/255;
train_classlabel = logical(yd');
num_train = size(train_data ,2);

test_data = double(X_test')/255;
test_classlabel = logical(yd_test');
yd_test = double(yd_test);
num_test = size(test_data,2);

my_data = double(X_my')/255;
my_classlabel = logical(yd_my');
yd_my = double(yd_my);
num_my = size(my_data,2);
%% 计算随机聚类中心和G
M = 209;
[center, sigma_random] = get_center(train_data, M);
sigma_random = 0.08*sigma_random;%% 0.05不够好 0.08不改了  
G = G_matrix_cal(train_data, M,center,sigma_random);
%% train
alfa = 1.2;
eita = 0;
 miu =1;%S形函数参数
num_label = size(train_data,2);
yd = double(yd);
num_center = M;
w0 = -10 + (10+10)*rand(1,num_center)';
w_old = w0;
derta_w = 1;
LL = [];
L_old = 1;
dL = 1;
L1 = 1;
Dw_k_1 = zeros(num_center,1);
count = 1;
 while(dL>1e-6&&count<100000)
   %%% --loss function---------------
    yt = G*w_old; 
    for j = 1:209
       yt_seigma(j,1) = 1./(1+exp(-miu*yt(j,1)));
    end
   L_new = - sum(yd.*log(yt_seigma+1e-10*ones(num_label,1))+(ones(num_label,1)-yd).*log(ones(num_label,1)-yt_seigma+1e-10*ones(num_label,1)))/num_label;
%-更新w---动量项梯度下降---------------------
      DL_yt =(miu/num_label)*(yt_seigma - yd);
      Dw = (DL_yt')*G;%w1 w2 w3
      Dw_k = Dw';
      Dw = (1-eita)*Dw_k + eita*Dw_k_1;
      Dw_k_1 = Dw_k;
      w_new = w_old - alfa*Dw;     
      w_old = w_new;
     dL = abs(L_new - L_old);
     L_old = L_new ;
  count = count + 1;
  L(count,1) = L_new;
  L_new
 end
 %% 学习曲线
figure(1)
 plot(2:count,L(2:size(L,1))');
 xlabel('\ititeration','fontsize',20,'FontName', 'Times New Roman')    
%     '\itE\rm(\it\rmx)'
ylabel('\itLoss','fontsize',20,'FontName', 'Times New Roman')

      title(['learning curve ' ],'fontsize',20,'FontName', 'Times New Roman')
      
 set(gca,'fontsize',16,'FontName', 'Times New Roman')
 w_plus =  w_new;
% %%%训练结果%%%---------------------------------------------------------
   yt = G*w_plus;
   yt_seigma = 1./(1+exp(-miu*yt));%保证了分母不为0
    L_new = - sum(yd.*log(yt_seigma+1e-10*ones(num_label,1))+(ones(num_label,1)-yd).*log(ones(num_label,1)-yt_seigma+1e-10*ones(num_label,1)))/num_label;
 [  yt_seigma train_classlabel']
 AA_train = abs( yt_seigma - train_classlabel');
 AA_train( AA_train>0.5)=1;
 AA_train( AA_train<0.5)=0;
 uu_train=find( AA_train==0);
 vv_train=find( AA_train==1);
 size(uu_train,1)/num_test;
 L_new
 %% test
G_test = G_matrix_cal(test_data,M,center,sigma_random);
  yt_test = G_test*w_plus ;
    for j = 1:num_test
       yt_seigma_test(j,1) = 1./(1+exp(-miu*yt_test(j,1)));
    end
 L_test = - sum(yd_test.*log(yt_seigma_test+1e-10*ones(num_test,1))+(ones(num_test,1)-yd_test).*log(ones(num_test,1)- yt_seigma_test+1e-10*ones(num_test,1)))/num_test;
 [ yt_seigma_test test_classlabel']

u=find(test_classlabel==1)';
 AA = abs(yt_seigma_test- test_classlabel');
 AA(AA>0.5)=1;
 AA(AA<0.5)=0;
 uu=find(AA==0);
 vv=find(AA==1);
 save vv vv
 size(uu,1)/num_test
%% test my figure
G_my= G_matrix_cal(my_data, M,center, sigma_random);
  yt_my= G_my*w_plus ;
    for j = 1:num_my
       yt_seigma_my(j,1) = 1./(1+exp(-miu*yt_my(j,1)));
    end
 L_my= - sum(yd_my.*log(yt_seigma_my+1e-10*ones(num_my,1))+(ones(num_my,1)-yd_my).*log(ones(num_my,1)- yt_seigma_my+1e-10*ones(num_my,1)))/num_my;
 [ yt_seigma_my my_classlabel']
uu_my=find( yt_seigma_my>0.5);
disp('训练Loss')
L_new
disp('迭代次数')
count-1
disp('训练准确率')
 size(uu_train,1)/num_train
disp('测试准确率')
size(uu,1)/num_test
disp('13张准确率')
size(uu_my,1)/num_my









