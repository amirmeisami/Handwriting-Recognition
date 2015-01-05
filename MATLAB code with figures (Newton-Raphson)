close all; clc;
load mnist_49_3000.mat;
[d,n] = size(x);
%get the data set

%initialize the coefficient vector
beta_old=zeros(d+1,1);
beta_new=zeros(d+1,1);
%size of the training set
k=2000;
pred_prob=zeros(k,1);

%add a row of 1's to x matrix to create
x_new=[ones(1,n);x];
x_train=x_new(:,[1:k]);
%get the test data
x_test=x_new(:,[k+1:end]);
%construct the W matrix with diagonal elements equal to p*(1-p)
W=zeros(k,k);
%construct the hessian, (d+1)*(d+1) matrix
H=zeros(d+1,d+1);
%Jacobian, (d+1)*1
J=zeros(d+1,1);
y=y';

%stopping criteria
epsilon=0.005;
%lambda
lambda=10;

for i=1:n
    if y(i,1)==-1
        y(i,1)=0;
    end
end

%get the actual outcome for test and training
y_train=y([1:k],1);
y_test=y([k+1:end],1);
iteration=100;
opt_value=zeros(iteration,1);
log_likelihood=zeros(iteration,1);

%start the iterations
for k=1:iteration-1
    for i=1:2000
    pred_prob(i,1)=1/(1+exp(-beta_old'*x_train(:,i)));
    W(i,i)=pred_prob(i,1)*(1-pred_prob(i,1));
    J=J+x_train(:,i)*(y_train(i,1)-pred_prob(i,1)); 
    end
    log_likelihood(k,1)=log_likelihood(k,1)+(y_train(i,1)*beta_old'*x_train(:,i)-log(1+exp(beta_old'*x_train(:,i))));
    opt_value(k,1)=-log_likelihood(k,1)+lambda*norm(beta_old)*norm(beta_old);
    %Calculate the Hessian
    H=(x_train*W*x_train')+2*lambda*eye(d+1);
    beta_new=beta_old-inv(H)*(-J+2*lambda*beta_old);
    for i=1:2000
    %Calculate the objective value at optimum
    log_likelihood(k+1,1)=log_likelihood(k+1,1)+(y_train(i,1)*beta_new'*x_train(:,i)-log(1+exp(beta_new'*x_train(:,i))));
    opt_value(k+1,1)=-log_likelihood(k+1,1)+lambda*norm(beta_new)*norm(beta_new);
    end
    %if opt_value(k,1)-opt_value(k+1,1)<0 break;
    if norm(beta_new-beta_old) <epsilon break;
    end
    beta_old=beta_new;
end

%Label the objects in test data
y_est=zeros(1000,1);
y_sorted=zeros(1000,1);
pred_prob_test=zeros(1000,1);
test_error=zeros(1000,1);
total_error=0;

for i=1:1000
    if (beta_new'*x_test(:,i)) >=0 
    y_est(i,1)=1;
    else y_est(i,1)=0;
    end
end

B=[];
%Total error is 58
for i=1:1000
    pred_prob_test(i,1)=1/(1+exp(-beta_new'*x_test(:,i)));
    if abs(y_est(i,1)-y_test(i,1)==0)
        test_error(i,1)=0;
    else test_error(i,1)=1;
        total_error=total_error+1;
        B(:,total_error)=[repmat(x_test(:,i),1,1);pred_prob_test(i,1)];
    end;
end

total_error

%find the objects with highest probabilities among the misclassified
%objects
A=-B(786,:);
index_max=[];
for i=1:20
     [max,index] = min(A);
     index_max(i)=index;
     A(index)=0;
end
  

figure;
for i = 1:20
    subplot(4,5,i)
    imagesc(reshape(B(2:d+1,index_max(i)),[sqrt(d),sqrt(d)])');
end


