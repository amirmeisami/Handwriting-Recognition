load mnist_49_3000

% Data preparation 
n_x=x';
x_new=[ones(3000,1),n_x];
train_data=x_new(1:2000,:);
test_data=x_new(2001:3000,:);
y_trn=y';
for i=1:3000
    if y_trn(i,1)==-1
        y_trn(i,1)=0;
    end
end
y_train=y_trn(1:2000,:);
y_test=y_trn(2001:3000,:);
size_train=size(train_data);
m=size_train(1);
n=size_train(2);
theta=zeros(n,1);
lambda=10;
alpha=0.1;
mask = ones(size(theta));
mask(1) = 0;

% Cost function initialization
J(1)=inf;
J(2)=(1./m)*(-y_train'*log(1./(1+exp(-train_data*theta))))-(1-y_train')*log(1-1./(1+exp(-train_data*theta)))+(lambda./(2*m))*(theta'*theta-theta(1)^2 );

% Gradient-descent method
i=2;
hold on;
while J(i-1)-J(i)>0.01
    JJ=(1./m) * train_data' * (1./(1+exp(-train_data*theta)) - y_train)+lambda * (theta .* mask)./m;
    theta=theta-alpha*JJ;
    i=i+1;
    J(i)=(1./m)*(-y_train'*log(1./(1+exp(-train_data*theta))))-(1-y_train')*log(1-1./(1+exp(-train_data*theta)))+(lambda./(2*m))*(theta'*theta-theta(1)^2 );
    plot(i,J(i))
end
hold off;
% prediction probabilities 
pred_prob=1./(1+exp(-test_data*theta));

% predictions
pred=[];
for i=1:1000
    if pred_prob(i)<0.5
        pred(i)=0;
    else pred(i)=1;
    end  
end
% # of test errors
er_count=0;
 for i=1:1000
    if pred(i)~=y_test(i)
        er_count=er_count+1;
    end
end
er_count
