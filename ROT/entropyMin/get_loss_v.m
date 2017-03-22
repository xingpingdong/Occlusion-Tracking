function loss = get_loss_v(response,lambda_l,mean_v,var_v,v_predict,lambda_v)

p = response(:);
p = p - min(p)+1e-6;
p = p/sum(p);
logP = log(p);
L = max(logP);
H = sum(p.*logP);

d1 = mean_v*mean_v';
d2 = v_predict*v_predict';
d12 = mean_v*v_predict';

loss_v = (d12 - sqrt(d1*d2))^2/(var_v*d1*d2);
loss = -L+lambda_l*H+lambda_v*loss_v;
end