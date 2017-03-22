function loss = get_loss(response,lambda_l)

p = response(:);
p = p - min(p)+1e-6;
p = p/sum(p);
logP = log(p);
L = max(logP);
% L = log( max(p));
H = sum(p.*logP);
loss = -L-lambda_l*H;
end