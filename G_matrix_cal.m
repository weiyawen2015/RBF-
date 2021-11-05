function phi = G_matrix_cal(X,M,mu,sigma)
nb_samples = length(X(1,:));
phi = zeros(nb_samples,M);
for i = 1:nb_samples
    for j = 1:M
       phi(i,j) = exp( - norm(X(:,i) - mu(:,j))^2 /(2*sigma^2));  
    end
end
end