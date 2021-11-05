function [center, sigma] = get_center(X, M)
nb_samples = length(X(1,:));
permutation = randperm(nb_samples); 
center_numbers = permutation(1:M); 
center = X(:,center_numbers);
distance = zeros(M,M);
for j = 1:M
    for i = 1:j
        distance(i,j) = sum((center(:,i) - center(:,j)) .* (center(:,i) - center(:,j)));
    end
end
dmax = max(max(distance));
sigma = dmax / sqrt(2*M);
end

