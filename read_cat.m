function [X_test X yd_test yd X_my yd_my] = read_cat()
train_set_x_orig = hdf5read('train_catvnoncat.h5', 'train_set_x');
train_set_y_orig = hdf5read('train_catvnoncat.h5', 'train_set_y');
test_set_x_orig = hdf5read('test_catvnoncat.h5', 'test_set_x');
test_set_y_orig = hdf5read('test_catvnoncat.h5', 'test_set_y');
classes = hdf5read('test_catvnoncat.h5', 'list_classes');
m_train = size(train_set_x_orig, 4);
m_test = size(test_set_x_orig, 4);
num_px = size(train_set_x_orig, 2);
num_channel = size(train_set_x_orig, 1);
fprintf('Dataset dimensions: \n');
fprintf('Number of training examples: m_train = %d \n', m_train);
fprintf('Number of testing examples: m_test  = %d \n', m_test);
fprintf('Height/Width of each image: num_px  = %d \n', num_px);
fprintf('Each image is of size:       (%d, %d) \n', num_px, num_px);
fprintf('train_set_x shape: (%s) \n', num2str(size(train_set_x_orig)));
fprintf('train_set_y shape: (%s) \n', num2str(size(train_set_y_orig)));

% If you want to display an image, uncomment the following code
X = [];
for i = 1:size(train_set_x_orig,4)
img = train_set_x_orig(:, :, :, i);
img = permute(img, [2 3 1]);
img_x = img(:)';
X = [X;img_x];
end
X_test = [];
for i = 1:size(test_set_x_orig,4)
img = test_set_x_orig(:, :, :, i);
img = permute(img, [2 3 1]);
img_x_test = img(:)';
X_test = [X_test;img_x_test];
end
yd = train_set_y_orig;
yd_test =test_set_y_orig;

%% ∂¡»°test_my
X_my = [];
for i =1:13
    filename = ['mao', num2str(i), '.jpg'];
    Imag = imread(filename);
     img_1 = imrotate(Imag, 90);
    img_x = img_1(:)';
    X_my = [X_my;img_x];
end
yd_my = ones(13,1);




% load vv
% for i = 1:size(vv,1)
% 
% img = test_set_x_orig(:, :, :,vv(i));
% img = permute(img, [2 3 1]);
%  filename = ['fail', num2str(i), '.jpg'];
% imwrite(img ,filename)
% end






