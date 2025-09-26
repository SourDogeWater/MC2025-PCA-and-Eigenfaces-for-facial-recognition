clc;
clear;

disp("Program has been initiated...");

filenames = dir('face/*.bmp');

numTestFiles = 28;

ACTUALnumTrainingFiles = length(filenames);

numTrainingFiles = ACTUALnumTrainingFiles - numTestFiles;


img_x_dim = 256;
img_y_dim = 256;
dimension = img_x_dim * img_y_dim;

D = zeros(dimension, numTrainingFiles);
meanFace1 = zeros(dimension,1);

for i = 1:numTrainingFiles
    
    currentFilePath = strcat('face/', filenames(i).name);

    %disp(currentFilePath)

    img = imread(currentFilePath);

    
    D(:, i) = double(img(:)); % : <== this turns matrix into single column

    meanFace1 = D(:, i) + meanFace1;
   

end

meanFace1 = meanFace1./numTrainingFiles; % ./ means element by element



for i = 1:numTrainingFiles
    centeredFaces(:, i) = D(:, i) - meanFace1;
end

%centeredD = D - meanFace1; (This does the same action as for loop above)


%covarM = cov(trueMean);


covarMatrix = (centeredFaces' * centeredFaces)/(numTrainingFiles);

[eig_vect,eig_value] = eig(covarMatrix);

eig_vals_diag = diag(eig_value);
[sorted_vals, index] = sort(eig_vals_diag, 'descend');

eig_vals_sorted = diag(sorted_vals);       
eig_vecs_sorted = eig_vect(:, index); 


%Project eigenvectors back into orginal high-dimensional space
sorted_eigenfaces = centeredFaces * eig_vecs_sorted; 


%Normalize eigenfaces
normalizedEF = zeros(dimension , numTrainingFiles);

for i = 1:numTrainingFiles
    normalizedEF(: , i) = sorted_eigenfaces(: , i) / norm(sorted_eigenfaces(: , i));
end

% percent calculations
totVar = sum(eig_vals_sorted);
totVar2 = sum(eig_vals_diag);
csum = 0;

for i = 1:ACTUALnumTrainingFiles 
    csum = csum + sorted_vals(i);
    tV = csum / totVar2;
    if tV > 0.99
        k = i;
        break
    end
end

percent_variance = (eig_vals_sorted/totVar) * 100;

figure;
for i = 1:16
    subplot(4,4,i);
    
    eigenface = sorted_eigenfaces(:, i);
    eigenface_img = reshape(eigenface, img_x_dim, img_y_dim);
    eigenface_img = mat2gray(eigenface_img);

    imshow(eigenface_img);
    title({['Eigenface ', num2str(i)], [num2str(percent_variance(i), '%.2f'), '% variance']}, 'FontSize', 8);
end


test_start_index = 150;
num_test_images_to_display = 28;  % total test images to loop through

% Initialize error array before loop
reconstruction_errors = zeros(num_test_images_to_display, 1);

for i = 1:num_test_images_to_display
    test_img_index = test_start_index + i - 1;
    test_img_path = fullfile('face', filenames(test_img_index).name);
    test_img = imread(test_img_path);
    test_col = double(reshape(test_img, dimension, 1));

    centered_test_col = test_col - meanFace1;
    weights = normalizedEF' * centered_test_col;

    reconstructed_col = meanFace1 + normalizedEF(:, 1:k) * weights(1:k);
    reconstructed_img = reshape(reconstructed_col, img_x_dim, img_y_dim);
    original_img = reshape(test_col, img_x_dim, img_y_dim);

    % Compute relative error %
    reconstruction_errors(i) = 100 * norm(test_col - reconstructed_col) / norm(test_col);

    % Open a new figure every 4 images
    if mod(i-1, 4) == 0
        figure;
    end

    subplot_idx = mod(i-1, 4) + 1;

    subplot(2, 4, subplot_idx);  % Original
    imshow(original_img, []);
    title(['Original ', num2str(test_img_index)]);

    subplot(2, 4, subplot_idx + 4);  % Reconstructed
    imshow(reconstructed_img, []);
    title(['Reconstructed ', num2str(test_img_index)]);
end

% === Final Figure: Reconstruction Error Bar Plot ===
% figure;
% bar(reconstruction_errors, 'FaceColor', [0.2 0.6 0.8]);
% xlabel('Test Image Index');
% ylabel('Relative Reconstruction Error (%)');
% title('Reconstruction Error for Each Test Image');
% grid on;



% === ERROR VS. NUMBER OF EIGENFACES USED — SPECIFIC TEST IMAGE ===
test_image_index = 151;  % <<== CHANGE THIS TO THE TEST IMAGE YOU WANT

% Load the specific test image
test_img_path = fullfile('face', filenames(test_image_index).name);
test_img = imread(test_img_path);
test_col = double(reshape(test_img, dimension, 1));
centered_test_col = test_col - meanFace1;

% Compute reconstruction error vs number of eigenfaces
error_vs_var_single = zeros(numTrainingFiles, 1);

for k = 1:numTrainingFiles
    weights_k = normalizedEF(:, 1:k)' * centered_test_col;
    reconstruction_k = meanFace1 + normalizedEF(:, 1:k) * weights_k;

    error_vs_var_single(k) = 100 * norm(reconstruction_k - test_col) / norm(test_col);
end

% Plot error vs. number of eigenfaces for this specific test image
figure;
plot(1:numTrainingFiles, error_vs_var_single, 'o-', 'LineWidth', 1.5);
xlabel('Number of Eigenfaces Used');
ylabel('Reconstruction Error (%)');
title(['Reconstruction Error vs. # Eigenfaces (Test Image ', num2str(test_image_index), ')']);
grid on;


% === ERROR VS. NUMBER OF EIGENFACES USED — AVERAGED ACROSS ALL TEST IMAGES ===

disp("Graph time...");

avg_error_vs_var = zeros(numTrainingFiles, 1);  % to hold summed error for each k

for testIdx = test_start_index : test_start_index + num_test_images_to_display - 1
    % Load and preprocess test image
    test_img_path = fullfile('face', filenames(testIdx).name);
    test_img = imread(test_img_path);
    test_col = double(reshape(test_img, dimension, 1));
    centered_test_col = test_col - meanFace1;

    for k = 1:numTrainingFiles
        weights_k = normalizedEF(:, 1:k)' * centered_test_col;
        reconstruction_k = meanFace1 + normalizedEF(:, 1:k) * weights_k;

        % Accumulate reconstruction error
        avg_error_vs_var(k) = avg_error_vs_var(k) + 100 * norm(reconstruction_k - test_col) / norm(test_col);
    end
end

% Average over the number of test images
avg_error_vs_var = avg_error_vs_var / num_test_images_to_display;

% Plot average error vs. number of eigenfaces
disp("Plotting average reconstruction error...");
figure;
plot(1:numTrainingFiles, avg_error_vs_var, 's-', 'LineWidth', 1.5, 'Color', [0.2 0.4 0.8]);
xlabel('Number of Eigenfaces Used');
ylabel('Average Reconstruction Error (%)');
title('Average Reconstruction Error vs. # of Eigenfaces (All Test Images)');
grid on;

