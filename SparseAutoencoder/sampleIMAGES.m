function patches = sampleIMAGES()
% sampleIMAGES
% Returns 10000 patches for training

load IMAGES;    % load images from disk 

patchsize = 8;  % we'll use 8x8 patches 
numpatches = 10000;

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize*patchsize, numpatches); %patches is a 64 x 10,000 dimensional matrix

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data 
%  from IMAGES.  
%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1


%this code is for displaying an image from the dataset
%h = imshow(IMAGES(:,:,10), [-1 1]);
%
%axis image off
%
%drawnow;
%
%warning on all
%---------------



for i = 1:10000,
  %randomly pick 1 of the 10 images
  img = ceil(rand(1) * 10); %generates random number between 1 and 10 
  %pick a random 8 x 8 block of pixels from the random image
  A = 1; B = 504;
  start = ceil(rand(1) * 501);
  end_point = start + (patchsize - 1);
  matrix = IMAGES(start:end_point, start:end_point, img);
  %convert the randomly selected matrix of pixels into a vector 
  patches(:, i) = matrix(:);
end






%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
patches = normalizeData(patches);

end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end
