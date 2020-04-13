clear all, close all,

filenames{1,1} = '3096_color.jpg';
filenames{1,2} = '42049_color.jpg';     

Kvalues = [2]; % desired numbers of clusters
K = 10;
regWeight = 1e-10;
delta = 1e-2;

for imageCounter = 1:2 %size(filenames,2)
    imdata = imread(filenames{1,imageCounter}); 
    figure(1), subplot(size(filenames,2),length(Kvalues)+1,(imageCounter-1)*(length(Kvalues)+1)+1), imshow(imdata);
    if length(size(imdata))==3 % color image with RGB color values
        [R,C,D] = size(imdata); N = R*C; imdata = double(imdata);
        rowIndices = [1:R]'*ones(1,C); colIndices = ones(R,1)*[1:C];
        features = [rowIndices(:)';colIndices(:)']; % initialize with row and column indices
        for d = 1:D
            color = imdata(:,:,d); % pick one color at a time
            features = [features;color(:)'];
        end
        minf = min(features,[],2); maxf = max(features,[],2);
        ranges = maxf-minf;
        x = diag(ranges.^(-1))*(features-repmat(minf,1,N)); % each feature normalized to the unit interval [0,1]
    end
    d = size(x,1); % feature dimensionality
   
    X1 = x;
    gm = fitgmdist(X1',2,'CovarianceType','full');
    alpha = gm.ComponentProportion;
    mu = (gm.mu)';
    sigma = gm.Sigma;
        
    %%% Use MLE to segment image into two parts
    post = posterior(gm,X1');
    [~,label] = max(post',[],1);
    
    subplot(size(filenames,2),length(Kvalues)+1,imageCounter)
    imagesc(reshape(label,R,C))
    if imageCounter == 1
        title('Segmented Plane Picture (2 components)'), xlabel('xPixel index'), ylabel('yPixel index');
    else if imageCounter == 2
        title('Segmented Bird Picture (2 components)'), xlabel('xPixel index'), ylabel('yPixel index');
        end
    end    
    % Start 10-fold cross validation
  dummy = ceil(linspace(0,N,K+1));
    for k = 1:K
        indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
    end
    
    for M = 1:15
    %[M,N],
    
    % K-fold cross validation
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        xValidate = x(indValidate); % Using folk k as validation set
        if k == 1
            indTrain = [indPartitionLimits(k,2)+1:N];
        elseif k == K
            indTrain = [1:indPartitionLimits(k,1)-1];
        else
            indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
        end
        xTrain = x(indTrain); % using all other folds as training set
        Ntrain = length(indTrain); Nvalidate = length(indValidate);
        for m = 1:M
            %find a best fit GMM for M components
            GMModel = fitgmdist(xTrain',M,'RegularizationValue',regWeight);
            alpha = GMModel.ComponentProportion;
            mu = (GMModel.mu)';
            sigma = GMModel.Sigma;
            perf_array(k,m) = sum(log(evalGMM(xValidate,alpha,mu,sigma)));
        end
        avg_perf = sum(perf_array)/K;
        [~,best_GMM] = max(avg_perf); % find the number of components for best_GMM
    end
    end
    % Generate GMM with best_GMM number of components
    gm2 = fitgmdist(X1',best_GMM,'CovarianceType','full');
    alpha = gm2.ComponentProportion;
    mu = (gm2.mu)';
    sigma = gm2.Sigma;
        
    %%% Use MLE to segment image into two parts
    post = posterior(gm2,X1');
    [~,label2] = max(post',[],1);
    
    %Plot the segmented images
    figure(2), subplot(size(filenames,2),length(Kvalues)+1,imageCounter)
    imagesc(reshape(label2,R,C));
    if imageCounter == 1
        title(['Segmented Plane Picture (',num2str(best_GMM),' components)']), xlabel('xPixel index'), ylabel('yPixel index');
    else if imageCounter == 2
        title(['Segmented Bird Picture (',num2str(best_GMM),' components)']), xlabel('xPixel index'), ylabel('yPixel index');
        end
    end  
end

function g = evalGaussian(x, mu, Sigma)
    [n,N] = size(x);
    invSigma = inv(Sigma);
    C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
    g = C * exp(E);
end
    
function gmm = evalGMM(x,alpha,mu,Sigma)
    gmm = zeros(1,size(x,2));
    for m = 1:length(alpha)
        gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
    end
end
    
