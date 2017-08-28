# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 12:47:27 2017

@author: Dan
LK gradient decent : http://www.ri.cmu.edu/pub_files/pub3/baker_simon_2004_1/baker_simon_2004_1.pdf
https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html
http://iatool.net/tutorials/
https://www.mathworks.com/matlabcentral/fileexchange/62921-ecc-registration-100x-faster

Big thanks to professor Georgios Evangelidis - this is practically a copy of his 
matlab implemenation in python.
"""

import numpy as np
from numpy import linalg as LA
import cv2

def iterativeReg( fixed, moving, niter = 50, npyramids = 3 , showProgress = False):    
    
    # initialize tform 
    tform = np.eye(3, 3, dtype=np.float32)
    deltas = [];
    tforms = [];
    fixedGrads = None
    #fixed = cv2.GaussianBlur(fixed,(5,5),0) # helps with stability
    #moving = cv2.GaussianBlur(moving,(5,5),0) # helps with stability
    
    # crete pyrmid
    pyr_fixed = [fixed]
    pyr_moving = [moving]
    for pyrlevel in range(npyramids):
        pyr_fixed.append(cv2.pyrDown(pyr_fixed[-1]))
        pyr_moving.append(cv2.pyrDown(pyr_moving[-1]))
        
    
    # use lstm to estimate tform delta iteratively
    for pyrlevel in range(npyramids-1,-1,-1):
        fixed_curr = pyr_fixed[pyrlevel]
        moving_curr = pyr_moving[pyrlevel]
        fixedGrads = None
        
        if showProgress:
            imshowpair(fixed_curr,moving_curr,tform, figureTitle = 'before level')

        for iter in range(niter):

            LKdelta, fixedGrads = LKParamUpdateEstimate(   fixed_curr, moving_curr,  tform, fixedGrads)
            deltas.append(LKdelta)
            tform = tform + np.vstack([LKdelta,0]).reshape((3,3)).T
            tform[-1,-1] = 1
            tforms.append(tform)
            
        # plot result of level
        if showProgress:
            imshowpair(fixed_curr,moving_curr,tform, figureTitle = 'after level')

        # update tform for next level up 
        if pyrlevel > 0 :
            tform = pyrUpTform(tform)  
            
    return tforms, deltas 

def imshowpair(fixed,moving,tform, figureTitle = ''):
    moving_tformed = cv2.warpPerspective(moving, np.linalg.inv(tform) ,
                    (moving.shape[1], moving.shape[0]), 
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    im_falsecolor = np.dstack((moving_tformed, fixed, fixed))
    
    cv2.imshow(figureTitle,im_falsecolor)
    cv2.waitKey(0)


def LKParamUpdateEstimate(fixed, moving, tform, fixedGrads = None, method = 'classical_LK'):
   '''
   fixed = the image which is reference. moving = the image which will be transformed inorder to match fixed such that 
   moving[tform*[x,y,1]_moving] ~= fixed. 
   
   tform = 3x3 homography transformation.
   fixedGrads = [gradient in the x direction, gradient in the y direction of fixed] ( each being the same size as fixed)
   
    # calc gradient of both images ( if none supplied)
    # trasform moving image with tform
    # transform gradient with tform
    # compue jacobian 
    # compute steepest decent  
    # compute hessian + inverse
    # compute compute error vector 
    # project error onto jacobian 
    # compute paramaters update 
        
        % mean remove from images 
        
        im = fixed- mean(fixed(:));
        temp = moving- mean(moving(:));
        
        wim = iat_inverse_warping(im, warp, transform, nx, ny, str); %inverse (backward) warping
         
        if (i == noi) % the algorithm is executed (noi-1) times
            break;
        end
        
        % Gradient Image interpolation (warped gradients)
        wvx = iat_inverse_warping(vx, warp, transform, nx, ny, str);
        wvy = iat_inverse_warping(vy, warp, transform, nx, ny, str);
        
        % Compute the jacobian of warp transform
        J = warp_jacobian_onPts(nx, ny, warp, transform);
        
        % Compute the jacobian of warped image wrt parameters (steepest
        % descent image)
        G = image_jacobian_onPts(wvx, wvy, J, 8);
        
        % Compute Hessian and its inverse
        C= G' * G;% C: Hessian matrix
        %i_C = inv(C);
               
        % Compute error vector
        imerror = temp - wim;
        
        % Compute the projection of error vector into Jacobian G
        Ge = G' * imerror(:);
        
        % Compute the optimum parameter correction vector
        delta_p = C\Ge;
   '''    
   if fixedGrads is None:
       fixedGrads = np.gradient(fixed.astype(float), axis = (0,1))
   
   im = fixed- fixed[np.where(fixed >0)].mean(); # mean removed fixed = image
   temp = moving- moving[np.where(moving >0)].mean(); # mean removed moving = template
   
   
   vy, vx = fixedGrads; 
   
   nPts = 1000
   # fetch nPts with high gradient
   highGradInds = np.fliplr(np.asanyarray(np.where((vx > vx.mean()) & (vy > vy.mean()) )).T)
   

   inds = highGradInds[np.random.randint(0,highGradInds.shape[0],(nPts)),:]
   # fetch a random nPts number of pts
#      inds = np.hstack([np.random.randint(0,im.shape[1],(nPts,1)), 
#                    np.random.randint(0,im.shape[0],(nPts,1))])
   # fetch entire image
#   iinds , jinds = np.meshgrid(np.linspace(0,  fixed.shape[0]-1, fixed.shape[0]) ,np.linspace(0, fixed.shape[1]-1, fixed.shape[1]))
#   inds = np.hstack([jinds.reshape(-1,1) , iinds.reshape(-1,1)]).astype(int)
   
   wim = applyWarpOnPts(inds,im, tform )
   wvx = applyWarpOnPts(inds,vx, tform )
   wvy = applyWarpOnPts(inds,vy, tform )
   
   J = warp_jacobian_onPts(inds[:,0], inds[:,1], tform)
         
   G = image_jacobian_onPts(wvx, wvy, J);
   
   C = G.T.dot(G) # C: Hessian matrix
   
   i_C = np.linalg.inv(C)
   
   if method == 'ecc':
       '''
       % Compute projections of images into G
        Gt = G' * tempzm(:);
        Gw = G' * wim(:);
        
       % Compute lambda parameter
        num = (norm(wim(:))^2 - Gw' * i_C * Gw);
        den = (tempzm(:)'*wim(:) - Gt' * i_C * Gw);
        lambda = num / den;
        
        % Compute error vector
        imerror = lambda * tempzm - wim;
       '''
       Gt =  G.T.dot(temp[inds[:,1],inds[:,0]].reshape((-1,1)))
       Gw =  G.T.dot(wim)
       num = LA.norm(wim)**2 - Gw.T.dot(i_C).dot(Gw);
       den = temp[inds[:,1],inds[:,0]].reshape((-1,1)).T.dot(wim) - Gt.T.dot(i_C).dot(Gw)
       lambda_weight = num / den;
       imerror = lambda_weight * temp[inds[:,1],inds[:,0]].reshape((-1,1))  - wim;
    
   else:    
       imerror = temp[inds[:,1],inds[:,0]].reshape((-1,1)) - wim;
   
   
   Ge = G.T.dot(imerror)
   
   #delta_p = np.linalg.solve(C,Ge); #if a is square; 
   
   delta_p = i_C.dot(Ge); # otherwise;  # C = 8x8 Ge = 8x150
   
   return delta_p, fixedGrads
   
   
   
def applyWarpOnPts(inds,imIn,warp):
    '''
    inds = [x coord, y coord] to be sampled. 
    imIn - mxn ( grayscale ) image 
    warp = 3x3 homography transformation. 
    
    
    [yy,xx] = ind2sub(size(imIn),inds);
    xy=[xx(:)';yy(:)';ones(1,length(yy(:)))];
    
    %3x3 matrix transformation
    A = warp;
    A(3,3) = 1;
    
    % new coordinates
    xy_prime = A * xy;
    
    if strcmp(transform,'homography')
    
        % division due to homogeneous coordinates
        xy_prime(1,:) = xy_prime(1,:)./xy_prime(3,:);
        xy_prime(2,:) = xy_prime(2,:)./xy_prime(3,:);
    end
    
    % Ignore third row
    xy_prime = xy_prime(1:2,:);
    
    % Subpixel interpolation
    valueAtInds = lininterp2_fast(imIn, xy_prime(1,:)', xy_prime(2,:)');
    
    valueAtInds(isnan(valueAtInds))=0;%replace Nan
    '''
    xx = inds[:,0];
    yy = inds[:,1];
    xy = np.array([xx,yy,np.ones_like(xx)])
    
    A = warp
    warp[-1,-1] = 1
    
    xy_prime = np.matmul(A,xy)
    
    xy_prime[0,:] = xy_prime[0,:]/xy_prime[2,:]
    xy_prime[1,:] = xy_prime[1,:]/xy_prime[2,:]
    
    xy_prime = xy_prime[:2,:];
    
    
    valueAtInds = lininterp2_fast(imIn, xy_prime[0,:].reshape((-1,1)), xy_prime[1,:].reshape((-1,1)));
    
    valueAtInds = np.nan_to_num(valueAtInds); # may not be a good idea
    
    return valueAtInds
    

def lininterp2_fast(V,x,y):
    '''
    V is a mxn matrix
    x,y = corrdinates of points to sample. ( each is a mx1 np array )
    
    x0 = floor(x);
    x1 = ceil(x);
    y0 = floor(y);
    y1 = ceil(y);
    szV = size(V);
    x = mod(x,1);
    y = mod(y,1);
    
    %% find valid outputs 
    validInds = x0> 0 & x1 < szV(2) & y0 > 0 & y1 < szV(1);
    x(~validInds) = [];
    y(~validInds) = [];
    x0(~validInds) = [];
    y0(~validInds) = [];
    x1(~validInds) = [];
    y1(~validInds) = [];
    
    %% calc near by values
    f00 = V(sub2ind(szV,y0,x0));
    f01 = V(sub2ind(szV,y1,x0));
    f10 = V(sub2ind(szV,y0,x1));
    f11 = V(sub2ind(szV,y1,x1));
    
    %% calc 
    valueOfValidPts = f00.*(1-mod(x,1)).*(1-y)+f10.*x.*(1-y)+f01.*(1-x).*y+f11.*x.*y;
    
    %% deal with poitns out of range
    value = nan(numel(validInds),1);
    value(validInds) = valueOfValidPts;
    '''
    
    x0, x1, y0, y1, szV, x, y  = [np.floor(x), np.ceil(x), np.floor(y), np.ceil(y), V.shape[:2], np.mod(x,1), np.mod(y,1)]
     
    validInds = (x0 >= 0) & (x1 < szV[1]-1) & (y0 >= 0) & (y1 < szV[0]-1);
    
    x  = x[np.where(validInds)]
    y  = y[np.where(validInds)]
    x0  = x0[np.where(validInds)]
    y0  = y0[np.where(validInds)]
    x1  = x1[np.where(validInds)]
    y1  = y1[np.where(validInds)]

    
    f00 = V[y0.astype(int), x0.astype(int)]
    f01 = V[y1.astype(int), x0.astype(int)];
    f10 = V[y0.astype(int), x1.astype(int)]; 
    f11 = V[y1.astype(int), x1.astype(int)];
    
    valueOfValidPts = f00 * (1-np.mod(x,1)) * (1-y) + f10 * x * (1-y) + f01 * (1-x) * y + f11 * x * y;
    
    value = np.ones_like(validInds).astype(float)
    value.fill(np.nan)
    
    
    value[np.where(validInds)] = valueOfValidPts;
    
    return value
    
    
def warp_jacobian_onPts(nx, ny, warp):
    '''
    nx, ny = the points which are being warped
    warp = 3x3 homography transformation matrix 
    
    numelPts = numel(nx);
    
    Jx=nx;
    Jy=ny;
    J0=0*Jx;
    J1=J0+1;


    xy=[Jx(:)';Jy(:)';ones(1,numelPts)];


    %3x3 matrix transformation
    A = warp;
    A(3,3) = 1;

    % new coordinates
    xy_prime = A * xy;



    % division due to homogeneous coordinates
    xy_prime(1,:) = xy_prime(1,:)./xy_prime(3,:);
    xy_prime(2,:) = xy_prime(2,:)./xy_prime(3,:);

    den = xy_prime(3,:)';

    Jx(:) = Jx(:) ./ den;
    Jy(:) = Jy(:) ./ den;
    J1(:) = J1(:) ./ den;

    Jxx_prime = Jx;
    Jxx_prime(:) = Jxx_prime(:) .* xy_prime(1,:)';
    Jyx_prime = Jy;
    Jyx_prime(:) = Jyx_prime(:) .* xy_prime(1,:)';

    Jxy_prime = Jx;
    Jxy_prime(:) = Jxy_prime(:) .* xy_prime(2,:)';
    Jyy_prime = Jy;
    Jyy_prime(:) = Jyy_prime(:) .* xy_prime(2,:)';


    J = [Jx, J0, -Jxx_prime, Jy, J0, - Jyx_prime, J1, J0;...
        J0, Jx, -Jxy_prime, J0, Jy, -Jyy_prime, J0, J1];
    
    '''
    numelPts = nx.size
    
    Jx = nx.astype(float).reshape((-1,1))
    Jy = ny.astype(float).reshape((-1,1))
    J0 = np.zeros_like(Jx)
    J1 = np.ones_like(Jx)
    
    xy=np.vstack([Jx.reshape((1,-1)), Jy.reshape((1,-1)), np.ones((1,numelPts))])
    
    A = warp;
    A[2,2] = 1;
    
    xy_prime = A.dot(xy)
    
    xy_prime[0,:] = xy_prime[0,:] / xy_prime[-1,:];
    xy_prime[1,:] = xy_prime[1,:] / xy_prime[-1,:];

    den = xy_prime[2,:].reshape((-1,1))
    
    JxNorm = Jx / den;
    JyNorm = Jy/ den;
    J1Norm = J1 / den;
    
    Jxx_prime = JxNorm * xy_prime[0,:].reshape((-1,1));
    Jyx_prime = JyNorm * xy_prime[0,:].reshape((-1,1));

    Jxy_prime = JxNorm * xy_prime[1,:].reshape((-1,1));
    Jyy_prime = JyNorm * xy_prime[1,:].reshape((-1,1));
    
    
    J = np.vstack([np.hstack([Jx, J0, -Jxx_prime, Jy, J0, - Jyx_prime, J1Norm, J0]),
        np.hstack([J0, Jx, -Jxy_prime, J0, Jy, -Jyy_prime, J0, J1Norm])])
    
    return J
    
    
    
def image_jacobian_onPts(gx,gy,jac):
    '''
    gx, gy  = mx1 vectors each of the values of the x and y gradients after a warpping. 
    jac = jacobian = mx8 matrix 
    
    [h,w]=size(jac);
    
    if nargin<4
        error('Not enough input arguments');
    end
        
    gx=repmat(gx,1,nop);
    gy=repmat(gy,1,nop);
        
    G=gx.*jac(1:h/2,:)+gy.*jac(h/2+1:end,:);
    '''
    
    h, w = jac.shape[:2];
    
    gx = np.tile(gx,(1,w))
    gy = np.tile(gy,(1,w))
    
    G = gx * jac[0:int(h/2),:] + gy * jac[int(h/2):,:]
    
    return G

def pyrUpTform(tform, factor = 2 ):
    
    tform[[0,1],[2,2]] *= factor;
    tform[[2,2],[0,1]] /= factor;
    
    return tform
    