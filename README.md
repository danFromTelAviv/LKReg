# LKReg
image registration using Lucas Kanade forward additive method.

overview:
The following is a simple pure python implementation of forward addative registration ( image alignment ). 

simple example : 

import LKForwardAddativeImageReg as LKReg 

#load images 
fixed = cv2.cvtColor(cv2.imread('fixed.png'), cv2.COLOR_BGR2GRAY)
moving = cv2.cvtColor(cv2.imread('moving.png'), cv2.COLOR_BGR2GRAY)

# perform registration
tforms, deltas  = LKReg.iterativeReg(fixed, moving, niter = 150, npyramids = 2)

# show before and after
LKReg.imshowpair(fixed,moving,tforms[0], figureTitle = 'before')
LKReg.imshowpair(fixed,moving,tforms[-1], figureTitle = 'after')

In depth  :
This module allows for fast, subpixel accurate, robust registration. It essentially performs stochatic gradient decent using a first order taylor polynomial aproximation of the gradient. This is very similar to whats found in cv2 / matlab imregister. The ECC ( enhanced correlation coefficient ) algorithm is also implemented which adds robustness and reduces the numbers of needed itterations. This registration can use pyramiding ( or not - npyramids = 1 ) to deal with relatively large transformations. 


references:
1) http://cseweb.ucsd.edu/classes/sp02/cse252/lucaskanade81.pdf
2) http://www.ri.cmu.edu/pub_files/pub3/baker_simon_2004_1/baker_simon_2004_1.pdf
3) https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html
4) http://iatool.net/tutorials/
5) https://www.mathworks.com/matlabcentral/fileexchange/62921-ecc-registration-100x-faster

