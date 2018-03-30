Algorithm description
=====================

This section provides a brief overview of the DTCWT-based fusion algorithm as
used in this project. The input to the algorithm is a set of input images, :math:`\mathcal{I}`.

Alignment
---------

This step aligns each input image with a single translation to match as well as
possible a template image.  The central image of :math:`\mathcal{I}` is
selected as the template image :math:`T`. For each image :math:`I \in
\mathcal{I}`:

1. Compute the cross-correlation image :math:`C = (I \cdot w) \star (T \cdot
   w)` where :math:`w` is a two-dimensional Hamming window, :math:`\cdot`
   denotes pixel-wise multiplication and :math:`\star` is the cross-correlation
   operator. Normalise this cross-correlation, :math:`C \rightarrow C / (w
   \star w)` where :math:`/` denotes element-wise division.

2. Find the location of the maximum of :math:`C` and compute the corresponding
   translational shift for that location. The maximum is found ignoring an
   apron around the edge of the image to avoid over matching of small
   overlap-regions.

3. Warp :math:`I` according to that translation.

Combine all aligned images into the set of aligned images, :math:`\mathcal{I}_a`.

Registration
------------

This step locally warps each aligned image to best match the same template
image as above. For each image :math:`I \in \mathcal{I}_a`:

1. Compute the local affine warp mapping :math:`I` to :math:`T` as described in
   [1, 2].

2. Warp :math:`I` according to the registration.

Combine all registered images into the set of registered images,
:math:`\mathcal{I}_r`.

Fusion
------

This step combines all images in :math:`\mathcal{I}_r` into a single fused
image. The fusion is performed in the wavelet domain and is based on the
technique in[3]. The overall lowpass image is computed by taking the mean of
the lowpass images corresponding to each image in :math:`\mathcal{I}_r`.
Letting :math:`\theta^{(i)}_{d,\ell,j}` correspond to the :math:`j`-th highpass
coeffiecient in direction :math:`d` at level :math:`\ell` of the DTCWT
transform of the :math:`i`-th image in :math:`\mathcal{I}_r` we can construct
the fused wavelet coefficients :math:`\theta_{d,\ell,j}` in the following way:

1. Compute :math:`\Theta_{d,\ell,j} = \sum_{i} \theta^{(i)}_{d,\ell,j}` and
   then :math:`\phi_{d,\ell,j} = \Theta_{d,\ell,j} / \left| \Theta_{d,\ell,j}
   \right|`. These unit-magnitude complex numbers represent the average phase
   of corresponding wavelet coefficients over all registered images.

2. Form the set :math:`\mathcal{T}_{d,\ell,j} = \left\{  \left|
   \theta^{(1)}_{d,\ell,j} \right|, \left| \theta^{(2)}_{d,\ell,j} \right|,
   \ \dots\ , \left| \theta^{(N)}_{d,\ell,j} \right| \right\}` for the :math:`N`
   images in :math:`\mathcal{I}_r`. Select :math:`T_{d, \ell, j}` from this set
   using some heuristic. In the current implementation this can be one of: mean
   value, maximum value or maximum value after 2-sigma outliers are removed.
   Which strategy is best may depend on input imagery.

3. Compute :math:`\theta_{d,\ell,j} = T_{d, \ell, j} \, \phi_{d,\ell,j}`.

4. Inverse DTCWT to form the fused image :math:`I_f`.

Shrinkage
---------

The wavelet coefficients of the fused image :math:`I_f` were selected to
maximise sharpness. This may cause noise to be incorrectly preserved in the
output image. A wavelet coefficient shrinkage method based on that in [4] is
then applied to give the final fused and denoised image.

References
----------

1. Pang, Derek, Huizhong Chen, and Sherif Halawa. "Efficient video
   stabilization with dual-tree complex wavelet transform." EE368 Project
   Report, Spring (2010).

2. Chen, Huizhong, and Nick Kingsbury. "Efficient registration of nonrigid 3-D
   bodies." Image Processing, IEEE Transactions on 21.1 (2012): 262-272.

3. Anantrasirichai, Nantheera, et al. "Atmospheric Turbulence Mitigation using
   Complex Wavelet-based Fusion." IEEE transactions on image processing: a
   publication of the IEEE Signal Processing Society (2013).

4. Loza, Artur, et al. "Non-Gaussian model-based fusion of noisy images in the
   wavelet domain." Computer Vision and Image Understanding 114.1 (2010):
   54-65.

