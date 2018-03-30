# vim: set fileencoding=utf8 :

"""Align, register and fuse frames specified on the command line using the
DTCWT fusion algorithm.

Usage:
    fuseimages [options] <frames>...
    fuseimages (-h | --help)

Options:
    -n, --normalise                     Re-normalise input frames to lie on the
                                        interval [0,1].
    -v, --verbose                       Increase logging verbosity.
    -o PREFIX, --output-prefix=PREFIX   Prefix output filenames with PREFIX.
                                        [default: fused-]
    -r STRATEGY, --ref=STRATEGY         Use STRATEGY to select reference
                                        frame. [default: middle]
    --register-to=REFERENCE             Which frame to use as registration reference.
                                        [default: mean-aligned]
    --save-registered-frames            Save registered frames in npz format.
    --save-registered-frame-images      Save registered frames, one image per frame.
    --save-input-frames                 Save input frames in npz format.
    --save-input-frame-images           Save input frames, one image per frame.

The frame within <frames> used as a reference frame can be selected via the
--ref flag.  The strategy can be one of:

    middle      Use the middle frame.
    first       Use the first frame.
    last        Use the last frame.
    max-mean    Use the frame with the maximum mean value. This is useful in
                the situation where you have a large number of blank frames in
                the input sequence.
    max-range   Use the frame with the maximum range of values.

The frame to be used as reference for the DTCWT-based image registration can be
selected via the --register-to flag. It can take the following values:

    mean-aligned    Use the mean of the aligned frames. This is useful if your
                    input images are very noisy since it avoids
                    "over-registering" to the noise.
    reference       Use the same image as the bulk alignment step.

"""

from __future__ import print_function, division, absolute_import
import cv2
import logging
from os import walk
from _images2gif import writeGif
import sys
import dtcwt
import dtcwt.registration
import dtcwt.sampling
from docopt import docopt
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from six.moves import xrange
'''
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
'''
def load_frames(path, filenames, normalise=True):
    """Load frames from *filenames* returning a 3D-array with all pixel values.
    If *normalise* is True, then input frames are normalised onto the range
    [0,1].

    """
    frames = []

    for fn in filenames:
        # Load image with PIL
        logging.info('Loading "{fn}"'.format(fn=fn))
        im = Image.open(path + '\\' + fn)
        
        #im.show()
        #print(len(im.getdata()[0]))
        data  = np.asarray(im)
        data = data[:,:,0]
        # Extract data and store as floating point data
        im_array = np.array(data, dtype=np.float32).reshape(im.size[::-1])
 
        # If we are to normalise, do so
        if normalise:
            im_array -= im_array.min()
            im_array /= im_array.max()

        # If this isn't the first image, check that shapes are consistent
        if len(frames) > 0:
            if im_array.shape != frames[-1].shape:
                logging.warn('Skipping "{fn}" with inconsistent shape'.format(fn=fn))
                continue

        frames.append(im_array)

    logging.info('Loaded {0} image(s)'.format(len(frames)))

    return np.dstack(frames)

def align(frames, template):
    """
    Warp each slice of the 3D array frames to align it to *template*.

    """
    if frames.shape[:2] != template.shape:
        raise ValueError('Template must be same shape as one slice of frame array')

    # Calculate xs and ys to sample from one frame
    xs, ys = np.meshgrid(np.arange(frames.shape[1]), np.arange(frames.shape[0]))

    # Calculate window to use in FFT convolve
    w = np.outer(np.hamming(template.shape[0]), np.hamming(template.shape[1]))

    # Calculate a normalisation for the cross-correlation
    ccnorm = 1.0 / fftconvolve(w, w)

    # Set border of normalisation to zero to avoid overfitting. Borser is set so that there
    # must be a minimum of half-frame overlap
    ccnorm[:(template.shape[0]>>1),:] = 0
    ccnorm[-(template.shape[0]>>1):,:] = 0
    ccnorm[:,:(template.shape[1]>>1)] = 0
    ccnorm[:,-(template.shape[1]>>1):] = 0

    # Normalise template
    tmpl_min = template.min()
    norm_template = template - tmpl_min
    tmpl_max = norm_template.max()
    norm_template /= tmpl_max

    warped_ims = []
    i = 0
    for frame_idx in xrange(frames.shape[2]):
        logging.info('Aligning frame {0}/{1}'.format(frame_idx+1, frames.shape[2]))
        frame = frames[:,:,frame_idx]
        '''
        if i == 0:
            im = Image.fromarray(tonemap(frame).copy(), 'L')
            im.show()
        '''
        # Normalise frame
        norm_frame = frame - tmpl_min
        norm_frame /= tmpl_max
        '''
        if i == 0:
            im = Image.fromarray(tonemap(norm_frame).copy(), 'L')
            im.show()
        '''
        # Convolve template and frame
        ex1 = norm_template*w
        ex2 = np.fliplr(np.flipud(norm_frame*w))
        '''
        if i == 0:
            im1 = Image.fromarray(tonemap(ex1).copy(), 'L')
            im1.show()
            im2 = Image.fromarray(tonemap(ex2).copy(), 'L')
            im2.show()
        '''
        conv_im = fftconvolve(ex1, ex2)
        '''
        if i == 0:
            im = Image.fromarray(tonemap(conv_im).copy(), 'L')
            im.show()
        '''
        conv_im *= ccnorm
        '''
        if i == 0:
            im = Image.fromarray(tonemap(conv_im).copy(), 'L')
            im.show()
        '''
        # Find maximum location
        max_loc = np.unravel_index(conv_im.argmax(), conv_im.shape)
        
        # Convert location to shift
        dy = max_loc[0] - template.shape[0] + 1
        dx = max_loc[1] - template.shape[1] + 1
        print(dy, dx)
        logging.info('Offset computed to be ({0},{1})'.format(dx, dy))
        curr_img = dtcwt.sampling.sample(frame, xs-dx, ys-dy, method='bilinear')
        # Warp image
        warped_ims.append(curr_img)
        #save_image('cur_img- ' + str(i), curr_img)
        i += 1
    return np.dstack(warped_ims)

def register(frames, template, nlevels=7):
    """
    Use DTCWT registration to return warped versions of frames aligned to template.

    """
    # Normalise template
    tmpl_min = template.min()
    norm_template = template - tmpl_min
    tmpl_max = norm_template.max()
    norm_template /= tmpl_max

    # Transform template
    transform = dtcwt.Transform2d()
    template_t = transform.forward(norm_template, nlevels=nlevels)

    warped_ims = []
    i = 0
    for frame_idx in xrange(frames.shape[2]):
        logging.info('Registering frame {0}/{1}'.format(frame_idx+1, frames.shape[2]))
        frame = frames[:,:,frame_idx]

        # Normalise frame
        norm_frame = frame - tmpl_min
        norm_frame /= tmpl_max

        # Transform frame
        frame_t = transform.forward(norm_frame, nlevels=nlevels)

        # Register
        reg = dtcwt.registration.estimatereg(frame_t, template_t)
        ex = dtcwt.registration.warp(frame, reg, method='bilinear')
        
        if i == 0:
            im = Image.fromarray(tonemap(ex).copy(), 'L')
            im.show()
        kernel = np.zeros( (13,13), np.float32)
        kernel[4,4] = 2.0    
        boxFilter = np.ones( (13,13), np.float32) / 169.0

        #Subtract the two:
        kernel = kernel - boxFilter
        imgIn = Image.fromarray(tonemap(ex).copy(), 'L')
        custom = cv2.filter2D(ex, -1, kernel)
        data  = np.asarray(custom)
        # Extract data and store as floating point data
        sharpened = np.array(data, dtype=np.float32).reshape(im.size[::-1])
        '''
        blurred_f = ndimage.gaussian_filter(ex, 3)
        filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
        alpha = 30
        sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
        '''
        if i == 0:
            im = Image.fromarray(tonemap(sharpened).copy(), 'L')
            im.show()
            
        warped_ims.append(sharpened)
        i += 1
    return np.dstack(warped_ims)

def tonemap(array):
    # The normalisation strategy here is to let the middle 98% of
    # the values fall in the range 0.01 to 0.99 ('black' and 'white' level).
    black_level = np.percentile(array,  1)
    white_level = np.percentile(array, 99)

    norm_array = array - black_level
    norm_array /= (white_level - black_level)
    norm_array = np.clip(norm_array + 0.01, 0, 1)

    return np.array(norm_array * 255, dtype=np.uint8)

def save_image(filename, array):
    # Copy is workaround for http://goo.gl/8fuOJA
    im = Image.fromarray(tonemap(array).copy(), 'L')

    logging.info('Saving "{0}"'.format(filename + '.png'))
    im.save(filename + '.png')

def transform_frames(frames, nlevels=7):
    # Transform each registered frame storing result
    lowpasses = []
    highpasses = []
    for idx in xrange(nlevels):
        highpasses.append([])

    transform = dtcwt.Transform2d()
    for frame_idx in xrange(frames.shape[2]):
        logging.info('Transforming frame {0}/{1}'.format(frame_idx+1, frames.shape[2]))
        frame = frames[:,:,frame_idx]
        frame_t = transform.forward(frame, nlevels=nlevels)

        lowpasses.append(frame_t.lowpass)
        for idx in xrange(nlevels):
            highpasses[idx].append(frame_t.highpasses[idx][:,:,:,np.newaxis])

    return np.dstack(lowpasses), tuple(np.concatenate(hp, axis=3) for hp in highpasses)

def reconstruct(lowpass, highpasses):
    transform = dtcwt.Transform2d()
    t = dtcwt.Pyramid(lowpass, highpasses)
    return transform.inverse(t)

def shrink_coeffs(highpasses):
    """Implement Bivariate Laplacian shrinkage as described in [1].
    *highpasses* is a sequence containing wavelet coefficients for each level
    fine-to-coarse. Return a sequence containing the shrunken coefficients.

    [1] A. Loza, D. Bull, N. Canagarajah, and A. Achim, “Non-gaussian model-
    based fusion of noisy frames in the wavelet domain,” Comput. Vis. Image
    Underst., vol. 114, pp. 54–65, Jan. 2010.

    """
    shrunk_levels = []

    # Estimate noise from first level coefficients:
    # \sigma_n = MAD(X_1) / 0.6745

    # Compute median absolute deviation of wavelet magnitudes. This is more than
    # a little magic compared to the 1d version.
    level1_mad_real = np.median(np.abs(highpasses[0].real - np.median(highpasses[0].real)))
    level1_mad_imag = np.median(np.abs(highpasses[0].imag - np.median(highpasses[0].imag)))
    sigma_n = np.sqrt(level1_mad_real*level1_mad_real + level1_mad_imag+level1_mad_imag) / (np.sqrt(2) * 0.6745)

    # In this context, parent == coarse, child == fine. Work from
    # coarse to fine
    shrunk_levels.append(highpasses[-1])
    for parent, child in zip(highpasses[-1:0:-1], highpasses[-2::-1]):
        # We will shrink child coefficients.

        # Rescale parent to be the size of child
        parent = dtcwt.sampling.rescale(parent, child.shape[:2], method='nearest')

        # Construct gain for shrinkage separately per direction and for real and imag
        real_gain = np.ones_like(child.real)
        imag_gain = np.ones_like(child.real)
        for dir_idx in xrange(parent.shape[2]):
            child_d = child[:,:,dir_idx]
            parent_d = parent[:,:,dir_idx]

            # Estimate sigma_w and gain for real
            real_sigma_w = np.sqrt(np.maximum(1e-8, np.var(child_d.real) - sigma_n*sigma_n))
            real_R = np.sqrt(parent_d.real*parent_d.real + child_d.real*child_d.real)
            real_gain[:,:,dir_idx] = np.maximum(0, real_R - (np.sqrt(3)*sigma_n*sigma_n)/real_sigma_w) / real_R

            # Estimate sigma_w and gain for imag
            imag_sigma_w = np.sqrt(np.maximum(1e-8, np.var(child_d.imag) - sigma_n*sigma_n))
            imag_R = np.sqrt(parent_d.imag*parent_d.imag + child_d.imag*child_d.imag)
            imag_gain[:,:,dir_idx] = np.maximum(0, imag_R - (np.sqrt(3)*sigma_n*sigma_n)/imag_sigma_w) / imag_R

        # Shrink child levels
        shrunk = (child.real * real_gain) + 1j * (child.imag * imag_gain)
        shrunk_levels.append(shrunk)

    return shrunk_levels[::-1]


options = docopt(__doc__)
imprefix = options['--output-prefix']
# Set up logging according to command line options
loglevel = logging.INFO if options['--verbose'] else logging.WARN
logging.basicConfig(level=loglevel)

# Load inputs
logging.info('Loading input frames')
path = sys.argv[1]
files = []
for (dirpath, dirnames, filenames) in walk(path):
    files.extend(filenames)
print(files)
input_frames = load_frames(path, files)

if options['--save-input-frames']:
    logging.info('Saving input frames')
    np.savez_compressed(imprefix + 'input-frames.npz', frames=input_frames)

if options['--save-input-frame-images']:
    logging.info('Saving input frame images')
    for idx in xrange(input_frames.shape[2]):
        save_image(imprefix + 'input-frame-{0:05d}'.format(idx), input_frames[:,:,idx])

# Select the reference frame
ref_strategy = options['--ref']
if ref_strategy == 'middle':
    reference_frame = input_frames[:,:,input_frames.shape[2]>>1]
elif ref_strategy == 'first':
    reference_frame = input_frames[:,:,0]
elif ref_strategy == 'last':
    reference_frame = input_frames[:,:,-1]
elif ref_strategy == 'max-mean':
    means = np.array(list(np.mean(input_frames[:,:,idx]) for idx in xrange(input_frames.shape[2])))
    reference_frame = input_frames[:,:,means.argmax()]
elif ref_strategy == 'max-range':
    ranges = np.array(list(input_frames[:,:,idx].max() - input_frames[:,:,idx].min()
                           for idx in xrange(input_frames.shape[2])))
    reference_frame = input_frames[:,:,ranges.argmax()]
else:
    logging.error('Unknown reference strategy: {0}'.format(ref_strategy))
    

logging.info('Using reference strategy: {0}'.format(ref_strategy))

# Save sample frame
logging.info('Saving sample frame')
save_image(imprefix + 'sample-frame', reference_frame)

# Align frames to *centre* frame
logging.info('Aligning frames')
aligned_frames = align(input_frames, reference_frame)
mean_aligned_frame = np.mean(aligned_frames, axis=2)

# Save mean aligned frame
logging.info('Saving mean aligned frame')
save_image(imprefix + 'mean-aligned', mean_aligned_frame)

# Register frames
registration_ref_src = options['--register-to']
if registration_ref_src == 'mean-aligned':
    registration_reference = mean_aligned_frame
elif registration_ref_src == 'reference':
    registration_reference = reference_frame
else:
    logging.error('Unknown registration source: {0}'.format(registration_ref_src))
    
registered_frames = register(aligned_frames, registration_reference)

# Save mean registered frame
logging.info('Saving mean registered frame')
save_image(imprefix + 'mean-registered', np.mean(registered_frames, axis=2))

if options['--save-registered-frames']:
    logging.info('Saving registered frames')
    np.savez_compressed(imprefix + 'registered-frames.npz', frames=registered_frames)

if options['--save-registered-frame-images']:
    logging.info('Saving registered frame images')
    for idx in xrange(registered_frames.shape[2]):
        save_image(imprefix + 'registered-frame-{0:05d}'.format(idx), registered_frames[:,:,idx])

# Transform registered frames
lowpasses, highpasses = transform_frames(registered_frames)

# Compute mean lowpass image
lowpass_mean = np.mean(lowpasses, axis=2)

# Get mean direction for each subband
phases = []
for level_sb in highpasses:
    # Calculate mean direction by adding all subbands together and normalising
    sum_ = np.sum(level_sb, axis=3)
    sum_mag = np.abs(sum_)
    sum_ /= np.where(sum_mag != 0, sum_mag, 1)
    phases.append(sum_)

# Compute mean, maximum and maximum-of-inliers magnitudes
mean_mags, max_mags, max_inlier_mags = [], [], []
for level_sb in highpasses:
    mags = np.abs(level_sb)

    mean_mags.append(np.mean(mags, axis=3))
    max_mags.append(np.max(mags, axis=3))

    thresh = 2*np.repeat(np.median(mags, axis=3)[:,:,:,np.newaxis], level_sb.shape[3], axis=3)
    outlier_suppressed = np.where(mags < thresh, mags, 0)
    max_inlier_mags.append(np.max(outlier_suppressed, axis=3))

# Reconstruct frames
logging.info('Computing mean magnitude fused image')
mean_recon = reconstruct(lowpass_mean, tuple(mag*phase for mag, phase in zip(mean_mags, phases)))
save_image(imprefix + 'fused-mean-dtcwt', mean_recon)

logging.info('Computing maximum magnitude fused image')
max_recon = reconstruct(lowpass_mean, tuple(mag*phase for mag, phase in zip(max_mags, phases)))
save_image(imprefix + 'fused-max-dtcwt', max_recon)

logging.info('Computing maximum of inliners magnitude fused image')
max_inlier_recon = reconstruct(lowpass_mean, tuple(mag*phase for mag, phase in zip(max_inlier_mags, phases)))
save_image(imprefix + 'fused-max-inlier-dtcwt', max_inlier_recon)

logging.info('Computing maximum of inliners magnitude fused image w/ shrinkage')
max_inlier_shrink_recon = reconstruct(lowpass_mean,
        shrink_coeffs(tuple(mag*phase for mag, phase in zip(max_inlier_mags, phases))))
save_image(imprefix + 'fused-max-inlier-shrink-dtcwt', max_inlier_shrink_recon)

# Save final animation
shape = np.min((
    input_frames.shape[:2], aligned_frames.shape[:2],
    registered_frames.shape[:2], max_inlier_shrink_recon.shape
), axis=0)
anim_frames = tonemap(np.vstack((
    np.hstack((input_frames[:shape[0],:shape[1]], aligned_frames[:shape[0],:shape[1]])),
    np.hstack((registered_frames[:shape[0],:shape[1]],
               np.repeat(max_inlier_shrink_recon[:shape[0],:shape[1],np.newaxis],
                         input_frames.shape[2], axis=2))),
)))
anim_filename = imprefix + 'aligned-and-registered-anim10.gif'
logging.info('Saving animation to "{0}"'.format(anim_filename))
# Copy is workaround for http://goo.gl/8fuOJA
writeGif(anim_filename,
         list(Image.fromarray(anim_frames[:,:,idx].copy(), 'L')
              for idx in xrange(anim_frames.shape[2])),
         duration=0.1)
         
