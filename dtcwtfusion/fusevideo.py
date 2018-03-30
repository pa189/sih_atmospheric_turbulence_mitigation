# vim: set fileencoding=utf8 :

"""Align, register and fuse frames specified on the command line using the
DTCWT fusion algorithm.

Usage:
    fusevideo [options] -o <output> <frames>...
    fusevideo (-h | --help)

Options:
    -n, --normalise                     Re-normalise input frames to lie on the
                                        interval [0,1].
    -v, --verbose                       Increase logging verbosity.
    -w FRAMES, --window=FRAMES          Sliding half-window size. [default: 5]

Output will be saved in HDF5 format to <output>. Input is read from the TIFF
files listed as <frames>...

"""

from __future__ import print_function, division, absolute_import

import logging
import sys

import dtcwt
import dtcwt.registration
import dtcwt.sampling
from docopt import docopt
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from six.moves import xrange
import h5py

def load_frames(h5, filenames, normalise=False, dataset_name='input'):
    """Load frames from *filenames* returning a 3D-array with all pixel values.
    If *normalise* is True, then input frames are normalised onto the range
    [0,1]. A dataset will be created in the HDF5 file/group *h5* with name
    *dataset_name*.

    """
    dataset = None

    for f_idx, fn in enumerate(filenames):
        # Load image with PIL
        logging.info('Loading "{fn}"'.format(fn=fn))
        im = Image.open(fn)

        # Extract data and store as floating point data
        im_array = np.array(im.getdata(), dtype=np.float32).reshape(im.size[::-1])

        # If we've not yet created the dataset, do so
        if dataset is None:
            dataset = h5.create_dataset(dataset_name, im_array.shape + (len(filenames),),
                    dtype=im_array.dtype, chunks=im_array.shape + (1,),
                    compression='gzip')

        # If we are to normalise, do so
        if normalise:
            im_array -= im_array.min()
            im_array /= im_array.max()

        # Write result to HDF5
        dataset[:,:,f_idx] = im_array

    if dataset is not None:
        logging.info('Loaded {0} image(s)'.format(dataset.shape[2]))
    else:
        logging.warn('No images were loaded')

    return dataset

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
    for frame_idx in xrange(frames.shape[2]):
        logging.info('Aligning frame {0}/{1}'.format(frame_idx+1, frames.shape[2]))
        frame = frames[:,:,frame_idx]

        # Normalise frame
        norm_frame = frame - tmpl_min
        norm_frame /= tmpl_max

        # Convolve template and frame
        conv_im = fftconvolve(norm_template*w, np.fliplr(np.flipud(norm_frame*w)))
        conv_im *= ccnorm

        # Find maximum location
        max_loc = np.unravel_index(conv_im.argmax(), conv_im.shape)

        # Convert location to shift
        dy = max_loc[0] - template.shape[0] + 1
        dx = max_loc[1] - template.shape[1] + 1
        logging.info('Offset computed to be ({0},{1})'.format(dx, dy))

        # Warp image
        warped_ims.append(dtcwt.sampling.sample(frame, xs-dx, ys-dy, method='bilinear'))

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
        warped_ims.append(dtcwt.registration.warp(frame, reg, method='bilinear'))

    return np.dstack(warped_ims)

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

def main():
    options = docopt(__doc__)

    # Set up logging according to command line options
    loglevel = logging.INFO if options['--verbose'] else logging.WARN
    logging.basicConfig(level=loglevel)

    nlevels = 4

    # Open output
    logging.info('Creating output HDF5 file at "{0}"'.format(options['<output>']))
    output = h5py.File(options['<output>'], 'w')

    # Load inputs
    logging.info('Loading input frames')
    input_frames = load_frames(output, options['<frames>'], options['--normalise'])
    input_frames.attrs.create('description', 'Input frames'.encode('utf-8'))

    # Check there are enough frames
    half_window_size = int(options['--window'])
    if input_frames.shape[2] - (2*half_window_size + 1) <= 0:
        logging.error('Too few frames ({0}) for half window size ({1})'.format(
            input_frames.shape[2], half_window_size))
        sys.exit(1)

    # Create storage for output
    output_shape = list(input_frames.shape)
    output_shape[2] -= (2*half_window_size + 1)
    output_shape = tuple(output_shape)

    # These datasets will be created in the loop below
    mean_frames = None
    median_frames = None
    ninety_frames = None
    max_inlier_frames = None

    mean_shrink_frames = None
    median_shrink_frames = None
    ninety_shrink_frames = None
    max_inlier_shrink_frames = None

    # Select reference frames according to window
    frame_indices = output.create_dataset('processed_indices',
            data=np.arange(input_frames.shape[2])[half_window_size:-(half_window_size+1)],
            compression='gzip')
    frame_indices.attrs.create('description', 'Slice indices into /frames for each frame of output'.encode('utf-8'))
    for ref_idx in frame_indices:
        logging.info('Processing frame {0}'.format(ref_idx))

        reference_frame = input_frames[:,:,ref_idx]
        stack = input_frames[:,:,ref_idx-half_window_size:ref_idx+half_window_size+1]

        # Align frames to *centre* frame
        logging.info('Aligning frames')
        aligned_frames = align(stack, reference_frame)

        # Register frames
        logging.info('Registering frames')
        registration_reference = reference_frame
        registered_frames = register(aligned_frames, registration_reference, nlevels)

        # Transform registered frames
        lowpasses, highpasses = transform_frames(registered_frames, nlevels)

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
        mean_mags, median_mags, ninety_mags, max_inlier_mags = [], [], [], []
        for level_sb in highpasses:
            mags = np.abs(level_sb)

            mean_mags.append(np.mean(mags, axis=3))
            median_mags.append(np.median(mags, axis=3))
            ninety_mags.append(np.percentile(mags, 90, axis=3))

            thresh = 2*np.repeat(np.median(mags, axis=3)[:,:,:,np.newaxis], level_sb.shape[3], axis=3)
            outlier_suppressed = np.where(mags < thresh, mags, 0)
            max_inlier_mags.append(np.max(outlier_suppressed, axis=3))

        # Reconstruct frames
        logging.info('Computing fused image')
        mean_recon = reconstruct(lowpass_mean,
                tuple(mag*phase for mag, phase in zip(mean_mags, phases)))
        median_recon = reconstruct(lowpass_mean,
                tuple(mag*phase for mag, phase in zip(median_mags, phases)))
        ninety_recon = reconstruct(lowpass_mean,
                tuple(mag*phase for mag, phase in zip(ninety_mags, phases)))
        max_inlier_recon = reconstruct(lowpass_mean,
                tuple(mag*phase for mag, phase in zip(max_inlier_mags, phases)))

        if mean_frames is None:
            mean_frames = output.create_dataset('mean_fused',
                    mean_recon.shape + (output_shape[2],),
                    chunks=mean_recon.shape + (1,), compression='gzip',
                    dtype=mean_recon.dtype)
            mean_frames.attrs.create('description',
                    'Aligned, registered and wavelet fused frames (mean)'.encode('utf-8'))
            mean_frames.attrs.create('frame_count', 0)

        if median_frames is None:
            median_frames = output.create_dataset('median_fused',
                    median_recon.shape + (output_shape[2],),
                    chunks=median_recon.shape + (1,), compression='gzip',
                    dtype=median_recon.dtype)
            median_frames.attrs.create('description',
                    'Aligned, registered and wavelet fused frames (median)'.encode('utf-8'))
            median_frames.attrs.create('frame_count', 0)

        if ninety_frames is None:
            ninety_frames = output.create_dataset('ninety_fused',
                    ninety_recon.shape + (output_shape[2],),
                    chunks=ninety_recon.shape + (1,), compression='gzip',
                    dtype=ninety_recon.dtype)
            ninety_frames.attrs.create('description',
                    'Aligned, registered and wavelet fused frames (ninetieth percentile)'.encode('utf-8'))
            ninety_frames.attrs.create('frame_count', 0)

        if max_inlier_frames is None:
            max_inlier_frames = output.create_dataset('max_inlier_fused',
                    max_inlier_recon.shape + (output_shape[2],),
                    chunks=max_inlier_recon.shape + (1,), compression='gzip',
                    dtype=max_inlier_recon.dtype)
            max_inlier_frames.attrs.create('description',
                    'Aligned, registered and wavelet fused frames (max inlier)'.encode('utf-8'))
            max_inlier_frames.attrs.create('frame_count', 0)

        mean_frames[:,:,mean_frames.attrs['frame_count']] = mean_recon
        mean_frames.attrs['frame_count'] += 1
        median_frames[:,:,median_frames.attrs['frame_count']] = median_recon
        median_frames.attrs['frame_count'] += 1
        ninety_frames[:,:,ninety_frames.attrs['frame_count']] = ninety_recon
        ninety_frames.attrs['frame_count'] += 1
        max_inlier_frames[:,:,max_inlier_frames.attrs['frame_count']] = max_inlier_recon
        max_inlier_frames.attrs['frame_count'] += 1

        logging.info('Computing fused image w/ shrinkage')
        mean_shrink_recon = reconstruct(lowpass_mean,
                shrink_coeffs(tuple(mag*phase for mag, phase in zip(mean_mags, phases))))
        median_shrink_recon = reconstruct(lowpass_mean,
                shrink_coeffs(tuple(mag*phase for mag, phase in zip(median_mags, phases))))
        ninety_shrink_recon = reconstruct(lowpass_mean,
                shrink_coeffs(tuple(mag*phase for mag, phase in zip(ninety_mags, phases))))
        max_inlier_shrink_recon = reconstruct(lowpass_mean,
                shrink_coeffs(tuple(mag*phase for mag, phase in zip(max_inlier_mags, phases))))

        if mean_shrink_frames is None:
            mean_shrink_frames = output.create_dataset('mean_shrink',
                    mean_shrink_recon.shape + (output_shape[2],),
                    chunks=output_shape[:2] + (1,), compression='gzip',
                    dtype=mean_shrink_recon.dtype)
            mean_shrink_frames.attrs.create('description',
                    'Fused frames after wavelet shrinkage (mean)'.encode('utf-8'))
            mean_shrink_frames.attrs.create('frame_count', 0)

        if median_shrink_frames is None:
            median_shrink_frames = output.create_dataset('median_shrink',
                    median_shrink_recon.shape + (output_shape[2],),
                    chunks=output_shape[:2] + (1,), compression='gzip',
                    dtype=median_shrink_recon.dtype)
            median_shrink_frames.attrs.create('description',
                    'Fused frames after wavelet shrinkage (median)'.encode('utf-8'))
            median_shrink_frames.attrs.create('frame_count', 0)

        if ninety_shrink_frames is None:
            ninety_shrink_frames = output.create_dataset('ninety_shrink',
                    ninety_shrink_recon.shape + (output_shape[2],),
                    chunks=output_shape[:2] + (1,), compression='gzip',
                    dtype=ninety_shrink_recon.dtype)
            ninety_shrink_frames.attrs.create('description',
                    'Fused frames after wavelet shrinkage (ninetieth percentile)'.encode('utf-8'))
            ninety_shrink_frames.attrs.create('frame_count', 0)

        if max_inlier_shrink_frames is None:
            max_inlier_shrink_frames = output.create_dataset('max_inlier_shrink',
                    max_inlier_shrink_recon.shape + (output_shape[2],),
                    chunks=output_shape[:2] + (1,), compression='gzip',
                    dtype=max_inlier_shrink_recon.dtype)
            max_inlier_shrink_frames.attrs.create('description',
                    'Fused frames after wavelet shrinkage (max inlier)'.encode('utf-8'))
            max_inlier_shrink_frames.attrs.create('frame_count', 0)

        mean_shrink_frames[:,:,mean_shrink_frames.attrs['frame_count']] = mean_shrink_recon
        mean_shrink_frames.attrs['frame_count'] += 1
        median_shrink_frames[:,:,median_shrink_frames.attrs['frame_count']] = median_shrink_recon
        median_shrink_frames.attrs['frame_count'] += 1
        ninety_shrink_frames[:,:,ninety_shrink_frames.attrs['frame_count']] = ninety_shrink_recon
        ninety_shrink_frames.attrs['frame_count'] += 1
        max_inlier_shrink_frames[:,:,max_inlier_shrink_frames.attrs['frame_count']] = \
                max_inlier_shrink_recon
        max_inlier_shrink_frames.attrs['frame_count'] += 1
