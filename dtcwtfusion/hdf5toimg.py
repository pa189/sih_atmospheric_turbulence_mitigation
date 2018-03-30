# vim: set fileencoding=utf8 :

"""Convert output from fusevideo to an y-t slice image

Usage:
    hdf5toimg [options] <hdf5> <image>
    hdf5toimg (-h | --help)

Options:
    -v, --verbose                       Increase logging verbosity.
    -g NAME, --group=NAME               Which group from the HDF5 should be rendered
                                        [default: median_shrink]

    --column=INDEX                      Produce output for column at 0-based offset
                                        INDEX. If omitted, use central column.

    <hdf5>                              HDF5 file as produced by fusevideo.
    <image>                             If specified, file to write output to in PNG format.

"""

from __future__ import print_function, division, absolute_import

import itertools
import logging
import shlex
import subprocess
import sys

from PIL import Image
from docopt import docopt
from six.moves import xrange
import h5py
import numpy as np

def tonemap(array):
    # The normalisation strategy here is to let the middle 95% of
    # the values fall in the range 0.025 to 0.975 ('black' and 'white' level).
    black_level = np.percentile(array,  2.5)
    white_level = np.percentile(array, 97.5)

    norm_array = array - black_level
    norm_array *= 0.7 / (white_level - black_level)
    norm_array = np.clip(norm_array + 0.15, 0, 1)

    return np.array(norm_array * 255, dtype=np.uint8)

def main():
    options = docopt(__doc__)

    # Set up logging according to command line options
    loglevel = logging.INFO if options['--verbose'] else logging.WARN
    logging.basicConfig(level=loglevel)

    logging.info('Opening "{0}"'.format(options['<hdf5>']))
    h5 = h5py.File(options['<hdf5>'])

    denoised_frames = h5[options['--group']]
    output_cols = []
    logging.info('Processing {0} frames'.format(denoised_frames.shape[2]))
    for t in range(denoised_frames.shape[2]):
        frame = denoised_frames[:,:,t]

        if options['--column'] is not None:
            col_idx = int(options['--column'])
        else:
            col_idx = frame.shape[1] >> 1

        frame_col = frame[:, [col_idx,]]
        output_cols.append(frame_col)

    logging.info('Saving output to "{0}"'.format(options['<image>']))
    output_frame = tonemap(np.hstack(output_cols))
    output_im = Image.fromarray(output_frame)
    output_im.save(options['<image>'])

