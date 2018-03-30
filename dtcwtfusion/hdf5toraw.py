# vim: set fileencoding=utf8 :

"""Convert output from fusevideo to raw video frames.

Usage:
    hdf5toraw [options] <hdf5> [<output>]
    hdf5toraw (-h | --help)

Options:
    -v, --verbose                       Increase logging verbosity.
    --write-input=FILE                  Save input frames to FILE
    -c, --comparison                    Generate side-by-side comparison
    --fps=NUM                           Set frames per second. [default: 10]
    -g NAME, --group=NAME               Which group from the HDF5 should be rendered
                                        [default: median_shrink]

    <hdf5>                              HDF5 file as produced by fusevideo.
    <output>                            If specified, file to write output to. If not
                                        specified, write to standard output.

Video encoding:

    -f, --ffmpeg                        Use ffmpeg to encode raw data. In which
                                        case, <output> becomes the file which
                                        ffmpeg encodes to.
    --ffmpeg-binary=FILE                Use FILE as the ffmpeg binary. If not
                                        an absolute path, it is resolved using
                                        the current PATH value.
                                        [default: ffmpeg]
    --bitrate=BITRATE                   Bitrate to pass to ffmpeg binary. [default: 10M]
    --fps=FPS                           Frame rate. [default: 10]
    --ffmpeg-extra-opts=OPTS            Extra options to pass to the ffmpeg binary.

"""

from __future__ import print_function, division, absolute_import

import itertools
import logging
import shlex
import subprocess
import sys

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

def output_shape_for_frames(left, right=None):
    if right is None:
        output_shape = left.shape[:2]
    else:
        output_shape = (
                max(left.shape[0], right.shape[0]),
                left.shape[1] + right.shape[1]
        )
    return output_shape

def write_output(fobj, fps, left, right=None, leftidxs=None, rightidxs=None):
    output_shape = output_shape_for_frames(left, right)

    if right is None:
        n_frames = left.shape[2]
    else:
        n_frames = min(left.shape[2], right.shape[2])

    output_frame = np.zeros(output_shape, dtype=np.uint8)

    if leftidxs is None:
        leftidxs = np.arange(left.shape[2])
    if rightidxs is None:
        if right is not None:
            rightidxs = np.arange(right.shape[2])
        else:
            rightidxs = leftidxs

    rgba_output = np.ones(output_frame.shape + (3,), dtype=np.uint8)
    for lidx, ridx in zip(leftidxs, rightidxs):
        output_frame[:left.shape[0], :left.shape[1]] = tonemap(left[:,:,lidx])

        if right is not None:
            output_frame[:right.shape[0], left.shape[1]:(left.shape[1]+right.shape[1])] = \
                    tonemap(right[:,:,ridx])

        for cidx in xrange(3):
            rgba_output[:,:,cidx] = output_frame

        fobj.write(np.ravel(rgba_output, order='C').tostring())

def my_open(filename):
    """Like open() except that a) the file is opened in binary mode and b) if
    filename is '-' or None then stdout is returned.
    """
    if filename == '-' or filename is None:
        return sys.stdout.buffer
    return open(filename, 'wb')

def main():
    options = docopt(__doc__)
    fps = int(options['--fps'])

    # Set up logging according to command line options
    loglevel = logging.INFO if options['--verbose'] else logging.WARN
    logging.basicConfig(level=loglevel)

    logging.info('Opening "{0}"'.format(options['<hdf5>']))
    h5 = h5py.File(options['<hdf5>'])

    input_frames = h5['input']
    denoised_frames = h5[options['--group']]

    if options['--write-input'] is not None:
        logging.info('Writing input frames to "{0}"'.format(options['--write-input']))

        write_output(my_open(options['--write-input']), fps, input_frames)

    logging.info('Writing "{1}" frames to "{0}"'.format(options['<output>'], options['--group']))

    if options['--comparison']:
        left_frames, right_frames = input_frames, denoised_frames
        left_idxs = h5['processed_indices']
    else:
        left_frames, right_frames = denoised_frames, None
        left_idxs = None

    # Handle wiring up ffmpeg
    if options['--ffmpeg']:
        # Base FFMPEG command
        ffmpeg_cmd = [
            options['--ffmpeg-binary'],
            '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', '{0[1]}x{0[0]}'.format(output_shape_for_frames(left_frames, right_frames)),
            '-i', '-', '-r', str(int(options['--fps'])), '-vb', options['--bitrate'],
        ]

        # Extend with any extra options
        if options['--ffmpeg-extra-opts'] is not None:
            ffmpeg_cmd.extend(shlex.split(options['--ffmpeg-extra-opts']))

        # Write to output
        ffmpeg_cmd.append(options['<output>'] if options['<output>'] is not None else '-')

        logging.info('Using FFMPEG command line: {0}'.format(ffmpeg_cmd))

        ffmpeg_subp = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        output_fobj = ffmpeg_subp.stdin
    else:
        output_fobj = my_open(options['<output>'])

    write_output(output_fobj, fps, left=left_frames, right=right_frames, leftidxs=left_idxs)

    if ffmpeg_subp is not None:
        output_fobj.close()
        ffmpeg_subp.wait()
