
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.colors import LinearSegmentedColormap as cm


from scipy import ndimage
import matplotlib.gridspec as gridspec

from gantools import utils
import os
import warnings


def draw_images(images,
                nx=1,
                ny=1,
                axes=None,
                *args,
                **kwargs):
    """
    Draw multiple images. This function conveniently draw multiple images side
    by side.

    Parameters
    ----------
    x : List of images
        - Array  [ nx*ny , px, py ]
        - Array  [ nx*ny , px, py , 3]
        - Array  [ nx*ny , px, py , 4]
    nx : number of images to be ploted along the x axis (default = 1)
    ny : number of images to be ploted along the y axis (default = 1)
    px : number of pixel along the x axis (If the images are vectors)
    py : number of pixel along the y axis (If the images are vectors)
    axes : axes

    """
    ndim = len(images.shape)
    nimg = images.shape[0]

    if ndim == 1:
        raise ValueError('Wrong data shape')
    elif ndim == 2:
        images = np.expand_dims(np.expand_dims(images, axis=0), axis=3)
    elif ndim == 3:
        images = np.expand_dims(images, axis=3)
    elif ndim > 4:
        raise ValueError('The input contains too many dimensions')

    px, py, c = images.shape[1:]

    images_tmp = images.reshape([nimg, px, py, c])
    mat = np.zeros([nx * px, ny * py, c])
    for j in range(ny):
        for i in range(nx):
            if i + j * nx >= nimg:
                warnings.warn("Not enough images to tile the entire area!")
                break
            mat[i * px:(i + 1) * px, j * py:(
                j + 1) * py] = images_tmp[i + j * nx, ]
    # make lines to separate the different images
    # Code used to check the lines...
    #     imgs2 = np.zeros([25,32,32])
    #     imgs2[::2,:,:] =1
    #     plt.figure(figsize=(15, 15))
    #     plot.draw_images(imgs2,5,5)
    xx = []
    yy = []
    for j in range(1, ny):
        xx.append([py * j-0.5, py * j-0.5])
        yy.append([0, nx * px - 1])
    for j in range(1, nx):
        xx.append([0, ny * py - 1])
        yy.append([px * j-0.5, px * j-0.5])

    if axes is None:
        axes = plt.gca()
    axes.imshow(np.squeeze(mat), *args, **kwargs)
    for x, y in zip(xx, yy):
        axes.plot(x, y, color='r', linestyle='-', linewidth=2)
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    return axes





# Plot a line with a shade representing either the confidence interval or the standard error
# If confidence is None then the standard error is shown
def plot_with_shade(ax, x, y, label, color, confidence=None, **linestyle):
    transparency = 0.2

    n = y.shape[0]
    y_mean = np.mean(y, axis=0)
    error = (np.var(y, axis=0) / n)**0.5
    if confidence == 'std':
        error = np.std(y, axis=0)
    elif confidence is not None:
        error = error * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    ax.plot(x, y_mean, label=label, color=color, **linestyle)
    ax.fill_between(
        x, y_mean - error, y_mean + error, alpha=transparency, color=color)

    
def planck_cmap(ncolors=256):
    """
    Returns a color map similar to the one used for the "Planck CMB Map".
    Parameters
    ----------
    ncolors : int, *optional*
    Number of color segments (default: 256).
    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap instance
    Linear segmented color map.
    """
    segmentdata = {
        "red": [(0.0, 0.00, 0.00), (0.1, 0.00, 0.00), (0.2, 0.00, 0.00),
                (0.3, 0.00, 0.00), (0.4, 0.00, 0.00), (0.5, 1.00, 1.00),
                (0.6, 1.00, 1.00), (0.7, 1.00, 1.00), (0.8, 0.83, 0.83),
                (0.9, 0.67, 0.67), (1.0, 0.50, 0.50)],
        "green": [(0.0, 0.00, 0.00), (0.1, 0.00, 0.00), (0.2, 0.00, 0.00),
                  (0.3, 0.30, 0.30), (0.4, 0.70, 0.70), (0.5, 1.00, 1.00),
                  (0.6, 0.70, 0.70), (0.7, 0.30, 0.30), (0.8, 0.00, 0.00),
                  (0.9, 0.00, 0.00), (1.0, 0.00, 0.00)],
        "blue": [(0.0, 0.50, 0.50), (0.1, 0.67, 0.67), (0.2, 0.83, 0.83),
                 (0.3, 1.00, 1.00), (0.4, 1.00, 1.00), (0.5, 1.00, 1.00),
                 (0.6, 0.00, 0.00), (0.7, 0.00, 0.00), (0.8, 0.00, 0.00),
                 (0.9, 0.00, 0.00), (1.0, 0.00, 0.00)]
    }
    return cm("Planck-like", segmentdata, N=int(ncolors), gamma=1.0)

def tile_cube_to_2d(cube):
    '''
    cube = [:, :, :]
    arrange cube as tile of squares
    '''
    x_dim = cube.shape[0]
    y_dim = cube.shape[1]
    z_dim = cube.shape[2]
    v_stacks = []
    num = 0
    num_images_in_each_row = utils.num_images_each_row(x_dim)

    for i in range(x_dim//num_images_in_each_row):
        h_stacks = []
        for j in range(num_images_in_each_row): # show 'num_images_in_each_row' squares from the cube in one row
            h_stacks.append(cube[num, :, :])
            num += 1
        v_stacks.append( np.hstack(h_stacks) )

    tile = np.vstack(v_stacks)
    return tile


def tile_and_plot_3d_image(image, axis=None, **kwargs):
    '''
    Take a 3d cube as input.
    Tile the cube as slices, and display it.
    '''
    if axis is None:
        axis = plt.gca()
    tile = tile_cube_to_2d(image)
    axis.imshow(tile, **kwargs)

def cubes_to_animation(cubes, clim=None, figsize=(10,11), title=None, fontsize=24, fps=16, **kwargs):
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage
    if len(cubes.shape)<3:
        cubes = cubes.reshape([1, *cubes.shape])
    if clim is None:
        clim = (np.min(cubes[0]), np.max(cubes[0]))

    fig = plt.figure(figsize=figsize)
    
    nframe = cubes.shape[1]

    def make_frame(t):
        ind = int(round(t*fps))
        plt.cla()
        plt.imshow(cubes[0, ind, :, :], interpolation='none', clim=clim, **kwargs )
#         plt.axis('off')
        titlestr = 'Frame no. {}'.format(ind)
        if title:
            titlestr = title + ' - ' + titlestr
        plt.title(titlestr, fontsize=fontsize)

        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=nframe/fps)
    plt.clf()

    return animation

def animate_cubes(cubes, output_name='clip', output_format='gif', clim=None, 
    figsize=(10,11), title=None, fontsize=24, fps=16, **kwargs):
    animation = cubes_to_animation(cubes, clim =clim, figsize=figsize, title=title, 
        fontsize=fontsize, fps=fps, **kwargs)
    if output_format=='gif':
        if not (output_name[-4:]=='.gif'):
            output_name += '.gif'
        animation.write_gif(output_name, fps=fps)
        plt.clf()
    elif output_format=='mp4':
        if not (output_name[-4:]=='.mp4'):
            output_name += '.mp4'
        animation.write_videofile(output_name, fps=fps)
        plt.clf()
    # elif output_format=='ipython_display': 
    #     animation.ipython_display(fps=16, loop=True, autoplay=True)
    else:
        raise ValueError('Unknown output_format')

def get_animation(fig, real_cube, fake_cube, real_downsampled=None, clim = None, fps=16, axis=0, names=['Real', 'Downsampled', 'Fake'], 
    fontsize=20):
    '''
    Given real and fake 3d sample, create animation with slices along all 3 dimensions
    Return animation object
    
    By default, the figure will be in HD: 1920×1080 px
    '''
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage

    #ax = plt.axes([0,0,1,1], frameon=True)
    #plt.autoscale(tight=False)

    dim = fake_cube.shape[0]

    if real_downsampled is not None:
        dim_downsampled = real_downsampled.shape[0]
        factor = dim // dim_downsampled
        grid = (1, 3)
        if clim is None:
            cmin = min([np.min(fake_cube[:, :, :]), np.min(real_cube[:, :, :]), np.min(real_downsampled)])
            cmax = max([np.max(fake_cube[:, :, :]), np.max(real_cube[:, :, :]), np.max(real_downsampled)])
            clim =(cmin, cmax)

    else:
        grid = (1, 2)
        if clim is None:
            cmin = min([np.min(fake_cube[:, :, :]), np.min(real_cube[:, :, :])])
            cmax = max([np.max(fake_cube[:, :, :]), np.max(real_cube[:, :, :])])
            clim =(cmin, cmax)

    gridspec.GridSpec(grid[0], grid[1])
    
    duration = dim//fps
    
    def make_frame(t):

        i = 0
        ind = int(np.round(t*fps))
        plt.subplot2grid( grid, (0, i), rowspan=1, colspan=1)
        plt.imshow(real_cube[ind % dim, :, :], interpolation='nearest', cmap=plt.cm.plasma, clim=clim )
        plt.title(names[0] + ' {0}x{0}x{0}'.format(dim), fontsize=fontsize, color='white')
        i = i + 1


        if real_downsampled is not None:
            plt.subplot2grid( grid, (0, i), rowspan=1, colspan=1)
            plt.imshow(real_downsampled[(ind // factor) % dim_downsampled, :, :], interpolation='nearest', cmap=plt.cm.plasma, clim=clim )
            plt.title(names[1] + ' {0}x{0}x{0}'.format(dim_downsampled), fontsize=fontsize, color='white')
            i = i + 1


        plt.subplot2grid( grid, (0, i), rowspan=1, colspan=1)
        plt.imshow(fake_cube[ind % dim, :, :], interpolation='nearest', cmap=plt.cm.plasma, clim=clim )
        plt.title(names[-1] + ' {0}x{0}x{0}'.format(dim), fontsize=fontsize, color='white')
        plt.tight_layout()

        return mplfig_to_npimage(fig)
    

    animation = VideoClip(make_frame, duration=duration)
    return animation

def clip_title(fig, title='title', fontsize=40):
    '''
    Make a videoclip with a centered white title
    '''
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage
    
    duration = 1
    ax = plt.gca()
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    def make_frame(t):
        ax = plt.gca()
        ax = fig.add_axes([0,0,1,1])
        ax.text(0.5*(left+right), 0.5*(bottom+top), title,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=fontsize, color='white',
                transform=ax.transAxes)
        return mplfig_to_npimage(fig)
    

    animation = VideoClip(make_frame, duration=duration)
    return animation

def save_animation(real_cube, fake_cube, real_downsampled=None, figsize=(10, 6), dpi=96, fps=16, 
    format='gif', output_file_name='test', names=['real ', 'real downsampled ', 'fake '],
    fontsize=20, clim=None):
    '''
    Given real and fake 3d sample, create animation with slices along all 3 dimensions, and save it as gif.
    '''
    
    plt.style.use('dark_background')

    from moviepy.editor import concatenate_videoclips
    assert(real_cube.shape==fake_cube.shape)
    if real_downsampled is not None:
        assert(real_cube.shape[0]==real_downsampled.shape[0])
    if len(real_cube.shape)<=3:
        real_cube = np.expand_dims(real_cube, axis=0)
        fake_cube = np.expand_dims(fake_cube, axis=0)
        if real_downsampled:
            if len(real_downsampled.shape)<=3:
                real_downsampled = np.expand_dims(real_downsampled, axis=0)
    if real_downsampled is None:
        real_downsampled = [None] * len(real_cube)
    animations = []
    fig = plt.figure(figsize=figsize, dpi=dpi)

    for i in range(len(real_cube)):
        animations.append(clip_title(fig, title='Sample {}'.format(i+1), fontsize=100))
        animations.append(get_animation(fig, real_cube[i], fake_cube[i], real_downsampled[i], 
        clim=clim, fps=fps, names=names, fontsize=fontsize))
    
    animation = concatenate_videoclips(animations)
    if format == 'gif':
        animation.write_gif(output_file_name + '.gif', fps=fps)
    else:
        animation.write_videofile(output_file_name, fps=fps)
    plt.style.use('default')
    plt.clf()
    return animation
    
def plot_histogram(x, histo, yscale='log', tick_every=10, bar_width=1):
    plt.bar(np.arange(len(histo)), histo, bar_width)
    positions, labels = ([], [])
    for idx in range(len(x)):
        if idx == len(x) - 1:
            positions.append(idx)
            labels.append(np.round(x[idx], 2))
        if idx % tick_every == 0:
            positions.append(idx)
            labels.append(np.round(x[idx], 2))
    plt.xticks(positions, labels)
    plt.yscale(yscale)

    

# Comparison plot between two curves real and fake corresponding to axis x
def plot_cmp(x, fake, real=None, xscale='linear', yscale='log', xlabel="", ylabel="", ax=None, title="", shade=False, confidence=None, xlim=None, ylim=None, fractional_difference=False, algorithm='classic', loc=3):
    if ax is None:
        ax = plt.gca()
    if real is None:
        # Plot a line that will be canceled by fake
        # In this way the fractional difference is zero
        plot_single(x, fake, color='b', ax=ax, label="Real", xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel, title=title, shade=False, confidence=confidence, xlim=xlim, ylim=ylim)
    else:
        plot_single(x, real, color='b', ax=ax, label="Real", xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel, title=title, shade=shade, confidence=confidence, xlim=xlim, ylim=ylim)
    plot_single(x, fake, color='r', ax=ax, label="Fake", xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel, title=title, shade=shade, confidence=confidence, xlim=xlim, ylim=ylim)
    ax.legend(fontsize=14, loc=loc)
    if fractional_difference:
        if real is None:
            real = fake
        if shade:
            real = np.mean(real, axis=0)
            fake = np.mean(fake, axis=0)
        plot_fractional_difference(x, real, fake, xscale=xscale, yscale='linear', ax=ax, color='g', ylim=ylim[1] if isinstance(ylim, list) else None, algorithm=algorithm, loc=loc)
        
# Plot an image
def plot_img(img, x=None, title="", ax=None, cmap=plt.cm.plasma, vmin=None, vmax=None, tick_every=10, colorbar=False, log_norm=False):
    if ax is None:
        ax = plt.gca()
    img = np.squeeze(img)
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, norm=LogNorm() if log_norm else None)
    ax.set_title(title)
    ax.title.set_fontsize(16)
    if x is not None:
        # Define axes
        ticklabels = []
        for i in range(len(x)):
            if i % tick_every == 0:
                ticklabels.append(str(int(round(x[i], 0))))
        ticks = np.linspace(0, len(x) - (len(x) % tick_every), len(ticklabels))
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels)
    else:
        ax.axis('off')
    if colorbar:
        plt.colorbar(im, ax=ax)
    return im


# Plot fractional difference
# if algorithm='classic' then the classic fractional difference abs(real - fake) / real is computed
# if algorithm='relative' then abs(real-fake) / std(real) is computed
def plot_fractional_difference(x, real, fake, xlim=None, ylim=None, xscale="log", yscale="linear", title="", ax=None, color='b', algorithm='classic', loc=3):
    if ax is None:
        ax1 = plt.gca()
    else:
        ax1 = ax.twinx()
    diff = np.abs(real - fake)
    if algorithm == 'classic':
        diff = diff / real
    elif algorithm == 'relative':
        diff = diff / np.std(real, axis=0)
    else:
        raise ValueError("Unknown algorithm name " + str(algorithm))
    plot_single(x, diff, ax=ax1, xscale=xscale, yscale=yscale, title=title, color=color, label='Frac. diff.' if algorithm == 'classic' else 'Rel. diff.', ylim=ylim)
    ax1.tick_params(axis='y', labelsize=12, labelcolor=color)

    # Adjust legends
    if ax is not None:
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax1.get_legend_handles_labels()
        ax.get_legend().remove()
        ax1.legend(lines + lines2, labels + labels2, loc=loc, fontsize=14)



# Plot the scores as a heatmap on the parameter grid
# test_scores: array containing the scores for different parameters
# test_params: array of shape (n, 2) containing the parameters
# thresholds: if None a single heatmap with continuous values is produced
#             if a list is provided a subplot containing different discrete heatmaps 
#             (red points if below threshold and green otherwise) are produced
def plot_heatmap(test_scores, test_params, train_scores=None, train_params=None, vmin=0, vmax=0.25, thresholds=None, labels=['$\Omega_M$', '$\sigma_8$']):
    if thresholds is None:
        plt.scatter(test_params[:, 0], test_params[:, 1], c=test_scores, vmin=vmin, vmax=vmax, cmap=plt.cm.RdYlGn_r, edgecolor='k')
        if train_scores is not None and train_params is not None:
            plt.scatter(train_params[:, 0], train_params[:, 1], c=train_scores, vmin=vmin, vmax=vmax, cmap=plt.cm.RdYlGn_r, edgecolor='k', marker='s')
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.colorbar()
    else:
        _, ax = plt.subplots(nrows=1, ncols=len(thresholds), figsize=(len(thresholds) * 5, 5))
        for j in range(len(thresholds)):
            if train_params is not None and train_scores is not None:
                for i in range(len(train_params)):
                    ax[j].scatter(train_params[i, 0], train_params[i, 1], c='g' if train_scores[i] <= thresholds[j] else 'r', marker='s')
            for i in range(len(test_params)):
                ax[j].scatter(test_params[i, 0], test_params[i, 1], c='g' if test_scores[i] <= thresholds[j] else 'r')
            ax[j].set_xlabel(labels[0])
            ax[j].set_ylabel(labels[1])
            ax[j].set_title('Threshold: ' + str(thresholds[j]))
            
# Plot a single curve
def plot_single(x, y, color='b', ax=None, label=None, xscale='linear', yscale='log', xlabel="", ylabel="", title="", shade=False, confidence=None, xlim=None, ylim=None):
    if ax is None:
        ax = plt.gca()
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    linestyle = {
        "linewidth": 1,
        "markeredgewidth": 0,
        "markersize": 3,
        "marker": "o",
        "linestyle": "-"
    }
    if shade:
        plot_with_shade(ax, x, y, label=label, color=color, confidence=confidence, **linestyle)
    else:
        if confidence is not None:
            y = np.mean(y, axis=0)
        ax.plot(x, y, label=label, color=color, **linestyle)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        if isinstance(ylim, list):
            ylim = ylim[0]
        ax.set_ylim(ylim)
    ax.title.set_text(title + "\n")
    ax.title.set_fontsize(16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if label:
        ax.legend(fontsize=14, loc=3)

