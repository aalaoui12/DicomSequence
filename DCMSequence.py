import os
import numpy as np
import pydicom
import glob
import cv2
import matplotlib.pyplot as plt
from scipy import interpolate


# SLICE VIEWER
def multi_slice_viewer(volume):
    """
    Go slice-by-slice through a sequence of images stacked as a volume. Recommended to
    use for viewing the output of DcmSequence.interpolate_volume()
    :param volume: Stack of images
    :return: None
    """
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] - 1
    ax.set_title(str(ax.index))
    ax.imshow(volume[ax.index], cmap='gray')
    fig.canvas.mpl_connect('key_press_event', process_key)


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    ax.set_title(str(ax.index))
    fig.canvas.draw()


def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


# #---------------------------------------------------------------------------------# #
# #---------------------------------------------------------------------------------# #
# IMAGE PROCESSING FUNCTIONS

def convert_int_to_uint(img):
    """
    Conversion of int16 to uint16
    :param img: numpy array to convert
    :return: numpy array as type uint16
    """
    if img.dtype == np.int16:
        img_min = np.min(img)
        img += abs(img_min)
        return img.astype(np.uint16)


def apply_clahe(img, clip_lim=40, tile_grid_size=(8, 8)):
    """
    Applies CV2's clahe algorithm to an image array.
    :param img: Image to apply clahe to
    :param clip_lim: All bins in the color histogram above the limit are clipped
    and distributed to the existing bins.
    :param tile_grid_size: tile shape
    :return: Clahe image as a numpy array. Still retains whatever dtype the img started with
    """
    clahe = cv2.createCLAHE(clipLimit=clip_lim, tileGridSize=tile_grid_size)
    img = convert_int_to_uint(img)
    clahe_img = clahe.apply(img)
    return clahe_img


def apply_fiji_normalization(img):
    """
    Applies a fiji normalization to reduce the given image to a 255 range. Looks exactly
    the same as the original image.
    :param img: Image to normalize
    :return: Normalized image as an 8-bit numpy array.
    """
    img_min, img_max = int(np.min(img)), int(np.max(img))
    scale = 256. / (img_max - img_min + 1)
    x = img & 0xffff
    x -= img_min
    x[x < 0] = 0
    x = x * scale + 0.5
    x[x > 255] = 0
    return x.astype(np.uint8)


def apply_cr_normalization(img):
    """
    Applies the following normalization to reduce the image to a 0-1 range:
    img / (abs(image mean) + 3 * (image standard deviation))
    Then multiplies by 255 and clips the image between 0-255.
    :param img:
    :return: Normalized image as an 8-bit numpy array.
    """
    mu = np.mean(img)
    sigma = np.std(img)
    tmp = img / (abs(mu) + 3 * sigma)
    tmp *= 255
    uint8_img = np.clip(tmp, 0, 255).astype(np.uint8)
    return uint8_img


# #---------------------------------------------------------------------------------# #
# #---------------------------------------------------------------------------------# #


def plot_comparisons(original, cr=None, fiji=None, clahe=None, name="UNNAMED"):
    """
    Visualization of the different image processing algorithms in a 2x2 grid using
    matplotlib.
    :param original: original image
    :param cr: cr processed version of the image
    :param fiji: fiji processed version of the image
    :param clahe: clahe processed version of the image
    :param name: Name of the image. Default is UNNAMED
    :return: plotted figure
    """
    fig = plt.figure()
    plt.axis('off')
    plt.tick_params(axis='both')
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    fig.suptitle(name, fontsize=16)
    ax1.title.set_text('Original')
    ax2.title.set_text('CR')
    ax3.title.set_text('FIJI')
    ax4.title.set_text('CLAHE')

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)

    if cr is None or fiji is None or clahe is None:
        cr = apply_cr_normalization(original)
        clahe = apply_clahe(original)
        fiji = apply_fiji_normalization(original)

    ax1.imshow(original, cmap='gray')
    ax2.imshow(cr, cmap='gray')
    ax3.imshow(fiji, cmap='gray')
    ax4.imshow(clahe, cmap='gray')

    plt.show()
    return fig


# #---------------------------------------------------------------------------------# #
# #---------------------------------------------------------------------------------# #


class DcmSequence:

    def __init__(self):
        """
        Can either load in dicoms and masks with their respective load functions or
        feed them to the constructor in lists. If the length of the image list is not equal
        to the length of the filename list then the difference is made up by appending None
        to the images or appending a number string to the files.
        """

        self.dcm_files = []
        self.collection = []
        self.mask_files = []
        self.masks = []

    def load_dcm(self, src):
        """
        Add a dicom to the collection
        :param src: Source directory to read the dicoms in from.
        :return: None
        """
        for file in sorted(glob.glob(os.path.normpath(src + "/*.dcm"))):
            if file not in self.dcm_files:
                ds = pydicom.dcmread(file)
                self.collection.append(ds)
                self.dcm_files.append(file)

    def save_dcm(self, dest):
        """
        Save the dicom to the destination folder.
        :param dest: destination folder
        :return: None
        """
        dest = os.path.normpath(dest + '/')
        for i, path in enumerate(self.dcm_files):
            filename = os.path.basename(path)
            self.collection[i].save_as(os.path.join(dest, filename))

    def load_mask(self, src):
        """
        Add a mask to the masks
        :param src: Source directory to read the masks in from.
        :return: None
        """
        for file in glob.glob(os.path.normpath(src + "/*")):
            if file not in self.mask_files:
                img = cv2.imread(file)
                self.masks.append(img)
                self.mask_files.append(file)

    def save_mask(self, dest):
        """
        Save the masks to the destination folder.
        :param dest: destination folder
        :return: None
        """
        dest = os.path.normpath(dest + '/')
        for i, path in enumerate(self.mask_files):
            filename = os.path.basename(path)
            self.collection[i].save_as(os.path.join(dest, filename))

    def remove_dcm(self, **kwargs):  # works, useful with for loop of files
        """
        Remove a dicom and path from the collection.
        :param kwargs: Expecting to receive name or idx of file to remove
        :return: None
        """
        if 'name' not in kwargs.keys() and 'idx' not in kwargs.keys():
            raise KeyError('Expected either filename or index to delete')
        elif 'name' in kwargs.keys():
            name = kwargs['name']
            if name not in self.dcm_files:
                raise ValueError(name + ' is not in files')
            else:
                idx = self.dcm_files.index(name)
                self.dcm_files.remove(name)
                self.collection.remove(self.collection[idx])
        elif 'idx' in kwargs.keys():
            idx = kwargs['idx']
            if idx >= len(self.dcm_files):
                raise ValueError('Index out of bounds')
            else:
                self.dcm_files.remove(self.dcm_files[idx])
                self.collection.remove(self.collection[idx])

    def resize(self, dim):
        """
        Resize all dicoms and masks to the same dimension.
        :param dim: desired dimension of images (rows, columns)
        :return: None
        """
        for i in range(len(self.dcm_files)):
            ds = self.collection[i]
            image = ds.pixel_array
            downsampled = cv2.resize(image, dim)
            ds.PixelData = downsampled.tobytes()
            ds.Rows, ds.Columns = downsampled.shape

            self.collection[i] = ds

        for i in range(len(self.mask_files)):
            img = self.masks[i]
            img = cv2.resize(img, dim)
            self.masks[i] = img

    def imshow(self, start=0, end=None):
        """
        View the dicom images in the collection from start to end indices. Press enter
        to view next image.
        :param start: Which index to start at in the collection.
        :param end: Which index to stop at in the collection (exclusive). Leave empty to view to the
        end.
        :return: None
        """

        if end is None:
            clip_lim = 40
            tile_grid_size = (8, 8)

            for i in range(len(self.collection)):
                img = self.collection[i].pixel_array
                img_cr = apply_cr_normalization(img)
                img_clahe = apply_clahe(img, clip_lim=clip_lim, tile_grid_size=tile_grid_size)
                img_fiji = apply_fiji_normalization(img)

                name = self.dcm_files[i]
                name = os.path.basename(name)

                plot_comparisons(img, cr=img_cr, fiji=img_fiji, clahe=img_clahe, name=name)
                print('Press q to close this plot and view next image')
                plt.waitforbuttonpress()

        elif isinstance(end, int):
            clip_lim = 40
            tile_grid_size = (8, 8)

            for i in range(start, end):
                img = self.collection[i].pixel_array
                img_cr = apply_cr_normalization(img)
                img_clahe = apply_clahe(img, clip_lim=clip_lim, tile_grid_size=tile_grid_size)
                img_fiji = apply_fiji_normalization(img)

                name = self.dcm_files[i]
                name = os.path.basename(name)

                plot_comparisons(img, cr=img_cr, fiji=img_fiji, clahe=img_clahe, name=name)
                print('Press q to close this plot and view next image')
                plt.waitforbuttonpress()

        else:
            raise ValueError('end must be None or int')

    def interpolate_volume(self, num_slices=4, clahe=False, norm_alg=1):
        """
        Create an interpolated volume from the image stack. This will interpolate slices of
        images between every consecutive pair of slices. The num_slices determines how
        many interpolated slices are between the original slices and the separation between them.
        :param num_slices: Number of interpolated slices between the original slices
        :param clahe: whether or not to perform clahe on the images
        :param norm_alg: which normalization algorithm to use to get the image between 0-255.
        If using clahe, recommended to set norm_alg = 0. norm_alg = 1 is for the fiji
        normalization. norm_alg = 2 is for CR normalization.
        :return: the entire interpolated volume
        """
        _, images = self.get_png(clahe=clahe, norm_alg=norm_alg)
        volume = np.array(images)

        depth, img_width, img_height = volume.shape

        # set up interpolator
        points = (np.arange(depth), np.arange(img_height), np.arange(img_width))  # (z, y, x)
        rgi = interpolate.RegularGridInterpolator(points, volume)

        # get slices with separation of 1/(num_slices + 1)
        g = np.mgrid[1:num_slices + 1, :img_height, :img_width]
        coords = np.vstack(map(np.ravel, g)).transpose().astype(np.float16)
        coords[:, 0] *= 1 / (num_slices + 1)

        stack = np.zeros((depth + num_slices * (depth - 1), img_height, img_width), dtype=np.uint8)

        # visualize whole volume as slices
        for n in range(depth):
            stack[n * (num_slices + 1)] = volume[n]
            print("SLICE NUMBER:", n)
            if n < depth - 1:
                interp_slices = rgi(coords).reshape((num_slices, img_height, img_width)).astype(np.uint8)
                for i in range(num_slices):
                    print("\t", i)
                    stack[n * (num_slices + 1) + i + 1] = interp_slices[i]
                coords[:, 0] += 1

        return stack

    def get_png(self, clahe=False, norm_alg=1):
        """
        Get list of png images and list of file names by converting the current dicoms in the
        collection to 8-bit using the preferred norm-alg.
        :param clahe: whether or not to perform clahe on the images
        :param norm_alg: which normalization algorithm to use to get the image between 0-255.
        If using clahe, recommended to set norm_alg = 0. norm_alg = 1 is for the fiji
        normalization. norm_alg = 2 is for CR normalization.
        :return: List of file names (not the path), list of png images
        """
        # use if statements for the 3 dif algorithms for normalization
        png_names = []
        images = []

        for i, path in enumerate(self.dcm_files):
            png_names.append(os.path.splitext(os.path.basename(path))[0] + '.png')

            if self.collection[i].pixel_array.dtype == np.uint8:
                images.append(self.collection[i].pixel_array)

            elif self.collection[i].pixel_array.dtype == np.uint16 \
                    or self.collection[i].pixel_array.dtype == np.int16:
                image = self.collection[i].pixel_array
                image = convert_int_to_uint(image)
                if clahe:
                    clip_lim = 40
                    tile_grid_size = (8, 8)
                    image = apply_clahe(image, clip_lim, tile_grid_size)

                if norm_alg == 0:
                    image = np.uint8((image / np.max(image)) * 255)
                elif norm_alg == 1:
                    image = apply_fiji_normalization(image)
                elif norm_alg == 2:
                    image = apply_cr_normalization(image)

                images.append(image)

        return png_names, images

    def convert_to_8bit(self, clahe=False, norm_alg=1):
        """
        Convert 16-bit dicoms to 8-bit dicoms.
        :clahe: whether or not to use the CLAHE algorithm on the image beforehand
        :norm_alg: which normalization algorithm to use to get the image between 0-255.
        If using clahe, recommended to set norm_alg = 0. norm_alg = 1 is for the fiji
        normalization. norm_alg = 2 is for CR normalization.
        :return: None
        """
        for i, path in enumerate(self.dcm_files):
            ds = self.collection[i]

            if ds.pixel_array.dtype == np.uint16 or ds.pixel_array.dtype == np.int16:
                name = os.path.basename(path)
                new_ds = self.get_new_ds(ds, name)
                image = ds.pixel_array
                image = convert_int_to_uint(image)
                if clahe:
                    clip_lim = 40
                    tile_grid_size = (8, 8)
                    image = apply_clahe(image, clip_lim, tile_grid_size)

                if norm_alg == 0:
                    image = np.uint8((image / np.max(image)) * 255)
                elif norm_alg == 1:
                    image = apply_fiji_normalization(image)
                elif norm_alg == 2:
                    image = apply_cr_normalization(image)

                new_ds.PixelData = image.tobytes()

                self.collection[i] = new_ds

    @staticmethod
    def get_new_ds(ds, name):
        """
        Create a new dicom header. Creates a new dataset object and sets all existing
        fields in ds.
        :param ds: old dataset to base this new one off of
        :param name: name for this file
        :return: The new dicom dataset object with all header information. No pixel data.
        """
        file_meta = ds.file_meta

        new_ds = pydicom.dataset.FileDataset(name, {}, file_meta=file_meta,
                                             preamble=b"\0" * 128, is_implicit_VR=ds.is_implicit_VR,
                                             is_little_endian=ds.is_little_endian)

        a = [attr for attr in dir(ds) if not attr.startswith('__') and not callable(getattr(ds, attr))]

        for attr in a:
            if attr in ['_character_set', 'is_original_encoding', 'pixel_array',
                        'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation',
                        'PixelData']:
                continue
            else:
                new_ds.__setattr__(attr, ds.__getattr__(attr))

        # fix bits
        new_ds.BitsAllocated = 8
        new_ds.BitsStored = 8
        new_ds.HighBit = 7
        new_ds.PixelRepresentation = 0

        return new_ds
