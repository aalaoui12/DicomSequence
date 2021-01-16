import os
import numpy as np
import pydicom
import glob
import cv2
import matplotlib.pyplot as plt


def convert_int_to_uint(img):
    if img.dtype == np.int16:
        img_min = np.min(img)
        img += abs(img_min)
        return img.astype(np.uint16)


def plot_comparisons(name, original, cr, fiji, clahe):
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

    ax1.imshow(original, cmap='gray')
    ax2.imshow(cr, cmap='gray')
    ax3.imshow(fiji, cmap='gray')
    ax4.imshow(clahe, cmap='gray')

    plt.show()
    return fig


def apply_clahe(img, clip_lim=40, tile_grid_size=(8, 8)):
    '''
    Applies CV2's clahe algorithm to an image array. Helps with
    :param img: Image to apply clahe to
    :param clip_lim: All bins in the color histogram above the limit are clipped
    and distributed to the existing bins.
    :param tile_grid_size: tile shape
    :return: Clahe image as a numpy array. Still retains whatever dtype the img started with
    '''
    clahe = cv2.createCLAHE(clipLimit=clip_lim, tileGridSize=tile_grid_size)
    img = convert_int_to_uint(img)
    clahe_img = clahe.apply(img)
    return clahe_img


def apply_linear_normalization(img):
    '''
    Applies a linear normalization to reduce the given image to a 255 range. Looks exactly
    the same as the original image.
    :param img: Image to normalize
    :return: Normalized image as an 8-bit numpy array.
    '''
    img_min, img_max = int(np.min(img)), int(np.max(img))
    scale = 256. / (img_max - img_min + 1)
    x = img & 0xffff
    x -= img_min
    x[x < 0] = 0
    x = x * scale + 0.5
    x[x > 255] = 0
    return x.astype(np.uint8)


def apply_cr_normalization(img):
    '''
    Applies the following normalization to reduce the image to a 0-1 range:
    img / (abs(image mean) + 3 * (image standard deviation))
    Then multiplies by 255 and clips the image between 0-255.
    :param img:
    :return: Normalized image as an 8-bit numpy array.
    '''
    mu = np.mean(img)
    sigma = np.std(img)
    tmp = img / (abs(mu) + 3 * sigma)
    tmp *= 255
    uint8_img = np.clip(tmp, 0, 255).astype(np.uint8)
    return uint8_img


class DcmSequence:

    def __init__(self, dcm_files=None, collection=None, mask_files=None, masks=None):
        '''
        Can either load in dicoms and masks with their respective load functions or
        feed them to the constructor in lists. If the length of the image list is not equal
        to the length of the filename list then the difference is made up by appending None
        to the images or appending a number string to the files.
        :param dcm_files: list of path names for dicoms
        :param collection: list of dicom images
        :param mask_files: list of path names for masks
        :param masks: list of mask images
        '''
        if dcm_files is None:
            dcm_files = []
        if collection is None:
            collection = []
        if mask_files is None:
            mask_files = []
        if masks is None:
            masks = []

        if len(dcm_files) > len(collection):
            for i in range(len(collection), len(dcm_files)):
                collection.append(None)
        if len(dcm_files) < len(collection):
            for i in range(len(dcm_files), len(collection)):
                dcm_files.append(str(i))

        self.dcm_files = dcm_files
        self.collection = collection

        if len(mask_files) > len(masks):
            for i in range(len(masks), len(mask_files)):
                masks.append(None)
        if len(mask_files) < len(masks):
            for i in range(len(mask_files), len(masks)):
                mask_files.append(str(i))

        self.mask_files = mask_files
        self.masks = masks


    def load_dcm(self, src):
        '''
        Add a dicom to the collection
        :param src: Source directory to read the dicoms in from.
        :return: None
        '''
        for file in glob.glob(os.path.normpath(src + "/*.dcm")):
            if file not in self.dcm_files:
                ds = pydicom.dcmread(file)
                self.collection.append(ds)
                self.dcm_files.append(file)


    def save_dcm(self, dest):
        '''
        Save the dicom to the destination folder.
        :param dest: destination folder
        :return: None
        '''
        dest = os.path.normpath(dest + '/')
        for i, path in enumerate(self.dcm_files):
            filename = os.path.basename(path)
            self.collection[i].save_as(os.path.join(dest, filename))


    def load_mask(self, src):
        '''
        Add a mask to the masks
        :param src: Source directory to read the masks in from.
        :return: None
        '''
        for file in glob.glob(os.path.normpath(src + "/*")):
            if file not in self.mask_files:
                img = cv2.imread(file)
                self.masks.append(img)
                self.mask_files.append(file)


    def save_mask(self, dest):
        '''
        Save the masks to the destination folder.
        :param dest: destination folder
        :return: None
        '''
        dest = os.path.normpath(dest + '/')
        for i, path in enumerate(self.mask_files):
            filename = os.path.basename(path)
            self.collection[i].save_as(os.path.join(dest, filename))


    def remove_dcm(self, **kwargs):  # works, useful with for loop of files
        '''
        Remove a dicom and path from the collection.
        :param kwargs: Expecting to receive name or idx of file to remove
        :return: None
        '''
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
        '''
        Resize all dicoms and masks to the same dimension.
        :param dim: desired dimension of images (rows, columns)
        :return: None
        '''
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


    def imshow(self, start=0, end=None, **kwargs):
        '''
        View the dicom images in the collection from start to end indices. Press enter
        to view next image.
        :param start: Which index to start at in the collection.
        :param end: Which index to stop at in the collection (exclusive). Leave empty to view to the
        end.
        :param kwargs: includes clip_lim and tile_grid_size for apply_clahe function
        :return: None
        '''

        if end is None:
            clip_lim = 40
            tile_grid_size = (8, 8)
            if 'clip_lim' in kwargs.keys():
                clip_lim = kwargs['clip_lim']
            if 'tile_grid_size' in kwargs.keys():
                tile_grid_size = kwargs['tile_grid_size']

            for i in range(len(self.collection)):
                img = self.collection[i].pixel_array
                img_cr = apply_cr_normalization(img)
                img_clahe = apply_clahe(img)
                img_linear = apply_linear_normalization(img)

                name = self.dcm_files[i]
                name = os.path.basename(name)

                fig = plot_comparisons(name, img, img_cr, img_linear, img_clahe)
                print('Press q to close this plot and view next image')
                plt.waitforbuttonpress()

        elif isinstance(end, int):
            clip_lim = 40
            tile_grid_size = (8, 8)
            if 'clip_lim' in kwargs.keys():
                clip_lim = kwargs['clip_lim']
            if 'tile_grid_size' in kwargs.keys():
                tile_grid_size = kwargs['tile_grid_size']

            for i in range(start, end):
                img = self.collection[i].pixel_array
                img_cr = apply_cr_normalization(img)
                img_clahe = apply_clahe(img)
                img_linear = apply_linear_normalization(img)

                name = self.dcm_files[i]
                name = os.path.basename(name)

                fig = plot_comparisons(name, img, img_cr, img_linear, img_clahe)
                print('Press q to close this plot and view next image')
                plt.waitforbuttonpress()

        else:
            raise ValueError('end must be None or int')


    def get_png(self, clahe=False, norm_alg=0, **kwargs):
        '''
        Get list of png images and list of file names by converting the current dicoms in the
        collection to 8-bit using the preferred norm-alg.
        :param clahe: whether or not to perform clahe on the images
        :norm_alg: which normalization algorithm to use to get the image between 0-255.
        If using clahe, recommended to set norm_alg = 0. norm_alg = 1 is for the linear
        normalization. norm_alg = 2 is for CR normalization.
        :param kwargs: includes clip_lim and tile_grid_size for apply_clahe function
        :return: List of file names (not the path), list of png images
        '''
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
                if clahe:
                    clip_lim = 40
                    tile_grid_size = (8, 8)
                    if 'clip_lim' in kwargs.keys():
                        clip_lim = kwargs['clip_lim']
                    if 'tile_grid_size' in kwargs.keys():
                        tile_grid_size = kwargs['tile_grid_size']
                    image = apply_clahe(image, clip_lim, tile_grid_size)

                if norm_alg == 0:
                    image = np.uint8((image / np.max(image)) * 255)
                elif norm_alg == 1:
                    image = apply_linear_normalization(image)
                elif norm_alg == 2:
                    image = apply_cr_normalization(image)

                images.append(image)

        return png_names, images


    def convert_to_8bit(self, clahe=False, norm_alg=1, **kwargs):
        '''
        Convert 16-bit dicoms to 8-bit dicoms.
        :clahe: whether or not to use the CLAHE algorithm on the image beforehand
        :norm_alg: which normalization algorithm to use to get the image between 0-255.
        If using clahe, recommended to set norm_alg = 0. norm_alg = 1 is for the linear
        normalization. norm_alg = 2 is for CR normalization.
        :return: None
        '''
        for i, path in enumerate(self.dcm_files):
            ds = self.collection[i]

            if ds.pixel_array.dtype == np.uint16 or ds.pixel_array.dtype == np.int16:
                name = os.path.basename(path)
                new_ds = self.get_new_ds(ds, name)
                image = ds.pixel_array
                if clahe:
                    clip_lim = 40
                    tile_grid_size = (8, 8)
                    if 'clip_lim' in kwargs.keys():
                        clip_lim = kwargs['clip_lim']
                    if 'tile_grid_size' in kwargs.keys():
                        tile_grid_size = kwargs['tile_grid_size']
                    image = apply_clahe(image, clip_lim, tile_grid_size)

                if norm_alg == 0:
                    image = np.uint8((image / np.max(image)) * 255)
                elif norm_alg == 1:
                    image = apply_linear_normalization(image)
                elif norm_alg == 2:
                    image = apply_cr_normalization(image)

                new_ds.PixelData = image.tobytes()

                self.collection[i] = new_ds


    def get_new_ds(self, ds, name):
        '''
        Create a new dicom header. Creates a new dataset object and sets all existing
        fields in ds.
        :param ds: old dataset to base this new one off of
        :param name: name for this file
        :return: The new dicom dataset object with all header information. No pixel data.
        '''
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




