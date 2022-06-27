import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from skimage import measure

from typing import Any, List, Sequence, Tuple, Union, Optional


def ensure_tuple_rep(tup: Any, dim: int) -> Tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Raises:
        ValueError: When ``tup`` is a sequence and ``tup`` length is not ``dim``.

    Examples::

        >>> ensure_tuple_rep(1, 3)
        (1, 1, 1)
        >>> ensure_tuple_rep(None, 3)
        (None, None, None)
        >>> ensure_tuple_rep('test', 3)
        ('test', 'test', 'test')
        >>> ensure_tuple_rep([1, 2, 3], 3)
        (1, 2, 3)
        >>> ensure_tuple_rep(range(3), 3)
        (0, 1, 2)
        >>> ensure_tuple_rep([1, 2], 3)
        ValueError: Sequence must have length 3, got length 2.

    """
    if isinstance(tup, np.ndarray):
        tup = tup.tolist()
    if not issequenceiterable(tup):
        return (tup,) * dim
    if len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")
            



def get_bounding_box(
    image: np.ndarray, 
) -> list:
    """
    Get the n-dimensional bounding box for a mask image. Any non-zero values are used 
    to compute the bounding box.

    Args:
        image (np.ndarray): n-d array to compute bounding box from.

    Returns:
        list of ints: the bounding box indexes.
    """
    n = image.ndim
    out = []
    for ax in itertools.combinations(reversed(range(n)), n-1):
        nonzero = np.any(image, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return list(out)



def plot_contour(
    image: np.ndarray, 
    level=[1], 
    threshold=0.5, 
#     alpha=1.0, 
    color=None, 
    cmap_name='tab10', 
#     width=1,
    **kwargs
) -> None:
    """Plot image segmentation mask as a contour.

    """

    cmap = cm.get_cmap(cmap_name)

    for i in range(len(level)):
        current_color = color
        if current_color is None:
            current_color = cmap(i)

#         M = image==level[i]
        M = image
        contours = measure.find_contours(M, threshold)
        for n, contour in enumerate(contours):
            plt.plot(
                contour[:, 1], 
                contour[:, 0], 
#                 linewidth=width, 
                color=current_color, 
#                 alpha=alpha
                **kwargs
            )

            
            

            
class PlotSegmentation():
    """
    Assumes channel first 3D arrays of shape (CHWD). For labels this assumes segmentations are in one-hot encodings,
    where the number of classes is C.
    
    Args:
        num_slices: the number of slices to display.
            Defaults to 1.
        slice_spacing: the gap in pixels between the displayed slices.
            Defaults to 1.
            
        segm: (list) of segmentation to display as contour overlays.
        cmap_name: (list) of colormaps to use.
        bbox_image: image from which to compute a bounding box to focus display on areas of interst.
        kwargs: other arguments for the `plt.plot` function.
    
    """
    
    # TODO: add option for background cmap
    def __init__(
        self, 
        slice_axis: Optional[int] = -1, 
        num_slices: Optional[int] = 1, 
        slice_spacing: Optional[int] = 1, 
        plot_title: Optional[bool] = True,
#         line_width: Optional[float] = 1.0,
#         alpha: Optional[float] = 1.0,
        rotate: Optional[bool] = True,
        labels: Union[Sequence[Optional[int]], Optional[int]] = None,
        include_background: Optional[bool] = False,
        slice_indexes: Union[Sequence[Optional[int]], Optional[int]] = None,
        **kwargs
    ) -> None:
    
        self.num_slices = num_slices
        self.slice_spacing = slice_spacing
        self.include_background = include_background
        self.slice_axis = slice_axis
        self.rotate = rotate
        self.plot_title = plot_title
#         self.line_width = line_width
#         self.alpha = alpha
        self.labels = labels # TODO
        self.slice_indexes = slice_indexes
        self.kwargs = kwargs

        
        
    def __call__(
        self, 
        image: np.ndarray, 
        segm: Union[Sequence[Optional[np.ndarray]], Optional[np.ndarray]] = None, # TODO: ensure this takes single value not just lists!
        cmap_name: Union[Sequence[Optional[str]], Optional[str]] = None,
        contour_width: Union[Sequence[Optional[str]], Optional[str]] = 1,
        bbox_image: Optional[np.ndarray] = None
    ) -> None:
        """
        Args:
            image: the background image to display.
            segm: (list) of segmentation to display as contour overlays.
            cmap_name: (list) of colormaps to use.
            bbox_image: image from which to compute a bounding box to focus display on areas of interst.
        """
        
        # Ensure that image is CHWD
        # TODO: convert 2D to 3D
#         if image.ndims < 4:
#             image.
#         print('image.shape', image.shape)
                
        # Ensure the slice axis is valid, between [0,2]
        slice_axis = self.slice_axis
        if slice_axis >= image.ndim:
            slice_axis = image.ndim-1
        if slice_axis < 0:
            slice_axis = image.ndim-1
#         print('slice_axis', slice_axis)
        
    
        n_dims = len(image.shape)
        bbox = np.array(image.shape).T
        bbox = np.dstack((np.zeros(n_dims), bbox))
        bbox = bbox.flatten().astype(np.int)
#         print('bbox', bbox)
        
        # Check if a bounding box image is provided. If it is, then compute the bounding box of 
        # the non-zero values.
        if bbox_image is not None:
            bbox = np.array(get_bounding_box(bbox_image))
#         print('bbox', bbox)
    
        
        
        # Check for label values to use
        if segm!=None:
            # Remove the background (first) channel in the segmentation, if desired
            if not self.include_background:
                for i, s in enumerate(segm):
                    segm[i] = s[1:,...]
#                     print('segm[{:d}].shape, {}'.format(i, segm[i].shape))
            
            
            cmap = [colors.ListedColormap(['red'])]*len(segm)
            if cmap_name is not None:
                cmap = list()
                for c in cmap_name:
                    cmap.append(cm.get_cmap(c))
#             print('cmap', cmap)
            
            segm_values = None
            if self.labels is None:
                segm_values = np.arange(segm[0].shape[0])
            else:
                segm_values = self.labels
            
            
        slice_axis_start = 2*slice_axis
        slice_bounds = bbox[slice_axis_start:(slice_axis_start+2)]

#         print('slice_bounds', slice_bounds)

        if self.slice_indexes is not None:
            slices = np.array(self.slice_indexes)
            slices = slices[slices>=slice_bounds[0]]
            slices = slices[slices<=slice_bounds[1]]
            
        else:
            slice_range = slice_bounds[1]-slice_bounds[0]
            mid_slice_idx = int(slice_range)/2 + slice_bounds[0]

            n_slices = slice_range
            if self.num_slices < n_slices:
                n_slices = self.num_slices


            slices = (np.arange(n_slices)-n_slices//2)*self.slice_spacing + mid_slice_idx
            slices = slices.astype(np.int)
                
#         print('slices', slices)
            
        
        
        # Get the global image range for consistent display across multiple slices
        image_min = image.min()
        image_max = image.max()
        
        
        # Create the subplots object
#         fig, ax = plt.subplots(1, len(slices))
        
        # Extract the desired slices
        for i, idx in enumerate(slices):
#             print('idx', idx)
            disp_slice = np.take(image, indices=idx, axis=slice_axis)
            
            # Get the first channel
            disp_slice = disp_slice[0,...]
            
            if self.rotate:
                disp_slice = np.rot90(disp_slice)
                
#             print('disp_slice.shape', disp_slice.shape)
                
            ax = plt.subplot(1, len(slices), i+1)
#             ax[i].imshow(
            plt.imshow(
                disp_slice, 
                vmin=image_min,
                vmax=image_max,
                cmap=plt.cm.gray,
            )
            
            if self.plot_title:
                plt.title('Slice {}'.format(idx))
            plt.axis('off')
#                 ax[i].set_title('Slice {}'.format(idx))
#             ax[i].axis('off')

            
            if segm!=None:
                # Iterate over all the segmentations in the list
                for j, s in enumerate(segm):
                    

                    segm_slice = np.take(s, indices=idx, axis=slice_axis)
#                     print('segm_slice.shape', segm_slice.shape)

                    # Iterate over all channels in the segmentation
                    for c in range(segm_slice.shape[0]):
                        s_c = segm_slice[c,...]
#                         print('c, s_c.shape', c, s_c.shape)

                        if self.rotate:
                            s_c = np.rot90(s_c)
                            
                        plot_contour(
                            s_c, 
                            level=[1], # TODO: check for floating values?
                            color=cmap[j](c),
#                             width=self.line_width,
#                             alpha=self.alpha
                            **self.kwargs
                        )

                    
        # Return the fig and ax object handles
#         return fig, ax
