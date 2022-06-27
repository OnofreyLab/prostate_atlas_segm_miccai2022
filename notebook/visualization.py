import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from skimage import measure


from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.metrics import compute_meandice
from monai.metrics import compute_hausdorff_distance
from monai.metrics import compute_average_surface_distance


def get_bounding_box(image):
    """Get the n-dimensional bounding box for a mask image.

    Args:
        image (numpy array): 

    Returns:
        list of ints: the bounding box indexes.
    """
    n = image.ndim
    out = []
    for ax in itertools.combinations(reversed(range(n)), n-1):
        nonzero = np.any(image, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return list(out)



def plot_contour(image, level=[1], threshold=0.5, alpha=1.0, color=None, cmap_name='tab10'):
    """Plot image segmentation mask as a contour.

    """

    cmap = cm.get_cmap(cmap_name)

    for i in range(len(level)):
        current_color = color
        if current_color is None:
            current_color = cmap(i)

        M = image==level[i]
        contours = measure.find_contours(M, threshold)
        for n, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color=current_color, alpha=alpha)

            

def plot_segm_contour_slices(background, 
                             segm=None, 
                             ground_truth=None, 
                             slice_idx=None, display_values=None, alpha=1.0, cmap_name=None,
                             rotate=False
                            ):
    """ Plot 3D image as slices with optional segmentation results and ground-truth labels.
    
    This function allows for two sets of optional labels: (i) segmentation labels and (ii) ground-truth 
    annotations.
    
    """
    
    cmap = colors.ListedColormap(['red'])
    if cmap_name is not None:
        cmap = cm.get_cmap(cmap_name)
    

    B = background
    S = segm
    G = ground_truth
    if rotate:
        B = np.rot90(B)
        if S is not None:
            S = np.rot90(S)
        if G is not None:
            G = np.rot90(G)
    
    dims = np.array(B.shape)
    if slice_idx is None:
        if len(dims)>2:
            slice_idx = [dims[2]//2]
        else:
            slice_idx = [0]


    if display_values is None:
        unique_segm_values = np.unique(segm)
        segm_values = unique_segm_values[unique_segm_values != 0]
    else:
        segm_values = display_values
            
            
    for i, idx in enumerate(slice_idx):
        
        ax = plt.subplot(1, len(slice_idx), i+1)
        ax.clear()
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(B[...,idx], cmap=plt.cm.gray, norm=colors.Normalize(vmin=np.min(B), vmax=np.max(B)))

        if ground_truth is not None:
            for j, v in enumerate(segm_values):
                j_val = (j*2)+1
                plot_contour(G[...,idx], level=[v], color=cmap(j_val), alpha=alpha)
        
        if segm is not None:
            for j, v in enumerate(segm_values):
                j_val = j*2
                plot_contour(S[...,idx], level=[v], color=cmap(j_val), alpha=alpha)

    return ax



def compute_evaluation_metrics(y_pred, y):

    value = compute_meandice(y_pred=y_pred, y=y, include_background=False)
    dice_values = value[0].cpu().numpy()

    value = compute_hausdorff_distance(y_pred, y, percentile=95)
    hd_values = value[0].cpu().numpy()

    value = compute_average_surface_distance(y_pred, y)
    mad_values = value[0].cpu().numpy()

    return dice_values, hd_values, mad_values



def compute_pirads_zone_bounds(y):
    
    bbox = get_bounding_box(y)
    z_min = bbox[-2]
    z_max = bbox[-1]
    
    slices = np.arange(z_min, z_max)

    mid_start = np.int(np.percentile(slices,25))
    mid_stop = np.int(np.percentile(slices,75))
    
    wg =   [0, y.shape[-1]]
    apex = [z_min, mid_start]
    mid =  [mid_start, mid_stop]
    base = [mid_stop, y.shape[-1]]
    
    zone_bounds = list()
    zone_bounds.append(wg)
    zone_bounds.append(apex)
    zone_bounds.append(mid)
    zone_bounds.append(base)

    zone_names = ['WG','Apex','Mid','Base']
    
    return zone_bounds, zone_names


def plot_segmentation_results(I, S, G, num_slices=7):
    """Plot the segmentation results
    """

    bbox = get_bounding_box(G)
    slices = np.linspace(bbox[4], bbox[5], num_slices, dtype=np.int)

    plt.figure('result', (20,5))
    ax = plot_segm_contour_slices(
        background=I, 
        segm=S, 
        ground_truth=G,
        slice_idx=slices, display_values=None, alpha=1.0, cmap_name='tab20', rotate=True)
    plt.show()


    slices = np.linspace(bbox[2], bbox[3], num_slices, dtype=np.int)

    plt.figure('result', (20,5))
    ax = plot_segm_contour_slices(
        background=np.transpose(I,[0,2,1]), 
        segm=np.transpose(S,[0,2,1]), 
        ground_truth=np.transpose(G,[0,2,1]),
        slice_idx=slices, display_values=None, alpha=1.0, cmap_name='tab20', rotate=True)
    plt.show()


    slices = np.linspace(bbox[0], bbox[1], num_slices, dtype=np.int)

    plt.figure('result', (20,5))
    ax = plot_segm_contour_slices(
        background=np.transpose(I,[1,2,0]), 
        segm=np.transpose(S,[1,2,0]), 
        ground_truth=np.transpose(G,[1,2,0]),
        slice_idx=slices, display_values=None, alpha=1.0, cmap_name='tab20', rotate=True)
    plt.show()    
    
    