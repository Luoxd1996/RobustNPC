import numpy as np
import random
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from skimage import exposure
import matplotlib.pyplot as plt
try:
    from scipy.special import comb
except:
    from scipy.misc import comb
import SimpleITK as sitk


def withou_augmentation(image):
    return image

def random_noise(image, mean=0.0, var=1):
    gaussian_noise = np.random.normal(mean, var, (image.shape[0], image.shape[1], image.shape[2]))
    image = image + gaussian_noise
    return image

def random_rescale_intensity(image):
    image = exposure.rescale_intensity(image)
    return image


def random_equalize_hist(image):
    image = exposure.equalize_hist(image)
    return image


def random_sharpening(image):
    blurred = ndimage.gaussian_filter(image, 3)
    blurred_filter = ndimage.gaussian_filter(blurred, 1)
    alpha = random.randrange(1, 10)
    image = blurred + alpha * (blurred - blurred_filter)
    return image


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(x, prob=0.5):
    points = [[-1, -1], [random.random(), -random.random()], [-random.random(), random.random()], [1, 1]]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    print(xvals, yvals)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image
    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    return interp_t_values[bin_idx].reshape(oldshape)
    
def augmentation_invariant_transformtion(image, label):
    # images, labels have been augmented (spatial) via batchgenerator
    # without using hist_match, batch size is set to 1 or just using a patch
    image_cpu = image[0].squeeze(1).cpu().numpy()# batch size (1) * d * w * h
    func = random.sample([withou_augmentation, random_noise, random_equalize_hist, random_rescale_intensity, random_sharpening, nonlinear_transformation], 2)
    aug_image_one, aug_image_two = func[0](image_cpu), func[1](image_cpu)
    # using hist_match coming soon, batch size is set to 2
    return aug_image_one.unsqueeze(0).unsqueeze(1), aug_image_two.unsqueeze(0).unsqueeze(1), image[0].unsqueeze(0)
