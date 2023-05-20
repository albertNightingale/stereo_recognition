"""
This file holds the main code for disparity map calculations
"""
import torch
import numpy as np

from typing import Callable, Tuple


def calculate_disparity_map(left_img: torch.Tensor,
                            right_img: torch.Tensor,
                            block_size: int,
                            sim_measure_function: Callable,
                            max_search_bound: int = 50) -> torch.Tensor:
    """
    Calculate the disparity value at each pixel by searching a small 
    patch around a pixel from the left image in the right image

    Note: 
    1.  It is important for this project to follow the convention of search
        input in left image and search target in right image
    2.  While searching for disparity value for a patch, it may happen that there
        are multiple disparity values with the minimum value of the similarity
        measure. In that case we need to pick the smallest disparity value.
        Please check the numpy's argmin and pytorch's argmin carefully.
        Example:
        -- diparity_val -- | -- similarity error --
        -- 0               | 5
        -- 1               | 4
        -- 2               | 7
        -- 3               | 4
        -- 4               | 12

        In this case we need the output to be 1 and not 3.
    3. The max_search_bound is defined from the patch center.

    Args:
    -   left_img: image from the left stereo camera. Torch tensor of shape (H,W,C).
                C will be >= 1.
    -   right_img: image from the right stereo camera. Torch tensor of shape (H,W,C)
    -   block_size: the size of the block to be used for searching between
                left and right image
    -   sim_measure_function: a function to measure similarity measure between
                            two tensors of the same shape; returns the error value
    -   max_search_bound: the maximum horizontal distance (in terms of pixels) 
                            to use for searching
    Returns:
    -   disparity_map: The map of disparity values at each pixel. 
                        Tensor of shape (H-2*(block_size//2),W-2*(block_size//2))
    """

    assert left_img.shape == right_img.shape
    disparity_map = torch.zeros(1) #placeholder, this is not the actual size
    ############################################################################
    # Student code begin
    ############################################################################    
    H, W, C = left_img.shape
#     print("left", left_img)
#     print("right", right_img)
#     print("max_search_bound", max_search_bound)
    
    hbs = block_size//2
    d_shape = (H-2*(hbs),W-2*(hbs))    
    disparity_map = torch.zeros(d_shape)
    for y_idx in range(d_shape[0]):
        y = y_idx + hbs
        for x_idx in range(d_shape[1]):
            x = x_idx + hbs
            x_low  = x_idx
            x_high = x_low + block_size
            
            # crop the image region around x and y
            crop_region_left = left_img[y-hbs:y+hbs+1,x_low:x_high,:] # index does not work here
#             print("x==", x, "crop_region shape should match block_size", crop_region_left.shape, "is equal to", block_size)

            # starting in the (y,x) on the right image and crop the region
            # and shift one pixel at a time to the left until reaching max_search_bound
            # for each shift, store the index away from (y,x) as disparity and the error 
            # between the crop_region_right with crop_region_left.
            # after all iterations, find the argmin of the array and that will be the disparity value
            if max_search_bound == 0:
                disparity_map[y_idx][x_idx] = 0
            else:
                _min_err = 999999999
                _min_disp_idx = 0
                for idx in range(max_search_bound):
                    x_lower  = x-hbs-idx
                    x_higher = x_lower + block_size # adding block size here will ensure the size stay the same
                    if x_lower < 0:
                        x_lower = 0
                    if x_higher > W:
                        x_higher = W   

                    crop_region_right = right_img[y-hbs:y+hbs+1,x_lower:x_higher,:]
                    if crop_region_right.shape != crop_region_left.shape:
#                         print("x", x, "idx", idx, "lower bound", x_lower, "upper bound", x_higher)
#                         print("not matching:", crop_region_right.shape)
                        break
                    
                    err = sim_measure_function(crop_region_left, crop_region_right)
                    if err < _min_err:
                        _min_err = err
                        _min_disp_idx = idx
#                 print("x", x, "_min_disp_idx", _min_disp_idx)
                disparity_map[y_idx][x_idx] = _min_disp_idx
    
    
    ############################################################################
    # Student code end
    ############################################################################
    return disparity_map

def calculate_cost_volume(left_img: torch.Tensor,
                            right_img: torch.Tensor,
                            max_disparity: int,
                            sim_measure_function: Callable,
                            block_size: int = 9):
    """
    Calculate the cost volume. Each pixel will have D=max_disparity cost values
    associated with it. Basically for each pixel, we compute the cost of
    different disparities and put them all into a tensor.

    Note: 
    1.  It is important for this project to follow the convention of search
        input in left image and search target in right image
    2.  If the shifted patch in the right image will go out of bounds, it is
        good to set the default cost for that pixel and disparity to be something
        high(we recommend 255), so that when we consider costs, valid disparities will have a lower
        cost. 

    Args:
    -   left_img: image from the left stereo camera. Torch tensor of shape (H,W,C).
                    C will be 1 or 3.
    -   right_img: image from the right stereo camera. Torch tensor of shape (H,W,C)
    -   max_disparity:  represents the number of disparity values we will consider.
                    0 to max_disparity-1
    -   sim_measure_function: a function to measure similarity measure between
                    two tensors of the same shape; returns the error value
    -   block_size: the size of the block to be used for searching between
                    left and right image
    Returns:
    -   cost_volume: The cost volume tensor of shape (H,W,D). H,W are image
                    dimensions, and D is max_disparity. cost_volume[x,y,d] 
                    represents the similarity or cost between a patch around left[x,y]  
                    and a patch shifted by disparity d in the right image. 
                    
    """
    #placeholder
    H = left_img.shape[0]
    W = right_img.shape[1]
    cost_volume = torch.zeros(H, W, max_disparity)
    ############################################################################
    # Student code begin
    ############################################################################

    H, W, C = left_img.shape
#     print("left", left_img)
#     print("right", right_img)
#     print("max_search_bound", max_search_bound)
    
    hbs = block_size//2
    d_shape = (H-2*(hbs),W-2*(hbs), max_disparity)    
#     cost_volume = torch.zeros(d_shape)
    for y_idx in range(d_shape[0]):
        y = y_idx + hbs
#         print("y", y)
        for x_idx in range(d_shape[1]): 
            x = x_idx + hbs
            x_low  = x_idx
            x_high = x_low + block_size
            # crop the image region around x and y
            crop_region_left = left_img[y-hbs:y+hbs+1,x_low:x_high,:] # index does not work here
#             print("x==", x, "crop_region shape should match block_size", crop_region_left.shape, "is equal to", block_size)

            # starting in the (y,x) on the right image and crop the region
            # and shift one pixel at a time to the left until reaching max_search_bound
            # for each shift, store the index away from (y,x) as disparity and the error 
            # between the crop_region_right with crop_region_left.
            # after all iterations, find the argmin of the array and that will be the disparity value
            if max_disparity == 0:
                cost_volume[y_idx][x_idx] = 0
            else:
                _err = torch.zeros(max_disparity)
                for idx in range(max_disparity):
                    x_lower  = x-hbs-idx
                    x_higher = x_lower + block_size # adding block size here will ensure the size stay the same
                    if x_lower < 0:
                        x_lower = 0
                    if x_higher > W:
                        x_higher = W                
                    
                    crop_region_right = right_img[y-hbs:y+hbs+1,x_lower:x_higher,:]
                    if crop_region_right.shape != crop_region_left.shape:
#                         print("size not matching:", x_lower, x_higher)  
                        break
                    
                    _error = sim_measure_function(crop_region_left, crop_region_right)
                    _err[idx] = _error
#                     print("crop_region_left", crop_region_left)
#                     print("crop_region_right", crop_region_right)
#                     print(_error)
#                 print("x:", x, "error:", _err)
                cost_volume[y_idx][x_idx] = _err
    
    ############################################################################
    # Student code end
    ############################################################################
    return cost_volume
