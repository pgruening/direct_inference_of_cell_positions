import torch
import numpy as np

from DLBio.process_image_patchwise import whole_image_segmentation

# rm?
from DLBio.pytorch_helpers import cuda_to_numpy, image_batch_to_tensor
import torchvision.transforms as transforms
import torch.nn.functional as F


class CellSegmentationObject():
    """segmenting images 
        with already trained models (eg. for evaluation)
    """

    def __init__(self,
                 device,
                 model,
                 in_shape,
                 num_classes,
                 normalization=None
                 ):
        """constructor

        Parameters
        ----------
        device : torch device
        model : pytorch model
            the model to predict the images
        in_shape : list
            the height/width of the trained models input
        num_classes: int
            the number of classes to predict
        normalization: function
            normalization function that may be applied to the input
        """
        self.device = device
        self.model = model
        self.in_shape = in_shape
        self.num_classes = num_classes
        self.norm = normalization

    def do_task(self, image):
        """ prepares the image for prediction 

        if image is larger than model input dlbio's process_image_patchwise is used

        Parameters
        ----------
        image : numpy ndarray
            the image to predic

        Returns
        -------
        numpy ndarray
            the predicted output 
            with 1 Dimension for each class [image_y, image_x, class_n]
        """
        if image.shape[0] > self.in_shape[0] or image.shape[1] > self.in_shape[1]:
            # use process_image_patchwise if image is larger than model input
            pred = whole_image_segmentation(self, image)
        else:
            # otherwise pad image to net size and predict
            image = self.pad_image(image)
            pred = self._predict(image)

        return pred

    def _predict(self, input, do_pre_proc=False, predict_patch=False):
        """the actual prediction

        Parameters
        ----------
        input : numpy ndarray
            the input to predict
        do_pre_proc : bool, optional
            not used right now but necessary for process_image_patchwise
        predict_patch : bool, optional      
            not used right now but necessary for process_image_patchwise

        Returns
        -------
        numpy ndarray
            the predicted model output
        """
        to_tensor = transforms.ToTensor()
        do_unsqueeze = False

        if input.ndim == 3:
            do_unsqueeze = True
            input = to_tensor(input)
            input = input.to(self.device).unsqueeze(0)

        else:
            input = image_batch_to_tensor(input).to(self.device)

        if self.norm is not None:
            input = [
                self.norm(input[i, ...]) for i in range(input.shape[0])
            ]
            input = torch.stack(input, 0)

        # actual prediction
        with torch.no_grad():
            net_out = self.model(input)
        out_seg = F.softmax(net_out, dim=1)

        output = cuda_to_numpy(out_seg)

        if do_unsqueeze:
            output = output.squeeze(0)
        return output

    def pad_image(self, image):
        target_x = self.in_shape[1]
        target_y = self.in_shape[0]
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
            print('image dim changed')
        h, w, _ = image.shape
        x_pad = target_x - w
        y_pad = target_y - h
        padded_image = np.pad(
            image, [(0, y_pad), (0, x_pad), (0, 0)], mode='constant')
        return padded_image

    # ------------------------------------------------
    # methods necessary for whole_image_segmentation
    # ------------------------------------------------

    def get_input_shape(self):
        # whole image segmentation expects list with 'x': list[2], 'y': list[1]
        inpt = [-1]
        inpt.extend(self.in_shape)
        return inpt

    def get_output_shape_for_patchwise_processing(self):
        # whole image segmentation expects list with 'x': list[2], 'y': list[1]
        inpt = [-1]
        inpt.extend(self.in_shape)
        return inpt

    def get_num_classes(self):
        return self.num_classes
