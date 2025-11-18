import torch
import kornia as K
from kornia_moons.viz import draw_LAF_matches
import cv2


class Matcher:
    def __init__(self, image_size):
        self.image_size = image_size

    def match(self, image0, image1, confidence_min=0.8, accurate=False):
        # Convert images to tensors
        image0 = self._convert_image(image0)
        image1 = self._convert_image(image1)

        # Convert images to grayscale for LoFTR input
        input_dict = {
            'image0': K.color.rgb_to_grayscale(image0),
            'image1': K.color.rgb_to_grayscale(image1)
        }

        # Initialize pretrained LoFTR matcher (outdoor dataset)
        matcher_model = K.feature.LoFTR(pretrained='outdoor').eval()
        with torch.inference_mode():
            corresp = matcher_model(input_dict)

        # Create a mask to select keypoints with confidence above the threshold
        mask = corresp['confidence'] > confidence_min

        # Apply the mask to filter keypoints and confidence
        kpts0 = corresp['keypoints0'][mask].cpu().numpy()
        kpts1 = corresp['keypoints1'][mask].cpu().numpy()
        confidence = corresp['confidence'][mask].cpu().numpy()

        # Compute fundamental matrix and inliers
        fmat, inliers = cv2.findFundamentalMat(kpts0, kpts1, cv2.USAC_ACCURATE, 1, 0.99, 100000)
        inliers = inliers > 0

        # Return results as a dictionary
        results = {'image0': image0, 'image1': image1, 'keypoints0': kpts0, 'keypoints1': kpts1,
                   'confidence': confidence, 'inliers': inliers}

        return results

    def show_keypoints_matches(self, feature_matches):
        # Convert numpy keypoints to tensors
        keypoints0 = torch.from_numpy(feature_matches['keypoints0']).unsqueeze(0)
        keypoints1 = torch.from_numpy(feature_matches['keypoints1']).unsqueeze(0)

        num_points0 = keypoints0.shape[1]
        num_points1 = keypoints1.shape[1]

        # Set default scale and orientation for lafs
        scales0 = torch.ones(1, num_points0, 1, 1)
        scales1 = torch.ones(1, num_points1, 1, 1)
        orients0 = torch.ones(1, num_points0, 1)
        orients1 = torch.ones(1, num_points1, 1)

        # Generate LAFs from keypoints
        laf0 = K.feature.laf_from_center_scale_ori(keypoints0, scales0, orients0)
        laf1 = K.feature.laf_from_center_scale_ori(keypoints1, scales1, orients1)

        # Prepare matches as consecutive index pairs
        matches = torch.arange(num_points0).unsqueeze(1).repeat(1, 2)

        # Draw matches using Kornia visualization
        output_figure = draw_LAF_matches(
            laf0,
            laf1,
            matches,
            K.tensor_to_image(feature_matches['image0']),
            K.tensor_to_image(feature_matches['image1']),
            feature_matches['inliers'],
            draw_dict={
                'inlier_color': (0.2, 1, 0.2),
                'tentative_color': (1, 0.1, 0.1),
                'feature_color': (0.2, 0.5, 1),
                'vertical': False
            }
        )
        return output_figure

    def _convert_image(self, input_image):
        # convert image to tensor and normalize it to [0,1]. After that resize it
        tensor_img = K.utils.image_to_tensor(input_image).float()
        new_image = tensor_img.unsqueeze(0) / 255.0
        resized_img = K.geometry.resize(new_image, self.image_size, interpolation='area')
        return resized_img