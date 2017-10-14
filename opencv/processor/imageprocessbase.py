import numpy as np
import cv2


class ImageProcessorBase():

    smoothings = {
        'kernel': 'Kernel Smoothing',
        'gaussian': 'Gaussian Blur Smoothing',
        'median': 'Median Blur Smoothing'
    }

    morphs = {
        'erosion': 'Erosion Morphology',
        'dilation': 'Dilation Morphology',
        'opening': 'Opening Morphology',
        'closing': 'Closing Morphology'
    }

    gradients = {
        'laplace': 'Laplacian Gradient',
        'sobelx': 'X-axis Sobel Gradient',
        'sobely': 'Y-axis Sobel Gradient'
    }

    def returnThreshold(self, image):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)

        return ret, img, mask, inv_mask

    def addBasicSmoothing(self, frame, smoothing_method='median'):
        kernel = np.ones((15, 15), np.float) / 255
        smoothed = cv2.filter2D(frame, -1, kernel)
        gblur = cv2.GaussianBlur(frame, (15, 15), 0)
        median_blur = cv2.medianBlur(frame, 15)

        if smoothing_method == 'kernel':
            return smoothed, self.smoothings[smoothing_method]
        elif smoothing_method == 'gaussian':
            return gblur, self.smoothings[smoothing_method]
        elif smoothing_method == 'median':
            return median_blur, self.smoothings[smoothing_method]
        elif smoothing_method == 'all':
            cv2.imshow('Kernel Smoothing', smoothed)
            cv2.imshow('Gaussian Blur Smoothing', gblur)
            cv2.imshow('Median Blur Smoothing', median_blur)

    def addMorphology(self, frame, morph_medthod='closing'):
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(frame, kernel, iterations=1)
        dilation = cv2.dilate(frame, kernel, iterations=1)

        opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

        if morph_medthod == 'erosion':
            return erosion, self.morphs[morph_medthod]
        elif morph_medthod == 'dilation':
            return dilation, self.morphs[morph_medthod]
        elif morph_medthod == 'opening':
            return opening, self.morphs[morph_medthod]
        elif morph_medthod == 'closing':
            return closing, self.morphs[morph_medthod]
        elif morph_medthod == 'all':
            cv2.imshow('Erosion Morphology', erosion)
            cv2.imshow('Dilation Morphology', dilation)
            cv2.imshow('Opening Morphology', opening)
            cv2.imshow('Closing Morphology', closing)

    def addGradient(self, frame, grad_method='laplace', kernel_size=5):
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=kernel_size)

        if grad_method == 'laplace':
            return laplacian, self.gradients[grad_method]
        elif grad_method == 'sobelx':
            return sobelx, self.gradients[grad_method]
        elif grad_method == 'sobely':
            return sobely, self.gradients[grad_method]
        elif grad_method == 'all':
            cv2.imshow('Laplacian Gradient', laplacian)
            cv2.imshow('X-axis Sobel Gradient', sobelx)
            cv2.imshow('Y-axis Sobel Gradient', sobely)

    def extractForeground(self, image, rectangle):
        mask = np.zeros(image.shape[:2], np.uint8)

        bgdModel = np.zeros((1, 65), np.float)
        fgdModel = np.zeros((1, 65), np.float)

        rect = rectangle

        cv2.grabCut(image, mask, rect, bgdModel,
                    fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        extracted = image * mask2[:, :, np.newaxis]

        return extracted

    def detectCorners(self, image, feature_count, quality, minimum_distance):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray32 = np.float32(gray)

        corners = cv2.goodFeaturesToTrack(
            gray32, feature_count, quality, minimum_distance)
        corners_int = np.int_(corners)

        for corner in corners_int:
            x, y = corner.ravel()
            cv2.circle(image, (x, y), 3, 255, -1)

        return image

    def matchFeatures(self, main_image, feature_image, feature_count):

        orb = cv2.ORB_create()

        keypoint1, descriptor1 = orb.detectAndCompute(main_image, None)
        keypoint2, descriptor2 = orb.detectAndCompute(feature_image, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(descriptor1, descriptor2)
        matches = sorted(matches, key=lambda x: x.distance)

        match_img = cv2.drawMatches(
            main_image, keypoint1, feature_image, keypoint2, matches[:feature_count], None, flags=2)

        return match_img
