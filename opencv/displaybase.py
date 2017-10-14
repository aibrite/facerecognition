import cv2
import numpy as np
import matplotlib.pyplot as plt

from opencv.processor.imagetextbase import ImageTextInput
# from processor.imageprocessbase import ImageProcessorBase

from opencv.processor.imageprocessbase import ImageProcessorBase


class BaseDisplay():

    def displayAndClose(self, image, title='', closeWhenFinished=True):
        cv2.imshow(title, image)
        cv2.waitKey(0)
        if closeWhenFinished is True:
            cv2.destroyAllWindows()

    def printVideoMessage(self, message='', key_message=''):
        if message == '':
            print('Starting Filtered Video Feed...')
            print('Press ESC to quit')
        else:
            print(message)
            print(key_message)

    def displayCvImage(self, imageName, color=True, save=False, legend=True):
        colorOption = cv2.IMREAD_COLOR if color is True else cv2.IMREAD_GRAYSCALE
        img = cv2.imread(imageName, colorOption)

        if legend is True:
            ImageTextInput().addText(img, 'Press any key to continue...')

        if color is False & save is True:
            cv2.imwrite('img/savedimage.png', img)

        self.displayAndClose(img, 'Image')

    def displayPlotImage(self, imageName, color=True):
        colorOption = cv2.IMREAD_COLOR if color is True else cv2.IMREAD_GRAYSCALE
        img = cv2.imread(imageName, colorOption)

        plt.imshow(img, cmap='gray', interpolation='bicubic')
        plt.show()

    def displayVideoFeed(self, videoSource=0, grayscale=False, save=False):
        print('Starting Video Feed...')
        print('Press Q to quit')
        cap = cv2.VideoCapture(videoSource)

        if save is True:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('img/savedvideo.avi',
                                  fourcc, 20.0, (640, 480))

        while True:
            _, frame = cap.read()
            if grayscale is True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Gray', gray)

            if save is True:
                out.write(frame)

            cv2.imshow('Video Feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if save is True:
            out.release()
        cv2.destroyAllWindows()

    def displayThreshold(self, image):
        ret, img, mask, inv_mask = ImageProcessorBase().returnThreshold(image)
        self.displayAndClose(img, 'Original Image', closeWhenFinished=False)
        self.displayAndClose(mask, 'Threshold', closeWhenFinished=False)
        self.displayAndClose(inv_mask, 'InverseThreshold')

    def displayFilteredCam(self, videoSource=0, lower_color=[0, 0, 0], upper_color=[255, 255, 255], smoothing=None, morphology=None):
        self.printVideoMessage()

        cap = cv2.VideoCapture(videoSource)

        while True:
            _, frame = cap.read()

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_red = np.array(lower_color)
            upper_red = np.array(upper_color)

            mask = cv2.inRange(hsv, lower_red, upper_red)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow('Original Video Output', frame)
            cv2.imshow('Mask', mask)
            cv2.imshow('Filtered Video Output', result)

            if smoothing is not None and morphology is None:
                if smoothing == 'all':
                    ImageProcessorBase().addBasicSmoothing(result, smoothing)
                else:
                    smoothed_frame, smoothing_name = ImageProcessorBase().addBasicSmoothing(
                        result, smoothing_method=smoothing)
                    cv2.imshow(smoothing_name, smoothed_frame)
            elif smoothing is None and morphology is not None:
                if morphology == 'all':
                    ImageProcessorBase().addMorphology(result, morphology)
                else:
                    morphed, morph_name = ImageProcessorBase().addMorphology(
                        result, morph_medthod=morphology)
                    cv2.imshow(morph_name, morphed)
            if cv2.waitKey(27) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def displayGradients(self, videoSource=0, method='laplace'):
        cap = cv2.VideoCapture(videoSource)
        self.printVideoMessage()

        while True:
            _, frame = cap.read()

            cv2.imshow('Original Video Output', frame)

            if method == 'all':
                ImageProcessorBase().addGradient(frame, grad_method='all', kernel_size=5)
            else:
                result, method_name = ImageProcessorBase().addGradient(
                    frame, grad_method=method, kernel_size=5)
                cv2.imshow(method_name, result)

            if cv2.waitKey(27) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def detectEdges(self, videoSource=0, row_size=100, column_size=200):
        cap = cv2.VideoCapture(videoSource)
        self.printVideoMessage()

        while True:
            _, frame = cap.read()

            cv2.imshow('Original Video Output', frame)

            edges = cv2.Canny(frame, row_size, column_size)
            cv2.imshow('Edges', edges)

            if cv2.waitKey(27) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def matchTemplate(self, image, template):
        img_bgr = cv2.imread(image)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        template_img = cv2.imread(template, 0)
        width, height = template_img.shape[::-1]

        result = cv2.matchTemplate(
            img_gray, template_img, cv2.TM_CCOEFF_NORMED)
        match_quality = 0.8
        location = np.where(result >= match_quality)

        for point in zip(*location[::-1]):
            cv2.rectangle(
                img_bgr, point, (point[0] + width, point[1] + height), (0, 255, 255), 2)

        self.displayAndClose(img_bgr, 'Matches', closeWhenFinished=False)
        self.displayAndClose(template_img, 'Template')

    def displayForeground(self, image, rectangle=(161, 79, 150, 150)):
        img = cv2.imread(image)

        foreground = ImageProcessorBase().extractForeground(img, rectangle)

        plt.imshow(foreground)
        plt.colorbar()
        plt.show()

    def displayCorners(self, image, feature_count=100, quality=0.01, minimum_distance=10):
        img = cv2.imread(image)
        self.displayAndClose(img, 'Original Image', closeWhenFinished=False)

        corners = ImageProcessorBase().detectCorners(
            img, feature_count, quality, minimum_distance)

        self.displayAndClose(corners, 'Corners')

    def displayFeatureMatch(self, main_image, feature_image, feature_number=10):
        main = cv2.imread(feature_image, 0)
        feature_img = cv2.imread(main_image, 0)

        match_img = ImageProcessorBase().matchFeatures(
            main, feature_img, feature_count=feature_number)

        plt.imshow(match_img)
        plt.show()

    def displayMotionReduction(self, videoSource=0, noise_reduction=None):
        cap = cv2.VideoCapture(videoSource)
        self.printVideoMessage()

        bgfg = cv2.createBackgroundSubtractorMOG2()

        while True:
            _, frame = cap.read()
            fgmask = bgfg.apply(frame)

            if noise_reduction is not None:
                smoothedMask, smoothing_method = ImageProcessorBase().addBasicSmoothing(
                    fgmask, smoothing_method=noise_reduction)
                cv2.imshow('Original Video Feed', frame)
                cv2.imshow('Reduced Motion', fgmask)
                cv2.imshow('Smoothed Motion Reduction ({})'.format(
                    noise_reduction), smoothedMask)

            else:
                cv2.imshow('Original Video Feed', frame)
                cv2.imshow('Reduced Motion', fgmask)

            if cv2.waitKey(27) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
