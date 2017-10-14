import numpy as np
import cv2


class ImageTextInput():

    def addText(self, image, text='', size=1, thickness=1, color=(0, 0, 0)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, (50, 50), font, size,
                    color, thickness, cv2.LINE_AA)
