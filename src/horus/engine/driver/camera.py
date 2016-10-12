# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
import math
import time
import glob
from time import sleep
import platform
import subprocess
import select
from threading import Thread

import logging
logger = logging.getLogger(__name__)

system = platform.system()

try:
    import v4l2capture
except ImportError:
    print("Install v4l2capture module and retry ! Get it here:\nhttps://github.com/rmca/python-v4l2capture/archive/py3k.zip")
    raise SystemExit(0)

# Need to patch v4l2capture to get rid of libwebcam requirement
def ctl_param(dev, param, val=None, show=False):
#    print("Calling %r %r %r"%(dev, param, val))
    try:
        p = ['uvcdynctrl', '-d', 'video%s'%dev]
        if val:
            p.extend(['-s', param, str(val)])
        else:
            p.extend(['-g', str(param)])
        ret = subprocess.check_output(p)
        if not val:
            if show:
                print("%d"%int(ret))
            else:
                return int(ret)
    except Exception as e:
        print("Error calling %s: %s"%(' '.join(param), e))


import numpy as np

class Camcorder(Thread):
    def __init__(self, dev, width, height):
        if not v4l2capture:
            raise RuntimeError("Can't find v4l2capture")
        Thread.__init__(self)
        self.setDaemon(True)
        self.dev = dev
        video = v4l2capture.Video_device('/dev/video%d'%dev)
        size_x, size_y = video.set_format(width, height, 0)
        self.size = (size_x, size_y)
        self.fps = video.set_fps(30)

        print("Got %sx%s @ %s"%(size_x, size_y, self.fps))

        video.create_buffers(1)
        video.queue_all_buffers()
        self.video = video
        video.start() # start capture in background
        self.terminate = False
        print("Waiting for cam to be ready...")
        for n in range(10):
            try:
                self._cap()
                break
            except Exception as e:
                if not ( e.args and e.args[0] == 11):
                    import traceback
                    traceback.print_exc()
                sleep(1)
        else:
            raise RuntimeError("Can't init camera")

    def __getattr__(self, name):
        return getattr(self.video, name)

    def stop(self):
        self.terminate = True

    def get(self):
        return self.buff

    def _cap(self):
        image_data = self.video.read_and_queue()
        buff = np.fromstring(image_data, dtype=np.uint8)
        sz = list(reversed(self.size))
        sz.append(-1)
        self.buff = buff.reshape(*sz)

    def run(self):
        # Start the device. This lights the LED if it's a camera that has one.
        print("Starting capture")
        size_x, size_y = self.size

        while not self.terminate:
            select.select((self.video,), (), ())
            self._cap()

        self.video.close()
        cv2.destroyAllWindows()


def temp_expoinit():
    pass

def temp_getExposure():
    return int(subprocess.check_output(['uvcdynctrl', '-g', 'Exposure (Absolute)']))

def temp_setExposure(value):
    print("SetExpo(%s)"%value)
    print subprocess.check_output(['uvcdynctrl', '-s', 'Exposure (Absolute)', "%d"%int(value)])

if system == 'Darwin':
    import uvc
    from uvc.mac import *


class WrongCamera(Exception):

    def __init__(self):
        Exception.__init__(self, "Wrong Camera")


class CameraNotConnected(Exception):

    def __init__(self):
        Exception.__init__(self, "Camera Not Connected")


class InvalidVideo(Exception):

    def __init__(self):
        Exception.__init__(self, "Invalid Video")


class WrongDriver(Exception):

    def __init__(self):
        Exception.__init__(self, "Wrong Driver")


class InputOutputError(Exception):

    def __init__(self):
        Exception.__init__(self, "V4L2 Input/Output Error")


class Camera(object):

    """Camera class. For accessing to the scanner camera"""

    def __init__(self, parent=None, camera_id=0):
        self.parent = parent
        self.camera_id = camera_id
        self.unplug_callback = None

        self._capture = None
        self._is_connected = False
        self._reading = False
        self._updating = False
        self._last_image = None
        self._video_list = None
        self._tries = 0  # Check if command fails
        self._luminosity = 1.0

        self.initialize()

        if system == 'Windows':
            self._number_frames_fail = 3
            self._max_brightness = 1.
            self._max_contrast = 1.
            self._max_saturation = 1.
        elif system == 'Darwin':
            self._number_frames_fail = 3
            self._max_brightness = 255.
            self._max_contrast = 255.
            self._max_saturation = 255.
            self._rel_exposure = 10.
        else:
            self._number_frames_fail = 3
            self._max_brightness = 255.
            self._max_contrast = 255.
            self._max_saturation = 255.
            self._max_exposure = 1000.
            self._rel_exposure = 100.

    def initialize(self):
        self._brightness = 0
        self._contrast = 0
        self._saturation = 0
        self._exposure = 0
        self._frame_rate = 0
        self._width = 0
        self._height = 0
        self._rotate = True
        self._hflip = True
        self._vflip = False

    def connect(self):
        logger.info("Connecting camera {0}".format(self.camera_id))
        self._is_connected = False
        self.initialize()
        if system == 'Darwin':
            for device in uvc.mac.Camera_List():
                if device.src_id == self.camera_id:
                    self.controls = uvc.mac.Controls(device.uId)
        if self._capture is not None:
            self._capture.stop()
        self._capture = Camcorder(self.camera_id, 1280, 960)
        self._capture.start()
        self._is_connected = True

    def disconnect(self):
        tries = 0
        if self._is_connected:
            logger.info("Disconnecting camera {0}".format(self.camera_id))
            if self._capture is not None:
                self._is_connected = False
                self._capture.stop()
                logger.info(" Done")

    def set_unplug_callback(self, value):
        self.unplug_callback = value

    def _check_video(self):
        """Check correct video"""
        frame = self.capture_image(flush=1)
        if frame is None or (frame == 0).all():
            raise InvalidVideo()

    def _check_camera(self):
        """Check correct camera"""
        c_exp = False
        c_bri = False

        try:
            # Check exposure
            if system == 'Darwin':
                self.controls['UVCC_REQ_EXPOSURE_AUTOMODE'].set_val(1)
            temp_expoinit()
            self.set_exposure(2)
            exposure = self.get_exposure()
            if exposure is not None:
                c_exp = exposure >= 1.9

            # Check brightness
            self.set_brightness(2)
            brightness = self.get_brightness()
            if brightness is not None:
                c_bri = brightness >= 2
        except:
            raise WrongCamera()

        if not c_exp or not c_bri:
            raise WrongCamera()

    def _check_driver(self):
        """Check correct driver: only for Windows"""
        if system == 'Windows':
            self.set_exposure(10)
            frame = self.capture_image(flush=1)
            mean = sum(cv2.mean(frame)) / 3.0
            if mean > 200:
                raise WrongDriver()

    def capture_image(self, flush=0, auto=False):
        """Capture image from camera"""
        if self._is_connected:
            if self._updating:
                return self._last_image
            else:
                d = max(self._exposure / 64.0 * 2, 1/self._capture.fps)
                self._reading = True
                for n in range(flush):
                    time.sleep(d)
                image = self._capture.get()
                self._reading = False
                if True:
                    if self._rotate:
                        image = cv2.transpose(image)
                    if self._hflip:
                        image = cv2.flip(image, 1)
                    if self._vflip:
                        image = cv2.flip(image, 0)
                    self._success()
                    self._last_image = image
                    return image
                else:
                    self._fail()
                    return None
        else:
            return None

    def save_image(self, filename, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, image)

    def set_rotate(self, value):
        self._rotate = value

    def set_hflip(self, value):
        self._hflip = value

    def set_vflip(self, value):
        self._vflip = value

    def set_brightness(self, value):
        if self._is_connected:
            if self._brightness != value:
                self._updating = True
                ctl_param(self.camera_id, 'Brightness', value)
                self._brightness = value
                self._updating = False

    def set_contrast(self, value):
        if self._is_connected:
            if self._contrast != value:
                self._updating = True
                self._contrast = value
                ctl_param(self.camera_id, 'Contrast', value)
                self._updating = False

    def set_saturation(self, value):
        if self._is_connected:
            if self._saturation != value:
                self._updating = True
                self._saturation = value
                ctl_param(self.camera_id, 'Saturation', value)
                self._updating = False

    def set_exposure(self, value, force=False):
        if self._is_connected:
            if self._exposure != value or force:
                self._updating = True
                self._exposure = value
                ctl_param(self.camera_id, 'Exposure (Absolute)', value/64.0 * 10000)
                self._updating = False

    def set_luminosity(self, value):
        possible_values = {
            "High": 0.5,
            "Medium": 1.0,
            "Low": 2.0
        }
        self._luminosity = possible_values[value]
        self.set_exposure(self._exposure, force=True)

    def set_frame_rate(self, value):
        return
        if self._is_connected:
            if self._frame_rate != value:
                self._frame_rate = value
                self._updating = True
                self._capture.set(cv2.CAP_PROP_FPS, value)
                self._updating = False

    def set_resolution(self, width, height):
        if self._is_connected:
            if self._width != width or self._height != height:
                self._updating = True
                self._set_width(width)
                self._set_height(height)
                self._update_resolution()
                self._updating = False

    def _set_width(self, value):
        return
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, value)

    def _set_height(self, value):
        return
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, value)

    def _update_resolution(self):
        self._width = 1280 #int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = 960 #int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_brightness(self):
        return ctl_param(self.camera_id, 'Brightness')
        if self._is_connected:
            if system == 'Darwin':
                ctl = self.controls['UVCC_REQ_BRIGHTNESS_ABS']
                value = ctl.get_val()
            else:
                value = self._capture.get(cv2.CAP_PROP_BRIGHTNESS)
                value *= self._max_brightness
            return value

    def get_exposure(self):
        return ctl_param(self.camera_id, 'Exposure (Absolute)')
        if self._is_connected:
            if system == 'Darwin':
                ctl = self.controls['UVCC_REQ_EXPOSURE_ABS']
                value = ctl.get_val()
                value /= self._rel_exposure
            elif system == 'Windows':
                value = self._capture.get(cv2.CAP_PROP_EXPOSURE)
                value = 2 ** -value
            else:
                value = temp_getExposure()
#                value = self._capture.get(cv2.CAP_PROP_EXPOSURE)
                value /= self._rel_exposure
            return value

    def get_resolution(self):
        if self._rotate:
            return int(self._height), int(self._width)
        else:
            return int(self._width), int(self._height)

    def _success(self):
        self._tries = 0

    def _fail(self):
        logger.debug("Camera fail")
        self._tries += 1
        if self._tries >= self._number_frames_fail:
            self._tries = 0
            if self.unplug_callback is not None and \
               self.parent is not None and \
               not self.parent.unplugged:
                self.parent.unplugged = True
                self.unplug_callback()

    def _line(self, value, imin, imax, omin, omax):
        ret = 0
        if omin is not None and omax is not None:
            if (imax - imin) != 0:
                ret = int((value - imin) * (omax - omin) / (imax - imin) + omin)
        return ret

    def _count_cameras(self):
        for i in xrange(5):
            cap = cv2.VideoCapture(i)
            res = True
            cap.release()
            if res:
                return i
        return 5

    def get_video_list(self):
        baselist = []
        if system == 'Windows':
            if not self._is_connected:
                count = self._count_cameras()
                for i in xrange(count):
                    baselist.append(str(i))
                self._video_list = baselist
        elif system == 'Darwin':
            for device in uvc.mac.Camera_List():
                baselist.append(str(device.src_id))
            self._video_list = baselist
        else:
            for device in ['/dev/video*']:
                baselist = baselist + glob.glob(device)
            self._video_list = baselist
        return self._video_list
