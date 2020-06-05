import sys
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolov3_classes import get_cls_dict
from utils.yolov3 import TrtYOLOv3
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
import subprocess as sp

ADDR_IN = 'rtmp://192.168.1.68/live/test'
ADDR_OUT = 'rtmp://192.168.1.68:1936/live/test'
# rtmp://192.168.1.68/live/test
WINDOW_NAME = 'TrtYOLOv3Demo'

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLOv3 model on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--model', type=str, default='yolov3-416',
                        choices=['yolov3-288', 'yolov3-416', 'yolov3-608',
                                 'yolov3-tiny-288', 'yolov3-tiny-416'])
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolov3, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolov3: the TRT YOLOv3 object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    counter = 0
    capt = cv2.VideoCapture(ADDR)

    while True:
        # if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
        #     break
        img = capt.read()
        img = cv2.resize(img, (416, 416))
        print(img.shape)

        if img is not None:
            boxes, confs, clss = trt_yolov3.detect(img, conf_th)
            print('detect')
            img = vis.draw_bboxes(img, boxes, confs, clss)
            img = show_fps(img, fps)
            # cv2.imshow(WINDOW_NAME, img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc

            counter += 1
            cv2.imwrite('/home/shared_folder/' + str(counter) + '.png', img)

        # key = cv2.waitKey(1)
        # if key == 27:  # ESC key: quit program
        #     break
        # elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
        # full_scrn = not full_scrn
        # set_display(WINDOW_NAME, full_scrn)


def check(trt_yolov3, conf_th, vis):
    fps = 0.0
    tic = time.time()
    counter = 0
    capt = cv2.VideoCapture(ADDR_IN)

    width = int(capt.get(cv2.CAP_PROP_FRAME_WIDTH ))
    height = int(capt.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    fps =  int(capt.get(cv2.CAP_PROP_FPS))
    dimension = '{}x{}'.format(width, height)

    command = ['ffmpeg',
               '-y',
               '-f', 'rawvideo',
               '-vcodec','rawvideo',
               '-pix_fmt', 'bgr24',
               '-s', dimension,
               '-i', '-',
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               # '-preset', 'veryfast',
               '-f', 'flv',
               ADDR_OUT]

    proc = sp.Popen(command, stdin=sp.PIPE, shell=False)

    while True:
        f, img = capt.read()

        if img is not None:
            # img = cv2.resize(img, (416, 416))

            boxes, confs, clss = trt_yolov3.detect(img, conf_th)
            img = vis.draw_bboxes(img, boxes, confs, clss)
            img = show_fps(img, fps)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc

            counter += 1
            proc.stdin.write(img.tostring())


    capt.release()
    capt.stop()
    proc.stdin.close()
    proc.stderr.close()
    proc.wait()


def main():
    args = parse_args()

    cls_dict = get_cls_dict('coco')
    yolo_dim = int(args.model.split('-')[-1])  # 416 or 608
    trt_yolov3 = TrtYOLOv3(args.model, (yolo_dim, yolo_dim))
    vis = BBoxVisualization(cls_dict)

    check(trt_yolov3, conf_th=0.3, vis=vis)


if __name__ == '__main__':
    main()
