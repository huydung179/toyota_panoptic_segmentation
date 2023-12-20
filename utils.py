import cv2

from threading import Thread


class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.thread = None
        self.desired_fps = 1
        self.frame_interval = int(self.stream.get(
            cv2.CAP_PROP_FPS) / self.desired_fps)

    def start(self):
        self.thread = Thread(target=self.get, args=())
        self.thread.start()
        return self

    def get(self):
        while not self.stopped:
            ret, self.frame = self.stream.read()
            if not ret:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
        self.stream.release()
        if self.thread is not None:
            self.thread.join()
