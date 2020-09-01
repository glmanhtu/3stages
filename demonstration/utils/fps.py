# import the necessary packages
import datetime


class FPS:
	def __init__(self, max_frames=30):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
		self._max_frames = max_frames
		self._current_fps = -1

	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		self._numFrames = 0
		return self

	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
		if self._numFrames == self._max_frames:
			self._current_fps = self._numFrames / self.elapsed()
			self.start()

	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		end = datetime.datetime.now()
		return (end - self._start).total_seconds()

	def fps(self):
		# compute the (approximate) frames per second
		return self._current_fps
