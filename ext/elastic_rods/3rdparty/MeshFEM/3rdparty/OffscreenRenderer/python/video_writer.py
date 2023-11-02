from enum import Enum
import os
import subprocess as sp

# Enum class whose values hold the corresponding FFmpeg command line options
class Codec(Enum):
    ImgSeq = 0
    H264   = ['h264']
    HEVC   = ['hevc',  '-tag:v', 'hvc1'] # We need to set the hvc1 tag for Quicktime compatibility

class VideoWriter:
    """
    Writes a raw image stream to an image sequence or a compressed video
    """
    def __init__(self, outPath, inWidth, inHeight, codec=Codec.H264, framerate=30, quality='-crf 23', streaming=False, outWidth=None, outHeight=None, additionalFlags=[]):
        """
        streaming - Whether to write a fragmented MP4 that can be watched as it
        is being written (and is usable if the writing job dies before it finishes).
        """
        self. outPath     = outPath
        self. inWidth     = inWidth
        self. inHeight    = inHeight
        self.outWidth     = inWidth  if outWidth  is None else outWidth
        self.outHeight    = inHeight if outHeight is None else outHeight
        self.codec        = codec
        self.framerate    = framerate
        self.frameCounter = 0

        self.ffmpegProc = None

        if codec == Codec.ImgSeq:
            os.makedirs(outPath, exist_ok=True)
        else:
            ffmpegCommand = ['ffmpeg', '-y']
            if streaming:
                ffmpegCommand += ['-probesize', '32', '-flags', 'low_delay']
            # Input settings (read raw pixel data from stdin)
            ffmpegCommand += [
                    '-f', 'rawvideo', '-pixel_format', 'rgba',
                    '-video_size', f'{inWidth}x{inHeight}',
                    '-framerate', str(framerate),
                    '-i', '-']
            # Output settings
            ffmpegCommand += ['-pix_fmt', 'yuv420p'] # Quicktime-compatible YUV pixel format
            if self.isResizing():
                ffmpegCommand += ['-vf', f'scale={self.outWidth}:{self.outHeight}']
            ffmpegCommand += ['-vcodec'] + codec.value # Codec CLI settings are stored in enum
            ffmpegCommand += quality.split()
            if streaming:
                ffmpegCommand += ['-g', '15', '-movflags', 'frag_keyframe+empty_moov', '-blocksize', '2048', '-flush_packets', '1']
                ffmpegCommand += ['-tune', 'zerolatency']
                # Alternatively, we could use the subset of the x264 options selected by `-tune zerolatency` that are needed to ensure
                # the video file on disk is updated every "group of pictures" frames (15 here). The full list of options is:
                #       '--bframes 0 --force-cfr --no-mbtree --sync-lookahead 0 --sliced-threads --rc-lookahead 0'
                # of these, `--bframes 0` is not as crucial, but does seem to reduce the frame latency by
                # by about 5 frames without substantially increasing file size.
                # ffmpegCommand += ['-x264opts', 'force-cfr=1:no-mbtree=1:sliced-threads=1:sync-lookahead=0:rc-lookahead=0']
            ffmpegCommand.extend(additionalFlags)
            ffmpegCommand.append(outPath)

            # print(f"ffmpeg command: {' '.join(ffmpegCommand)}")
            self.ffmpegProc = sp.Popen(ffmpegCommand, stdin=sp.PIPE)

    def frameDataSize(self):
        return self.inWidth * self.inHeight * 4 # We assume RGBA input

    def isResizing(self):
        return (self.outWidth != self.inWidth) or (self.outHeight != self.inHeight)

    def writeFrame(self, frameData):
        """
        Write frame data held in a numpy array (in scaline order from top-left to bottom-right
        """
        frameData = frameData.ravel()
        if len(frameData) != self.frameDataSize(): raise Exception('Unexpected frame data size')

        if self.ffmpegProc is not None:
            self.ffmpegProc.stdin.write(frameData)
        else:
            from PIL import Image
            img = Image.fromarray(frameData.reshape((self.inHeight, self.inWidth, 4)))
            if self.isResizing(): img = img.resize((self.outWidth, self.outHeight))
            img.save(f'{self.outPath}/frame_{self.frameCounter:06d}.png')

        self.frameCounter += 1

    def finish(self):
        if self.ffmpegProc is not None:
            self.ffmpegProc.stdin.close()

    def __del__(self):
        self.finish()

class ContextVideoWriter(VideoWriter):
    """
    Renders an OpenGL context to an image sequence or a compressed video
    """
    def __init__(self, outPath, ctx, codec=Codec.H264, framerate=30, streaming=False, outWidth=None, outHeight=None):
        super().__init__(outPath, ctx.width, ctx.height, codec=codec, framerate=framerate, streaming=streaming, outWidth=outWidth, outHeight=outHeight)
        self.ctx = ctx

    def writeFrame(self):
        super().writeFrame(self.ctx.array())

class MeshRendererVideoWriter(VideoWriter):
    """
    Creates an image sequence or a compressed video from  a MeshRenderer
    """
    def __init__(self, outPath, mrenderer, codec=Codec.H264, framerate=30, streaming=False, outWidth=None, outHeight=None):
        super().__init__(outPath, mrenderer.ctx.width, mrenderer.ctx.height, codec=codec, framerate=framerate, streaming=streaming, outWidth=outWidth, outHeight=outHeight)
        self.mrenderer = mrenderer
        if codec != Codec.ImgSeq:
            # FFmpeg doesn't support transparent H264/HEVC output--it just
            # composites over a black background :(
            self.mrenderer.transparentBackground = False

    def writeFrame(self):
        """
        Render a new frame into the video.
        """
        self.mrenderer.render(True)
        super().writeFrame(self.mrenderer.array())

import numpy as np
class PlotVideoWriter(VideoWriter):
    """
    Creates an image sequence or a compressed video from  a MeshRenderer
    """
    def __init__(self, outPath, fig, dpi = 72, codec=Codec.H264, framerate=30, streaming=False, outWidth=None, outHeight=None):
        fig.set_dpi(dpi)
        self.dpi = dpi
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.buffer_rgba().tobytes()
        super().__init__(outPath, *fig.canvas.get_width_height(), codec=codec, framerate=framerate, streaming=streaming, outWidth=outWidth, outHeight=outHeight)

    def writeFrame(self, fig):
        """
        Render a new frame into the video.
        """
        fig.set_dpi(self.dpi)
        fig.tight_layout()
        if self.codec != Codec.ImgSeq:
            # FFmpeg doesn't support transparent H264/HEVC output, and using
            # a transparent plot background results in ugly plot labels.
            fig.patch.set_facecolor('white')

        w, h = fig.canvas.get_width_height()
        if (w != self.inWidth) or (h != self.inHeight): raise Exception('Plot dimensions changed')
        fig.canvas.draw()
        super().writeFrame(np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8))

def countFrames(vidPath):
    """
    Count the number of frames in video file `vidPath`.
    """
    p = sp.Popen(['ffmpeg', '-i', vidPath, '-map', '0:v:0', '-c', 'copy', '-f', 'null', '-'], stderr=sp.PIPE)
    out, err = p.communicate()
    for l in err.split(b'\n'):
        tokens = l.split()
        if len(tokens) == 0: continue
        if tokens[0] == b'frame=':
            return int(tokens[1])
    return -1
