import opencv_wrapper as cvw
import subprocess

class VidReadWrapper():
    def load_video(vid_file):
        return LoadedVidWrapper(vid_file)


def execute(cmd, **kwargs):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, **kwargs)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    # return_code = popen.wait()
    # if return_code:
    #     raise subprocess.CalledProcessError(return_code, cmd)



class LoadedVidWrapper():
    def __init__(self, vid_file=None):
        self.vid_file = vid_file
        ffprobe_res = [line for line in execute(f"ffprobe -select_streams v -show_streams {self.vid_file}".split(' '))]
        self.ffprobe_attrs = {key: val for key, val in [line.strip().split('=') for line in ffprobe_res if "=" in line and not line.startswith("DISPOSITION:")]}
        #see https://superuser.com/a/768420/489445

    def __enter__(self):
        self.cvw_vid = cvw.load_video(self.vid_file)
        self.cvw_ref = self.cvw_vid.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.cvw_vid.__exit__(exc_type, exc_value, exc_traceback)

    def __iter__(self):
        return self.cvw_ref.__iter__()

    @property
    def total_frames(self):
        return int(self.ffprobe_attrs['nb_frames'])

    @property
    def fps(self):
        print("here!")
        assert False
        return self.cvw_ref.fps

    @property
    def height(self):
        return self.cvw_ref.height

    @property
    def width(self):
        return self.cvw_ref.width

    def __getattr__(self,attr):
        #pass everything that isn't implemented here to the original cvw
        return self.cvw_ref.__getattribute__(attr)