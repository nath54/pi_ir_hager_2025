#
### Import Modules. ###
#
from torch import Tensor
#
import numpy as np
from numpy.typing import NDArray


#
### Global variables. ###
#
# Use a singleton class to manage tracking state
class TrackingState:
    def __init__(self):
        self.track_texts = []
        self.track_traces = []
        self.track_distances_abs = []
        self.track_distances_rel = []
        self.track_distances_cos = []
        self.track_id_trace = 0
        self.track_mode_pytorch = True

# Global singleton instance
_tracking_state = None

def get_tracking_state():
    global _tracking_state
    if _tracking_state is None:
        _tracking_state = TrackingState()
    return _tracking_state

def init_track() -> None:
    #
    state = get_tracking_state()
    state.track_texts = []
    state.track_traces = []
    state.track_distances_abs = []
    state.track_distances_rel = []
    state.track_distances_cos = []
    state.track_id_trace = 0
    state.track_mode_pytorch = True

#
### Tracking functions. ###
#
def track(variable: Tensor | NDArray[np.float32], descr_txt: str = "") -> None:

    #
    state = get_tracking_state()

    #
    ### Pytorch Mode. ###
    #
    if state.track_mode_pytorch:
        #
        state.track_traces.append( variable.detach().cpu().numpy() )
        #
        state.track_texts.append( descr_txt )

    #
    ### Interpretor Mode. ###
    #
    else:

        #
        if state.track_id_trace >= len(state.track_traces):
            #
            # No reference traces available - this means the model doesn't have track() calls
            # Just return without doing anything
            return

        #
        ref_var: NDArray[np.float32] = state.track_traces[state.track_id_trace]

        #
        ref_var_flat: NDArray[np.float32] = np.ndarray.flatten(ref_var)
        ext_var_flat: NDArray[np.float32] = np.ndarray.flatten(variable)

        #
        if ref_var_flat.shape != ext_var_flat.shape:
            #
            raise UserWarning("ref_var_flat.shape != ext_var_flat.shape")

        #
        abs_dist: float = np.linalg.norm(ref_var_flat - ext_var_flat)
        #
        ref_norm: float = np.linalg.norm(ref_var_flat)
        ext_norm: float = np.linalg.norm(ext_var_flat)
        #
        rel_dist: float = float("nan")
        #
        if abs(ref_norm) > 1e-6:
            rel_dist = abs_dist / ref_norm
        #
        dot_prod = np.dot(
            ext_var_flat,
            ref_var_flat.T
        )
        #
        cos_sim: float = 0.0
        #
        if abs(ref_norm * ext_norm) > 1e-6:
            #
            cos_sim: float = dot_prod / (ref_norm * ext_norm)

        #
        state.track_distances_abs.append( abs_dist )
        state.track_distances_rel.append( rel_dist )
        state.track_distances_cos.append( cos_sim )

        #
        state.track_id_trace += 1


#
### Function to reset the trace for one of the two modes. ###
#
def reset_trace(is_pytorch: bool) -> None:

    #
    state = get_tracking_state()

    #
    state.track_mode_pytorch = is_pytorch

    #
    if is_pytorch:
        #
        state.track_traces = []
        state.track_texts = []
    #
    else:
        #
        # Don't clear track_traces and track_texts - they contain the reference data
        # Only clear the distance arrays for fresh comparison
        state.track_distances_abs = []
        state.track_distances_rel = []
        state.track_distances_cos = []

    #
    state.track_id_trace = 0


#
### File to log distances. ###
#
def log_distances(output_file: str) -> None:

    #
    state = get_tracking_state()

    #
    if len(state.track_traces) == 0:
        #
        return

    #
    txt: str = ""

    #
    mlt: int = max([len(txt) for txt in state.track_texts])

    #
    for i in range(len(state.track_distances_abs)):
        #
        t: str = state.track_texts[i]
        #
        txt += f"{i:03d} | {t+' '*(mlt-len(t))} | {state.track_distances_rel[i]} | {state.track_distances_cos[i]} | {state.track_distances_abs[i]}\n"

    #
    with open(output_file, "w", encoding="utf-8") as f:
        #
        f.write(txt.strip())

