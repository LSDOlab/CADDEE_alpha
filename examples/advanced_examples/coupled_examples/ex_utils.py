import matplotlib.pyplot as plt
import csdl_alpha as csdl
import numpy as np

def plot_vlm(vlm_outputs):
    print("total drag", vlm_outputs.total_drag.value)
    print("total lift", vlm_outputs.total_lift.value)
    print("total forces", vlm_outputs.total_force.value)
    print("total moments", vlm_outputs.total_moment.value)
    plt.rcParams['text.usetex'] = False
    fig, axs = plt.subplots(3, 1)
    # plt.rc('text', usetex=False)
    for i in range(len(vlm_outputs.surface_CL)):
        panel_forces = csdl.sum(vlm_outputs.surface_panel_forces[i][0, :, :, :], axes=(0, ))
        shape = panel_forces.shape
        norm_span = np.linspace(-1, 1, shape[0])

        axs[0].plot(norm_span, panel_forces[:, 0].value)
        axs[0].set_xlabel("norm span")
        axs[0].set_ylabel("Fx")
        axs[1].plot(norm_span, panel_forces[:, 1].value)
        axs[1].set_xlabel("norm span")
        axs[1].set_ylabel("Fy")
        axs[2].plot(norm_span, panel_forces[:, 2].value)
        axs[2].set_xlabel("norm span")
        axs[2].set_ylabel("Fz")

        print(f"surface {i} CL", vlm_outputs.surface_CL[i].value)
        print(f"surface {i} CDi", vlm_outputs.surface_CDi[i].value)
        print(f"surface {i} L", vlm_outputs.surface_lift[i].value)
        print(f"surface {i} Di", vlm_outputs.surface_drag[i].value)

    plt.show()