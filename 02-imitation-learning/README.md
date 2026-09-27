# Seminar 02 — Imitation Learning

Materials for a 90-minute seminar on behavior cloning, distribution shift,
and DAgger with the SO-101 robot in MuJoCo.

The student notebook is located at
`notebooks/seminar_02_imitation_learning_student.ipynb`. It contains the graded
TODOs for Behavior Cloning and DAgger and the optional goal-conditioned task.

## Run in Google Colab

1. Open the student notebook from GitHub in Colab, or download
   `notebooks/seminar_02_imitation_learning_student.ipynb` and upload it at
   <https://colab.research.google.com/>.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run the notebook from the first cell. The setup cells:
   - install MuJoCo, plotting, widget, and video dependencies;
   - set `MUJOCO_GL=egl` for headless rendering;
   - locate the seminar files or clone
     `https://github.com/wingrune/multimodal-vla-course.git` into
     `/content/multimodal-vla-course` when only the notebook was uploaded;
   - add `notebooks/` to `sys.path` and import `seminar_il.py`;
   - report the Python, PyTorch, and accelerator configuration.
4. Check that the setup output contains `device=cuda`. If it prints
   `device=cpu`, attach a GPU runtime before starting the training tasks.
5. Continue with **Runtime → Run all**, or execute the cells sequentially while
   completing the student TODOs.

If a previous Colab attempt left an incomplete `/content/multimodal-vla-course`
checkout, rerun the two setup cells. The notebook updates an existing Git
checkout and prints the exact module path it imports.
If that path exists but is not a Git checkout, remove or rename it first.

PyTorch is supplied by the Colab GPU runtime and is deliberately not reinstalled.
The remaining packages installed by the notebook are:

```text
mujoco>=3.2,<4
matplotlib>=3.8
pandas>=2
ipywidgets>=8
imageio>=2.34
imageio-ffmpeg>=0.5
pillow>=10
```

The MP4 visualizations are embedded in notebook outputs, so no browser video
files need to be downloaded separately. The browser teleoperation widget is an
optional warm-up; the required BC and DAgger experiments use reproducible
scripted demonstrations and do not require interactive desktop windows.

The full teacher notebook has been tested with CUDA. A complete student run is
designed for approximately 10–15 minutes on a Colab T4, although runtime can
vary with Colab load.

## Run locally

Create an environment with CUDA-enabled PyTorch and the packages listed above,
start Jupyter from this repository, and run either notebook. The setup code
automatically finds `02-imitation-learning` when the repository layout is kept
intact. MuJoCo rendering remains offscreen through EGL.

The SO-101 model assets retain their upstream Apache-2.0 license; see
`assets/PROVENANCE.md`.
