# Asset provenance

The SO-101 / SO-ARM100 MuJoCo assets in `so101_gym/` were copied from the
ETH Zürich course repository snapshot provided with this project:
`ethz-course-2026/hw3_imitation_learning/so101_gym/assets/`.

The robot model is derived from The Robot Studio's public SO-ARM100 model and
is distributed under Apache License 2.0. The complete upstream license is kept
at `so101_gym/trs_so_arm100/LICENSE` and the accompanying model README is kept
beside it.

The seminar notebook, environment wrapper, scripted expert, experiments, and
explanatory text were written specifically for this course. They do not copy
the ETH homework's Python implementation or assignment prose.

`trs_so_arm100/so_arm100_ee.xml` contains one seminar modification: mesh paths
were made explicit so the included model resolves correctly from the notebook's
scene directory. The model geometry and dynamics were not changed.
