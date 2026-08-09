# Cover Letter - Machine Learning: Science and Technology

Dear Editors,

We submit our manuscript, *"Information-limited fidelity estimation of composed quantum channels: when to compose, measure, or learn"*, for consideration as a
research article in *Machine Learning: Science and Technology*.

The paper studies a basic question in scientific machine learning. When is a
learned estimator useful once the information available at inference and the
cost of measurements or simulations are taken into account? We examine this
question for the end-to-end entanglement fidelity of composed quantum channels.

Our results separate three cases. When complete Markovian Choi descriptions are
available, direct superoperator composition is both numerically exact and faster
than the learned models at the dimensions studied. When the deployed process can
be measured, low-shot Direct Fidelity Estimation is a strong baseline. In the
non-Markovian collision benchmark, the observed marginal channels omit bath
retention. The same observed input can therefore correspond to several target
fidelities.

The main contributions are as follows.

1. We formulate a representation audit that tests whether the supplied input
   identifies the requested target before model fitting. It gives fibre-wise
   minimax and distribution-dependent Bayes error floors for ambiguous inputs.
2. We construct a counterfactual collision experiment that replays 2048 fixed
   observable inputs at 15 hidden retention values. This experiment measures the
   error caused by missing information without changing the model input.
3. We combine a calibrated marginal prediction with a same-query measurement
   pilot. On a fresh split generated with independent random streams, the
   32-shot and 64-shot hybrids approach the accuracy of 64-shot and 96-shot
   Direct Fidelity Estimation. Their full calibration costs amortise after 192
   and 256 deployment queries.
4. We compare these results with exact composition, analytic approximations,
   Monte Carlo estimators, label-budget controls, and Direct Fidelity Estimation
   under explicit cost accounting.

The contribution is methodological rather than architecture-centred. The
representation audit and information-conditioned comparison apply to scientific
surrogates whenever different physical states can produce the same model input.
This focus fits the journal's coverage of machine-learning methodology developed
through scientific applications.

The manuscript is original, has not been published elsewhere, and is not under
consideration by another journal. The authors declare no competing interests.
The public repository includes the manuscript-specific release tagged
mlst-submission-v1. It contains the data generators, evaluation scripts, random
seeds, and machine-readable result tables used in the study. The tag identifies
the exact commit, and the software environment is documented in the repository.

Thank you for considering the manuscript.

Sincerely,

Shuchang Wang, Xiaoman Liu, and Wei Yang
University of Science and Technology of China, Hefei, China
Corresponding author: Wei Yang, qubit@ustc.edu.cn
