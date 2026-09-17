# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Wrapper around the affine-invariant ensemble sampler of Goodman and
# Weare (2010), as implemented in emcee, with an HDF5 backend that provides
# checkpointing, resume and a wall-time guard.
# Manuscript: Section 4.4 for the two-dimensional synthetic runs and
# Section 4.5 for the wolf application.
# Inputs: a log posterior callable and sampler settings. Outputs: the chain of
# every walker and the acceptance fraction.

import time  # Wall clock measurement used by the wall-time guard.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and random number generation.


# Run the affine-invariant ensemble sampler with checkpointing and resume.
# Arguments:
#   log_posterior (callable): unnormalised log posterior in log space.
#   initial (numpy.ndarray): starting positions of shape (n_walkers, n_dim).
#   n_steps (int): number of ensemble steps requested in total.
#   backend_path (str or pathlib.Path): HDF5 file that stores the chain.
#   resume (bool): whether an existing backend is continued.
#   wall_time_limit (float): wall time in seconds after which the run stops.
#   seed (int): seed of the sampler, recorded in the output metadata.
#   progress_every (int): number of steps between two wall-time checks.
# Returns:
#   dict: the chain of shape (steps, walkers, parameters), the log posterior
#   trace, the acceptance fraction and the number of completed steps.
def run_ensemble(
    log_posterior,  # Unnormalised log posterior density in log space.
    initial,  # Starting positions of the walkers.
    n_steps,  # Number of ensemble steps requested in total.
    backend_path=None,  # HDF5 file that stores the chain between blocks.
    resume=False,  # Whether an existing backend is continued.
    wall_time_limit=None,  # Wall time in seconds after which the run stops.
    seed=None,  # Seed of the sampler, recorded in the output metadata.
    progress_every=50,  # Number of steps between two wall-time checks.
):  # End of the argument list.
    import emcee  # Imported lazily so that the package imports without emcee.

    started = time.perf_counter()  # Wall clock reading at the start of the run.
    positions = np.array(initial, dtype=float)  # Starting positions of the walkers.
    n_walkers, n_dim = positions.shape  # Number of walkers and of sampled parameters.
    backend = None  # HDF5 backend of the sampler, created only when requested.
    if backend_path is not None:  # Checkpointing to an HDF5 backend was requested.
        location = Path(backend_path)  # Normalise the argument into a path object.
        location.parent.mkdir(parents=True, exist_ok=True)  # Create the containing directory.
        backend = emcee.backends.HDFBackend(str(location))  # Backend bound to that file.
        if not (resume and location.is_file()):  # A fresh run, or no backend to resume.
            backend.reset(n_walkers, n_dim)  # Clear any previous contents of the backend.
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior, backend=backend)  # Sampler.
    completed = sampler.iteration if backend is not None else 0  # Steps already stored.
    state = None  # Starting state, taken from the backend when a run is resumed.
    if completed > 0:  # A previous block of this run has already been stored.
        state = sampler.get_last_sample()  # Continue from the last stored ensemble state.
    else:  # No previous block exists, so the supplied positions are used.
        state = positions  # Starting positions supplied by the caller.
    remaining = int(n_steps) - completed  # Number of ensemble steps still to be performed.
    while remaining > 0:  # Perform the remaining steps in blocks of progress_every.
        block = min(int(progress_every), remaining)  # Number of steps in the current block.
        state = sampler.run_mcmc(state, block, progress=False)  # Advance the ensemble.
        remaining = remaining - block  # Update the number of steps still to be performed.
        elapsed = time.perf_counter() - started  # Wall time consumed by the run so far.
        if wall_time_limit is not None and elapsed > wall_time_limit:  # Guard has fired.
            break  # Stop cleanly so that a dependent job can continue the run.
    chain = sampler.get_chain()  # Chain of shape (steps, walkers, parameters).
    log_prob = sampler.get_log_prob()  # Log posterior value of every stored state.
    acceptance = float(np.mean(sampler.acceptance_fraction))  # Mean acceptance fraction.
    # Assemble the chain together with the diagnostics of the run.
    return {
        "chain": chain,  # Chain of every walker, one slab per ensemble step.
        "log_prob": log_prob,  # Log posterior value of every stored state.
        "acceptance_rate": acceptance,  # Mean acceptance fraction over the walkers.
        "iterations": int(sampler.iteration),  # Number of ensemble steps completed.
        "complete": int(sampler.iteration) >= int(n_steps),  # Whether the run finished.
        "seed": seed,  # Seed recorded for the output metadata.
    }


# Draw starting positions of the walkers from the prior.
# Arguments:
#   prior (nmi.priors.LogUniformPrior): the prior on the sampled parameters.
#   n_walkers (int): number of walkers of the ensemble.
#   rng (numpy.random.Generator): generator used for the draw.
# Returns:
#   numpy.ndarray: starting positions of shape (n_walkers, n_dim), in log space.
def initial_positions(prior, n_walkers, rng):
    return prior.sample(rng, size=int(n_walkers))  # Independent draws from the log-uniform prior.
