"""Truth-free LHS warm start evaluated by the unchanged author likelihood.

Only preprocessing changes: the author plume, posterior and M-MALA sampler stay
unchanged. Sampling uses SciPy's public LHS implementation rather than pyDOE.
"""

import hashlib

import numpy as np
from scipy.stats import qmc

from paper_newman_adapter import verified_notebook_function


def initialize_from_author_likelihood(inputs, notebook, expected_sha256, samples=1000, seed=1729):
    import jax
    import jax.numpy as jnp
    if samples < 1:
        raise ValueError('Positive LHS sample count required')
    scope = dict(inputs['log_posterior'].__globals__)
    scope.update(initial_betas=inputs['gibbs_params']['background'], data=inputs['data'])
    likelihood = verified_notebook_function(notebook, expected_sha256, scope, 'latin_hypercube_likelihood')
    # Geometry bounds describe the observed experiment terrain. The rate limit
    # uses the published log-rate prior, not the release's known rate or position.
    upper_rate = float(np.exp(inputs['priors'].log_slab_mean+3*np.sqrt(inputs['priors'].log_slab_var)))
    lower = np.array([1e-8, 0., 0., 1e-8, .5, .5, .5, .5])
    upper = np.array([upper_rate, 110., 110., 1e-4, 1.2, 1.2, 1.01, 1.01])
    design = qmc.scale(qmc.LatinHypercube(8, seed=seed).random(samples), lower, upper)
    minutes = len(inputs['data'])//7

    def step(carry, parameters):
        rate, x, y, variance, ah, av, bh, bv = parameters
        score = likelihood(jnp.array([rate]), jnp.array([x]), jnp.array([y]),
                           variance, ah, av, bh, bv, 'Draxler', False, minutes)
        return carry, score

    scores = np.asarray(jax.jit(lambda values: jax.lax.scan(step, 0., values)[1])(jnp.asarray(design)))
    if not np.isfinite(scores).any():
        raise ValueError('No finite author-likelihood initialization')
    index = int(np.argmax(np.where(np.isfinite(scores), scores, -np.inf)))
    rate, x, y, variance, ah, av, bh, bv = design[index]
    inputs['mh_params'] = dict(log_a_H=jnp.log(ah), log_a_V=jnp.log(av),
        log_b_H=jnp.log(bh), log_b_V=jnp.log(bv), log_s=jnp.array([np.log(rate)]),
        source_x=jnp.array([x]), source_y=jnp.array([y]))
    inputs['gibbs_params'] = dict(background=inputs['gibbs_params']['background'],
                                 sigma_squared=jnp.array(variance))
    return dict(method='SciPy LHS with unchanged author latin_hypercube_likelihood',
        samples=samples, seed=seed, parameter_order=['s', 'x', 'y', 'variance', 'aH', 'aV', 'bH', 'bV'],
        lower=lower.tolist(), upper=upper.tolist(), selected_index=index,
        selected_parameters=design[index].tolist(), selected_log_likelihood=float(scores[index]),
        design_sha256=hashlib.sha256(design.tobytes()).hexdigest(), truth_used=False,
        bounds_source='0..110m terrain; rate prior +3SD; author dispersion/initial variance ranges',
        difference_from_author='broad truth-free terrain search, SciPy rather than pyDOE LHS')
