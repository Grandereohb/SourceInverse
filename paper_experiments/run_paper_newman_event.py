"""Single-event adapter to Newman's unchanged Gaussian plume and M-MALA code.

Uses our prespecified observation windows, not author concatenated releases.
Posterior and wind-processing functions are taken unchanged from verified author
notebooks. Short runs validate plumbing only, never posterior convergence.
"""

import argparse
from dataclasses import fields
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import circmean

from audit_paper_controlled_release_data import parse_measurement, parse_measurement_start_label
from paper_newman_adapter import verified_notebook_function, verified_posterior, run_author_sampler

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'result/paper_experiments_20261017'
AUTHOR = ROOT/'deployment_runs/newman_author_verified_v1'


def event_weather(event_dir):
    public = STUDY/'public_data'
    provenance = json.loads((public/'additional_author_sources.json').read_text(encoding='utf-8'))
    expected = next(item['sha256'] for item in provenance['sources']
                    if item['file'] == 'author_source2_processing.ipynb')
    notebook = public/'author_source2_processing.ipynb'
    scope = dict(np=np, pd=pd, circmean=circmean)
    wind = verified_notebook_function(notebook, expected, scope, 'calculate_wind_direction_and_speed')
    resample = verified_notebook_function(notebook, expected, scope, 'resample_and_average')
    average = verified_notebook_function(notebook, expected, scope, 'average_wind_fields')
    manifest = json.loads((event_dir/'manifest.json').read_text(encoding='utf-8'))
    event = manifest['selection']
    frames = []
    with zipfile.ZipFile(public/'chilbolton_original.zip') as archive:
        for beam in range(1, 8):
            name = f'{event["archive_directory"]}/POS{beam}_{event["release"]}_processed.txt'
            lines, table, columns = parse_measurement(archive.read(name).decode('ascii'))
            base = parse_measurement_start_label(lines[lines.index('Time stamp for record start')+1])
            df = pd.DataFrame(table, columns=columns)
            required = ['Elapsedtime_sec', 'WindX_m/s', 'WindY_m/s', 'WindZ_m/s', 'CH4vmr_ppmv']
            df = df.loc[np.isfinite(df[required].to_numpy()).all(axis=1)].copy()
            df['Clock_Time'] = pd.Timestamp(base)+pd.to_timedelta(df['Elapsedtime_sec'], unit='s')
            df['Measurements'] = df['CH4vmr_ppmv']
            frames.append(resample(wind(df, 'WindX_m/s', 'WindY_m/s', 'WindZ_m/s')))
    return average(frames)


def build_event(event_dir, height, minutes=None):
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import sourceinversion.atmospheric_measurements as gp
    import sourceinversion.mcmc as mcmc
    import tensorflow_probability.substrates.jax as tfp

    observations = pd.read_csv(event_dir/'observations.csv', parse_dates=['time'])
    pivot = observations.pivot(index='time', columns='beam', values='residual_ppm').sort_index()
    if minutes is not None:
        pivot = pivot.iloc[:minutes]
    if list(pivot.columns) != list(range(1, 8)) or not np.isfinite(pivot.to_numpy()).all():
        raise ValueError('Author seven-beam adapter requires complete retained minutes')
    weather = event_weather(event_dir).reindex(pivot.index)
    if not np.isfinite(weather.to_numpy()).all() or (weather['Average Speed'] <= 0).any():
        raise ValueError('Author wind statistics must be finite with positive speed')
    # Keep the same pre-window enhancement as IA, restore the author reference
    # background so the unchanged posterior can estimate seven background terms.
    data = (pivot.to_numpy().T.reshape(-1, 1)+1.92)
    geometry = pd.read_csv(event_dir/'beams.csv').sort_values('beam')
    counts = [90, 165, 110, 245, 210, 145, 85]
    locations = []
    for row, count in zip(geometry.itertuples(), counts):
        locations.extend(np.linspace([row.start_x_m, row.start_y_m, row.start_z_m],
                                     [row.end_x_m, row.end_y_m, row.end_z_m], count).tolist())
    grid = gp.Grid((jnp.array(0.), jnp.array(110.)), (jnp.array(0.), jnp.array(110.)),
                   (jnp.array(0.), jnp.array(0.)), jnp.array(10.), jnp.array(10.), jnp.array(1.))
    # Placeholder xy and rate are not truth and are not used by the parameterized
    # grid-free unit-response call; source height is a prescribed sensitivity.
    location = gp.SourceLocation(jnp.array([50.]), jnp.array([50.]), jnp.array([height]))
    atmosphere_args = {item.name: None for item in fields(gp.AtmosphericState)}
    atmosphere_args.update(emission_rate=jnp.array([1.]), source_half_width=jnp.array(1.),
                           max_abl=jnp.array(1000.))
    atmosphere = gp.AtmosphericState(**atmosphere_args)
    sensors = gp.SensorsSettings(None, jnp.array(7), None, None, None, locations)
    plume = gp.GaussianPlume(grid, location, weather, atmosphere, sensors)
    fixed = plume.fixed_objects_of_gridfree_chilbolton_coupling_matrix(
        simulation=False, wind_direction=weather['Average Direction'].values,
        wind_speed=weather['Average Speed'].values, tangamma_ts=weather,
        number_of_time_steps=len(pivot))
    provenance = json.loads((AUTHOR/'provenance.json').read_text(encoding='utf-8'))
    expected = next(item['sha256'] for item in provenance['files']
                    if item['file'] == 'author_source2_inversion.ipynb')
    scope = dict(gaussianplume=plume, jnp=jnp, tfd=tfp.distributions)
    for i in range(7):
        scope[f'fixed_ref{i+1}'] = (fixed[i], fixed[7+i], fixed[14+3*i], fixed[15+3*i],
            fixed[35], fixed[36], fixed[16+3*i], fixed[37+i], fixed[44+i])
    posterior = verified_posterior(AUTHOR/'author_source2_inversion.ipynb', expected, scope)
    priors = mcmc.Priors(theta=.1, log_spike_mean=-25., log_spike_var=10.,
        log_slab_mean=-7.5, log_slab_var=1.5, source_location_x_mean=50.,
        source_location_y_mean=50., source_location_x_var=25., source_location_y_var=25.,
        sigma_squared_con=1e-11, sigma_squared_rate=1e-8, mean_background_prior=1.92,
        variance_background_prior=.1**2, a_mean=jnp.log(.6), a_var=.5**2,
        b_mean=jnp.log(.6), b_var=.2**2)
    mh = dict(log_a_H=jnp.log(.6), log_a_V=jnp.log(.6), log_b_H=jnp.log(.6),
              log_b_V=jnp.log(.6), log_s=jnp.array([-7.5]),
              source_x=jnp.array([50.]), source_y=jnp.array([50.]))
    gibbs = dict(background=jnp.full(7, 1.92), sigma_squared=jnp.array(.01))
    return dict(gaussianplume=plume, data=data, log_posterior=posterior,
                priors=priors, mh_params=mh, gibbs_params=gibbs, fixed=fixed), weather


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event', default='source2_rel06')
    parser.add_argument('--height', type=float, choices=(.3, 1.6), default=.3)
    parser.add_argument('--minutes', type=int)
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--initialization', choices=('generic', 'author-lhs'), default='generic')
    parser.add_argument('--lhs-samples', type=int, default=1000)
    parser.add_argument('--lhs-seed', type=int, default=1729)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    event_dir = STUDY/'controlled_release_preparation_v2'/args.event
    started = time.perf_counter()
    inputs, weather = build_event(event_dir, args.height, args.minutes)
    initialization = dict(method='fixed generic initial state', truth_used=False)
    if args.initialization == 'author-lhs':
        from paper_newman_initialization import initialize_from_author_likelihood
        provenance = json.loads((AUTHOR/'provenance.json').read_text(encoding='utf-8'))
        expected = next(item['sha256'] for item in provenance['files']
                        if item['file'] == 'author_source2_inversion.ipynb')
        initialization = initialize_from_author_likelihood(inputs,
            AUTHOR/'author_source2_inversion.ipynb', expected, args.lhs_samples, args.lhs_seed)
    (args.output/'initialization.json').write_text(json.dumps(initialization, indent=2), encoding='utf-8')
    weather.to_csv(args.output/'author_wind_statistics.csv', index_label='time')
    np.save(args.output/'observations_author_order.npy', inputs['data'])
    chains = run_author_sampler(**inputs, iterations=args.iterations, initial_step_size=.5)
    arrays = {key: np.asarray(value) for key, value in chains.items()}
    np.savez_compressed(args.output/'chains.npz', **arrays)
    required = ('source_x', 'source_y', 's', 'll', 'sigma_squared', 'background')
    finite = all(np.isfinite(arrays[key]).all() for key in required)
    report = dict(author_implementation='sourceinversion0.1.5 Manifold_MALA_Within_Gibbs',
        event=args.event, height_m=args.height, minutes=len(weather), iterations=args.iterations,
        finite_required_chains=finite, seconds=time.perf_counter()-started,
        status='technical smoke only; posterior convergence NOT assessed',
        adaptation=['single event, not author concatenated releases',
            'same pre-window residual plus1.92ppm reference; author estimates background',
            'unchanged author wind processing, plume, posterior and sampler',
            'author Source2 normal location priors retained, NOT uniform IA search box',
            'initialization recorded separately; never initialized from true coordinates'],
        initialization=initialization,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        last_acceptance_rate=np.asarray(arrays['acceptance_rate'][-1]).tolist())
    report['runtime_versions'] = {name: importlib.metadata.version(name)
        for name in ('sourceinversion', 'jax', 'jaxlib', 'numpy', 'scipy', 'tensorflow-probability')}
    report['author_provenance'] = json.loads((AUTHOR/'provenance.json').read_text(encoding='utf-8'))['files']
    report['input_sha256'] = {name: hashlib.sha256((event_dir/name).read_bytes()).hexdigest()
        for name in ('observations.csv', 'beams.csv', 'manifest.json')}
    if args.minutes is None and args.iterations >= 10000:
        import tensorflow_probability.substrates.jax as tfp
        burn = args.iterations//2
        posterior = {key: np.asarray(arrays[key][burn:]).reshape(args.iterations-burn, -1)
                     for key in ('source_x', 'source_y', 's')}
        report['status'] = 'full-window fixed-budget author sampling; not automatic convergence certification'
        report['burn_in'] = burn
        report['posterior_mean'] = {key: value.mean(axis=0).tolist() for key, value in posterior.items()}
        report['posterior_95_interval'] = {key: np.quantile(value, [.025, .975], axis=0).tolist()
                                           for key, value in posterior.items()}
        report['effective_sample_size'] = {key: np.asarray(tfp.mcmc.effective_sample_size(value)).tolist()
                                            for key, value in posterior.items()}
        report['convergence_warning'] = 'one chain; ESS reported, no independent-chain R-hat available'
        # Evaluation truth is read only after the sampler has returned its output.
        source = 'source_'+args.event.split('_')[0].removeprefix('source')
        truth = json.loads((STUDY/'public_data/author_truth_values.json').read_text(encoding='utf-8'))
        reference = np.asarray(truth['values_as_stored'][source+'_location'][:2])
        estimate = np.array([posterior['source_x'].mean(), posterior['source_y'].mean()])
        report['localization_error_m'] = float(np.linalg.norm(estimate-reference))
        report['posterior_mean_q_kg_h'] = float(posterior['s'].mean()*3600.)
    (args.output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if not finite:
        raise FloatingPointError('Nonfinite author chains; do not use as completed fit')


if __name__ == '__main__':
    main()
