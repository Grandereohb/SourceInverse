"""Thin entry to the published sourceinversion M-MALA-within-Gibbs sampler.

Reuse the author's posterior function from a hash-pinned notebook, never execute
the notebook's data loading, plotting or truth-based setup cells. Construction of
case inputs and priors is a separate, auditable preprocessing step.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import urllib.request
import zipfile

REPOSITORY = 'https://raw.githubusercontent.com/NewmanTHP/Probabilistic-Inversion-Modeling-of-Gas-Emissions/master/'
NOTEBOOK = ('Code/Chilbolton_Case_Study (Section 5 + SM.B 2)/'
            'Inversion (Section 5 + SM.B 2.5-2.7)/Chilbolton Source 2 inversion.ipynb')


def verified_notebook_function(notebook, expected_sha256, namespace, name):
    """Compile one unchanged author function, never its surrounding setup cells."""
    content = Path(notebook).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('Author notebook hash mismatch')
    document = json.loads(content)
    functions = []
    for cell in document['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if f'def {name}(' not in source:
            continue
        tree = ast.parse(source)
        functions.extend(node for node in tree.body
                         if isinstance(node, ast.FunctionDef) and node.name == name)
    if len(functions) != 1 or functions[0].decorator_list:
        raise ValueError('Expected exactly one undecorated author posterior')
    module = ast.Module(body=functions, type_ignores=[])
    scope = dict(namespace)
    exec(compile(ast.fix_missing_locations(module), str(notebook), 'exec'), scope)
    return scope[name]


def verified_posterior(notebook, expected_sha256, namespace):
    """Compile only the unchanged author log_posterior definition."""
    return verified_notebook_function(notebook, expected_sha256, namespace, 'log_posterior')


def run_author_sampler(*, gaussianplume, data, log_posterior, priors,
                       mh_params, gibbs_params, fixed, iterations, initial_step_size):
    """The exact sampler entry used by the Source 2 inversion notebook."""
    if iterations < 1 or initial_step_size <= 0:
        raise ValueError('Positive sampling budget and step size required')
    from jax.flatten_util import ravel_pytree
    import sourceinversion.mcmc as mcmc

    mh_flat, _ = ravel_pytree(mh_params)
    sampler = mcmc.Manifold_MALA_Within_Gibbs(
        False, gaussianplume, data, log_posterior, priors, mh_params, gibbs_params,
        fixed, chilbolton=True, wind_sigmas=True, release_17=False,
        step_size_tuning='False')
    return sampler.manifold_mala_chains(gibbs_params, mh_flat, iterations,
                                        initial_step_size, release_17=False)


def prepare_sources(output):
    """Download original code only; do not install or run the author notebook."""
    from urllib.parse import quote
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with urllib.request.urlopen('https://pypi.org/pypi/sourceinversion/0.1.5/json', timeout=30) as request:
        metadata = json.load(request)
    wheel = next(item for item in metadata['urls'] if item['filename'].endswith('.whl'))
    files = [('author_source2_inversion.ipynb', REPOSITORY+quote(NOTEBOOK, safe='/'), None),
             (wheel['filename'], wheel['url'], wheel['digests']['sha256'])]
    records = []
    for name, url, expected in files:
        with urllib.request.urlopen(url, timeout=60) as request:
            content = request.read()
        digest = hashlib.sha256(content).hexdigest()
        if expected and digest != expected:
            raise ValueError('Author distribution download checksum mismatch')
        (output/name).write_bytes(content)
        records.append(dict(file=name, url=url, sha256=digest))
    with zipfile.ZipFile(output/wheel['filename']) as archive:
        source = archive.read('sourceinversion/mcmc/mcmc.py').decode()
    module = ast.parse(source)
    classes = [node.name for node in module.body if isinstance(node, ast.ClassDef)]
    if 'Manifold_MALA_Within_Gibbs' not in classes:
        raise ValueError('Expected author sampler missing from distribution')
    report = dict(author_version='0.1.5', files=records,
        python_requirement=metadata['info']['requires_python'],
        dependencies=metadata['info']['requires_dist'], current_python=platform.python_version(),
        runtime_verified=False, notebook_executed=False, sampler_classes=classes,
        next_step='isolated compatible environment, author demo, then single-event input adaptation')
    (output/'provenance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-sources', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_sources(args.prepare_sources), indent=2))
