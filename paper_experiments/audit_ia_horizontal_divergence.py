"""Read-only wind-only audit; does not infer a source or tune an inversion."""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from run_metec_comparison import ROOT, OUT, AUDIT, STUDY, read, save
from run_ia_wind_ensemble_pilot import load_wind_members
from ia_metec_point_adapter import xy_from_lon_lat
from ia_horizontal_divergence import fit_horizontal_affine
from paper_affine_wind import fit_incompressible_wind

DEST = STUDY / 'ia_horizontal_divergence_audit_20261028'


def run():
    for name, digest in read(ROOT / 'paper_experiments/nls_baseline_frozen.json')['sha256_lf'].items():
        if hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise ValueError('Frozen NLS changed: ' + name)
    protocol = read(OUT / 'protocol.json')
    raw = pd.read_csv(AUDIT / 'raw/input_data/ADED_data_clean_SAMPLE.csv')
    raw['timestamp'] = pd.to_datetime(raw.time, utc=True) - pd.Timedelta(minutes=1)
    sensors = pd.read_csv(AUDIT / 'raw/input_data/sensor_locations.csv').set_index('name')
    names = ['E', 'NW', 'SW']
    rows = []
    for identifier in protocol['event_ids']:
        data = dict(np.load(OUT / 'inputs' / identifier / 'input.npz'))
        meta = read(OUT / 'inputs' / identifier / 'metadata.json')
        winds, missing = load_wind_members(raw, pd.to_datetime(data['time_utc'], utc=True), names, data['wind_xy'])
        wind = np.stack(winds, axis=1)
        positions = xy_from_lon_lat(sensors.loc[names].lon, sensors.loc[names].lat, meta['origin'])
        fit = fit_horizontal_affine(positions, wind)
        _, constrained, fitted = fit_incompressible_wind(positions, wind)
        div = fit['divergence']
        row = dict(event_id=identifier, minutes=len(div),
            median_abs_horizontal_divergence_per_s=float(np.median(np.abs(div))),
            p95_abs_horizontal_divergence_per_s=float(np.quantile(np.abs(div), .95)),
            median_signed_horizontal_divergence_per_s=float(np.median(div)),
            hypothetical_w_at_2m_rms_m_s=float(2 * np.sqrt(np.mean(div ** 2))),
            constrained_wind_rms_m_s=float(np.sqrt(np.mean((fitted - wind) ** 2))),
            unconstrained_wind_rms_m_s=float(np.sqrt(np.mean((fit['fitted'] - wind) ** 2))),
            normalized_design_condition=fit['design_condition'],
            residual_dof_per_time=fit['residual_degrees_of_freedom_per_time'],
            global_wind_fallback_minutes=sum(item['global_fallback_minutes'] for item in missing))
        save(DEST / identifier / 'wind_fit.json', dict(**fit, missing=missing,
             source_truth_accessed=False, inverted_source_not_modified=True))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(DEST / 'summary.csv', index=False, encoding='utf-8-sig')
    save(DEST / 'status.json', dict(wind_audit_complete=True, events=len(rows),
        source_truth_accessed=False, concentration_fits=False, nls_called=False,
        production_promoted=False, vertical_wind_observed=False,
        caveat='Nonzero fitted horizontal divergence can reflect measurement error or nonlinear flow; three stations leave no independent residual degrees of freedom',
        next_step='Do not adopt a 3D vertical closure solely because a saturated horizontal fit has zero residual',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(frame.to_string(index=False))


if __name__ == '__main__':
    run()
