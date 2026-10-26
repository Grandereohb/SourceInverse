"""Inventory downloaded Chilbolton originals without inventing source geometry."""

import ast
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import pickle
import pickletools
import re
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "result/paper_experiments_20261017/public_data"


def parse_measurement_start_label(label):
    # Generator pads a one-digit seconds field with a space after the colon.
    for format_string in ('%m/%d/%Y %H:%M:%S.%f','%m/%d/%Y %H:%M: %S.%f'):
        try:
            return datetime.strptime(label,format_string)
        except ValueError:
            continue
    raise ValueError(f'Unsupported original measurement timestamp: {label!r}')


def parse_measurement(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    matches = [i for i,line in enumerate(lines) if line.startswith('Elapsed')]
    if len(matches) != 1:
        raise ValueError('Exactly one measurement-table header required')
    header = matches[0]
    old_header = lines[header].startswith('Elapsed time')
    data = np.loadtxt(io.StringIO('\n'.join(lines[header+(2 if old_header else 1):])),ndmin=2)
    columns = (['Elapsedtime_sec','Temp_K','Press_Pa','Humidity_ppmv',
                'WindX_m/s','WindY_m/s','WindZ_m/s','Speedofsound_m/s',
                'CH4vmr_ppmv','CH4vmrerror_ppmv'] if old_header else lines[header].split())
    if data.shape[1] != len(columns) or len(columns) != 10:
        raise ValueError('Unexpected measurement-table shape')
    return lines,data,columns


def main():
    archive = OUTPUT / "chilbolton_original.zip"
    raw = archive.read_bytes()
    metadata = json.loads((OUTPUT / "original_bitstream.json").read_text(encoding="utf-8"))
    if len(raw) != metadata["sizeBytes"] or hashlib.md5(raw).hexdigest() != metadata["checkSum"]["value"]:
        raise ValueError("Original repository checksum or size does not match")
    records = []
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.lower().endswith('.txt') or 'processed' not in name.lower():
                continue
            text = z.read(name).decode('ascii')
            lines,data,columns = parse_measurement(text)
            length_index = next(i for i, line in enumerate(lines) if line == 'Open Path length (m)')
            angle_index = next(i for i, line in enumerate(lines) if line == 'Beam angle (degrees)')
            start_index = next(i for i, line in enumerate(lines) if line == 'Time stamp for record start')
            concentration = data[:, columns.index('CH4vmr_ppmv')]
            records.append({"archive_member": name, "sha256": hashlib.sha256(z.read(name)).hexdigest(),
                "date": name.split('/')[0], "release": name.split('/')[1], "beam": name.split('/')[-1].split('_')[0],
                "start_label_original": lines[start_index+1], "rows": len(data),
                "duration_seconds": float(data[-1, 0]-data[0, 0]),
                "path_length_m": float(lines[length_index+1]),
                "beam_angle_original_degrees": float(lines[angle_index+1]),
                "nonfinite_entries": int((~np.isfinite(data)).sum()),
                "ch4_min_ppm": float(np.nanmin(concentration)),
                "ch4_max_ppm": float(np.nanmax(concentration))})
        license_readme = z.read('README.txt').decode('ascii')
        if 'CC-BY' not in license_readme:
            raise ValueError("Expected license declaration is absent")
        mass_files = [name for name in z.namelist() if name.lower().endswith('.csv')]
    with (OUTPUT / "inventory.csv").open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {"audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_title": "Methane Emissions: Remote Mapping and Source Quantification using an Open-path Laser Dispersion Spectrometer",
        "original_record": "https://edata.stfc.ac.uk/items/5c88d121-0e19-4840-a26b-499dba49419a/full",
        "download_url": metadata['_links']['content']['href'],
        "attribution": "Damien Weidmann, STFC Chilbolton Observatory, May8-12 2017",
        "license": "CC-BY (README); original record links CC-BY4.0",
        "archive_bytes": len(raw), "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "repository_md5_verified": True, "measurement_files": len(records),
        "measurement_rows": sum(r['rows'] for r in records),
        "release_directories": sorted({r['date'] + '/' + r['release'] for r in records}),
        "cylinder_mass_files": mass_files,
        "observation_type": "open-path methane mixing ratio, NOT point concentration",
        "analysis_status": "downloaded and inventoried; NO inversion performed",
        "required_before_section_4_3": [
            "Verify source locations, exact release schedules, sensor origin and beam coordinate rotation",
            "Implement and test path-averaged observation operator, not beam midpoint approximation",
            "Align methane mixing ratio and the model amplitude definition",
            "For kg/h comparison establish a vertical/mixing-height physical model; do not calibrate on evaluation truth",
            "Handle background and second-scale acquisition without imposing hourly averaging",
            "Freeze event inclusion and distinguish single-source from simultaneous multi-source releases"]}
    truth_path = OUTPUT / 'author_source_truth.pkl'
    if truth_path.exists():
        truth_raw = truth_path.read_bytes()
        allowed = {'PROTO', 'FRAME', 'EMPTY_DICT', 'MEMOIZE', 'MARK', 'SHORT_BINUNICODE',
                   'EMPTY_LIST', 'BINFLOAT', 'APPENDS', 'SETITEMS', 'STOP'}
        if any(op.name not in allowed for op, _, _ in pickletools.genops(truth_raw)):
            raise ValueError('Source truth contains non-primitive pickle operations; do not execute it')
        truth = pickle.loads(truth_raw)
        truth_record = {'values_as_stored': truth, 'sha256': hashlib.sha256(truth_raw).hexdigest(),
            'source_url': 'https://raw.githubusercontent.com/NewmanTHP/Probabilistic-Inversion-Modeling-of-Gas-Emissions/master/Data/Chilbolton_data_files/Postprocessed/Source_locations_and_emission_rates/Chilbolton_sources_locations_and_emission_rates.pkl',
            'units_and_coordinate_orientation': 'pending author-method verification; not used as inversion inputs'}
        (OUTPUT / 'author_truth_values.json').write_text(json.dumps(truth_record, indent=2), encoding='utf-8')
        report['author_truth_primitive_values_obtained'] = True
    notebook_path = OUTPUT / 'author_source1_processing.ipynb'
    if notebook_path.exists():
        notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
        metadata = {'notebook_sha256': hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
                    'notebook_executed': False, 'source_label': 'source_1'}
        release_records = set()
        for cell in notebook['cells']:
            source = ''.join(cell.get('source', []))
            release_records.update(re.findall(r'/Preprocessed/(\d{8})/(rel\d{2})/', source))
            for variable, key in (('instruments_location', 'instrument_coordinates_as_printed'),
                                  ('source_locations', 'source_coordinates_as_printed')):
                if f'print({variable})' in source:
                    for item in cell.get('outputs', []):
                        printed = ''.join(item.get('text', [])).strip()
                        if printed.startswith('{'):
                            value = ast.literal_eval(printed)
                            if not isinstance(value, dict) or not all(
                                isinstance(v, list) and len(v) == 3 and
                                all(isinstance(x, (int, float)) for x in v) for v in value.values()):
                                raise ValueError('Unexpected primitive coordinate dictionary')
                            metadata[key] = value
        metadata['source1_release_paths'] = sorted('/'.join(parts) for parts in release_records)
        metadata['author_horizontal_wind_direction_formula'] = 'atan2(wind_x, -wind_y) in degrees modulo360; physical from/to convention must be verified with the forward model'
        metadata['gap_handling_warning'] = 'Author concatenates separate sub-releases for a quasi-steady model; do not treat them as one continuous PDE trajectory'
        metadata['height_discrepancy'] = 'Printed processing-notebook source coordinates use z=1.6; author final truth pickle uses z=0.3. Preserve both; resolve before3D mass-rate validation.'
        (OUTPUT/'author_processing_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        report['processing_notebook_reviewed_not_executed'] = True
        report['height_metadata_conflict_pending'] = True
    (OUTPUT / 'provenance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('repository_md5_verified', 'measurement_files',
        'measurement_rows', 'analysis_status')}, ensure_ascii=False))


def extract_author_release_windows():
    """Read literal time filters and path dates; never execute downloaded code."""
    records, provenance = [], []
    for source in (1,2):
        path = OUTPUT/f'author_source{source}_processing.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        dates, windows = {}, {}
        for index,cell in enumerate(notebook['cells']):
            code = ''.join(cell.get('source',[]))
            for date,release in re.findall(r'/Preprocessed/(\d{8})/rel(\d{2})/',code):
                dates[int(release)] = date
            if cell.get('cell_type')!='code':
                continue
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node,ast.Assign):
                    continue
                names = [target.id for target in node.targets if isinstance(target,ast.Name)]
                for name in names:
                    match = re.fullmatch(r'reflector_1_rel(\d+)_filtered',name)
                    if match is None:
                        continue
                    stamps = [call.args[0].value for call in ast.walk(node.value)
                        if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                        and call.func.attr=='Timestamp' and call.args
                        and isinstance(call.args[0],ast.Constant) and isinstance(call.args[0].value,str)]
                    if len(stamps)==2:
                        release = int(match.group(1))
                        candidate = dict(start=stamps[0],end=stamps[1],notebook_cell=index)
                        if release in windows and windows[release]!=candidate:
                            raise ValueError('Conflicting author time filters')
                        windows[release] = candidate
        for release,date in sorted(dates.items()):
            record = dict(source=f'source_{source}',release=f'rel{release:02d}',date=date,
                          archive_directory=f'{date}/rel{release:02d}')
            if source==2 and release==16:
                record.update(eligible=False,exclusion='Author notebook cell51 documents bad reflector2 measurements; event excluded before inversion')
            elif release not in windows:
                raise ValueError(f'Missing documented release interval: {record}')
            else:
                record.update(eligible=True,**windows[release])
            records.append(record)
        provenance.append(dict(source=source,path=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),executed=False))
    result = dict(events=records,provenance=provenance,
        selection='All author-listed single-source releases with documented windows; no inversion outcomes used',
        warning='Eligibility is metadata-only; raw observation/background/cylinder checks still required')
    target = OUTPUT/'author_release_windows.json'
    if target.exists():
        raise FileExistsError('Preserve existing release metadata; use a new version for changes')
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-windows',action='store_true')
    arguments = parser.parse_args()
    extract_author_release_windows() if arguments.release_windows else main()
