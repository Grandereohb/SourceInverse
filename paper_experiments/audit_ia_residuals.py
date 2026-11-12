"""Read-only station-wise residual audit of frozen IA fits; no NLS or fitting."""

import hashlib

import numpy as np
from scipy.stats import kurtosis,norm

from run_metec_comparison import OUT,read,save


def run():
    records=[]
    normalized=[]
    for identifier in read(OUT/'protocol.json')['event_ids']:
        data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        metadata=read(OUT/'inputs'/identifier/'metadata.json')
        fit=read(OUT/'fits'/identifier/'ia_vpdi.json')
        values=data['concentration']
        valid=np.isfinite(values.ravel())
        prediction=np.full(values.size,np.nan)
        prediction[valid]=fit['prediction']
        residual=prediction.reshape(values.shape)-values
        for index,name in enumerate(metadata['station_names']):
            r=residual[:,index]
            r=r[np.isfinite(r)]
            center=float(np.median(r))
            scale=float(1.4826*np.median(np.abs(r-center)))
            item=dict(event_id=identifier,station=name,count=len(r),
                residual_median_ppm=center,residual_mad_scale_ppm=scale,
                excess_kurtosis=float(kurtosis(r,fisher=True,bias=False)) if len(r)>3 and np.std(r)>0 else None)
            if scale>0:
                z=(r-center)/scale
                item['fraction_above3_mad']=float(np.mean(np.abs(z)>3.))
                item['abs_standardized_p99']=float(np.quantile(np.abs(z),.99))
                normalized.extend(z)
            else:
                item['fraction_above3_mad']=None
                item['warning']='Zero robust scale; no artificial scale floor inserted'
            records.append(item)
    fractions=[row['fraction_above3_mad'] for row in records if row['fraction_above3_mad'] is not None]
    z=np.asarray(normalized)
    result=dict(events=17,station_event_records=len(records),fitting_performed=False,nls_called=False,
        gaussian_reference_fraction_above3=float(2*norm.sf(3)),
        station_median_fraction_above3_mad=float(np.median(fractions)),
        pooled_fraction_above3_mad=float(np.mean(np.abs(z)>3)),
        station_fraction_with_over1pct_above3mad=float(np.mean(np.asarray(fractions)>.01)),
        interpretation='Descriptive residual audit, not a calibrated normality test or proof of sensor outliers',
        caveats=['Residuals can be heavy-tailed because of wind/transport mismatch, not erroneous observations',
            'No observation or high-value station was deleted or changed',
            'A robust observation model would require separate location and peak-fit validation'],
        source_protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
        records=records)
    path=OUT.parent/'ia_residual_audit_20261028.json'
    save(path,result)
    print({key:value for key,value in result.items() if key not in ['records','caveats']},flush=True)


if __name__=='__main__':
    run()
