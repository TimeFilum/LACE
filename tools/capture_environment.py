"""Print package versions needed to complete the research environment record."""
import importlib.metadata
import json
import platform

PACKAGES = [
    'torch','uni2ts','transformers','chronos-forecasting','timesfm',
    'numpy','scipy','pandas','scikit-learn','einops','matplotlib',
    'huggingface-hub','lightning','gluonts','jax','python-dotenv',
]


def capture():
    result = {'python':platform.python_version(),'packages':{}}
    for name in PACKAGES:
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][name] = None
            continue
        result['packages'][name] = dist.version
        if name=='uni2ts':
            text = dist.read_text('direct_url.json')
            if text:
                info = json.loads(text)
                # Retain the commit, not local paths or authenticated URLs.
                if 'vcs_info' in info:
                    result['uni2ts_vcs'] = info['vcs_info']
                elif info.get('dir_info',{}).get('editable'):
                    result['uni2ts_editable_install'] = True
                    result['uni2ts_commit_note'] = 'Record git rev-parse HEAD from the uni2ts checkout.'
    return result


if __name__=='__main__':
    print(json.dumps(capture(),indent=2))
