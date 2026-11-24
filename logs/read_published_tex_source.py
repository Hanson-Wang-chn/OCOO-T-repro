import hashlib,json,os,tarfile
from pathlib import Path
from datetime import datetime,timezone,timedelta
import requests
out=Path('paper/tex_source_v2');out.mkdir(exist_ok=True)
url='https://arxiv.org/src/2606.12838v2';archive=Path('paper/OCOO-T_v2_source.tar.gz');temporary=archive.with_suffix('.download');limit=25<<20
record={'time':datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),'url':url,'purpose':'Inspect publicly published TeX comments/supplementary text for unreported hyperparameters. Read only; do not execute TeX or code. Active training sources/configs unchanged.'}
try:
 if not archive.exists():
  with requests.get(url,stream=True,timeout=(15,40)) as response:
   response.raise_for_status();size=0
   with temporary.open('wb') as stream:
    for chunk in response.iter_content(1<<18):
     size+=len(chunk)
     if size>limit:raise ValueError('Published source exceeds25MiB retrieval cap')
     stream.write(chunk)
    stream.flush();os.fsync(stream.fileno())
  temporary.replace(archive)
 record.update(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
 if not tarfile.is_tarfile(archive):raise ValueError('Published source response is not a readable tar archive; no extraction')
 files=[]
 with tarfile.open(archive) as bundle:
  for member in bundle.getmembers():
   if not member.isfile() or Path(member.name).suffix.lower() not in ['.tex','.bib','.sty','.txt']:continue
   if member.size>5<<20:raise ValueError('Oversized text member')
   path=out/member.name
   if not path.resolve().is_relative_to(out.resolve()):raise ValueError('Source archive member escapes destination')
   data=bundle.extractfile(member).read();path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
   files.append({'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
 record.update(status='downloaded_text_only_extracted',text_members=files)
except Exception as exc:
 record.update(status='retrieval_or_inspection_failed',error=repr(exc))
Path('paper/tex_source_v2_manifest.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
