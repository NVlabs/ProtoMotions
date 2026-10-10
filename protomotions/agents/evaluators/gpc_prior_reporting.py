"""W&B scalar/media reporting, reusable online or as a local offline W&B run."""
import json
from pathlib import Path


def report_media(report, directory):
    directory=Path(directory)
    items={}
    # Files belong to this report's epoch/evaluation, never older runs.
    stem=f'epoch_{report["agent_epoch"]}_eval_{report.get("eval_count",1)}'
    for path in sorted(directory.glob(stem+'_*.png')):
        items['eval_media/'+path.stem]=('image',path)
    for source,splits in report.get('weight_cohorts',{}).items():
        for split,record in splits.items():
            for condition,data in record.get('conditions',{}).items():
                for video in data.get('videos',[]):
                    path=Path(video['path'])
                    if not path.exists(): path=directory/path.name
                    if path.exists(): items[f'rollout_video/{source}/{split}/{condition}/{video["case"]}']=('video',path)
    return items


def log_to_run(run, report, directory, step, *, scalars=True):
    import wandb
    payload={key:float('nan') if value is None else value for key,value in report['metrics'].items()} if scalars else {}
    for key,(kind,path) in report_media(report,directory).items():
        payload[key]=wandb.Image(str(path)) if kind=='image' else wandb.Video(str(path),format='mp4')
    run.log(payload,step=step,commit=False)


def log_training_media(fabric, report, directory, step):
    if fabric.global_rank!=0: return
    for logger in getattr(fabric,'loggers',[]):
        if 'wandb' in type(logger).__module__.lower():
            log_to_run(logger.experiment,report,directory,step,scalars=False)


def main():
    import argparse
    import wandb
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True)
    parser.add_argument('--project',required=True)
    parser.add_argument('--name',default=None)
    parser.add_argument('--mode',choices=['online','offline','disabled'],default='offline')
    args=parser.parse_args()
    path=Path(args.report); report=json.loads(path.read_text())
    with wandb.init(project=args.project,name=args.name,mode=args.mode,dir=str(path.parent),
                    config={k:v for k,v in report.items() if k not in ('metrics','cohorts','weight_cohorts')}) as run:
        log_to_run(run,report,path.parent,report['agent_epoch'])
        artifact=wandb.Artifact('gpc-stage2-evaluation',type='evaluation')
        artifact.add_file(str(path))
        for csv in path.parent.glob(f'epoch_{report["agent_epoch"]}_eval_{report.get("eval_count",1)}*.csv'):
            artifact.add_file(str(csv))
        run.log_artifact(artifact)
        result=dict(mode=args.mode,run_id=run.id,url=run.url)
    path.with_suffix('.wandb.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__': main()
