"""Headless skeleton replay videos of real simulated trajectories (not camera RGB)."""
from pathlib import Path
import numpy as np


def render_cases(position, episodes, parents, dt, prefix, limit=3, fps=25):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    finite=[i for i,e in enumerate(episodes) if not e['invalid']]
    chosen=[]
    for label,pool in [('success',[i for i in finite if episodes[i]['survival_rate']]),
                       ('failure',[i for i in finite if not episodes[i]['survival_rate']]),
                       ('rare_tokens',sorted(finite,key=lambda i: episodes[i].get('token_rarity_nats') or 0,reverse=True))]:
        if pool: chosen.append((label,pool[0]))
    result=[]
    for label,index in chosen[:limit]:
        pos=position[index].numpy()
        frames=np.unique(np.minimum(np.arange(0,(len(pos)-1)*dt+dt*.01,1/fps)/dt,len(pos)-1).round().astype(int))
        path=Path(str(prefix)+'_'+label+'.mp4')
        fig=plt.figure(figsize=(6.4,4.8),dpi=100)
        ax=fig.add_subplot(111,projection='3d')
        lo=pos.min((0,1)); hi=pos.max((0,1)); center=(lo+hi)/2
        radius=max(float((hi-lo).max()/2),.7)
        with imageio.get_writer(path,fps=fps,codec='libx264',macro_block_size=16) as writer:
            for t in frames:
                ax.clear()
                for child,parent in enumerate(parents):
                    if parent>=0:
                        pair=pos[t,[child,parent]]
                        ax.plot(pair[:,0],pair[:,1],pair[:,2],color='#1675a8',linewidth=2)
                ax.scatter(*pos[t].T,s=8,color='#d85c26')
                ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),
                       zlim=(max(0,center[2]-radius),center[2]+radius),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
                ax.set_box_aspect((1,1,1)); ax.view_init(20,-60)
                ax.set_title(f'Physics trajectory replay | {label}\nmotion {episodes[index]["motion_id"]} | t={t*dt:.2f}s')
                fig.canvas.draw()
                writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
        plt.close(fig)
        result.append(dict(case=label,path=str(path),episode=index,rendering='3D skeleton replay of recorded physics states'))
    return result
