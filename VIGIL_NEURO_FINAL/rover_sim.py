"""Headless differential-drive chassis with a rendered camera and two virtual feelers."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pybullet as pb
from baseline_policy import decide


class RoverSim:
    def __init__(self, scene="open", seed=7, fault="none", fault_start_s=0.2, fault_end_s=0.5, delay_s=0.3):
        self.client = pb.connect(pb.DIRECT)
        self.tick = 0
        self.speed = self.turn = 0.
        self.events = 0
        self.touch = False
        self.fault, self.window, self.delay = fault,(fault_start_s,fault_end_s),delay_s
        self.queue = []
        self.packet = None
        self.last_center = self.growth = 0.
        self.rgb0 = np.zeros((96,128,3),dtype=np.uint8)
        pb.setTimeStep(1/120,physicsClientId=self.client)
        self.rover = self.box([0,0,0.05],[0.10,0.08,0.05],1,[0.1,0.3,0.5,1])
        self.solids = []
        blocks = {"open":[], "wall":[(0.55,0,.22)],"left":[(.55,.07,.028)],
                  "right":[(.55,-.07,.028)],"post":[(.55,0,.02)]}
        if scene not in blocks: raise ValueError(scene)
        for x,y,half in blocks[scene]:
            self.solids.append(self.box([x,y,.12],[.03,half,.12],0,[.86,.17,.09,1]))
        self.box([0,0,-.035],[4,4,.03],0,[.55,.57,.58,1])
        self.tint = np.random.default_rng(seed).uniform(.85,1.)
        self.update_camera()

    def box(self,pos,half,mass,color):
        c=self.client
        shape=pb.createCollisionShape(pb.GEOM_BOX,halfExtents=half,physicsClientId=c)
        visual=pb.createVisualShape(pb.GEOM_BOX,halfExtents=half,rgbaColor=color,physicsClientId=c)
        return pb.createMultiBody(mass,shape,visual,pos,physicsClientId=c)

    def __enter__(self): return self

    def __exit__(self,*args): pb.disconnect(self.client)

    def pose(self):
        pos,q=pb.getBasePositionAndOrientation(self.rover,physicsClientId=self.client)
        return pos,pb.getEulerFromQuaternion(q)[2]

    def whiskers(self):
        (x,y,_),yaw=self.pose()
        c,s=math.cos(yaw),math.sin(yaw)
        def tip(length,side): return [x+length*c-side*s,y+length*s+side*c,.06]
        rays=pb.rayTestBatch([tip(.101,k) for k in (.07,-.07)],[tip(.20,k) for k in (.07,-.07)],physicsClientId=self.client)
        return tuple(r[0] in self.solids for r in rays)

    def render(self):
        (x,y,_),yaw=self.pose()
        c,s=math.cos(yaw),math.sin(yaw)
        eye=[x+.09*c,y+.09*s,.14]
        view=pb.computeViewMatrix(eye,[eye[0]+c,eye[1]+s,.10],[0,0,1])
        proj=pb.computeProjectionMatrixFOV(66,128/96,.015,8)
        rgba=pb.getCameraImage(128,96,view,proj,renderer=pb.ER_TINY_RENDERER,physicsClientId=self.client)[2]
        return (np.asarray(rgba).reshape(96,128,4)[:,:,:3]*self.tint).astype(np.uint8)

    def update_camera(self):
        now=self.tick/20
        fault=self.fault if self.window[0]<=now<self.window[1] else "none"
        if self.tick%2==0 and fault!="drop":
            if fault=="freeze" and self.packet is not None:
                capture,image=self.packet[:2]
            else: capture,image=now,self.render()
            self.queue.append((now+(self.delay if fault=="delay" else 0),capture,image))
        pending=[]
        for delivery,capture,image in self.queue:
            if delivery>now+1e-9:
                pending.append((delivery,capture,image))
            elif self.packet is None or capture>=self.packet[0]:
                center=image[:,44:84].astype(float)
                fraction=float(np.mean(center[:,:,0]>1.7*np.maximum(1,center[:,:,1])))
                self.growth=fraction-self.last_center
                self.last_center=fraction
                self.packet=(capture,image,now)
        self.queue=pending

    def observe(self):
        now=self.tick/20
        capture,image,delivery=self.packet or (None,self.rgb0,None)
        left,right=self.whiskers()
        return dict(time_s=now,rgb=image.copy(),capture_s=capture,delivery_s=delivery,frame_valid=capture is not None,
                    frame_age_s=now-capture if capture is not None else None,whisker_left=left,whisker_right=right,
                    brightness=float(image.mean()/255),contrast=float(image.std()/255),
                    obstacle_center=self.last_center,looming_center=self.growth)

    def step(self,action):
        targets={"forward":(.35,0),"slow":(.12,0),"brake":(0,0),"left":(0,1.),"right":(0,-1.)}
        target,turn=targets[action]
        if any(self.whiskers()): target,turn=0.,0.
        for _ in range(6):
            self.speed+=float(np.clip(target-self.speed,-.8/120,.8/120))
            _,yaw=self.pose()
            pb.resetBaseVelocity(self.rover,[self.speed*math.cos(yaw),self.speed*math.sin(yaw),0],[0,0,turn],physicsClientId=self.client)
            pb.stepSimulation(physicsClientId=self.client)
            touching=any(pb.getContactPoints(self.rover,o,physicsClientId=self.client) for o in self.solids)
            self.events+=int(touching and not self.touch)
            self.touch=touching
        self.tick+=1
        self.update_camera()
        return self.observe()

    def evaluator(self):
        (x,y,_),yaw=self.pose()
        velocity,_=pb.getBaseVelocity(self.rover,physicsClientId=self.client)
        return dict(x_m=x,y_m=y,yaw_rad=yaw,speed_m_s=math.hypot(*velocity[:2]),body_contact=self.touch,body_contact_events=self.events)


def run_episode(out,steps=80,scene="wall",seed=7,fault="none"):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    frames=[];captures=[];deliveries=[]
    with RoverSim(scene,seed,fault) as sim,(out/"replay.jsonl").open("w",encoding="utf-8") as log:
        obs=sim.observe()
        for tick in range(steps+1):
            image=obs.pop("rgb")
            if obs["frame_valid"] and (not deliveries or deliveries[-1]!=obs["delivery_s"]):
                frames.append(image);captures.append(obs["capture_s"]);deliveries.append(obs["delivery_s"])
            obs["frame_index"]=len(frames)-1 if obs["frame_valid"] else None
            action=decide(obs)[0] if tick<steps else None
            log.write(json.dumps(dict(sensors=obs,action=action,state=sim.evaluator()))+"\n")
            if action: obs=sim.step(action)
        summary=dict(scene=scene,seed=seed,fault=fault,steps=steps,delivered_frames=len(frames),final=sim.evaluator())
    np.savez_compressed(out/"frames.npz",rgb=frames,capture_s=captures,delivery_s=deliveries)
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True);parser.add_argument("--steps",type=int,default=80)
    parser.add_argument("--scene",choices=["wall","open","left","right","post"],default="wall")
    parser.add_argument("--fault",choices=["none","drop","freeze","delay"],default="none")
    args=parser.parse_args();print(run_episode(**vars(args)))
