import argparse, numpy as np, torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from src.algos.fep_ppo import FEP_PPO
from src.models.cnn_actor_critic import CNNActorCritic
from src.models.policy_prior import PolicyPrior
from src.storage import RolloutStorage
from src.model_G import WorldModelG

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--env-id",default="CarRacing-v2")
    p.add_argument("--n-envs",type=int,default=4)
    p.add_argument("--total-steps",type=int,default=200_000)
    p.add_argument("--rollout-steps",type=int,default=512)
    p.add_argument("--frame-stack",type=int,default=4)
    p.add_argument("--resize",type=int,nargs=2,default=[84,84])
    p.add_argument("--gray",action="store_true")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--efe-kl-coef",type=float,default=0.5)
    p.add_argument("--intrinsic-beta",type=float,default=0.05)
    return p.parse_args()

def main():
    args=parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    print("Device:",device)

    vec_env=make_vec_env(args.env_id,n_envs=args.n_envs)
    vec_env=VecFrameStack(vec_env,n_stack=args.frame_stack)
    obs_space=vec_env.observation_space
    act_space=vec_env.action_space
    c,h,w=obs_space.shape
    action_dim=act_space.shape[0]

    ac=CNNActorCritic((c,h,w),action_dim).to(device)
    prior=PolicyPrior((c,h,w),action_dim).to(device)
    algo=FEP_PPO(ac,prior,prior_coef=args.efe_kl_coef,lr=3e-4,
                 clip_param=0.2,ppo_epoch=4,num_mini_batch=4,
                 value_loss_coef=0.5,entropy_coef=0.01,max_grad_norm=0.5)

    WM=WorldModelG((c,h,w),action_dim,device=device).to(device)
    wm_opt=torch.optim.Adam(WM.parameters(),lr=2e-4)

    storage=RolloutStorage(args.rollout_steps,args.n_envs,(c,h,w),action_dim,device=device)
    obs=vec_env.reset()
    storage.obs[0].copy_(torch.as_tensor(obs,dtype=torch.uint8))
    h_state=WM.init_state(args.n_envs)

    num_updates=args.total_steps//(args.rollout_steps*args.n_envs)
    for update in range(1,num_updates+1):
        for step in range(args.rollout_steps):
            with torch.no_grad():
                obs_f=torch.as_tensor(obs,dtype=torch.float32,device=device)
                values,actions,logp,_=ac.act(obs_f,
                    torch.zeros(args.n_envs,1,device=device),
                    torch.ones(args.n_envs,device=device))
            next_obs,rewards,dones,infos=vec_env.step(actions.cpu().numpy())
            # world model intrinsic reward
            z_t=WM.encode(torch.as_tensor(obs,dtype=torch.uint8,device=device))
            z_tp1=WM.encode(torch.as_tensor(next_obs,dtype=torch.uint8,device=device))
            a_t=torch.as_tensor(actions,dtype=torch.float32,device=device)
            nll,_=WM.loss_step(z_t,a_t,h_state,z_tp1)
            intr=args.intrinsic_beta*float(nll.item())
            rewards+=intr
            rewards_t=torch.as_tensor(rewards,dtype=torch.float32,device=device)
            masks=torch.as_tensor(1.0-dones.astype(np.float32),device=device)
            storage.insert(torch.as_tensor(next_obs,dtype=torch.uint8),torch.zeros(args.n_envs,1,device=device),
                           actions,logp,values,rewards_t,masks)
            obs=next_obs
        with torch.no_grad():
            obs_f=torch.as_tensor(obs,dtype=torch.float32,device=device)
            _,_,next_value=ac.forward(obs_f)
        storage.compute_returns(next_value,gamma=0.99,gae_lambda=0.95)
        out=algo.update(storage); storage.after_update()
        # world model update
        wm_opt.zero_grad(); nll.backward(); wm_opt.step()
        print(f"Update {update}: loss={out}")
    vec_env.close()

if __name__=="__main__":
    main()

