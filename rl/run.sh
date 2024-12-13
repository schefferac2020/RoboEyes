python ppo_rgb_drew.py --env_id="PlaySoccer-v1" --seed=1 \
    --num_envs=64 --update_epochs=8 --num_minibatches=16 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=1 --num_steps=100 --num_eval_steps=100\
    --exp-name="custom-ppo-rgb-Socker-v1-pred-dist-Dec13_3" \
    --wandb_entity="stonet2000" --track

# python ppo.py --env_id="PickCube-v1" --seed=541 \
#     --num_envs=4 --update_epochs=8 --num_minibatches=8 \
#     --total_timesteps=50_000_000 \
#     --num_eval_envs=1 \
#     --exp-name="ppo-PickCube-v1-state-${541}-walltime_efficient" \
#     --wandb_entity="stonet2000" --track
